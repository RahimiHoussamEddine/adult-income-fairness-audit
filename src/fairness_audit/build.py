"""Run the pinned Adult source through the full audit and write aggregate outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from fairness_audit.constants import (
    PROCESSED_DIR,
    RANDOM_STATE,
    RAW_DIR,
    REPORTS_DIR,
    TARGET_COLUMN,
    VALIDATION_SIZE,
)
from fairness_audit.data import (
    data_quality_report,
    parse_archive,
    write_source_manifest,
)
from fairness_audit.metrics import calculate_group_metrics
from fairness_audit.modeling import build_model, feature_columns, prepare_features
from fairness_audit.policies import (
    apply_policy,
    choose_global_threshold,
    choose_group_thresholds,
)


def _split_training_rows(training: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    strata = (
        training[TARGET_COLUMN].astype("string")
        + "|"
        + training["race"].fillna("Unknown")
        + "|"
        + training["sex"].fillna("Unknown")
    )
    train_indices, validation_indices = train_test_split(
        np.arange(len(training)),
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=strata,
    )
    return train_indices, validation_indices


def _intersection_groups(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["race"].fillna("Unknown").astype("string")
        + " | "
        + frame["sex"].fillna("Unknown").astype("string")
    ).rename("race_sex")


def _group_views(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "race": frame["race"].fillna("Unknown").astype("string").rename("race"),
        "sex": frame["sex"].fillna("Unknown").astype("string").rename("sex"),
        "race_sex": _intersection_groups(frame),
    }


def _policy_threshold_rows(
    model_name: str,
    policy: str,
    global_threshold: float,
    group_thresholds: dict[str, float] | None,
    group_dimension: str | None,
    target_tpr: float | None,
) -> list[dict[str, object]]:
    if group_thresholds is None:
        return [
            {
                "model": model_name,
                "policy": policy,
                "group_dimension": "global",
                "group_value": "all",
                "threshold": global_threshold,
                "global_validation_threshold": global_threshold,
                "validation_tpr_target": None,
            }
        ]
    assert group_dimension is not None
    return [
        {
            "model": model_name,
            "policy": policy,
            "group_dimension": group_dimension,
            "group_value": group,
            "threshold": threshold,
            "global_validation_threshold": global_threshold,
            "validation_tpr_target": target_tpr,
        }
        for group, threshold in sorted(group_thresholds.items())
    ]


def _prediction_metrics(
    truth: np.ndarray, probabilities: np.ndarray, predictions: np.ndarray
) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(truth, probabilities)),
        "average_precision": float(average_precision_score(truth, probabilities)),
        "brier_score": float(brier_score_loss(truth, probabilities)),
        "log_loss": float(log_loss(truth, probabilities, labels=[0, 1])),
        "accuracy": float(accuracy_score(truth, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "precision": float(precision_score(truth, predictions, zero_division=0)),
        "recall": float(recall_score(truth, predictions, zero_division=0)),
        "selection_rate": float(np.mean(predictions == 1)),
    }


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(value) for value in frame.columns]
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False)]
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    header = (
        "| "
        + " | ".join(value.ljust(widths[index]) for index, value in enumerate(headers))
        + " |"
    )
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [
        "| "
        + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _markdown_report(
    model_metrics: pd.DataFrame,
    group_metrics: pd.DataFrame,
    quality: dict[str, Any],
    thresholds: pd.DataFrame,
) -> str:
    headline_values = model_metrics[
        ["model", "policy", "roc_auc", "average_precision", "balanced_accuracy"]
    ].copy()
    for column in ("roc_auc", "average_precision", "balanced_accuracy"):
        headline_values[column] = headline_values[column].map(
            lambda value: f"{value:.3f}"
        )
    headline = _markdown_table(headline_values)
    primary_groups = group_metrics[
        (group_metrics["policy"] == "global") & (group_metrics["model"] == "primary")
    ]
    disparity_values = primary_groups.groupby("group_type", as_index=False).agg(
        selection_rate_gap=(
            "selection_rate",
            lambda values: values.max() - values.min(),
        ),
        tpr_gap=("true_positive_rate", lambda values: values.max() - values.min()),
        fpr_gap=("false_positive_rate", lambda values: values.max() - values.min()),
    )
    for column in ("selection_rate_gap", "tpr_gap", "fpr_gap"):
        disparity_values[column] = disparity_values[column].map(
            lambda value: f"{value:.3f}"
        )
    disparities = _markdown_table(disparity_values)
    primary_thresholds = thresholds[thresholds["model"] == "primary"]
    global_threshold = float(
        primary_thresholds.loc[
            primary_thresholds["policy"] == "global", "threshold"
        ].iloc[0]
    )
    row_counts = quality["rows"]
    duplicates = quality["duplicate_rows"]
    return f"""# Evaluation results

## Frozen source split and row counts

- Official source training file: {row_counts["adult.data"]:,} rows.
- Official source test file: {row_counts["adult.test"]:,} rows
  (evaluated after selection).
- The official test outcomes were not used for model fitting or threshold selection.
- The primary global validation threshold was {global_threshold:.6f}.
- Raw archive SHA-256: {quality["source"]["sha256"]}.

## Test-set utility

ROC-AUC and average precision are probability metrics. Balanced accuracy and
the other threshold metrics depend on the validation-selected policy. Brier
score and log loss summarize probabilistic prediction error; they do not
establish calibration by themselves.

{headline}

## Primary model group gaps under the global threshold

The gap is max minus min across groups within each listed audit dimension.
Group-level rates and bootstrap intervals appear in group_metrics.csv;
bootstrap intervals for max-minus-min differences appear in
group_differences.csv.

{disparities}

## Duplicate feature profiles

The source files contain {duplicates["adult.data_features"]} duplicate feature rows
in adult.data and {duplicates["adult.test_features"]} in adult.test. The files
also share {duplicates["unique_feature_fingerprints_shared_across_official_files"]}
unique feature profiles. These official rows are preserved; overlapping
profiles may make the test estimate somewhat optimistic. Counts ignore the
income target.

## Validation-selected policy thresholds

Equal-opportunity-style policies use the lowest validation true-positive rate
under the global threshold as their target. Group-specific thresholds are
selected on validation rows only. They can change error and selection rates
and require group membership at decision time; these diagnostic policies are
not deployment recommendations.

See threshold_policies.csv for every selected threshold and validation
target. The proxy sensitivity excludes marital-status and relationship.

## Limits

This is an educational audit of a historical U.S. benchmark. Its income label
is not a measure of merit or deservingness. Race and sex are excluded from
model features but used for group auditing; correlated proxies remain.
Metrics are unweighted and do not reproduce the Census survey design.
Uncertainty intervals describe resampling variation within this benchmark
split, not source bias or performance in a present-day population. See
docs/METHODOLOGY.md for the full scope and metric definitions.
"""


def run_build(
    archive_path: Path | None = None,
    *,
    bootstrap: bool = True,
) -> dict[str, Any]:
    archive_path = archive_path or RAW_DIR / "adult.zip"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    training, test = parse_archive(archive_path)
    manifest = write_source_manifest(archive_path, training, test)
    quality = data_quality_report(training, test, archive_path)
    train_indices, validation_indices = _split_training_rows(training)
    train = training.iloc[train_indices].reset_index(drop=True)
    validation = training.iloc[validation_indices].reset_index(drop=True)
    final_test = test.reset_index(drop=True)

    model_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    difference_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    model_specs = [
        ("primary", False),
        ("no_marital_relationship", True),
    ]

    for model_name, remove_proxy_fields in model_specs:
        columns = feature_columns(
            train.columns, remove_proxy_fields=remove_proxy_fields
        )
        model = build_model(columns)
        x_train = prepare_features(train, columns)
        x_validation = prepare_features(validation, columns)
        x_test = prepare_features(final_test, columns)
        y_train = train[TARGET_COLUMN].to_numpy(dtype="int8")
        y_validation = validation[TARGET_COLUMN].to_numpy(dtype="int8")
        model.fit(x_train, y_train)
        validation_scores = model.predict_proba(x_validation)[:, 1]
        global_threshold = choose_global_threshold(y_validation, validation_scores)

        validation_views = _group_views(validation)
        test_views = _group_views(final_test)
        policy_specs: list[
            tuple[str, dict[str, float] | None, str | None, float | None]
        ] = [("global", None, None, None)]
        for dimension in ("sex", "race"):
            thresholds, target_tpr = choose_group_thresholds(
                y_validation,
                validation_scores,
                validation_views[dimension],
                global_threshold,
            )
            policy_name = f"equal_opportunity_by_{dimension}"
            policy_specs.append((policy_name, thresholds, dimension, target_tpr))

        # Score the locked official test only after every threshold is fixed.
        y_test = final_test[TARGET_COLUMN].to_numpy(dtype="int8")
        test_scores = model.predict_proba(x_test)[:, 1]
        for policy, thresholds, dimension, target_tpr in policy_specs:
            protected = test_views[dimension] if dimension else None
            predictions = apply_policy(
                test_scores,
                protected,
                global_threshold=global_threshold,
                group_thresholds=thresholds,
            )
            metrics = _prediction_metrics(y_test, test_scores, predictions)
            model_rows.append(
                {
                    "model": model_name,
                    "policy": policy,
                    "n_test": len(final_test),
                    "n_features": len(columns),
                    "global_validation_threshold": global_threshold,
                    "validation_tpr_target": (
                        target_tpr if target_tpr is not None else None
                    ),
                    **metrics,
                }
            )
            threshold_rows.extend(
                _policy_threshold_rows(
                    model_name,
                    policy,
                    global_threshold,
                    thresholds,
                    dimension,
                    target_tpr,
                )
            )
            for group_type, groups in test_views.items():
                rows, differences = calculate_group_metrics(
                    y_test,
                    predictions,
                    groups,
                    group_type=group_type,
                    model=model_name,
                    policy=policy,
                    bootstrap=(
                        bootstrap and model_name == "primary" and policy == "global"
                    ),
                )
                group_rows.extend(rows)
                difference_rows.extend(differences)

    model_metrics = pd.DataFrame(model_rows)
    group_metrics = pd.DataFrame(group_rows)
    group_differences = pd.DataFrame(difference_rows)
    threshold_policies = pd.DataFrame(threshold_rows)
    model_metrics.to_csv(REPORTS_DIR / "model_metrics.csv", index=False)
    group_metrics.to_csv(REPORTS_DIR / "group_metrics.csv", index=False)
    group_differences.to_csv(REPORTS_DIR / "group_differences.csv", index=False)
    threshold_policies.to_csv(REPORTS_DIR / "threshold_policies.csv", index=False)
    (REPORTS_DIR / "data_quality.json").write_text(
        json.dumps(quality, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    primary_records = (
        model_metrics[model_metrics["model"] == "primary"]
        .astype(object)
        .where(pd.notna(model_metrics[model_metrics["model"] == "primary"]), None)
        .to_dict(orient="records")
    )
    summary = {
        "project": "Adult Income Fairness and Threshold Policy Audit",
        "source": manifest,
        "split": {
            "adult_data_rows": len(training),
            "model_train_rows": len(train),
            "policy_validation_rows": len(validation),
            "locked_official_test_rows": len(final_test),
            "random_state": RANDOM_STATE,
            "validation_size": VALIDATION_SIZE,
            "test_used_for_selection": False,
        },
        "uncertainty": {
            "method": "nonparametric person-level bootstrap with replacement",
            "replicates": 2_000,
            "ci_quantiles": [0.025, 0.975],
            "random_state": RANDOM_STATE,
        },
        "primary_model": primary_records,
        "interpretation": (
            "Historical benchmark results only; not a hiring, lending, benefits, "
            "or individual decision tool."
        ),
    }
    (REPORTS_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    (REPORTS_DIR / "RESULTS.md").write_text(
        _markdown_report(model_metrics, group_metrics, quality, threshold_policies),
        encoding="utf-8",
    )
    return {
        "models": model_metrics,
        "groups": group_metrics,
        "differences": group_differences,
        "thresholds": threshold_policies,
        "quality": quality,
    }


if __name__ == "__main__":
    output = run_build()
    print(output["models"].to_string(index=False))
