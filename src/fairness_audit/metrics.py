"""Fairlearn group metrics and deterministic person-level bootstrap intervals."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame
from sklearn.metrics import accuracy_score, precision_score

from fairness_audit.constants import BOOTSTRAP_REPLICATES, CI_QUANTILES, RANDOM_STATE


def _selection_rate(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    del y_true
    return float(np.mean(np.asarray(y_pred) == 1))


def _true_positive_rate(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    truth = np.asarray(y_true)
    predictions = np.asarray(y_pred)
    positives = truth == 1
    return (
        float(np.mean(predictions[positives] == 1)) if positives.any() else float("nan")
    )


def _false_positive_rate(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    truth = np.asarray(y_true)
    predictions = np.asarray(y_pred)
    negatives = truth == 0
    return (
        float(np.mean(predictions[negatives] == 1)) if negatives.any() else float("nan")
    )


def _precision(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    return float(precision_score(y_true, y_pred, zero_division=0))


def _accuracy(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    return float(accuracy_score(y_true, y_pred))


def _support(y_true: Iterable[int], y_pred: Iterable[int]) -> int:
    del y_pred
    return int(len(y_true))


GROUP_METRICS = {
    "selection_rate": _selection_rate,
    "true_positive_rate": _true_positive_rate,
    "false_positive_rate": _false_positive_rate,
    "precision": _precision,
    "accuracy": _accuracy,
    "support": _support,
}


def _group_label(value: object) -> str:
    if isinstance(value, tuple):
        return " | ".join(str(part) for part in value)
    return str(value)


def _bootstrap_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: pd.Series,
    group_labels: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Bootstrap group rates with vectorized confusion-count calculations."""
    truth = np.asarray(y_true, dtype="int8")
    predicted = np.asarray(y_pred, dtype="int8")
    values = groups.astype(str).to_numpy()
    sorted_groups = np.asarray(group_labels, dtype=str)
    group_codes = pd.Categorical(values, categories=sorted_groups).codes
    if np.any(group_codes < 0):
        raise ValueError("A group value could not be assigned a bootstrap code")

    metric_names = list(GROUP_METRICS)
    samples = {
        metric: np.full((BOOTSTRAP_REPLICATES, len(sorted_groups)), np.nan)
        for metric in metric_names
    }
    gaps = {metric: np.full(BOOTSTRAP_REPLICATES, np.nan) for metric in metric_names}
    rng = np.random.default_rng(RANDOM_STATE)
    n_rows = len(truth)
    n_groups = len(sorted_groups)

    for replicate in range(BOOTSTRAP_REPLICATES):
        draw = rng.integers(0, n_rows, size=n_rows)
        sampled_truth = truth[draw]
        sampled_prediction = predicted[draw]
        sampled_groups = group_codes[draw]

        support = np.bincount(sampled_groups, minlength=n_groups).astype(float)
        positives = np.bincount(
            sampled_groups, weights=(sampled_truth == 1), minlength=n_groups
        )
        negatives = support - positives
        selected = np.bincount(
            sampled_groups, weights=(sampled_prediction == 1), minlength=n_groups
        )
        true_positives = np.bincount(
            sampled_groups,
            weights=((sampled_truth == 1) & (sampled_prediction == 1)),
            minlength=n_groups,
        )
        false_positives = np.bincount(
            sampled_groups,
            weights=((sampled_truth == 0) & (sampled_prediction == 1)),
            minlength=n_groups,
        )
        true_negatives = negatives - false_positives

        with np.errstate(divide="ignore", invalid="ignore"):
            precision_rates = np.divide(
                true_positives,
                selected,
                out=np.zeros(n_groups, dtype=float),
                where=selected > 0,
            )
            current = {
                "selection_rate": selected / support,
                "true_positive_rate": true_positives / positives,
                "false_positive_rate": false_positives / negatives,
                "precision": precision_rates,
                "accuracy": (true_positives + true_negatives) / support,
                "support": support,
            }
        for metric, rates in current.items():
            samples[metric][replicate] = rates
            if np.isfinite(rates).any():
                gaps[metric][replicate] = np.nanmax(rates) - np.nanmin(rates)

    low_values = {
        metric: np.nanquantile(values, CI_QUANTILES[0], axis=0)
        for metric, values in samples.items()
    }
    high_values = {
        metric: np.nanquantile(values, CI_QUANTILES[1], axis=0)
        for metric, values in samples.items()
    }
    index = pd.Index(sorted_groups, name=groups.name or "group")
    low = pd.DataFrame(low_values, index=index)
    high = pd.DataFrame(high_values, index=index)
    difference_low = pd.Series(
        {
            metric: float(np.nanquantile(values, CI_QUANTILES[0]))
            for metric, values in gaps.items()
        }
    )
    difference_high = pd.Series(
        {
            metric: float(np.nanquantile(values, CI_QUANTILES[1]))
            for metric, values in gaps.items()
        }
    )
    return low, high, difference_low, difference_high


def calculate_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: pd.Series,
    *,
    group_type: str,
    model: str,
    policy: str,
    bootstrap: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    frame = MetricFrame(
        metrics=GROUP_METRICS,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=groups.reset_index(drop=True).rename(group_type),
    )
    points = frame.by_group
    if bootstrap:
        group_labels = [_group_label(value) for value in points.index]
        low, high, difference_low, difference_high = _bootstrap_intervals(
            y_true,
            y_pred,
            groups.reset_index(drop=True),
            group_labels,
        )
        intervals = [low, high]
    else:
        intervals = None
    rows: list[dict[str, object]] = []
    for group_value, values in points.iterrows():
        label = _group_label(group_value)
        row: dict[str, object] = {
            "model": model,
            "policy": policy,
            "group_type": group_type,
            "group_value": label,
        }
        for metric, value in values.items():
            row[metric] = float(value) if metric != "support" else int(value)
            if intervals:
                row[f"{metric}_ci_low"] = float(intervals[0].loc[group_value, metric])
                row[f"{metric}_ci_high"] = float(intervals[1].loc[group_value, metric])
            else:
                row[f"{metric}_ci_low"] = np.nan
                row[f"{metric}_ci_high"] = np.nan
        rows.append(row)

    differences = frame.difference()
    difference_rows: list[dict[str, object]] = []
    for metric, value in differences.items():
        difference_rows.append(
            {
                "model": model,
                "policy": policy,
                "group_type": group_type,
                "metric": metric,
                "max_minus_min": float(value),
                "ci_low": (float(difference_low.loc[metric]) if bootstrap else np.nan),
                "ci_high": (
                    float(difference_high.loc[metric]) if bootstrap else np.nan
                ),
            }
        )
    return rows, difference_rows
