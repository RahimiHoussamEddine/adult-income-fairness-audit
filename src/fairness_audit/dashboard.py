"""Streamlit report dashboard for aggregate results."""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from fairness_audit.constants import REPORTS_DIR


def main() -> None:
    st.set_page_config(
        page_title="Adult Income Fairness Audit",
        page_icon="⚖️",
        layout="wide",
    )
    st.title("Adult Income Fairness & Threshold Policy Audit")
    st.caption(
        "Historical benchmark audit only. The label is not a measure of merit, "
        "and these results are not suitable for individual decisions."
    )

    summary_path = REPORTS_DIR / "summary.json"
    group_path = REPORTS_DIR / "group_metrics.csv"
    model_path = REPORTS_DIR / "model_metrics.csv"
    if not all(path.is_file() for path in (summary_path, group_path, model_path)):
        st.warning("Build the aggregate reports first with make build.")
        return

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    model_metrics = pd.read_csv(model_path)
    group_metrics = pd.read_csv(group_path)
    source_split = summary["split"]
    st.caption(
        f"Official training rows: {source_split['adult_data_rows']:,} · "
        f"fit rows: {source_split['model_train_rows']:,} · "
        f"validation rows: {source_split['policy_validation_rows']:,} · "
        f"locked test rows: {source_split['locked_official_test_rows']:,}"
    )

    model = st.selectbox("Model", sorted(model_metrics["model"].unique()))
    visible_metrics = model_metrics[model_metrics["model"] == model]
    st.subheader("Utility by threshold policy")
    st.dataframe(
        visible_metrics[
            [
                "policy",
                "roc_auc",
                "average_precision",
                "brier_score",
                "balanced_accuracy",
                "precision",
                "recall",
                "selection_rate",
            ]
        ].round(3),
        hide_index=True,
        width="stretch",
    )

    group_type = st.selectbox(
        "Audit dimension",
        ["race", "sex", "race_sex"],
        format_func=lambda value: {
            "race": "Race",
            "sex": "Sex",
            "race_sex": "Race × sex",
        }[value],
    )
    policy = st.selectbox(
        "Threshold policy",
        sorted(visible_metrics["policy"].unique()),
    )
    selected = group_metrics[
        (group_metrics["model"] == model)
        & (group_metrics["policy"] == policy)
        & (group_metrics["group_type"] == group_type)
    ].copy()
    if selected.empty:
        st.info("No aggregate groups are available for this selection.")
        return
    st.subheader("Subgroup selection and error rates")
    chart_data = selected.melt(
        id_vars=["group_value", "support"],
        value_vars=[
            "selection_rate",
            "true_positive_rate",
            "false_positive_rate",
        ],
        var_name="metric",
        value_name="rate",
    )
    chart = px.bar(
        chart_data,
        x="group_value",
        y="rate",
        color="metric",
        barmode="group",
        range_y=[0, 1],
        labels={
            "group_value": "Audit group",
            "rate": "Rate",
            "metric": "Metric",
        },
        hover_data={"support": True, "rate": ":.3f"},
    )
    chart.update_layout(legend_title_text="", xaxis_title="")
    st.plotly_chart(chart, width="stretch")

    displayed = selected[
        [
            "group_value",
            "support",
            "selection_rate",
            "selection_rate_ci_low",
            "selection_rate_ci_high",
            "true_positive_rate",
            "true_positive_rate_ci_low",
            "true_positive_rate_ci_high",
            "false_positive_rate",
            "false_positive_rate_ci_low",
            "false_positive_rate_ci_high",
            "precision",
            "accuracy",
        ]
    ].sort_values("group_value")
    st.dataframe(displayed.round(3), hide_index=True, width="stretch")
    st.caption(
        "Intervals are shown for the primary model's global policy. Other rows "
        "have blank intervals. Overlap is not a test of equality."
    )

    quality_path = REPORTS_DIR / "data_quality.json"
    if quality_path.is_file():
        with st.expander("Data quality and source provenance"):
            st.json(json.loads(quality_path.read_text(encoding="utf-8")))
    st.markdown(
        "See the repository methodology for split rules, metric definitions, "
        "and limits. Group-specific thresholds are diagnostic and require "
        "protected-group membership at decision time."
    )


if __name__ == "__main__":
    main()
