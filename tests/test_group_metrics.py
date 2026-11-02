import numpy as np
import pandas as pd

from fairness_audit.metrics import calculate_group_metrics


def test_group_report_has_rates_and_support():
    truth = np.array([1, 0, 1, 0])
    predicted = np.array([1, 1, 0, 0])
    groups = pd.Series(["A", "A", "B", "B"])

    rows, differences = calculate_group_metrics(
        truth,
        predicted,
        groups,
        group_type="sex",
        model="primary",
        policy="global",
        bootstrap=True,
    )

    first = next(row for row in rows if row["group_value"] == "A")
    assert first["support"] == 2
    assert first["selection_rate"] == 1.0
    assert first["true_positive_rate"] == 1.0
    assert first["selection_rate_ci_low"] <= first["selection_rate_ci_high"]
    assert {row["metric"] for row in differences} >= {
        "selection_rate",
        "true_positive_rate",
        "false_positive_rate",
    }
