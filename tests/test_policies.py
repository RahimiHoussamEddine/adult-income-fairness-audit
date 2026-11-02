import numpy as np
import pandas as pd
import pytest

from fairness_audit.policies import (
    apply_policy,
    choose_global_threshold,
    choose_group_thresholds,
)


def test_global_threshold_is_deterministic_and_selects_a_candidate():
    truth = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9])

    first = choose_global_threshold(truth, scores)
    second = choose_global_threshold(truth, scores)

    assert first == second
    assert first in set(scores) | {0.0, 0.5, 1.0}


def test_group_thresholds_reach_validation_reference_tpr():
    truth = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    scores = np.array([0.9, 0.7, 0.8, 0.4, 0.5, 0.3, 0.9, 0.7])
    groups = pd.Series(["A"] * 4 + ["B"] * 4)

    thresholds, target = choose_group_thresholds(truth, scores, groups, 0.5)
    predictions = apply_policy(
        scores,
        groups,
        global_threshold=0.5,
        group_thresholds=thresholds,
    )

    assert target == 1.0
    assert thresholds == {"A": 0.7, "B": 0.5}
    for group in ("A", "B"):
        mask = groups.to_numpy() == group
        tpr = predictions[mask & (truth == 1)].mean()
        assert tpr >= target


def test_group_policy_fails_for_unseen_groups():
    with pytest.raises(ValueError, match="No validation threshold"):
        apply_policy(
            np.array([0.7]),
            np.array(["new group"]),
            global_threshold=0.5,
            group_thresholds={"known group": 0.6},
        )
