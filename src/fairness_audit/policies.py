"""Deterministic threshold selection using validation rows only."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score


def choose_global_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], scores)))
    ranked = []
    for threshold in candidates:
        predictions = (scores >= threshold).astype("int8")
        score = balanced_accuracy_score(y_true, predictions)
        ranked.append((float(score), -abs(float(threshold) - 0.5), float(threshold)))
    return max(ranked)[2]


def _tpr(y_true: np.ndarray, predictions: np.ndarray) -> float:
    positives = y_true == 1
    if not positives.any():
        return float("nan")
    return float(np.mean(predictions[positives] == 1))


def _threshold_for_minimum_tpr(
    y_true: np.ndarray, scores: np.ndarray, target_tpr: float
) -> float:
    candidates = np.unique(
        np.concatenate((scores, [np.nextafter(float(np.max(scores)), np.inf)]))
    )
    eligible = [
        float(threshold)
        for threshold in candidates
        if _tpr(y_true, (scores >= threshold).astype("int8")) + 1e-12 >= target_tpr
    ]
    if not eligible:
        raise ValueError("No score threshold can attain the validation TPR target")
    return max(eligible)


def choose_group_thresholds(
    y_true: np.ndarray,
    scores: np.ndarray,
    groups: pd.Series | np.ndarray,
    global_threshold: float,
) -> tuple[dict[str, float], float]:
    group_values = np.asarray(groups, dtype=str)
    global_predictions = (scores >= global_threshold).astype("int8")
    group_tprs = []
    for group in sorted(np.unique(group_values)):
        mask = group_values == group
        value = _tpr(y_true[mask], global_predictions[mask])
        if np.isnan(value):
            raise ValueError(f"Validation group {group!r} has no positive labels")
        group_tprs.append(value)
    target_tpr = min(group_tprs)
    thresholds = {}
    for group in sorted(np.unique(group_values)):
        mask = group_values == group
        thresholds[group] = _threshold_for_minimum_tpr(
            y_true[mask], scores[mask], target_tpr
        )
    return thresholds, target_tpr


def apply_policy(
    scores: np.ndarray,
    groups: pd.Series | np.ndarray | None,
    *,
    global_threshold: float,
    group_thresholds: dict[str, float] | None = None,
) -> np.ndarray:
    if group_thresholds is None:
        return (scores >= global_threshold).astype("int8")
    if groups is None:
        raise ValueError("Group labels are required for a group-threshold policy")
    group_values = np.asarray(groups, dtype=str)
    missing = sorted(set(group_values) - set(group_thresholds))
    if missing:
        raise ValueError(f"No validation threshold exists for groups: {missing}")
    thresholds = np.array([group_thresholds[value] for value in group_values])
    return (scores >= thresholds).astype("int8")
