"""Check that checked-in result artifacts are aggregate and internally consistent."""

from __future__ import annotations

import json
import re
from importlib.metadata import distributions
from pathlib import Path

import pandas as pd
from packaging.utils import canonicalize_name

from fairness_audit.constants import RAW_DIR, REPORTS_DIR, ROOT, SOURCE_SHA256
from fairness_audit.data import sha256_file

REQUIRED_REPORTS = [
    "RESULTS.md",
    "data_quality.json",
    "group_differences.csv",
    "group_metrics.csv",
    "model_metrics.csv",
    "summary.json",
    "threshold_policies.csv",
]


def verify_lock(lock_path: Path = ROOT / "requirements.lock") -> int:
    if not lock_path.is_file():
        raise FileNotFoundError(f"Dependency lock is missing: {lock_path}")
    expected: dict[str, str] = {}
    for line in lock_path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+-]+)", entry)
        if not match:
            raise ValueError(f"Lock entry is not an exact version pin: {entry}")
        expected[canonicalize_name(match.group(1))] = match.group(2)

    installed = {
        canonicalize_name(dist.metadata["Name"]): dist.version
        for dist in distributions()
    }
    mismatches = {
        name: {"expected": version, "installed": installed.get(name)}
        for name, version in expected.items()
        if installed.get(name) != version
    }
    if mismatches:
        raise ValueError(
            f"Installed packages do not match requirements.lock: {mismatches}"
        )
    return len(expected)


def verify_reports(reports_dir: Path = REPORTS_DIR) -> dict[str, int | bool]:
    missing = [name for name in REQUIRED_REPORTS if not (reports_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Required report files are missing: {missing}")
    groups = pd.read_csv(reports_dir / "group_metrics.csv")
    models = pd.read_csv(reports_dir / "model_metrics.csv")
    thresholds = pd.read_csv(reports_dir / "threshold_policies.csv")
    summary = json.loads((reports_dir / "summary.json").read_text(encoding="utf-8"))
    manifest_path = RAW_DIR.parent / "source_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("data/source_manifest.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    expected_group_columns = {
        "model",
        "policy",
        "group_type",
        "group_value",
        "selection_rate",
        "true_positive_rate",
        "false_positive_rate",
        "support",
    }
    if not expected_group_columns.issubset(groups.columns):
        raise ValueError("Aggregate group report is missing required metrics")
    prohibited_columns = {
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
        "prediction",
        "probability",
    }
    if prohibited_columns & set(groups.columns):
        raise ValueError("Person-level data leaked into the aggregate group report")
    if groups.duplicated(["model", "policy", "group_type", "group_value"]).any():
        raise ValueError("Aggregate group report has duplicate group keys")
    if set(groups["group_type"]) != {"race", "sex", "race_sex"}:
        raise ValueError("Expected race, sex, and intersectional audit dimensions")
    if len(models) != 6 or len(thresholds) < 6:
        raise ValueError("Expected two model variants with three policies each")
    if summary["split"]["test_used_for_selection"] is not False:
        raise ValueError("The locked test set must not be used for selection")
    if summary["uncertainty"]["replicates"] != 2_000:
        raise ValueError("Expected 2,000 person-level bootstrap replicates")
    if manifest["archive_sha256"] != SOURCE_SHA256:
        raise ValueError("Source manifest does not match the pinned archive")

    local_archive = RAW_DIR / "adult.zip"
    if local_archive.is_file() and sha256_file(local_archive) != SOURCE_SHA256:
        raise ValueError("Local ignored source archive checksum does not match")
    locked_distributions = verify_lock()
    return {
        "models": len(models),
        "group_rows": len(groups),
        "threshold_rows": len(thresholds),
        "locked_distributions": locked_distributions,
        "aggregate_only": True,
        "source_manifest_valid": True,
    }


if __name__ == "__main__":
    print(verify_reports())
