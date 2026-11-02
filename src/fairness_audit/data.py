"""Source parsing, provenance, and data-quality checks."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from fairness_audit.constants import (
    RAW_DIR,
    SOURCE_BYTES,
    SOURCE_COLUMNS,
    SOURCE_SHA256,
    SOURCE_URL,
    TARGET_COLUMN,
    TARGET_LABELS,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv_member(payload: bytes, *, test_file: bool) -> pd.DataFrame:
    frame = pd.read_csv(
        io.BytesIO(payload),
        header=None,
        names=SOURCE_COLUMNS,
        skiprows=1 if test_file else 0,
        skipinitialspace=True,
        na_values=["?"],
        skip_blank_lines=True,
        dtype="string",
    )
    for column in frame.columns:
        if column != TARGET_COLUMN:
            frame[column] = frame[column].str.strip()
            frame[column] = frame[column].replace("", pd.NA)

    labels = frame[TARGET_COLUMN].str.strip().str.removesuffix(".")
    unknown = sorted(set(labels.dropna().unique()) - set(TARGET_LABELS))
    if unknown or labels.isna().any():
        raise ValueError(f"Unexpected target labels: {unknown}")
    frame[TARGET_COLUMN] = labels.map(TARGET_LABELS).astype("int8")
    return frame


def parse_archive(
    archive_path: Path, *, verify_checksum: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not archive_path.is_file():
        raise FileNotFoundError(
            f"Missing source archive at {archive_path}; run make fetch first."
        )
    actual_hash = sha256_file(archive_path)
    if verify_checksum and actual_hash != SOURCE_SHA256:
        raise ValueError(
            f"Source checksum mismatch: expected {SOURCE_SHA256}, got {actual_hash}"
        )
    if verify_checksum and archive_path.stat().st_size != SOURCE_BYTES:
        raise ValueError("Source archive byte count differs from the pinned snapshot")

    with zipfile.ZipFile(archive_path) as archive:
        required = {"adult.data", "adult.test"}
        if not required.issubset(archive.namelist()):
            raise ValueError("UCI archive is missing an expected Adult source file")
        training = _read_csv_member(archive.read("adult.data"), test_file=False)
        test = _read_csv_member(archive.read("adult.test"), test_file=True)
    return training, test


def _record_fingerprints(frame: pd.DataFrame) -> set[int]:
    normalized = frame.astype("string").fillna("<MISSING>")
    return set(pd.util.hash_pandas_object(normalized, index=False).astype("uint64"))


def data_quality_report(
    training: pd.DataFrame, test: pd.DataFrame, archive_path: Path
) -> dict[str, Any]:
    feature_columns = [column for column in training if column != TARGET_COLUMN]
    training_features = training[feature_columns]
    test_features = test[feature_columns]
    train_fingerprints = _record_fingerprints(training_features)
    test_fingerprints = _record_fingerprints(test_features)
    overlap = train_fingerprints & test_fingerprints
    raw_missing = {
        str(column): int(count)
        for column, count in training_features.isna().sum().items()
        if count
    }
    raw_missing_test = {
        str(column): int(count)
        for column, count in test_features.isna().sum().items()
        if count
    }
    return {
        "source": {
            "url": SOURCE_URL,
            "archive": "adult.zip",
            "sha256": sha256_file(archive_path),
            "bytes": archive_path.stat().st_size,
        },
        "rows": {"adult.data": len(training), "adult.test": len(test)},
        "target_counts": {
            "adult.data": {
                str(key): int(value)
                for key, value in training[TARGET_COLUMN]
                .value_counts()
                .sort_index()
                .items()
            },
        },
        "missing_values": {
            "adult.data": raw_missing,
            "adult.test": raw_missing_test,
        },
        "duplicate_rows": {
            "adult.data_features": int(training_features.duplicated().sum()),
            "adult.test_features": int(test_features.duplicated().sum()),
            "unique_feature_fingerprints_shared_across_official_files": len(overlap),
        },
        "modeling": {
            "protected_attributes_in_features": False,
            "sampling_weights_used": False,
            "test_used_for_model_or_policy_selection": False,
        },
    }


def write_source_manifest(
    archive_path: Path, training: pd.DataFrame, test: pd.DataFrame
):
    manifest = {
        "dataset": "UCI Adult",
        "source_url": SOURCE_URL,
        "doi": "10.24432/C5XW20",
        "license": "CC BY 4.0",
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256_file(archive_path),
        "expected_sha256": SOURCE_SHA256,
        "adult_data_rows": len(training),
        "adult_test_rows": len(test),
        "dataset_credit": (
            "Becker, B. & Kohavi, R. (1996). Adult [Dataset]. "
            "UCI Machine Learning Repository."
        ),
    }
    manifest_path = RAW_DIR.parent / "source_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
