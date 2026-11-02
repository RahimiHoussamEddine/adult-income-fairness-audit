from __future__ import annotations

import zipfile

from fairness_audit.constants import SOURCE_COLUMNS
from fairness_audit.data import parse_archive


def _adult_row(*, label: str, missing_workclass: bool = False) -> str:
    values = [
        "39",
        "?" if missing_workclass else "State-gov",
        "77516",
        "Bachelors",
        "13",
        "Never-married",
        "Adm-clerical",
        "Not-in-family",
        "White",
        "Male",
        "2174",
        "0",
        "40",
        "United-States",
        label,
    ]
    return ", ".join(values)


def test_parse_archive_normalizes_official_target_labels(tmp_path):
    archive_path = tmp_path / "adult.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("adult.data", _adult_row(label="<=50K") + "\n")
        archive.writestr(
            "adult.test",
            "|1x3 Cross validator\n"
            + _adult_row(label=">50K.", missing_workclass=True)
            + "\n",
        )

    training, test = parse_archive(archive_path, verify_checksum=False)

    assert list(training.columns) == SOURCE_COLUMNS
    assert training["income"].tolist() == [0]
    assert test["income"].tolist() == [1]
    assert test["workclass"].isna().tolist() == [True]
