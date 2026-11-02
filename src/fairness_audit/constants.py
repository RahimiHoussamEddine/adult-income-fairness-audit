"""Pinned source and analysis settings."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
SOURCE_URL = "https://archive.ics.uci.edu/static/public/2/adult.zip"
SOURCE_SHA256 = "7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb"
SOURCE_BYTES = 620_237
RANDOM_STATE = 2026
VALIDATION_SIZE = 0.20
BOOTSTRAP_REPLICATES = 2_000
CI_QUANTILES = [0.025, 0.975]
TARGET_COLUMN = "income"
SENSITIVE_COLUMNS = ["race", "sex"]
SAMPLING_COLUMN = "fnlwgt"
TARGET_LABELS = {"<=50K": 0, ">50K": 1}
SOURCE_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]
