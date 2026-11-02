from fairness_audit.constants import SOURCE_COLUMNS
from fairness_audit.modeling import feature_columns


def test_target_sensitive_fields_and_sampling_weight_are_excluded():
    columns = feature_columns(SOURCE_COLUMNS)

    assert "income" not in columns
    assert "race" not in columns
    assert "sex" not in columns
    assert "fnlwgt" not in columns
    assert "marital-status" in columns
    assert "relationship" in columns


def test_proxy_sensitivity_removes_relationship_fields():
    columns = feature_columns(SOURCE_COLUMNS, remove_proxy_fields=True)

    assert "marital-status" not in columns
    assert "relationship" not in columns
