import pytest

from cfm_challenge.data.schema import (
    X_COLUMNS,
    Y_COLUMNS,
    SchemaValidationError,
    validate_label,
    validate_x_columns,
    validate_y_columns,
)


def test_exact_x_columns_are_accepted() -> None:
    assert validate_x_columns(X_COLUMNS) is None


def test_exact_y_columns_are_accepted() -> None:
    assert validate_y_columns(Y_COLUMNS) is None


def test_missing_column_is_rejected() -> None:
    with pytest.raises(SchemaValidationError, match="missing columns.*flux"):
        validate_x_columns(X_COLUMNS[:-1])


def test_extra_column_is_rejected() -> None:
    with pytest.raises(SchemaValidationError, match="extra columns.*unexpected"):
        validate_y_columns((*Y_COLUMNS, "unexpected"))


def test_incorrect_column_order_is_rejected() -> None:
    reordered = (X_COLUMNS[1], X_COLUMNS[0], *X_COLUMNS[2:])

    with pytest.raises(SchemaValidationError, match="column order at position 0"):
        validate_x_columns(reordered)


@pytest.mark.parametrize("label", [0, 23])
def test_boundary_labels_are_accepted(label: int) -> None:
    assert validate_label(label) == label


@pytest.mark.parametrize(
    "invalid_label",
    [None, float("nan"), True, False, "0", 0.0, -1, 24],
    ids=["none", "nan", "true", "false", "string", "float", "below", "above"],
)
def test_invalid_labels_are_rejected(invalid_label: object) -> None:
    with pytest.raises(SchemaValidationError, match="label"):
        validate_label(invalid_label)
