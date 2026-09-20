"""Schema constants and dependency-free validation helpers."""

from collections import Counter
from collections.abc import Iterable


OBSERVATION_KEY = "obs_id"
TARGET_COLUMN = "eqt_code_cat"
EXPECTED_SEQUENCE_LENGTH = 100
EXPECTED_CLASS_COUNT = 24
VALID_CLASSES = tuple(range(EXPECTED_CLASS_COUNT))

X_COLUMNS = (
    OBSERVATION_KEY,
    "venue",
    "order_id",
    "action",
    "side",
    "price",
    "bid",
    "ask",
    "bid_size",
    "ask_size",
    "trade",
    "flux",
)

Y_COLUMNS = (
    OBSERVATION_KEY,
    TARGET_COLUMN,
)


class SchemaValidationError(ValueError):
    """Raised when columns or a label do not match the challenge contract."""


def validate_x_columns(
    columns: Iterable[str], *, require_order: bool = True
) -> None:
    """Validate that feature columns match the exact X schema."""
    _validate_columns(columns, X_COLUMNS, "X", require_order=require_order)


def validate_y_columns(
    columns: Iterable[str], *, require_order: bool = True
) -> None:
    """Validate that target columns match the exact y schema."""
    _validate_columns(columns, Y_COLUMNS, "y", require_order=require_order)


def validate_label(value: object) -> int:
    """Return a valid built-in integer class label or raise a descriptive error."""
    if type(value) is not int:
        raise SchemaValidationError(
            "label must be a built-in int in the inclusive range 0..23; "
            f"received {value!r} ({type(value).__name__})"
        )
    if value not in VALID_CLASSES:
        raise SchemaValidationError(
            f"label must be in the inclusive range 0..23; received {value}"
        )
    return value


def _validate_columns(
    columns: Iterable[str],
    expected: tuple[str, ...],
    dataset_name: str,
    *,
    require_order: bool,
) -> None:
    if isinstance(columns, (str, bytes)):
        raise SchemaValidationError(
            f"{dataset_name} columns must be an iterable of column names, not a string"
        )

    actual = tuple(columns)
    counts = Counter(actual)
    missing = tuple(column for column in expected if column not in counts)
    extra = tuple(column for column in actual if column not in expected)
    duplicated = tuple(column for column, count in counts.items() if count > 1)

    problems: list[str] = []
    if missing:
        problems.append(f"missing columns: {missing!r}")
    if extra:
        problems.append(f"extra columns: {extra!r}")
    if duplicated:
        problems.append(f"duplicated columns: {duplicated!r}")
    if problems:
        raise SchemaValidationError(
            f"invalid {dataset_name} columns ({'; '.join(problems)}); "
            f"expected exactly {expected!r}"
        )

    if require_order and actual != expected:
        first_mismatch = next(
            index
            for index, (actual_name, expected_name) in enumerate(zip(actual, expected))
            if actual_name != expected_name
        )
        raise SchemaValidationError(
            f"invalid {dataset_name} column order at position {first_mismatch}: "
            f"received {actual[first_mismatch]!r}, expected {expected[first_mismatch]!r}; "
            f"expected order is {expected!r}"
        )
