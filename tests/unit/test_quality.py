import pandas as pd

from cfm_challenge.data.quality import Severity, assess_observation
from cfm_challenge.data.schema import X_COLUMNS


def _events() -> pd.DataFrame:
    rows = [(1, 2, 0 if i == 0 else i % 3, "A", "B", 1.0, 0.0 if i == 0 else 1.0, 2.0, 1, 1, True, 0) for i in range(100)]
    return pd.DataFrame(rows, columns=X_COLUMNS)


def _codes(frame: pd.DataFrame) -> dict[str, Severity]:
    return {issue.code: issue.severity for issue in assess_observation(frame)}


def test_valid_observation_has_no_validation_errors() -> None:
    codes = _codes(_events())
    assert "invalid_action" not in codes
    assert "invalid_side" not in codes
    assert "invalid_trade" not in codes


def test_invalid_modalities_and_trade_are_errors() -> None:
    frame = _events()
    frame.loc[0, ["action", "side"]] = ["X", "X"]
    frame["trade"] = frame["trade"].astype(object)
    frame.loc[0, "trade"] = "yes"
    codes = _codes(frame)
    assert codes["invalid_action"] is Severity.ERROR
    assert codes["invalid_side"] is Severity.ERROR
    assert codes["invalid_trade"] is Severity.ERROR


def test_missing_and_nonfinite_values_are_errors() -> None:
    frame = _events()
    frame.loc[1, "price"] = float("nan")
    frame.loc[2, "bid"] = float("inf")
    frame.loc[3, "ask"] = float("-inf")
    codes = _codes(frame)
    assert codes["missing_required_value"] is Severity.ERROR
    assert codes["nonfinite_numeric_value"] is Severity.ERROR


def test_first_bid_and_order_id_have_expected_severities() -> None:
    frame = _events()
    frame.loc[0, "bid"] = 0.5
    frame.loc[0, "order_id"] = 8
    codes = _codes(frame)
    assert codes["nonzero_first_bid"] is Severity.ERROR
    assert codes["nonzero_first_order_id"] is Severity.WARNING


def test_spread_sizes_price_and_repeated_order_are_not_errors() -> None:
    frame = _events()
    frame.loc[1, ["bid", "ask", "bid_size", "price"]] = [3.0, 2.0, -1, 4.0]
    frame.loc[2, ["bid", "ask", "price"]] = [2.0, 2.0, 1.0]
    codes = _codes(frame)
    assert codes["crossed_spread"] is Severity.WARNING
    assert codes["zero_spread"] is Severity.WARNING
    assert codes["negative_bid_size"] is Severity.WARNING
    assert codes["price_above_ask"] is Severity.INFO
    assert codes["price_below_bid"] is Severity.INFO
    assert codes["repeated_order_id"] is Severity.INFO


def test_flux_mixed_signs_is_a_warning() -> None:
    frame = _events()
    frame.loc[1, "flux"] = -1
    frame.loc[2, "flux"] = 1
    assert _codes(frame)["mixed_flux_signs"] is Severity.WARNING


def test_identifier_type_and_descriptive_flux_are_reported() -> None:
    frame = _events()
    frame["order_id"] = frame["order_id"].astype(object)
    frame.loc[1, "order_id"] = "bad"
    frame.loc[1, "flux"] = -1
    frame.loc[2, "flux"] = 1
    codes = _codes(frame)
    assert codes["incompatible_identifier_type"] is Severity.ERROR
    assert codes["flux_negative"] is Severity.INFO
    assert codes["flux_zero"] is Severity.INFO
    assert codes["flux_positive"] is Severity.INFO
    assert codes["transactions_present"] is Severity.INFO


def test_header_length_identity_venue_and_empirical_bounds_are_checked() -> None:
    frame = _events()
    frame.loc[0, "venue"] = 9
    frame.loc[0, "price"] = 9.0
    issues = assess_observation(
        frame,
        known_venues={2},
        numeric_extreme_bounds={"price": (0.0, 2.0)},
    )
    codes = {issue.code: issue.severity for issue in issues}
    assert codes["unseen_venue"] is Severity.WARNING
    assert codes["numeric_extreme"] is Severity.WARNING
    assert _codes(frame.iloc[:-1])["invalid_length"] is Severity.ERROR
    invalid_header = frame.rename(columns={"flux": "other"})
    assert _codes(invalid_header)["invalid_header"] is Severity.ERROR
