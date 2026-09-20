"""Non-destructive quality checks for one challenge observation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Collection, Mapping

import numpy as np
import pandas as pd

from cfm_challenge.data.schema import EXPECTED_SEQUENCE_LENGTH, OBSERVATION_KEY, X_COLUMNS


class Severity(str, Enum):
    """Severity assigned to a quality finding."""

    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass(frozen=True, slots=True)
class QualityIssue:
    """An aggregate, non-destructive quality finding."""

    severity: Severity
    code: str
    count: int
    message: str


DEFAULT_BID_TOLERANCE = 1e-9
_NUMERIC_COLUMNS = ("price", "bid", "ask", "bid_size", "ask_size", "flux")


def assess_observation(
    events: pd.DataFrame,
    *,
    known_venues: Collection[object] | None = None,
    numeric_extreme_bounds: Mapping[str, tuple[float, float]] | None = None,
    bid_tolerance: float = DEFAULT_BID_TOLERANCE,
) -> tuple[QualityIssue, ...]:
    """Classify one observation without altering its events.

    Extreme-value warnings are emitted only for caller-supplied empirical
    bounds; the function deliberately does not invent business thresholds.
    """
    issues: list[QualityIssue] = []
    if tuple(events.columns) != X_COLUMNS:
        return (_issue(Severity.ERROR, "invalid_header", 1, "X header is invalid."),)
    if len(events) != EXPECTED_SEQUENCE_LENGTH:
        issues.append(_issue(Severity.ERROR, "invalid_length", 1, "Observation must contain 100 events."))
    required_missing = int(events.isna().sum().sum())
    if required_missing:
        issues.append(_issue(Severity.ERROR, "missing_required_value", required_missing, "Required values are missing."))
    for column in _NUMERIC_COLUMNS:
        values = pd.to_numeric(events[column], errors="coerce")
        incompatible = int(values.isna().sum() - events[column].isna().sum())
        nonfinite = int((~np.isfinite(values.dropna())).sum())
        if incompatible:
            issues.append(_issue(Severity.ERROR, "incompatible_numeric_type", incompatible, f"{column} is not numeric."))
        if nonfinite:
            issues.append(_issue(Severity.ERROR, "nonfinite_numeric_value", nonfinite, f"{column} contains non-finite values."))
    _append_allowed_values(issues, events, "action", {"A", "D", "U"}, "invalid_action")
    _append_allowed_values(issues, events, "side", {"A", "B"}, "invalid_side")
    trade_is_boolean = events["trade"].map(
        lambda value: isinstance(value, (bool, np.bool_))
    )
    invalid_trade = int((trade_is_boolean == False).sum())
    if invalid_trade:
        issues.append(_issue(Severity.ERROR, "invalid_trade", invalid_trade, "trade must be boolean."))
    if events[OBSERVATION_KEY].nunique(dropna=False) != 1:
        issues.append(_issue(Severity.ERROR, "inconsistent_obs_id", 1, "obs_id varies within an observation."))
    for column in (OBSERVATION_KEY, "venue", "order_id"):
        converted = pd.to_numeric(events[column], errors="coerce")
        incompatible = int(converted.isna().sum() - events[column].isna().sum())
        if incompatible:
            issues.append(
                _issue(
                    Severity.ERROR,
                    "incompatible_identifier_type",
                    incompatible,
                    f"{column} must be numeric.",
                )
            )
    bid = pd.to_numeric(events["bid"], errors="coerce")
    if len(bid) and pd.notna(bid.iloc[0]) and abs(float(bid.iloc[0])) > bid_tolerance:
        issues.append(_issue(Severity.ERROR, "nonzero_first_bid", 1, "First bid is not zero within tolerance."))
    _append_relational_findings(issues, events)
    if known_venues is not None:
        unseen = int((~events["venue"].isin(known_venues)).sum())
        if unseen:
            issues.append(_issue(Severity.WARNING, "unseen_venue", unseen, "venue was not observed in the training reference."))
    if numeric_extreme_bounds:
        for column, bounds in numeric_extreme_bounds.items():
            values = pd.to_numeric(events[column], errors="coerce")
            outside = int(((values < bounds[0]) | (values > bounds[1])).sum())
            if outside:
                issues.append(_issue(Severity.WARNING, "numeric_extreme", outside, f"{column} is outside supplied empirical bounds."))
    return tuple(issues)


def _append_allowed_values(issues: list[QualityIssue], events: pd.DataFrame, column: str, allowed: set[str], code: str) -> None:
    invalid = int((~events[column].isin(allowed)).sum())
    if invalid:
        issues.append(_issue(Severity.ERROR, code, invalid, f"{column} has an invalid modality."))


def _append_relational_findings(issues: list[QualityIssue], events: pd.DataFrame) -> None:
    bid = pd.to_numeric(events["bid"], errors="coerce")
    ask = pd.to_numeric(events["ask"], errors="coerce")
    price = pd.to_numeric(events["price"], errors="coerce")
    for code, mask, severity, message in (
        ("crossed_spread", bid > ask, Severity.WARNING, "bid is greater than ask."),
        ("zero_spread", bid == ask, Severity.WARNING, "bid equals ask."),
        ("negative_bid_size", pd.to_numeric(events["bid_size"], errors="coerce") < 0, Severity.WARNING, "bid_size is negative."),
        ("negative_ask_size", pd.to_numeric(events["ask_size"], errors="coerce") < 0, Severity.WARNING, "ask_size is negative."),
        ("price_below_bid", price < bid, Severity.INFO, "price is below bid."),
        ("price_above_ask", price > ask, Severity.INFO, "price is above ask."),
        ("repeated_order_id", events["order_id"].duplicated(keep=False), Severity.INFO, "order_id repeats within the observation."),
        ("venue_varies", pd.Series([events["venue"].nunique(dropna=False) > 1] * len(events)), Severity.INFO, "venue varies within the observation."),
    ):
        count = int(mask.sum())
        if count:
            issues.append(_issue(severity, code, count, message))
    if len(events) and events["order_id"].iloc[0] != 0:
        issues.append(_issue(Severity.WARNING, "nonzero_first_order_id", 1, "First order_id is not zero."))
    flux = pd.to_numeric(events["flux"], errors="coerce")
    for sign, mask in (
        ("negative", flux < 0),
        ("zero", flux == 0),
        ("positive", flux > 0),
    ):
        count = int(mask.sum())
        if count:
            issues.append(_issue(Severity.INFO, f"flux_{sign}", count, f"flux is {sign}."))
    trade_count = int(
        events["trade"].map(lambda value: isinstance(value, (bool, np.bool_)) and value).sum()
    )
    if trade_count:
        issues.append(_issue(Severity.INFO, "transactions_present", trade_count, "Transactions are present."))
    else:
        issues.append(_issue(Severity.INFO, "transactions_absent", len(events), "Transactions are absent."))
    if (flux < 0).any() and (flux > 0).any():
        issues.append(_issue(Severity.WARNING, "mixed_flux_signs", int(((flux < 0) | (flux > 0)).sum()), "flux has both signs."))


def _issue(severity: Severity, code: str, count: int, message: str) -> QualityIssue:
    return QualityIssue(severity=severity, code=code, count=count, message=message)
