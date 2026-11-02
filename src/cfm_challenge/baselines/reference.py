"""Independent, deliberately scalar oracle for qualifying the 50 features.

This module does not call production aggregation routines. It is intended for
small, valid samples, not for processing the whole training file.
"""

from __future__ import annotations

from collections import Counter
from math import fsum
from statistics import pstdev
from typing import Sequence

import numpy as np
import pandas as pd


# A few float32 rounding units, including absolute tolerance around zero.
# These tolerances are fixed before any real-data comparison.
FEATURE_RTOL = 1e-6
FEATURE_ATOL = 1e-6

REFERENCE_NAMES = (
    tuple(
        f"{column}_{statistic}"
        for column in ("price", "bid", "ask", "bid_size", "ask_size", "flux")
        for statistic in ("mean", "std", "min", "max", "first", "last")
    )
    + tuple(f"venue_slot_{index}_frequency" for index in range(6))
    + (
        "venue_unknown_frequency",
        "action_A_frequency",
        "action_D_frequency",
        "action_U_frequency",
        "side_A_frequency",
        "side_B_frequency",
        "trade_false_frequency",
        "trade_true_frequency",
    )
)


def reference_features(
    events: pd.DataFrame,
    venue_slots: Sequence[int | None],
) -> np.ndarray:
    """Calculate population statistics and counts without production reducers."""
    if len(events) != 100 or len(venue_slots) != 6:
        raise ValueError("reference requires 100 events and six venue slots")
    values: list[float] = []
    for column in ("price", "bid", "ask", "bid_size", "ask_size", "flux"):
        series = [float(value) for value in events[column]]
        values.extend(
            (
                fsum(series) / 100,
                pstdev(series),
                min(series),
                max(series),
                series[0],
                series[-1],
            )
        )
    counts = Counter(events["venue"].tolist())
    known = {venue for venue in venue_slots if venue is not None}
    values.extend(
        0.0 if venue is None else counts[venue] / 100 for venue in venue_slots
    )
    values.append(
        sum(count for venue, count in counts.items() if venue not in known) / 100
    )
    for column, categories in (
        ("action", ("A", "D", "U")),
        ("side", ("A", "B")),
        ("trade", (False, True)),
    ):
        counts = Counter(events[column].tolist())
        values.extend(counts[category] / 100 for category in categories)
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (50,) or not np.isfinite(result).all():
        raise ValueError("invalid independent reference result")
    return result
