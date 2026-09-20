from pathlib import Path

import pandas as pd
import pytest

from cfm_challenge.data.profiling import profile_training_data, render_markdown
from cfm_challenge.data.schema import X_COLUMNS


def _write_features(path: Path) -> None:
    rows = []
    for obs_id in (111111, 222222):
        for position in range(100):
            rows.append((obs_id, position % 2, 0 if position == 0 else position % 3, "A" if position % 2 else "D", "B" if position % 2 else "A", 3.0, 0.0 if position == 0 else 2.0, 1.0 if position == 2 else 2.0, -1 if position == 3 else 1, 1, position % 2 == 0, (-1, 0, 1)[position % 3]))
    pd.DataFrame(rows, columns=X_COLUMNS).to_csv(path, index=False)


def test_profile_merges_chunk_statistics_and_conditional_counts(tmp_path: Path) -> None:
    path = tmp_path / "features.csv"
    _write_features(path)
    profile = profile_training_data(path, chunk_size=75)
    assert profile.event_count == 200
    assert profile.observation_count == 2
    assert profile.chunk_count == 3
    assert profile.spread["negative"] == 2
    assert profile.negative_sizes["bid_size"] == 2
    assert profile.first_bid == {"zero": 2}
    assert profile.first_order_id == {"zero": 2}
    assert profile.conditional["crossed_by_action"] == {"D": 2}
    assert profile.conditional["negative_bid_size_by_trade"] == {"False": 2}


def test_markdown_is_aggregate_only(tmp_path: Path) -> None:
    path = tmp_path / "features.csv"
    _write_features(path)
    markdown = render_markdown(profile_training_data(path, chunk_size=100))
    assert "111111" not in markdown
    assert "222222" not in markdown
    assert "# Profil agrégé" in markdown


def test_profile_rejects_non_contiguous_or_incomplete_observations(tmp_path: Path) -> None:
    path = tmp_path / "invalid.csv"
    _write_features(path)
    frame = pd.read_csv(path)
    frame = pd.concat((frame.iloc[:100], frame.iloc[:1], frame.iloc[100:]))
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="exactly 100|reappeared"):
        profile_training_data(path, chunk_size=75)
