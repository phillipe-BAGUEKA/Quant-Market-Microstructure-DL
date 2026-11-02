import numpy as np
import pandas as pd
import pytest

from cfm_challenge.baselines.aggregation import (
    AggregationError,
    VenueVocabulary,
    aggregate_observation,
    compose_features,
    feature_names,
    summarize_observation,
)


def frame():
    data = {
        "price": np.arange(100, dtype=float),
        "bid": np.zeros(100),
        "ask": np.ones(100),
        "bid_size": np.ones(100),
        "ask_size": np.ones(100) * 2,
        "flux": np.arange(100, dtype=float),
        "venue": [2, 1] * 50,
        "action": ["A", "D", "U", "A"] * 25,
        "side": ["A", "B"] * 50,
        "trade": [False, True] * 50,
        "obs_id": [9] * 100,
        "order_id": list(range(100)),
        "eqt_code_cat": [0] * 100,
    }
    return pd.DataFrame(data)


def test_aggregate_schema_values_and_input_are_deterministic():
    events = frame()
    before = events.copy(deep=True)
    vocab = VenueVocabulary.build([2, 1, 2])
    result = aggregate_observation(events, vocab)
    names = feature_names(vocab)
    assert vocab.slots == (1, 2, None, None, None, None)
    assert result.dtype == np.float32
    assert result.shape == (50,)
    assert names[:6] == (
        "price_mean",
        "price_std",
        "price_min",
        "price_max",
        "price_first",
        "price_last",
    )
    assert result[0] == np.float32(49.5)
    assert result[1] == np.float32(np.std(np.arange(100), ddof=0))
    assert (
        "obs_id" not in names
        and "order_id" not in names
        and "eqt_code_cat" not in names
    )
    pd.testing.assert_frame_equal(events, before)


@pytest.mark.parametrize("size", [99, 101])
def test_invalid_length_is_rejected(size):
    events = (
        frame().iloc[:size]
        if size < 100
        else pd.concat((frame(), frame().iloc[:1]), ignore_index=True)
    )
    with pytest.raises(AggregationError):
        aggregate_observation(events, VenueVocabulary.build([1, 2]))


def test_unknown_and_invalid_categories_are_explicit():
    events = frame()
    result = aggregate_observation(events, VenueVocabulary.build([1]))
    assert result[42] == np.float32(0.5)
    for column, value in [("action", "X"), ("side", "X"), ("trade", "yes")]:
        bad = frame()
        if column == "trade":
            bad[column] = bad[column].astype(object)
        bad.loc[0, column] = value
        with pytest.raises(AggregationError):
            aggregate_observation(bad, VenueVocabulary.build([1, 2]))


def test_missing_and_infinite_numeric_values_are_rejected():
    for value in [np.nan, np.inf]:
        bad = frame()
        bad.loc[0, "price"] = value
        with pytest.raises(AggregationError):
            aggregate_observation(bad, VenueVocabulary.build([1, 2]))


def test_six_known_venues_produce_fifty_features_and_normalized_frequencies():
    events = frame()
    events["venue"] = [index % 6 for index in range(100)]
    vector = aggregate_observation(events, VenueVocabulary.build(range(6)))
    assert vector.shape == (50,)
    assert np.isclose(vector[36:43].sum(), 1.0)
    assert np.isclose(vector[43:46].sum(), 1.0)
    assert np.isclose(vector[46:48].sum(), 1.0)
    assert np.isclose(vector[48:50].sum(), 1.0)


@pytest.mark.parametrize("values", [[], [1], list(range(5)), list(range(6))])
def test_venue_slots_have_constant_width(values):
    vocab = VenueVocabulary.build(values)
    vector = aggregate_observation(frame(), vocab)
    assert len(vocab.slots) == 6 and vector.shape == (50,)
    assert len(set(feature_names(vocab))) == 50


def test_venue_vocabulary_validates_values_and_rejects_seven():
    assert VenueVocabulary.build([np.int64(1)]).slots[0] == 1
    for value in [True, False, np.bool_(True), 1.0, "1", None, np.nan, np.inf, -np.inf]:
        with pytest.raises(AggregationError):
            VenueVocabulary.build([value])
    with pytest.raises(AggregationError):
        VenueVocabulary.build(range(7))


def test_summary_recomposition_matches_direct_aggregation_exactly():
    events = frame()
    vocabulary = VenueVocabulary.build([1])
    direct = aggregate_observation(events, vocabulary)
    recomposed = compose_features(summarize_observation(events), vocabulary)
    np.testing.assert_allclose(recomposed, direct, rtol=0.0, atol=0.0)
    assert recomposed.dtype == np.float32 and recomposed.shape == (50,)
