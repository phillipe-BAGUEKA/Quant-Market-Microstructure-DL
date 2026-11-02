from pathlib import Path
from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from cfm_challenge.baselines.dataset import (
    BaselineDatasetError,
    discover_venue_vocabulary,
    extract_selected_observations,
    localize_folds,
)
from cfm_challenge.baselines.dataset import build_development_dataset
from cfm_challenge.baselines.dataset import build_development_selection
from cfm_challenge.baselines.dataset import DevelopmentSelection
from cfm_challenge.baselines.dataset import FoldDataset, SelectedObservationDataset
from cfm_challenge.baselines.dataset import build_fold_dataset
from cfm_challenge.baselines.dataset import iter_fold_datasets
from cfm_challenge.baselines.dataset import (
    build_development_summaries,
    iter_fold_datasets_from_summaries,
)
from cfm_challenge.baselines.aggregation import VenueVocabulary
from cfm_challenge.evaluation.splitting import (
    Fold,
    Partition,
    SplitConfiguration,
    build_lockbox,
    grouped_folds,
    stratified_folds,
)


def write_x(path: Path) -> None:
    rows = []
    for obs, venue in ((20, 2), (10, 1), (30, 9)):
        for i in range(100):
            rows.append((obs, venue, 0, "A", "B", float(i), 0.0, 1.0, 1, 1, False, 0))
    pd.DataFrame(
        rows,
        columns=[
            "obs_id",
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
        ],
    ).to_csv(path, index=False)


def test_selected_extraction_is_ordered_and_excludes_unselected(tmp_path: Path):
    path = tmp_path / "x.csv"
    write_x(path)
    ids = (10, 20)
    vocab = discover_venue_vocabulary(path, ids, chunk_size=101)
    data = extract_selected_observations(
        path, ids, {10: 1, 20: 2}, vocab, chunk_size=101
    )
    assert vocab.known == (1, 2)
    assert data.features.dtype.name == "float32"
    assert data.observation_ids.tolist() == [10, 20]
    assert data.labels.tolist() == [1, 2]
    assert data.features.shape[1] == len(data.feature_names)
    assert "venue_unknown_frequency" in data.feature_names
    assert all(
        name not in data.feature_names
        for name in ("obs_id", "order_id", "eqt_code_cat")
    )


@pytest.mark.parametrize("ids,chunk", [([], 1), ((10, 10), 1), ((10,), 0)])
def test_invalid_selection_is_rejected(tmp_path: Path, ids, chunk):
    path = tmp_path / "x.csv"
    write_x(path)
    with pytest.raises(BaselineDatasetError):
        discover_venue_vocabulary(path, ids, chunk_size=chunk)


def test_missing_observation_and_label_are_rejected(tmp_path: Path):
    path = tmp_path / "x.csv"
    write_x(path)
    vocab = discover_venue_vocabulary(path, (10,), chunk_size=100)
    with pytest.raises(BaselineDatasetError):
        extract_selected_observations(path, (99,), {99: 1}, vocab, chunk_size=100)
    with pytest.raises(BaselineDatasetError):
        extract_selected_observations(path, (10,), {}, vocab, chunk_size=100)


@pytest.mark.parametrize("label", [None, True, False, 0.0, "0", float("nan"), -1, 24])
def test_invalid_labels_are_rejected(tmp_path: Path, label: object):
    path = tmp_path / "x.csv"
    write_x(path)
    vocab = discover_venue_vocabulary(path, (10,), chunk_size=137)
    with pytest.raises(BaselineDatasetError):
        extract_selected_observations(path, (10,), {10: label}, vocab, chunk_size=137)


def test_chunk_boundary_determinism_and_immutability(tmp_path: Path):
    path = tmp_path / "x.csv"
    write_x(path)
    ids = (10, 20)
    labels = {10: 0, 20: 1}
    first_vocab = discover_venue_vocabulary(path, ids, chunk_size=137)
    second_vocab = discover_venue_vocabulary(path, ids, chunk_size=137)
    first = extract_selected_observations(
        path, ids, labels, first_vocab, chunk_size=137
    )
    second = extract_selected_observations(
        path, ids, labels, second_vocab, chunk_size=137
    )
    assert first_vocab == second_vocab and isinstance(first_vocab.known, tuple)
    np.testing.assert_array_equal(first.features, second.features)
    np.testing.assert_array_equal(first.labels, second.labels)
    np.testing.assert_array_equal(first.observation_ids, second.observation_ids)
    with pytest.raises(FrozenInstanceError):
        first.venue_vocabulary = second_vocab


def test_nonfinite_and_wrong_width_are_rejected(tmp_path: Path, monkeypatch):
    path = tmp_path / "x.csv"
    write_x(path)
    vocab = discover_venue_vocabulary(path, (10,), chunk_size=100)
    bad = pd.read_csv(path)
    bad.loc[bad["obs_id"] == 10, "price"] = float("inf")
    bad.to_csv(path, index=False)
    with pytest.raises(BaselineDatasetError):
        extract_selected_observations(path, (10,), {10: 0}, vocab, chunk_size=100)


def test_wrong_aggregator_width_is_rejected(tmp_path: Path, monkeypatch):
    path = tmp_path / "x.csv"
    write_x(path)
    vocab = discover_venue_vocabulary(path, (10,), chunk_size=100)
    monkeypatch.setattr(
        "cfm_challenge.baselines.dataset.aggregate_observation",
        lambda events, vocabulary: np.zeros(1, dtype=np.float32),
    )
    with pytest.raises(BaselineDatasetError, match="invalid aggregated feature vector"):
        extract_selected_observations(path, (10,), {10: 0}, vocab, chunk_size=100)


def test_build_development_dataset_uses_only_development_observations(tmp_path: Path):
    y_path = tmp_path / "labels.csv"
    x_path = tmp_path / "features.csv"
    pairs = [(1000 + i, i // 5) for i in range(25)]
    pd.DataFrame(pairs, columns=["obs_id", "eqt_code_cat"]).to_csv(y_path, index=False)
    ids = tuple(item[0] for item in pairs)
    labels = tuple(item[1] for item in pairs)
    partition = build_lockbox(ids, labels, SplitConfiguration())
    development = tuple(ids[i] for i in partition.development_indices)
    lockbox = {ids[i] for i in partition.lockbox_indices}
    rows = []
    for obs_id, label in pairs:
        venue = 999 if obs_id in lockbox else label
        for position in range(100):
            rows.append(
                (obs_id, venue, 0, "A", "B", float(position), 0.0, 1.0, 1, 1, False, 0)
            )
    pd.DataFrame(
        rows,
        columns=[
            "obs_id",
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
        ],
    ).to_csv(x_path, index=False)
    first = build_development_dataset(x_path, y_path, chunk_size=137)
    second = build_development_dataset(x_path, y_path, chunk_size=137)
    expected_indices = np.asarray(partition.development_indices, dtype=np.int64)
    assert tuple(first.observation_ids) == development
    assert first.source_indices.tolist() == expected_indices.tolist()
    assert set(first.observation_ids).isdisjoint(lockbox)
    assert first.labels.tolist() == [labels[index] for index in expected_indices]
    assert all(
        first.observation_ids[local] == ids[source]
        for local, source in enumerate(first.source_indices)
    )
    assert set(first.source_indices).isdisjoint(partition.lockbox_indices)
    assert 999 not in first.venue_vocabulary.known
    assert first.features.dtype == np.float32
    assert np.isfinite(first.features).all()
    assert first.features.shape == (len(first.labels), 50)
    np.testing.assert_array_equal(first.features, second.features)
    np.testing.assert_array_equal(first.labels, second.labels)


def test_build_development_dataset_wraps_invalid_y_header(tmp_path: Path):
    y_path = tmp_path / "bad.csv"
    x_path = tmp_path / "x.csv"
    y_path.write_text("wrong,label\n1,0\n", encoding="utf-8")
    write_x(x_path)
    with pytest.raises(BaselineDatasetError) as error:
        build_development_dataset(x_path, y_path, chunk_size=137)
    assert error.value.__cause__ is not None


def test_build_development_dataset_wraps_impossible_stratification(tmp_path: Path):
    y_path = tmp_path / "labels.csv"
    x_path = tmp_path / "x.csv"
    pd.DataFrame(
        [(10, 0), (20, 1), (30, 1)], columns=["obs_id", "eqt_code_cat"]
    ).to_csv(y_path, index=False)
    write_x(x_path)
    with pytest.raises(BaselineDatasetError) as error:
        build_development_dataset(x_path, y_path, chunk_size=137)
    assert error.value.__cause__ is not None


def test_venue_discovery_passes_only_distinct_selected_values(
    tmp_path: Path, monkeypatch
):
    path = tmp_path / "repeated.csv"
    rows = []
    for obs, venue in ((10, 2), (20, 1), (30, 99)):
        for position in range(100):
            rows.append(
                (
                    obs,
                    venue,
                    position,
                    "A",
                    "B",
                    float(position),
                    0.0,
                    1.0,
                    1,
                    1,
                    False,
                    0,
                )
            )
    pd.DataFrame(
        rows,
        columns=[
            "obs_id",
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
        ],
    ).to_csv(path, index=False)
    captured = []
    original = VenueVocabulary.build.__func__

    def build(cls, values):
        captured.extend(values)
        return original(cls, captured)

    monkeypatch.setattr(VenueVocabulary, "build", classmethod(build))
    vocabulary = discover_venue_vocabulary(path, (10, 20), chunk_size=137)
    assert set(captured) == {1, 2} and len(captured) == 2 and 99 not in captured
    assert vocabulary.slots == (1, 2, None, None, None, None)


def test_localize_folds_preserves_noncontiguous_source_order():
    sources = [7, 1, 9, 3, 4]
    folds = (Fold((7, 9, 4), (1, 3)), Fold((1, 3), (7, 9, 4)))
    localized = localize_folds(folds, sources)
    assert localized[0] == Fold((0, 2, 4), (1, 3))
    assert localized[1] == Fold((1, 3), (0, 2, 4))


def test_localize_folds_preserves_full_noncontiguous_order():
    sources = [7, 1, 9, 3, 4, 11]
    folds = (Fold((9, 7, 1), (3, 4, 11)), Fold((3, 4, 11), (9, 7, 1)))
    localized = localize_folds(folds, sources)
    assert isinstance(localized, tuple)
    assert localized == (Fold((2, 0, 1), (3, 4, 5)), Fold((3, 4, 5), (2, 0, 1)))
    assert all(
        0 <= index <= 5
        for fold in localized
        for index in fold.train_indices + fold.validation_indices
    )


def test_localize_folds_materializes_generator_once():
    sources = [7, 1, 9, 3, 4, 11]
    folds = (Fold((9, 7, 1), (3, 4, 11)), Fold((3, 4, 11), (9, 7, 1)))
    calls = []

    def generator():
        for fold in folds:
            calls.append(fold)
            yield fold

    result = localize_folds(generator(), sources)
    assert calls == list(folds) and result == (
        Fold((2, 0, 1), (3, 4, 5)),
        Fold((3, 4, 5), (2, 0, 1)),
    )


def test_localize_folds_rejects_empty_collection():
    with pytest.raises(BaselineDatasetError, match="fold collection"):
        localize_folds([], [7, 1, 9, 3])


def test_localize_folds_rejects_unknown_global_index():
    with pytest.raises(BaselineDatasetError, match="absent"):
        localize_folds((Fold((7, 1), (99, 3)), Fold((9, 3), (7, 1))), [7, 1, 9, 3])


def test_localize_folds_rejects_partition_overlap():
    with pytest.raises(BaselineDatasetError):
        localize_folds((Fold((7, 1, 9), (9, 3)), Fold((9, 3), (7, 1))), [7, 1, 9, 3])


def test_localize_folds_rejects_invalid_validation_coverage():
    folds = (Fold((7, 1), (9, 3)), Fold((7, 1), (9, 3)))
    with pytest.raises(BaselineDatasetError, match="validations"):
        localize_folds(folds, [7, 1, 9, 3])


def test_localize_folds_integrates_stratified_global_folds():
    labels = [index % 2 for index in range(24)]
    source_indices = (1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 21, 22)
    partition = Partition(
        source_indices,
        tuple(index for index in range(24) if index not in source_indices),
    )
    global_folds = stratified_folds(labels, partition, SplitConfiguration(n_splits=2))
    local_folds = localize_folds(global_folds, source_indices)
    assert (
        isinstance(local_folds, tuple)
        and len(local_folds) == len(global_folds) == 2
        and max(source_indices) >= len(source_indices)
    )
    validations = []
    for global_fold, local_fold in zip(global_folds, local_folds):
        assert set(local_fold.train_indices).isdisjoint(local_fold.validation_indices)
        assert set(local_fold.train_indices) | set(
            local_fold.validation_indices
        ) == set(range(len(source_indices)))
        assert (
            tuple(source_indices[index] for index in local_fold.train_indices)
            == global_fold.train_indices
        )
        assert (
            tuple(source_indices[index] for index in local_fold.validation_indices)
            == global_fold.validation_indices
        )
        validations.extend(local_fold.validation_indices)
    assert set(validations) == set(range(len(source_indices))) and len(
        validations
    ) == len(source_indices)


def test_localize_folds_integrates_grouped_global_folds():
    labels = [index % 2 for index in range(24)]
    groups = [index // 2 for index in range(24)]
    source_indices = (1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 21, 22)
    partition = Partition(
        source_indices,
        tuple(index for index in range(24) if index not in source_indices),
    )
    global_folds = grouped_folds(
        labels, partition, groups, SplitConfiguration(n_splits=2)
    )
    localized = localize_folds(global_folds, source_indices)
    validations = []
    assert (
        isinstance(localized, tuple)
        and len(localized) == len(global_folds) == 2
        and max(source_indices) >= len(source_indices)
    )
    for global_fold, local_fold in zip(global_folds, localized):
        assert set(global_fold.train_indices) | set(
            global_fold.validation_indices
        ) == set(source_indices)
        assert (
            tuple(source_indices[index] for index in local_fold.train_indices)
            == global_fold.train_indices
        )
        assert (
            tuple(source_indices[index] for index in local_fold.validation_indices)
            == global_fold.validation_indices
        )
        assert set(local_fold.train_indices).isdisjoint(local_fold.validation_indices)
        assert set(local_fold.train_indices) | set(
            local_fold.validation_indices
        ) == set(range(len(source_indices)))
        assert not set(
            groups[source_indices[index]] for index in local_fold.train_indices
        ) & set(
            groups[source_indices[index]] for index in local_fold.validation_indices
        )
        validations.extend(local_fold.validation_indices)
    assert set(validations) == set(range(len(source_indices))) and len(
        validations
    ) == len(source_indices)


def test_build_development_selection_uses_y_only_and_preserves_indices(tmp_path: Path):
    path = tmp_path / "labels.csv"
    pairs = [(1000 + index, index // 5) for index in range(25)]
    pd.DataFrame(pairs, columns=["obs_id", "eqt_code_cat"]).to_csv(path, index=False)
    selection = build_development_selection(path)
    ids = tuple(item[0] for item in pairs)
    labels = [item[1] for item in pairs]
    partition = build_lockbox(ids, labels, SplitConfiguration())
    assert selection.source_indices.tolist() == list(partition.development_indices)
    assert selection.observation_ids == tuple(
        ids[index] for index in selection.source_indices
    )
    assert selection.labels.tolist() == [
        labels[index] for index in selection.source_indices
    ]
    assert set(selection.source_indices).isdisjoint(partition.lockbox_indices)
    assert (
        selection.labels.ndim == selection.source_indices.ndim == 1
        and selection.labels.flags.writeable is False
        and selection.source_indices.flags.writeable is False
    )


def test_development_selection_defensively_copies_arrays():
    labels = np.array([0, 1])
    sources = np.array([7, 9])
    selection = DevelopmentSelection((10, 20), labels, sources)
    labels[0] = 9
    sources[0] = 99
    assert selection.labels.tolist() == [
        0,
        1,
    ] and selection.source_indices.tolist() == [7, 9]
    with pytest.raises(ValueError):
        selection.labels[0] = 1
    with pytest.raises(ValueError):
        selection.source_indices[0] = 1


def test_build_development_selection_is_deterministic(tmp_path: Path):
    path = tmp_path / "labels.csv"
    pd.DataFrame(
        [(1000 + i, i // 5) for i in range(25)], columns=["obs_id", "eqt_code_cat"]
    ).to_csv(path, index=False)
    first = build_development_selection(path)
    second = build_development_selection(path)
    assert first.observation_ids == second.observation_ids
    np.testing.assert_array_equal(first.labels, second.labels)
    np.testing.assert_array_equal(first.source_indices, second.source_indices)


@pytest.mark.parametrize(
    "text",
    [
        "wrong,label\n1,0\n",
        "obs_id,eqt_code_cat\n1,0\n1,0\n",
        "obs_id,eqt_code_cat\n1,24\n",
    ],
)
def test_build_development_selection_wraps_invalid_y(tmp_path: Path, text: str):
    path = tmp_path / "bad.csv"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(BaselineDatasetError):
        build_development_selection(path)


def test_build_development_selection_does_not_use_x_pipeline(
    tmp_path: Path, monkeypatch
):
    path = tmp_path / "labels.csv"
    pd.DataFrame(
        [(1000 + i, i // 5) for i in range(25)], columns=["obs_id", "eqt_code_cat"]
    ).to_csv(path, index=False)
    monkeypatch.setattr(
        "cfm_challenge.baselines.dataset.discover_venue_vocabulary",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()),
    )
    monkeypatch.setattr(
        "cfm_challenge.baselines.dataset.extract_selected_observations",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()),
    )
    assert build_development_selection(path).observation_ids


def _fold_dataset() -> tuple[SelectedObservationDataset, VenueVocabulary]:
    vocabulary = VenueVocabulary.build([])
    data = SelectedObservationDataset(
        np.zeros((2, 50), dtype=np.float32),
        np.array([0, 1]),
        np.array([10, 20]),
        tuple(f"f{index}" for index in range(50)),
        vocabulary,
        np.array([0, 1], dtype=np.int64),
    )
    return data, vocabulary


def test_fold_dataset_accepts_valid_local_partition():
    data, vocabulary = _fold_dataset()
    result = FoldDataset(data, Fold((0,), (1,)), vocabulary)
    assert (
        result.development is data
        and result.fold == Fold((0,), (1,))
        and result.venue_vocabulary is vocabulary
    )
    with pytest.raises(FrozenInstanceError):
        result.fold = Fold((1,), (0,))


def test_fold_dataset_rejects_overlapping_partition():
    data, vocabulary = _fold_dataset()
    with pytest.raises(BaselineDatasetError):
        FoldDataset(data, Fold((0, 1), (1,)), vocabulary)


@pytest.mark.parametrize(
    "fold",
    [
        Fold((), (0, 1)),
        Fold((0, 1), ()),
        Fold((-1,), (0, 1)),
        Fold((True,), (1,)),
        Fold((0, 2), (1,)),
        Fold((0, 0), (1,)),
        Fold((0,), (1, 1)),
    ],
)
def test_fold_dataset_rejects_invalid_local_partition(fold):
    data, vocabulary = _fold_dataset()
    with pytest.raises(BaselineDatasetError):
        FoldDataset(data, fold, vocabulary)


def test_fold_dataset_rejects_invalid_field_types():
    data, vocabulary = _fold_dataset()
    for development, fold, venues in [
        (object(), Fold((0,), (1,)), vocabulary),
        (data, object(), vocabulary),
        (data, Fold((0,), (1,)), object()),
    ]:
        with pytest.raises(BaselineDatasetError):
            FoldDataset(development, fold, venues)


def test_fold_dataset_rejects_incomplete_three_row_partition():
    vocabulary = VenueVocabulary.build([])
    data = SelectedObservationDataset(
        np.zeros((3, 50), dtype=np.float32),
        np.array([0, 1, 0]),
        np.array([10, 20, 30]),
        tuple(f"f{index}" for index in range(50)),
        vocabulary,
        np.array([0, 1, 2], dtype=np.int64),
    )
    with pytest.raises(BaselineDatasetError):
        FoldDataset(data, Fold((0,), (1,)), vocabulary)


def test_build_fold_dataset_uses_train_vocabulary_and_one_extraction(monkeypatch):
    selection = DevelopmentSelection(
        (10, 20), np.array([0, 1]), np.array([7, 9], dtype=np.int64)
    )
    fold = Fold((0,), (1,))
    vocabulary = VenueVocabulary.build([1])
    development = SelectedObservationDataset(
        np.zeros((2, 50), dtype=np.float32),
        np.array([0, 1]),
        np.array([10, 20]),
        tuple(f"f{index}" for index in range(50)),
        vocabulary,
        np.array([7, 9], dtype=np.int64),
    )
    calls = []

    def discover(path, ids, *, chunk_size):
        calls.append(("discover", ids, chunk_size))
        return vocabulary

    def extract(path, ids, labels, received, *, chunk_size, source_indices):
        calls.append(("extract", ids, labels, received, chunk_size, source_indices))
        return development

    monkeypatch.setattr(
        "cfm_challenge.baselines.dataset.discover_venue_vocabulary", discover
    )
    monkeypatch.setattr(
        "cfm_challenge.baselines.dataset.extract_selected_observations", extract
    )
    result = build_fold_dataset("unused.csv", selection, fold, chunk_size=137)
    assert calls[0] == ("discover", (10,), 137)
    assert (
        calls[1][0] == "extract"
        and calls[1][1] == (10, 20)
        and calls[1][2] == {10: 0, 20: 1}
        and calls[1][3] is vocabulary
        and calls[1][4] == 137
        and np.array_equal(calls[1][5], selection.source_indices)
    )
    assert (
        isinstance(result, FoldDataset)
        and result.development is development
        and result.fold == fold
        and result.venue_vocabulary is vocabulary
    )


@pytest.mark.parametrize(
    "selection,fold,chunk_size",
    [
        (object(), Fold((0,), (1,)), 1),
        (None, Fold((0,), (1,)), 0),
        (None, Fold((0,), (1,)), -1),
        (None, Fold((), (0, 1)), 1),
        (None, Fold((0, 1), ()), 1),
        (None, Fold((0, 3), (1, 2)), 1),
        (None, Fold((0, 1), (1, 2)), 1),
        (None, Fold((0,), (1,)), 1),
    ],
)
def test_build_fold_dataset_rejects_invalid_inputs_before_x(
    monkeypatch, selection, fold, chunk_size
):
    valid = DevelopmentSelection(
        (10, 20, 30), np.array([0, 1, 0]), np.array([7, 9, 11], dtype=np.int64)
    )
    selection = valid if selection is None else selection
    calls = []
    monkeypatch.setattr(
        "cfm_challenge.baselines.dataset.discover_venue_vocabulary",
        lambda *args, **kwargs: calls.append("discover"),
    )
    monkeypatch.setattr(
        "cfm_challenge.baselines.dataset.extract_selected_observations",
        lambda *args, **kwargs: calls.append("extract"),
    )
    with pytest.raises(BaselineDatasetError):
        build_fold_dataset("unused.csv", selection, fold, chunk_size=chunk_size)
    assert calls == []


def test_build_fold_dataset_maps_validation_only_venue_to_unknown(tmp_path: Path):
    path = tmp_path / "x.csv"
    rows = []
    for obs, venue in ((10, 1), (20, 99)):
        for position in range(100):
            rows.append(
                (
                    obs,
                    venue,
                    position,
                    "A",
                    "B",
                    float(position),
                    0.0,
                    1.0,
                    1,
                    1,
                    False,
                    0,
                )
            )
    pd.DataFrame(
        rows,
        columns=[
            "obs_id",
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
        ],
    ).to_csv(path, index=False)
    selection = DevelopmentSelection(
        (10, 20), np.array([0, 1]), np.array([7, 9], dtype=np.int64)
    )
    result = build_fold_dataset(path, selection, Fold((0,), (1,)), chunk_size=137)
    names = result.development.feature_names
    slots = [names.index(f"venue_slot_{index}_frequency") for index in range(6)]
    unknown = names.index("venue_unknown_frequency")
    assert (
        result.venue_vocabulary.slots[0] == 1
        and 99 not in result.venue_vocabulary.known
        and len(result.venue_vocabulary.slots) == 6
    )
    assert (
        result.development.features.shape == (2, 50)
        and result.development.features.dtype == np.float32
        and np.isfinite(result.development.features).all()
    )
    assert (
        result.development.features[0, slots[0]] == 1.0
        and result.development.features[0, unknown] == 0.0
    )
    assert (
        np.all(result.development.features[1, slots] == 0.0)
        and result.development.features[1, unknown] == 1.0
    )
    assert np.allclose(
        result.development.features[:, slots + [unknown]].sum(axis=1), 1.0
    )
    assert (
        tuple(result.development.observation_ids) == selection.observation_ids
        and result.development.labels.tolist() == [0, 1]
        and result.development.source_indices.tolist() == [7, 9]
        and result.fold == Fold((0,), (1,))
    )


def test_iter_fold_datasets_is_lazy_and_materializes_generator_once(monkeypatch):
    selection = DevelopmentSelection(
        (10, 20), np.array([0, 1]), np.array([7, 9], dtype=np.int64)
    )
    folds = (Fold((0,), (1,)), Fold((1,), (0,)))
    consumed = []
    calls = []

    def source():
        for fold in folds:
            consumed.append(fold)
            yield fold

    monkeypatch.setattr(
        "cfm_challenge.baselines.dataset.build_fold_dataset",
        lambda path, current, fold, *, chunk_size: calls.append(fold) or fold,
    )
    iterator = iter_fold_datasets("unused.csv", selection, source(), chunk_size=1)
    assert consumed == list(folds) and calls == []
    assert next(iterator) == folds[0] and calls == [folds[0]]
    assert next(iterator) == folds[1] and calls == list(folds)
    with pytest.raises(StopIteration):
        next(iterator)
    assert calls == list(folds)


@pytest.mark.parametrize(
    "selection,folds,chunk_size",
    [
        (object(), (), 1),
        (
            DevelopmentSelection(
                (10, 20), np.array([0, 1]), np.array([7, 9], dtype=np.int64)
            ),
            (),
            1,
        ),
        (
            DevelopmentSelection(
                (10, 20), np.array([0, 1]), np.array([7, 9], dtype=np.int64)
            ),
            (Fold((0,), (1,)),),
            0,
        ),
        (
            DevelopmentSelection(
                (10, 20), np.array([0, 1]), np.array([7, 9], dtype=np.int64)
            ),
            (Fold((), (0, 1)),),
            1,
        ),
    ],
)
def test_iter_fold_datasets_validates_inputs_before_x(
    monkeypatch, selection, folds, chunk_size
):
    calls = []
    monkeypatch.setattr(
        "cfm_challenge.baselines.dataset.build_fold_dataset",
        lambda *args, **kwargs: calls.append(1),
    )
    with pytest.raises(BaselineDatasetError):
        iter_fold_datasets("unused.csv", selection, folds, chunk_size=chunk_size)
    assert calls == []


def test_iter_fold_datasets_validates_late_invalid_fold_before_x(monkeypatch):
    selection = DevelopmentSelection(
        (10, 20), np.array([0, 1]), np.array([7, 9], dtype=np.int64)
    )
    calls = []
    monkeypatch.setattr(
        "cfm_challenge.baselines.dataset.build_fold_dataset",
        lambda *args, **kwargs: calls.append(1),
    )
    with pytest.raises(BaselineDatasetError):
        iter_fold_datasets(
            "unused.csv", selection, (Fold((0,), (1,)), Fold((0,), (1,))), chunk_size=1
        )
    assert calls == []


def test_iter_fold_datasets_stops_and_propagates_build_error(monkeypatch):
    selection = DevelopmentSelection(
        (10, 20), np.array([0, 1]), np.array([7, 9], dtype=np.int64)
    )
    folds = (Fold((0,), (1,)), Fold((1,), (0,)))
    calls = []

    def build(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("build failure")
        return folds[0]

    monkeypatch.setattr("cfm_challenge.baselines.dataset.build_fold_dataset", build)
    iterator = iter_fold_datasets("unused.csv", selection, folds, chunk_size=1)
    assert next(iterator) == folds[0]
    with pytest.raises(RuntimeError, match="build failure"):
        next(iterator)
    assert calls == [1, 1]


def test_iter_fold_datasets_integrates_selection_folds_and_train_vocabularies(
    tmp_path: Path,
):
    all_ids = tuple(range(100, 110))
    labels = [index // 5 for index in range(10)]
    y_path = tmp_path / "y.csv"
    pd.DataFrame(zip(all_ids, labels), columns=["obs_id", "eqt_code_cat"]).to_csv(
        y_path, index=False
    )
    selection = build_development_selection(y_path)
    assert len(selection.observation_ids) == 8
    lockbox_ids = {
        all_ids[index]
        for index in set(range(len(all_ids))) - set(selection.source_indices.tolist())
    }
    first_train = tuple(range(4))
    second_train = tuple(range(4, 8))
    folds = (Fold(first_train, second_train), Fold(second_train, first_train))
    first_train_ids = {selection.observation_ids[index] for index in first_train}
    rows = []
    for obs_id in all_ids:
        venue = (
            10
            if obs_id in first_train_ids
            else 20 if obs_id in selection.observation_ids else 99
        )
        for order_id in range(100):
            rows.append(
                (
                    obs_id,
                    venue,
                    order_id,
                    "A",
                    "B",
                    float(order_id),
                    0.0,
                    1.0,
                    1,
                    1,
                    False,
                    0,
                )
            )
    x_path = tmp_path / "x.csv"
    pd.DataFrame(
        rows,
        columns=[
            "obs_id",
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
        ],
    ).to_csv(x_path, index=False)
    datasets = tuple(iter_fold_datasets(x_path, selection, folds, chunk_size=137))
    summaries = build_development_summaries(x_path, selection, chunk_size=137)
    optimized = tuple(iter_fold_datasets_from_summaries(summaries, folds))
    assert [dataset.fold for dataset in datasets] == list(folds)
    assert [dataset.venue_vocabulary.known for dataset in datasets] == [(10,), (20,)]
    assert datasets[0].venue_vocabulary is not datasets[1].venue_vocabulary
    assert [dataset.fold for dataset in optimized] == list(folds)
    assert [dataset.venue_vocabulary.known for dataset in optimized] == [(10,), (20,)]
    for dataset, optimized_dataset, fold in zip(datasets, optimized, folds):
        development = dataset.development
        optimized_development = optimized_dataset.development
        unknown = development.feature_names.index("venue_unknown_frequency")
        assert tuple(development.observation_ids) == selection.observation_ids
        np.testing.assert_array_equal(development.labels, selection.labels)
        np.testing.assert_array_equal(
            development.source_indices, selection.source_indices
        )
        assert set(development.observation_ids).isdisjoint(lockbox_ids)
        np.testing.assert_array_equal(
            development.labels[list(fold.train_indices)],
            selection.labels[list(fold.train_indices)],
        )
        np.testing.assert_array_equal(
            development.labels[list(fold.validation_indices)],
            selection.labels[list(fold.validation_indices)],
        )
        assert np.all(development.features[list(fold.train_indices), unknown] == 0.0)
        assert np.all(
            development.features[list(fold.validation_indices), unknown] == 1.0
        )
        assert development.feature_names == optimized_development.feature_names
        np.testing.assert_allclose(
            optimized_development.features, development.features, rtol=0.0, atol=0.0
        )
        np.testing.assert_array_equal(optimized_development.labels, development.labels)
        np.testing.assert_array_equal(
            optimized_development.source_indices, development.source_indices
        )
        np.testing.assert_array_equal(
            optimized_development.observation_ids, development.observation_ids
        )
    reordered = tuple(
        iter_fold_datasets_from_summaries(summaries, tuple(reversed(folds)))
    )
    assert [dataset.fold for dataset in reordered] == list(reversed(folds))
    for first, second in zip(reversed(optimized), reordered):
        assert first.venue_vocabulary == second.venue_vocabulary
        np.testing.assert_allclose(
            first.development.features, second.development.features, rtol=0.0, atol=0.0
        )


@pytest.mark.parametrize(
    "sources",
    [
        ([1],),
        ([1, 2, 3, 4],),
        ([[1, 2, 3]],),
        ([1, 1, 2],),
        ([-1, 2, 3],),
        ([True, 2, 3],),
        ([np.bool_(True), 2, 3],),
        ([1.0, 2, 3],),
        (["1", 2, 3],),
        ([None, 2, 3],),
    ],
)
def test_source_indices_invalid_contract_is_rejected(tmp_path: Path, sources):
    path = tmp_path / "x.csv"
    write_x(path)
    vocab = discover_venue_vocabulary(path, (30, 10, 20), chunk_size=137)
    with pytest.raises(BaselineDatasetError, match="source_indices"):
        extract_selected_observations(
            path,
            (30, 10, 20),
            {30: 0, 10: 1, 20: 2},
            vocab,
            chunk_size=137,
            source_indices=sources[0],
        )


def test_source_indices_preserve_order_default_and_are_immutable(tmp_path: Path):
    path = tmp_path / "x.csv"
    write_x(path)
    ids = (30, 10, 20)
    vocab = discover_venue_vocabulary(path, ids, chunk_size=137)
    supplied = np.array([7, 1, 9])
    data = extract_selected_observations(
        path, ids, {30: 2, 10: 0, 20: 1}, vocab, chunk_size=137, source_indices=supplied
    )
    assert data.source_indices.dtype == np.int64 and data.source_indices.tolist() == [
        7,
        1,
        9,
    ]
    assert (
        data.observation_ids.tolist() == [30, 10, 20]
        and data.labels.tolist() == [2, 0, 1]
        and data.features.shape == (3, 50)
    )
    supplied[0] = 99
    assert data.source_indices[0] == 7
    with pytest.raises(ValueError):
        data.source_indices[0] = 0
    default = extract_selected_observations(
        path, ids, {30: 2, 10: 0, 20: 1}, vocab, chunk_size=137
    )
    np.testing.assert_array_equal(default.source_indices, np.arange(3, dtype=np.int64))
