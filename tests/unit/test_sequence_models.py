from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cfm_challenge.sequences.cache import build_sequence_cache
from cfm_challenge.sequences.models import EventEncoder, PilotBiGRU, PilotCNN
from test_sequence_cache import sequence_fixture


def test_event_encoder_uses_only_training_rows_and_unknown_venue(
    tmp_path: Path,
) -> None:
    source = tmp_path / "x.csv"
    selection = sequence_fixture(source)
    cache = build_sequence_cache(source, selection, tmp_path / "cache", chunk_size=137)
    encoder = EventEncoder.fit(cache, [0, 1])
    assert encoder.venues == (1, 2)
    np.testing.assert_allclose(
        encoder.scaler.mean_, cache.numeric[[0, 1]].mean(axis=(0, 1))
    )
    assert not np.allclose(encoder.scaler.mean_, cache.numeric.mean(axis=(0, 1)))
    features = encoder.transform(cache, [2])
    assert features.shape == (1, 100, 20) and features.dtype == np.float32
    assert np.isfinite(features).all()
    np.testing.assert_array_equal(features[0, :, 12], np.ones(100))
    np.testing.assert_array_equal(features[:, :, 6:13].sum(axis=2), np.ones((1, 100)))


@pytest.mark.parametrize("factory", [PilotCNN, PilotBiGRU])
def test_pilot_network_forward_backward_and_deterministic_initialization(
    factory: object,
) -> None:
    torch.set_num_threads(2)
    features = (
        torch.arange(4 * 100 * 20, dtype=torch.float32).reshape(4, 100, 20) / 8000
    )
    labels = torch.tensor([0, 1, 2, 3])
    torch.manual_seed(42)
    model = factory()
    initial = model(features).detach()
    torch.manual_seed(42)
    torch.testing.assert_close(initial, factory()(features))
    assert initial.shape == (4, 24)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = torch.nn.CrossEntropyLoss()(model(features), labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    optimizer.step()
    assert not torch.equal(initial, model(features))
