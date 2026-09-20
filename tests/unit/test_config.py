import inspect
import re
from pathlib import Path

import pytest

import cfm_challenge.config as config_module
from cfm_challenge.config import DEFAULT_RANDOM_SEED, ProjectConfig


def test_challenge_dimensions_and_default_seed() -> None:
    config = ProjectConfig()

    assert config.sequence_length == 100
    assert config.class_count == 24
    assert config.random_seed == DEFAULT_RANDOM_SEED == 42


def test_paths_are_derived_from_explicit_root(tmp_path: Path) -> None:
    config = ProjectConfig(root=tmp_path)

    assert config.root == tmp_path
    assert config.raw_data_dir == tmp_path / "data" / "raw"
    assert config.interim_data_dir == tmp_path / "data" / "interim"
    assert config.processed_data_dir == tmp_path / "data" / "processed"
    assert config.models_dir == tmp_path / "models"
    assert config.reports_dir == tmp_path / "reports"


def test_configuration_does_not_create_directories(tmp_path: Path) -> None:
    config = ProjectConfig(root=tmp_path)

    derived_paths = (
        config.raw_data_dir,
        config.interim_data_dir,
        config.processed_data_dir,
        config.models_dir,
        config.reports_dir,
    )
    assert all(not path.exists() for path in derived_paths)


def test_configuration_contains_no_personal_absolute_path() -> None:
    source = inspect.getsource(config_module)

    assert re.search(r"[A-Za-z]:[\\/]", source) is None
    assert not ProjectConfig().root.is_absolute()


def test_configuration_is_immutable() -> None:
    config = ProjectConfig()

    with pytest.raises(AttributeError):
        config.random_seed = 7  # type: ignore[misc]
