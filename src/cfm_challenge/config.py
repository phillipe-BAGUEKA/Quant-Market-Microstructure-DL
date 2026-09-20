"""Small, side-effect-free project configuration."""

from dataclasses import dataclass
from pathlib import Path

from cfm_challenge.data.schema import EXPECTED_CLASS_COUNT, EXPECTED_SEQUENCE_LENGTH


DEFAULT_RANDOM_SEED = 42


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """Paths and fixed challenge dimensions derived from a configurable root."""

    root: Path = Path(".")
    sequence_length: int = EXPECTED_SEQUENCE_LENGTH
    class_count: int = EXPECTED_CLASS_COUNT
    random_seed: int = DEFAULT_RANDOM_SEED

    def __post_init__(self) -> None:
        """Normalize the root to a Path without touching the filesystem."""
        object.__setattr__(self, "root", Path(self.root))

    @property
    def raw_data_dir(self) -> Path:
        """Return the directory containing immutable source data."""
        return self.root / "data" / "raw"

    @property
    def interim_data_dir(self) -> Path:
        """Return the directory for intermediate data."""
        return self.root / "data" / "interim"

    @property
    def processed_data_dir(self) -> Path:
        """Return the directory for prepared data."""
        return self.root / "data" / "processed"

    @property
    def models_dir(self) -> Path:
        """Return the directory for model artifacts."""
        return self.root / "models"

    @property
    def reports_dir(self) -> Path:
        """Return the directory for reports."""
        return self.root / "reports"
