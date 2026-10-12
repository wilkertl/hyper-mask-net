"""Validated run configuration composed from Hydra groups (research R2, R12)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from hyperdime.contracts.schemas import Dtype, Id
from hyperdime.contracts.validation import ConfigError, format_config_error


class _Config(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ModelConfig(_Config):
    """The frozen encoder (constitution I): fixed name, width 1024, pinned revision."""

    name: Literal["Qwen/Qwen3-Embedding-0.6B"]
    revision: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
    embedding_dim: Literal[1024]
    frozen: Literal[True]
    normalize: bool
    max_length: Annotated[int, Field(gt=0)]
    dtype: Dtype


class SampleConfig(_Config):
    num_documents: Annotated[int, Field(ge=1)]
    num_queries: Annotated[int, Field(ge=1)]
    seed: Annotated[int, Field(ge=0)]


class DataConfig(_Config):
    environment: Id
    source: Annotated[str, StringConstraints(min_length=1)]
    instruction: str | None
    validation_fraction: Annotated[float, Field(gt=0, lt=1)]
    sample: SampleConfig | None = None


class RunSettings(_Config):
    seed: Annotated[int, Field(ge=0)]
    device: Annotated[str, StringConstraints(pattern=r"^(auto|cpu|cuda|cuda:\d+)$")]
    dtype: Dtype
    output_dir: Annotated[str, StringConstraints(min_length=1)]


class RunConfig(_Config):
    model: ModelConfig
    data: DataConfig
    run: RunSettings
    config_version: Literal[1]


def validate_config(raw: Mapping[str, Any]) -> RunConfig:
    """Validate a composed config; every problem is reported by its dotted field path."""
    try:
        return RunConfig.model_validate(raw)
    except ValidationError as error:
        raise ConfigError(format_config_error(error)) from error


def load_config(config_dir: Path, config_name: str, overrides: Sequence[str] = ()) -> RunConfig:
    """Compose ``config_name`` from the Hydra groups in ``config_dir`` and validate it.

    Uses Hydra's compose API, so the working directory is never changed and nothing is
    written; Hydra's global state is cleared when composition ends.
    """
    from hydra import compose, initialize_config_dir
    from hydra.errors import HydraException
    from omegaconf import OmegaConf
    from omegaconf.errors import OmegaConfBaseException

    try:
        with initialize_config_dir(
            config_dir=str(Path(config_dir).resolve()), version_base=None, job_name="hyperdime"
        ):
            composed = compose(config_name=config_name, overrides=list(overrides))
        raw = OmegaConf.to_container(composed, resolve=True)
    except (HydraException, OmegaConfBaseException) as error:
        raise ConfigError(f"{config_name}: {error}") from error
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_name}: the composed config is not a mapping")
    return validate_config({str(key): value for key, value in raw.items()})
