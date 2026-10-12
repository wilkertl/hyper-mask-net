"""Validated run configuration composed from Hydra groups (research R2, R12)."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from hyperdime.contracts.schemas import Dtype, Id


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
