# Contract: `hyperdime.contracts` Python API

The public surface later modules import. Field definitions are in
[data-model.md](../data-model.md). Signatures are normative; bodies are not specified here.

## `hyperdime.contracts.schemas`

```python
class Record(BaseModel):            # frozen, extra="forbid"
    schema_version: int

class QueryRecord(Record): ...
class DocumentRecord(Record): ...
class QrelsRecord(Record): ...
class EmbeddingArtifact(Record): ...
class SpaceDescriptor(Record): ...
class SelectorCheckpoint(Record): ...
class EvaluationResult(Record): ...
class PayloadRef(BaseModel): ...

def dump_record(record: Record, path: Path) -> None
def load_record(cls: type[R], path: Path) -> R
def dump_records(records: Iterable[Record], path: Path) -> None        # JSON Lines
def load_records(cls: type[R], path: Path) -> list[R]
```

- `load_record` / `load_records` raise `ContractError` for unsupported `schema_version`, unknown
  fields, or invalid values; the message names the record type, the field, and (for JSON Lines)
  the line number.

## `hyperdime.contracts.validation`

```python
class ContractError(ValueError): ...
class ConfigError(ContractError): ...

def validate_config(raw: Mapping[str, Any]) -> RunConfig
```

- `ContractError` is the only exception type raised for invalid records or configs.

## `hyperdime.contracts.config`

```python
class ModelConfig(BaseModel): ...
class DataConfig(BaseModel): ...
class RunSettings(BaseModel): ...
class RunConfig(BaseModel): ...

def load_config(config_dir: Path, config_name: str, overrides: Sequence[str] = ()) -> RunConfig
```

- Composes with Hydra's compose API, then calls `validate_config`. Never changes the working
  directory and writes nothing.

## `hyperdime.contracts.hashing`

```python
def canonical_json(obj: Any) -> str
def config_hash(config: RunConfig | Mapping[str, Any]) -> str      # 64 hex
def file_sha256(path: Path) -> str
def tree_sha256(root: Path) -> str
NO_DATASET: Final = "none"
```

- `config_hash(a) == config_hash(b)` whenever `a` and `b` are equal after validation, regardless of
  key order. Non-finite floats raise `ContractError`.

## `hyperdime.contracts.seeding`

```python
def seed_everything(seed: int, *, deterministic: bool = True) -> dict[str, str | bool | None]
```

- Seeds `random`, NumPy, and torch (CPU and CUDA). Returns what it set for the manifest.

## `hyperdime.contracts.manifests`

```python
class RunManifest(Record): ...

def start_manifest(command: str, config: RunConfig, *, dataset_hash: str = NO_DATASET,
                   model_revision: str | None = None, device: str | None = None) -> RunManifest
def finish_manifest(manifest: RunManifest, artifacts: Sequence[tuple[str, str]]) -> RunManifest

class ArtifactWriter:
    def __init__(self, root: Path, kind: str, manifest: RunManifest) -> None
    def path(self, name: str) -> Path              # staging path for a payload file
    def write_record(self, name: str, record: Record) -> None
    def commit(self) -> Path                        # publishes; returns the artifact directory
    def __enter__(self) -> ArtifactWriter
    def __exit__(self, *exc: object) -> None       # removes staging on error, commits otherwise

def read_manifest(artifact_dir: Path) -> RunManifest
```

Guarantees (tested):

1. `ArtifactWriter(root, kind, None)` raises `ContractError`; so does `commit()` if any staged
   file is missing from disk.
2. Nothing is visible under `root/kind/` until `commit()` returns; the published directory always
   contains `manifest.json`, whose `artifacts` list matches the payload files and their SHA-256.
3. An exception inside the `with` block leaves no directory under `root/kind/` except hidden
   `.tmp-` directories, which are removed on a best-effort basis.
4. `read_manifest` raises `ContractError` for a directory without `manifest.json`.
5. Committing to an existing `run_id` directory raises; published artifacts are never overwritten.

## Stability

Everything above is public API from this feature on. Changes require a schema-version bump (for
records) and justification in the pull request (constitution principle III).
