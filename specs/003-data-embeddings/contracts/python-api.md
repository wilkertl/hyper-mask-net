# Contract: F2 Python API

Records are in [data-model.md](../data-model.md). Signatures are normative; later features (F3+)
use only these.

```python
# hyperdime.data.environments
class EvalQuerySet(BaseModel): name: str; loader: Literal["ir_datasets"]; source: str
class EnvironmentSpec(BaseModel): ...            # spec_version 1
ENVIRONMENTS: Mapping[str, EnvironmentSpec]       # scifact, nfcorpus, msmarco
def environment_spec(name: str) -> EnvironmentSpec  # ContractError for an unknown name

# hyperdime.data.prepare
@dataclass(frozen=True)
class DatasetBundle:
    environment: str
    documents: Iterator[DocumentRecord] | list[DocumentRecord]   # streamed for large corpora
    queries: dict[str, QueryRecord]                                # split in QueryRecord.split
    qrels: dict[str, list[QrelsRecord]]                            # keyed by judgments file name
    manifest: DatasetManifest

def prepare_environment(spec: EnvironmentSpec, data: DataConfig, writer: ArtifactWriter,
                        source: SourceReader | None = None) -> DatasetManifest
def load_environment(dataset_dir: Path, *, include_test_qrels: bool = False,
                     stream_documents: bool = False) -> DatasetBundle
def iter_documents(dataset_dir: Path) -> Iterator[DocumentRecord]
```
- `load_environment(include_test_qrels=False)` returns `qrels` without `test` and without any
  evaluation query set, and opens none of their files; only evaluation passes `True`.
- `prepare_environment` drops judgments whose document is missing, excludes queries left without
  a positive, refuses duplicate ids, and records every count in the `DatasetManifest`.

```python
# hyperdime.embeddings.encoders
def encoder_settings(model: ModelConfig, backend: str, role: str, instruction: str | None) -> dict[str, Any]
def build_encoder(model: ModelConfig, backend: str, device: str, batch_size: int) -> Encoder

# hyperdime.embeddings.cache
def embed_sharded(ids: Sequence[str], texts: Iterable[str], encode: Callable[[Sequence[str]], Tensor],
                  work_dir: Path, settings: Mapping[str, Any], shard_size: int = 65_536) -> list[Path]
def publish_embeddings(writer: ArtifactWriter, name: str, ids: Sequence[str], parts: Sequence[Path],
                       settings: Mapping[str, Any], role: str, environment: str,
                       instruction: str | None, model: ModelConfig) -> EmbeddingArtifact
def load_embeddings(artifact_dir: Path, name: str, *, mmap: bool = True) -> tuple[EmbeddingArtifact, np.ndarray]
def check_settings(stored: Mapping[str, Any], requested: Mapping[str, Any]) -> None   # ContractError naming keys

# hyperdime.retrieval.exact
def exact_search_streamed(queries: Tensor, documents: np.ndarray, depth: int, *,
                          chunk_size: int = 1_000_000, device: str = "cpu") -> Tensor  # (Q, depth) row indices

# hyperdime.data.negatives
def sample_from_rankings(rankings: Tensor, document_ids: Sequence[str], excluded: Sequence[Collection[str]],
                         count: int, generator: torch.Generator) -> list[list[str]]
def build_negative_pools(bundle: DatasetBundle, query_vectors: np.ndarray, query_ids: Sequence[str],
                         documents: np.ndarray, document_ids: Sequence[str], *, depth: int = 1000,
                         count: int = 64, seed: int = 0, device: str = "cpu") -> list[NegativePool]

# hyperdime.artifacts
def latest_artifact(root: Path, kind: str, environment: str) -> Path   # ContractError if none
```
- `exact_search_streamed` returns the same indices as `exact_search` on the full matrix, ties
  broken by lower row index.
- `embed_sharded` keeps completed shards, skips them on restart, and refuses a work directory
  whose stored settings differ; the published vectors equal an uninterrupted run's.
