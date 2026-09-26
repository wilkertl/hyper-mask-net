# Contract: module interfaces across milestones

The boundaries between packages that different features, and different agents, must agree on
(constitution VII; workflow: "two branches must not change the same package unless both issues state
the contract between them"). Signatures are the target; each owning feature may add parameters with
defaults but must not change these without an ADR. Records are from
[data-model.md](../data-model.md) and [001 data-model](../../001-m0-foundation/data-model.md).

Existing functions keep their signatures: `split_ids`, `mine_hard_negatives`, `format_query`,
`last_token_pool`, `exact_search`, `query_metrics`, `evaluate_run`, `oracle_scores`,
`oracle_importance`, `resolve_k`, `top_k_mask`, `apply_mask`, `LinearSelector`, `selector_kl_loss`.

## F2 — `hyperdime.data`, `hyperdime.embeddings`

```python
# data/loaders.py
class DatasetBundle:                 # plan §5 M1
    environment: str
    corpus: dict[str, DocumentRecord]
    queries: dict[str, QueryRecord]
    qrels: list[QrelsRecord]
    instruction: str

def load_environment(spec: EnvironmentSpec, split: Split, *, allow_test_qrels: bool = False) -> DatasetBundle
```
- `allow_test_qrels=False` returns test queries with **no** qrels; only evaluation code passes `True`.

```python
# embeddings/qwen.py
class QwenEncoder:
    def __init__(self, config: ModelConfig, device: str) -> None
    def encode_queries(self, texts: Sequence[str], instruction: str | None) -> Tensor  # (N, 1024)
    def encode_documents(self, texts: Sequence[str]) -> Tensor                         # (N, 1024)

# embeddings/cache.py
def embed_environment(bundle: DatasetBundle, encoder: QwenEncoder, writer: ArtifactWriter) -> tuple[EmbeddingArtifact, EmbeddingArtifact]
def load_embeddings(artifact_dir: Path) -> tuple[EmbeddingArtifact, Tensor]
```

## F3 — `hyperdime.oracle`, `hyperdime.baselines`, `hyperdime.training`

```python
# oracle/targets.py
def build_targets(queries: Tensor, documents: Tensor, doc_index: Mapping[str, int],
                  positives: Mapping[str, Mapping[str, int]], negatives: Mapping[str, Sequence[str]],
                  query_ids: Sequence[str], temperature: float, m: int) -> Tensor   # (Q, 1024), rows sum to 1

# training/selector_trainer.py
def train_selector(targets: Tensor, queries: Tensor, val_targets: Tensor, val_queries: Tensor,
                   config: SelectorTrainConfig) -> tuple[LinearSelector, list[dict[str, float]]]

# baselines/*.py — every baseline implements:
class ImportanceModel(Protocol):
    def importance(self, query_embeddings: Tensor) -> Tensor                        # (B, 1024)
# mrl_prefix.PrefixImportance, random_mask.RandomImportance(seed),
# global_selector.GlobalImportance, OracleImportance(targets), LinearSelector (via adapter)
```

## F3/F4 — `hyperdime.retrieval`, `hyperdime.evaluation`

```python
# retrieval/scoring.py
def masked_search(model: ImportanceModel, k: int | float, queries: Tensor, documents: Tensor,
                  doc_ids: Sequence[str], query_ids: Sequence[str], depth: int = 100) -> dict[str, list[str]]

# evaluation/statistics.py
def paired_randomization_test(a: Mapping[str, float], b: Mapping[str, float], n: int = 10_000, seed: int = 0) -> float
def paired_bootstrap_ci(a: Mapping[str, float], b: Mapping[str, float], n: int = 10_000, level: float = 0.95, seed: int = 0) -> tuple[float, float]
def holm(p_values: Mapping[str, float]) -> dict[str, float]

# evaluation/selector_shift.py
def selector_properties(model: ImportanceModel, battery: Tensor, k_values: Sequence[int]) -> SelectorProperties
def transfer_matrix(...) -> list[TransferCell]
```

## F5 — `hyperdime.space`

```python
# space/descriptor.py — one builder per protocol; signatures enforce allowed inputs (research R11)
def build_descriptor_p0(documents: Tensor, *, sample: int, seed: int) -> SpaceDescriptor
def build_descriptor_p1(documents: Tensor, calib_queries: Tensor, *, sample: int, seed: int) -> SpaceDescriptor
def build_descriptor_p2(documents: Tensor, calib_queries: Tensor, labels: LabeledCalibration, *, sample: int, seed: int) -> SpaceDescriptor

class DescriptorNormalizer:
    @classmethod
    def fit(cls, descriptors: Sequence[SpaceDescriptor], split: MetaSplit) -> DescriptorNormalizer  # raises on non-meta-train input
    def transform(self, descriptor: SpaceDescriptor) -> Tensor                                     # (1024, h)
```
- Invariants (tested): document-order invariance; dimension permutation permutes rows.

## F6 — `hyperdime.space.encoder`, `hyperdime.hypernet`

```python
class SpaceEncoder(nn.Module):        # (1024, h) -> (1024, h_z)
class LowRankSelector(nn.Module):     # forward(e_q, W0, b0, A, B, delta_b) -> log-importance (B, 1024)
class HyperHead(nn.Module):           # (1024, h_z) -> (A, B, delta_b); zero-gated at init
def generate(descriptor: Tensor, encoder: SpaceEncoder, head: HyperHead, base: LinearSelector) -> ModulationPackage

# training/meta_trainer.py
def meta_train(config: MetaTrainConfig, split: MetaSplit) -> Path   # refuses meta-test environments
```

## F7 — `hyperdime.selection`, protocols

```python
# selection/rdime.py
def adaptive_k(importance: Tensor, **criterion_params: float) -> Tensor   # (B,) ints in [1, 1024]
```
- Policies consume importance and return masks; they never see labels or documents.
