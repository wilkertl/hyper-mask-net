# Research: F2 Data and Embeddings

Phase 0 output for [plan.md](plan.md). Facts were checked against the live sources on 2026-10-12.

## D1. Sources and pinned revisions

- **Decision**: BEIR corpora and queries from the Hugging Face Hub as parquet files, judgments as
  TSV files, each pinned by repository commit (research R2, confirmed by the researcher):

  | Environment | Corpus and queries | Revision | Judgments | Revision |
  |---|---|---|---|---|
  | scifact | `BeIR/scifact` | `b3b5335604bf5ee3c4447671af975ea25143d4f5` | `BeIR/scifact-qrels` (train, test) | `2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029` |
  | nfcorpus | `BeIR/nfcorpus` | `b5026a0e96e8a7ac4f95f482a596389289d46269` | `BeIR/nfcorpus-qrels` (train, dev, test) | `a451b3b26d3ae1358f259c1a3a4dd61fcea35a65` |
  | msmarco | `BeIR/msmarco` | `a918e0d11a77ed33f42f29d98340b655593b96ad` | `BeIR/msmarco-qrels` (train, dev, test) | `253fbf8a3f8d4a0932b63882b5162bedc84779f5` |

  Files are fetched with `huggingface_hub.hf_hub_download(..., revision=<sha>)` and read with
  `pyarrow`, both in the locked `data` extra. TREC DL 2019, DL 2020, and DL-HARD come from
  `ir_datasets` (`msmarco-passage/trec-dl-2019/judged`, `msmarco-passage/trec-dl-2020/judged`,
  `msmarco-passage/trec-dl-hard`, version 0.6.3).
- **Verified**: SciFact has 5,183 documents, 1,109 queries, and 300 test queries; NFCorpus 3,633,
  3,237, and 323; DL19, DL20, and DL-HARD have 43, 54, and 50 judged queries, every one with a
  positive judgment (relevance 0–3). The Hugging Face SciFact and NFCorpus documents and queries
  are identical to the UKP archives the F0 spike embedded, so SC-004 compares like with like.
  MS MARCO counts (8,841,823 passages, 6,980 dev queries) are checked when it is prepared.
- **Rationale**: Pinned commits make the dataset fingerprint meaningful and survive the UKP host
  disappearing; parquet reads in streamed batches, which MS MARCO needs.
- **Alternatives considered**: the UKP archives the spike uses (unversioned URLs; rejected by the
  researcher); the `datasets` library's `load_dataset` (adds a cache layer and Arrow conversion for
  no benefit over reading the parquet files directly).

## D2. Normalized form and leakage-safe layout

- **Decision**: A prepared environment is a published artifact `artifacts/datasets/<env>/<run_id>/`
  holding `documents.jsonl` (`DocumentRecord`), `queries.jsonl` (`QueryRecord`, every split),
  one judgments file per split — `qrels-train.jsonl`, `qrels-validation.jsonl`,
  `qrels-test.jsonl` — and one per evaluation query set (`qrels-trec-dl-2019.jsonl`, ...), plus
  `dataset.json` (the dataset manifest record). Test and evaluation-set judgments live only in their
  own files, and `load_environment(..., include_test_qrels=False)` never opens them; training,
  negative mining, and target construction call it without that flag.
- **Rationale**: A file that is never opened cannot leak (constitution II: enforced by code). The
  guard is testable by corrupting or deleting the test files and checking training outputs are
  unchanged, the pattern the spike used.
- **Document text**: `title + " " + text` when the title is non-empty, else `text`, as in the spike
  and BEIR's encoders; records keep title and text separately.

## D3. Splits

- **Decision**: Train queries are those with at least one positive judgment in the source's train
  split whose positive documents exist; validation is `split_ids(train, 0.1, seed 0)`; test is the
  judged queries of the source's test split (`dev` for MS MARCO, as in Learning to Select and the
  DIME papers). NFCorpus's own `dev` split is not used, so all three environments are split the same
  way. MS MARCO training queries are capped at a fixed-seed sample of 50,000 before the validation
  split (the Learning-to-Select 50,000-pair protocol, research R6), which also bounds query
  embedding and negative mining.

## D4. Embedding settings and the reuse guard

- **Decision**: Settings recorded with every embedding and compared before reuse: model name,
  revision `97b0c614…`, width 1024, last-token pooling, L2 normalization, `max_length` 512, compute
  dtype (`bfloat16` on GPU), storage dtype `float16`, backend (`local` or `vllm`), and, for queries,
  the instruction. Reuse with any difference is refused, naming the differing keys (FR-012).
  Including the backend means a vLLM-made cache is never mixed with a local one.

## D5. Sharded, resumable embedding

- **Decision**: Documents are embedded in shards of `shard_size` rows (default 65,536) in source
  order. Each completed shard is saved as `part-NNNNN.npy` in a work directory
  `artifacts/.work/embeddings/<env>/<role>/<settings-hash>/` with the settings file; a restart
  skips completed shards. Texts are streamed from `documents.jsonl`, so memory holds one shard of
  text, never the corpus. When all shards exist, they are concatenated into the published
  artifact's `documents.npy` through a NumPy memmap (no full copy in RAM), the `EmbeddingArtifact`
  record is written, and `ArtifactWriter` publishes; the work directory is then removed. Within a
  shard, texts are length-sorted into batches (the spike's `QwenEncoder.encode` does this).
- **Rationale**: Publication stays atomic (F1's gate) while the expensive part is resumable; at
  most one shard is lost on interruption (SC-005). The spike's `embed_cached` holds every text in
  memory and finalizes in place, so it is generalized rather than reused verbatim: its part files
  and settings guard are kept.
- **Memory**: a 0.6B model in bfloat16 is ~1.2 GB; batch 128 × 512 tokens fits easily in 24 GB.
  Storage: 8,841,823 × 1024 × 2 bytes ≈ 18.1 GB for MS MARCO documents.

## D6. Exact search for negatives at MS MARCO scale

- **Decision**: `exact_search_streamed(queries, documents, depth, chunk_size, device)` scores
  queries against document chunks read from the memmapped `documents.npy`, keeps a running top-
  `depth` per query, and returns global row indices; on a GPU, chunks of 1M documents (2 GB in
  float16) and query batches of 4,096 fit in 24 GB. Sampling follows ADR-0002 exactly: 64 with
  replacement from the top 1,000 minus judged positives and the query's own id, seed 0. The
  spike's `sample_hard_negatives` is split so the sampling step takes precomputed rankings.
- **Rationale**: 55,000 MS MARCO queries × 8.8M documents is ~5 × 10¹⁴ multiply-adds, minutes on a
  24 GB GPU; the full score matrix never exists in memory.

## D7. Commands

- **Decision**: `hyperdime prepare-data`, `hyperdime embed`, `hyperdime mine-negatives`, each a
  subcommand like `diagnose`: Hydra config (`experiment/prepare` with `data=<env>`), exit 2 on an
  invalid config with nothing written, exit 1 on other errors, artifact path on stdout. `embed` takes
  `--backend local|vllm`, `--shard-size`, `--batch-size`; the device comes from `run.device`.
  Inputs are found as the latest published artifact of the right kind for the environment, and the
  manifest records which one (`dataset_hash` = the dataset manifest's fingerprint).

## D8. Where F2 runs

- **Decision**: Implementation and tests run anywhere on CPU. `prepare-data` for MS MARCO,
  `embed`, and `mine-negatives` run on the RTX 4090 workstation over SSH, so every F2 vector comes
  from one machine (spec Assumptions). SC-006's 12-hour ceiling is replaced by a measured estimate
  from a throughput run on that machine before the full MS MARCO job.
