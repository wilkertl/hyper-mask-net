# Feature Specification: F2 Data and Embeddings

**Feature Branch**: `agent/F2-data-embeddings`

**Created**: 2026-10-12

**Status**: Draft

**Input**: User description: "F2 data-embeddings (roadmap feature F2, backlog M1.1–M1.4 and M2.1–M2.2; specs/002-hyperdime-roadmap plan and research R2, R5, R17; contracts from F1 in hyperdime.contracts). Load SciFact and NFCorpus (BEIR, train/dev/test qrels) and the full MS MARCO passage corpus with its train/dev queries, plus the TREC DL 2019, DL 2020 and DL-HARD query sets as extra evaluation-only sets over the MS MARCO corpus, into one normalized form of QueryRecord, DocumentRecord and QrelsRecord, generic over an EnvironmentSpec. Query-level train/validation/test splits (10% of training queries for validation) with guards that keep test labels out of every training path. Embed every environment once with the frozen Qwen3-Embedding-0.6B (pinned revision 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3) into a sharded, resumable embedding cache published as EmbeddingArtifact records through ArtifactWriter; MS MARCO (8.8M passages) must be embeddable in shards on one 24 GB GPU and resume after interruption, and all F2 embeddings are made on one machine. Hard-negative pools for training queries only, following ADR-0002. Reuse the spike's BEIR loader, encoders, and cache. CLI subcommands prepare-data, embed, mine-negatives. Exit: dataset manifests and cached embeddings for SciFact, NFCorpus and MS MARCO, ready for F3."

## User Scenarios & Testing *(mandatory)*

The users are the researcher, who runs the data and embedding steps once per environment on the
GPU workstation, and the coding agents implementing F3 (the Learning-to-Select baseline and gate
G1), who consume what this feature produces. Both need the three initial environments in one
shared form, with leakage impossible by construction and every vector traceable to the exact data,
model revision, and settings that produced it.

### User Story 1 - Environments in one shared, leakage-safe form (Priority: P1)

The researcher prepares SciFact and NFCorpus with one command each. Each becomes a set of queries,
documents, and relevance judgments in the project's shared record types, with a fixed
train/validation/test split of its queries and a dataset manifest that fingerprints exactly what
was loaded. When a training step loads an environment, it receives test queries without their
judgments; only an evaluation step that asks for them explicitly gets test judgments.

**Why this priority**: every later step (embedding, negatives, oracle targets, evaluation) reads
these records, and the constitution's non-negotiable rule — no test labels in any training path —
is enforced here or nowhere.

**Independent Test**: Prepare SciFact and NFCorpus, check their query, document, and judgment
counts against the published dataset statistics, load each for training and for evaluation, and
check that test judgments appear only in the evaluation load.

**Acceptance Scenarios**:

1. **Given** SciFact and NFCorpus, **When** they are prepared, **Then** their document and test
   query counts match the published statistics, and each has a dataset manifest with a fingerprint
   of the loaded data and the pinned source revision.
2. **Given** a prepared environment, **When** its training queries are split, **Then** 10% of them
   form a validation split chosen by a fixed seed, train, validation, and test queries are
   disjoint, and preparing again gives identical splits.
3. **Given** a prepared environment, **When** a training step loads it, **Then** test queries carry
   no judgments; **When** an evaluation step loads it and asks for test judgments, **Then** it
   receives them.
4. **Given** the same environment prepared twice from the same pinned source, **When** the dataset
   fingerprints are compared, **Then** they are identical.

---

### User Story 2 - Each environment embedded once, verifiably (Priority: P1)

The researcher embeds a prepared environment's documents and queries with the frozen model. The
vectors are published together with a record that says which ids they belong to, which model
revision and settings produced them, and a fingerprint of the stored vectors. Later steps load
them instead of re-embedding, and a request to reuse vectors made with different settings is
refused.

**Why this priority**: the frozen coordinate system (constitution I) only holds if each environment
is embedded exactly once and reused; F3 and every later milestone read these vectors.

**Independent Test**: Embed SciFact, reload its published vectors, check row count, width, unit
norm, and id order, and compare them with the F0 spike's SciFact vectors made with the same model
revision.

**Acceptance Scenarios**:

1. **Given** a prepared environment, **When** it is embedded, **Then** one document artifact and
   one query artifact are published, each with the row ids in order, width 1024, unit-norm rows,
   the model revision, the input-length limit, and (for queries) the instruction.
2. **Given** an environment already embedded with the same settings, **When** embedding is
   requested again, **Then** the existing vectors are reused and nothing is recomputed.
3. **Given** cached vectors made with a different model revision, input-length limit, or
   instruction, **When** a step asks to reuse them, **Then** it is refused with a message naming
   the differing setting.
4. **Given** SciFact embedded by this feature and by the F0 spike with the same model revision,
   **When** corresponding vectors are compared, **Then** they agree to within the tolerance of
   reduced-precision arithmetic.

---

### User Story 3 - The full MS MARCO passage corpus and its TREC query sets (Priority: P2)

The researcher prepares and embeds the full MS MARCO passage corpus (about 8.8 million passages)
with its training and development queries on one 24 GB GPU, in shards, over one or more sessions.
If the job stops, it resumes from the last completed shard. The TREC DL 2019, DL 2020, and DL-HARD
query sets are prepared as evaluation-only query sets over the same corpus and reuse its document
vectors rather than embedding the corpus again.

**Why this priority**: MS MARCO with the TREC DL sets is needed for the Learning-to-Select and DIME
comparisons at G1, but it is by far the largest cost; the smaller environments let F3 start first.

**Independent Test**: Embed a small slice of MS MARCO, interrupt the job, restart it, and check
that it continues from the last completed shard and that the result equals an uninterrupted run;
then check that the TREC DL query sets resolve their judgments against the MS MARCO passage ids.

**Acceptance Scenarios**:

1. **Given** the MS MARCO corpus, **When** it is embedded, **Then** the job runs within the memory of
   one 24 GB GPU, writes completed shards as it goes, and publishes the full document artifact only
   when every shard is complete.
2. **Given** an embedding job interrupted partway, **When** it is restarted with the same settings,
   **Then** completed shards are kept, work resumes after the last completed shard, and the
   published vectors equal those of an uninterrupted run.
3. **Given** the TREC DL 2019, DL 2020, and DL-HARD query sets, **When** they are prepared, **Then**
   each is an evaluation-only query set of the MS MARCO environment, its judged-query counts match
   the published statistics, and all its judgments refer to MS MARCO passage ids.
4. **Given** the TREC DL query sets, **When** they are embedded, **Then** only their queries are
   embedded and the MS MARCO document vectors are reused.

---

### User Story 4 - Hard-negative pools for training queries (Priority: P2)

The researcher builds, for each environment, a pool of hard negatives for its training and
validation queries, following the Learning-to-Select recipe adopted in ADR-0002: 64 negatives
sampled with replacement from the 1,000 highest-scoring non-relevant documents, by exact
full-dimension search with the frozen model. The pools are published with a manifest and are never
built for test queries.

**Why this priority**: F3's oracle targets need them, but they depend on Stories 1–3.

**Independent Test**: Build pools for SciFact, check that every pool has 64 entries, none is a
judged relevant document or the query's own id, the same seed gives the same pools, and the
command refuses test queries.

**Acceptance Scenarios**:

1. **Given** an embedded environment, **When** pools are built, **Then** each training and
   validation query has 64 negatives drawn from its top 1,000 non-relevant documents by exact
   search, recorded with the depth, pool size, and seed.
2. **Given** a request to build pools for test queries, **When** it runs, **Then** it is refused
   before reading any test judgment.
3. **Given** the same embeddings and seed, **When** pools are built twice, **Then** they are
   identical.

### Edge Cases

- A judgment refers to a document missing from the corpus: the judgment is dropped, the count of
  dropped judgments is recorded in the dataset manifest, and a query left with no relevant document
  is excluded from its split.
- A dataset has no training judgments (none of the three initial environments, but later ones such
  as ArguAna): preparation records that it has none, and no training split is produced.
- Duplicate query or document ids in a source: preparation fails and names the duplicated id.
- A query whose id equals a document id (ArguAna-style): that document is excluded from the query's
  negatives and from its retrieval results, as in the spike.
- An embedding job runs out of GPU memory: it fails with an explicit message and leaves completed
  shards intact; the shard size can be lowered and the job resumed.
- The source data changes upstream: the pinned revision keeps preparation reproducible; a
  different revision gives a different fingerprint and is never silently mixed with old vectors.
- Text longer than the input limit: it is truncated to the recorded limit, and the number of
  truncated texts is recorded.
- Two query sets share a corpus (MS MARCO and the TREC DL sets): the corpus is embedded once and
  shared; the query sets are never presented as separate environments.

## Requirements *(mandatory)*

### Functional Requirements

**Environments and records (M1.1, M1.2)**

- **FR-001**: Each environment MUST be described by a versioned environment specification: name,
  data source and pinned revision, judgment source, corpus group (for shared corpora),
  evaluation-only query sets, query instruction, license, and whether it has training judgments.
- **FR-002**: Preparing an environment MUST produce its queries, documents, and judgments as the
  shared record types of F1 (`QueryRecord`, `DocumentRecord`, `QrelsRecord`), with document text
  formed from title and body in one fixed way.
- **FR-003**: Preparation MUST publish a dataset manifest with the source revision, per-split query
  and judgment counts, document count, dropped-judgment and excluded-query counts, and a
  fingerprint of the prepared records.
- **FR-004**: SciFact, NFCorpus, and the full MS MARCO passage corpus with its training and
  development queries MUST be preparable, with the TREC DL 2019, DL 2020, and DL-HARD query sets as
  evaluation-only query sets over the MS MARCO corpus.

**Splits and leakage guards (M1.3)**

- **FR-005**: Each environment with training judgments MUST have disjoint train, validation, and
  test query splits; validation is a fixed-seed 10% of the training queries.
- **FR-006**: Loading an environment MUST return test queries without judgments unless the caller
  explicitly asks for test judgments; only evaluation code may ask, and training, negative mining,
  and target construction MUST fail if configured to read them.
- **FR-007**: The leakage guards MUST be covered by tests that would fail if any training path could
  read a test judgment.

**Embeddings (M2.1, M2.2)**

- **FR-008**: Documents and queries MUST be embedded with the frozen Qwen3-Embedding-0.6B at the
  pinned revision, width 1024, last-token pooling, unit norm, a recorded input limit of 512 tokens,
  and the environment's query instruction, through either the local GPU or the shared embedding
  service, with identical settings recorded either way.
- **FR-009**: Each environment's vectors MUST be published as embedding artifacts (one for
  documents, one per query set) that record row ids in order, settings, model revision, and a
  fingerprint of the stored vectors, together with a run manifest.
- **FR-010**: Embedding MUST proceed in shards that are kept as they complete, so an interrupted
  job resumes after the last completed shard; the final artifact is published only when all shards
  are complete, and its vectors equal an uninterrupted run's.
- **FR-011**: Embedding the full MS MARCO corpus MUST fit within one 24 GB GPU, with the shard size
  adjustable.
- **FR-012**: Reusing cached vectors whose settings or model revision differ from the request MUST
  be refused with a message naming the difference.
- **FR-013**: Environments that share a corpus MUST share one document embedding.

**Hard negatives (M1.4)**

- **FR-014**: Hard-negative pools MUST be built for training and validation queries only: 64
  negatives sampled with replacement, with a fixed seed, from the 1,000 highest-scoring documents
  by exact full-dimension search, excluding judged relevant documents and the query's own id.
- **FR-015**: Pools MUST be published with their depth, pool size, seed, and the embedding
  artifacts they were computed from.

**Commands and reuse**

- **FR-016**: The researcher MUST be able to run each step with one command per environment
  (prepare data, embed, mine negatives), following the conventions of F1's diagnostic command:
  validated configs, a manifest for every output, explicit exit codes, and nothing written on an
  invalid config.
- **FR-017**: The feature MUST reuse the spike's dataset download and parsing, its encoders, and
  its resumable cache, upgraded to F1's contracts, rather than introduce parallel implementations.
- **FR-018**: All tests for this feature MUST run on CPU without network access, using small
  fixtures in place of real datasets and the real model.

### Key Entities

- **Environment specification**: the description of one environment (FR-001); the unit later
  milestones refer to by name.
- **Prepared environment**: its query, document, and judgment records, its splits, and its dataset
  manifest.
- **Dataset manifest**: what was loaded, from which pinned revision, with counts, exclusions, and a
  fingerprint.
- **Embedding artifact**: published vectors for one environment's documents or one query set, with
  ids, settings, model revision, and fingerprint (F1's `EmbeddingArtifact`).
- **Embedding shard**: a completed part of an embedding job, kept so the job can resume.
- **Negative pool**: per training or validation query, 64 sampled hard negatives with the depth,
  pool size, and seed used.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: For 100% of prepared environments and query sets, document and test (or evaluation)
  query counts match the published statistics: SciFact 5,183 documents and 300 test queries;
  NFCorpus 3,633 documents and 323 test queries; MS MARCO 8,841,823 passages and 6,980
  development queries; TREC DL 2019, DL 2020, and DL-HARD 43, 54, and 50 judged queries.
- **SC-002**: 0 training, negative-mining, or target-building paths can read a test judgment,
  verified by tests that would fail otherwise.
- **SC-003**: Preparing or embedding the same environment twice from the same pinned source and
  settings gives identical fingerprints for records and identical vectors.
- **SC-004**: SciFact vectors from this feature agree with the F0 spike's SciFact vectors made with
  the same model revision: cosine similarity at least 0.999 for every document and query.
- **SC-005**: An embedding job interrupted at any point loses at most one shard of work when
  restarted.
- **SC-006**: The full MS MARCO corpus embeds on the single 24 GB GPU workstation in about 6 hours
  of wall time, in one or more sessions (measured on 2026-10-12: 154 s per 65,536-passage shard,
  about 425 passages/s, 135 shards, 16.8 GB of GPU memory; the earlier ceiling was 12 hours).
- **SC-007**: F3 can load each of the three environments' records, splits, vectors, and negative
  pools through the shared contracts alone, without reading any F2 internal file directly.

## Assumptions

- All F2 embeddings are made on one machine, the DataPolicy RTX 4090 workstation (`ceia40901`,
  reached over Tailscale), so every F2 vector comes from the same hardware. The other 4090
  (Corejur, `ceia40902`) was serving a vLLM engine at full load and was not used.
- Sources follow research R2: BEIR datasets and MS MARCO from the Hugging Face Hub pinned by
  revision, and the TREC DL 2019, DL 2020, and DL-HARD query sets and judgments from `ir_datasets`.
  The spike's UKP BEIR archives may serve SciFact and NFCorpus if their content is identical, which
  the dataset fingerprint makes checkable.
- MS MARCO splits: training judgments give train and validation queries; the 6,980-query
  development set is the test split, as in the Learning-to-Select and DIME papers. Hard-negative
  pools for MS MARCO cover a fixed-seed subsample of training queries sized for the
  Learning-to-Select protocol (about 50,000 query–document pairs, research R6) rather than all
  ~500,000 training queries; the subsample is recorded in the dataset manifest.
- Embeddings are stored at 16-bit precision (≈ 18 GB for MS MARCO), computed in bfloat16, matching
  the F0 spike; exact search for negatives streams document shards from disk.
- The spike's own commands and caches are left as they are; F2 adds the shared versions alongside.
- Robust04 and the other reference datasets are out of scope until F5 (research R17).
- The protocol rules in `docs/protocols.md` and the constitution apply unchanged; this feature adds
  no new scientific protocol.
