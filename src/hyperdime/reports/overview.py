"""The overview report: the end-to-end pipeline, the selector strategies, and their results.

The prose describes the pipeline and each strategy and carries no result number; every number
comes from the latest S3 `compare-selectors` run, which scores every strategy on the same queries
(F8). Protocol: in-domain. Scoring: exact.
"""

from __future__ import annotations

from pathlib import Path

from hyperdime.reports import s1, s3
from hyperdime.reports.e0 import _csv, _fmt, _markdown

DIM = 1024
# (method template, strategy, experiment that introduced it)
STRATEGIES = (
    ("full", "full query, all 1024 dimensions", "—"),
    ("prefix", "first k dimensions (MRL-style truncation)", "E0"),
    ("random", "k random dimensions", "E0"),
    ("global-importance", "one dimension ranking for every query: mean oracle target", "E0"),
    ("global-selector", "one L2S selector trained on all environments together", "F3"),
    ("l2s:{env}", "Learning-to-Select: linear map trained by KL to the oracle", "E0 / E1"),
    ("variance", "label-free geometric score q_d² · Var(x_d)", "P1"),
    ("retrieval:{env}", "linear map trained on masked retrieval", "S1"),
    ("geometric:{env}", "fixed geometric prior + linear correction", "S2"),
    ("weighted:{env}", "learned weight on the prior + linear correction", "S3"),
)
REPORTS = (
    ("E0", "Learning-to-Select reproduction, gate G1", "ADR-0006"),
    ("E1", "selector shift across environments, gate G2", "ADR-0007"),
    ("S1", "selector trained on masked retrieval", "ADR-0009"),
    ("P1", "what the corpus geometry alone does", "—"),
    ("S2", "selector started from the geometric prior", "ADR-0010"),
    ("S3", "learned weight on the geometric prior", "ADR-0011"),
)


def _value(results: s1.S1Results, key: tuple[str, str], method: str, k: int | None) -> float:
    """Mean nDCG@10 at budget ``k`` (the mean over 10%-90% when ``k`` is None); the full query
    has one value at every budget."""
    if method == "full":
        return _mean(s1.per_query(results, key, "full", DIM))
    if k is None:
        return _mean(s1.summary(results, key, method, s1.SUMMARIES[0]))
    return _mean(s1.per_query(results, key, method, k))


def results_table(results: s1.S1Results, query_sets: set[str]) -> list[list[str]]:
    budgets = results.budgets
    low, headline = budgets[0], s1.HEADLINE_K
    table = [[
        "test", "query set", "method", "strategy", "mean 10%-90%", f"10% (k={low})",
        f"30% (k={headline})", "best budget", "nDCG@10 at best",
    ]]  # fmt: skip
    for key in sorted(k for k in results.results if k[1] in query_sets):
        env = key[0]
        for template, strategy, _ in STRATEGIES:
            method = template.format(env=env)
            by_k = {k: _value(results, key, method, k) for k in budgets}
            by_k[DIM] = _value(results, key, "full", DIM)
            best = max(by_k, key=lambda k: (by_k[k], -k))
            table.append([
                *key, method, strategy, _fmt(_value(results, key, method, None)),
                _fmt(by_k[low]), _fmt(by_k[headline]), str(best), _fmt(by_k[best]),
            ])  # fmt: skip
    return table


def leaders(results: s1.S1Results) -> list[list[str]]:
    """The masked method with the highest mean over 10%-90% on each test set."""
    table = [["test", "best masked method", "mean 10%-90%", "full query"]]
    for key in sorted(k for k in results.results if k[1] == "test"):
        env = key[0]
        masked = [t.format(env=env) for t, _, _ in STRATEGIES if t != "full"]
        means = {m: _value(results, key, m, None) for m in masked}
        top = max(means, key=lambda m: (means[m], -masked.index(m)))
        table.append([env, top, _fmt(means[top]), _fmt(_value(results, key, "full", DIM))])
    return table


PIPELINE = """\
## The pipeline, end to end

Every stage is a `hyperdime` command that writes its artifact with a run manifest; reports and
decisions are generated from those result files only.

1. **Data** (`prepare-data`). SciFact, NFCorpus (BEIR), and MS MARCO with its full 8.8M-passage
   corpus, plus the TREC DL 2019, DL 2020, and DL-HARD query sets as extra MS MARCO evaluation
   sets. Training queries are split 90/10 into train and validation by a fixed seed; test
   judgments never reach any training step.
2. **Embeddings** (`embed`). Qwen3-Embedding-0.6B, frozen, 1024 dimensions, last-token pooling,
   L2-normalized; computed once and cached in shards.
3. **Hard negatives** (`mine-negatives`). For each training and validation query, 64 documents
   sampled from its top 1,000 non-relevant documents under the full query.
4. **Baseline selectors** (`train-selector --recipe official | candidates | seeded`,
   `train-global`). The oracle importance `e_q ⊙ (p − n)` from training labels, and the
   Learning-to-Select (L2S) linear selector trained by KL to it (E0, gate G1), three seeds per
   environment, and one global selector over all environments.
5. **Selector shift** (`transfer-matrix`). Every selector on every environment (E1, gate G2):
   in-domain selectors beat transferred ones and pick different dimensions.
6. **Selector track** (ADR-0008). New per-environment selectors, each chosen on validation from a
   grid and trained with three seeds: `train-selector --recipe retrieval-* | geometric-* |
   weighted-*` (S1, S2, S3); the label-free geometric probe `probe-geometry` (P1).
7. **Comparison** (`compare-selectors --experiment S1 | S2 | S3`). Every strategy on every test
   and TREC DL set at every 10% budget, in one corpus pass per query set.
8. **Reports and decisions** (`report <experiment>`, `decide <gate>`). Paired per-query tests,
   bootstrap intervals, Holm correction over environments; a drafted ADR per gate or
   experiment, accepted by the researcher.

**Evaluation protocol, shared by every stage.** The selector scores the query's 1024 dimensions;
the k highest are kept and the rest zeroed (the masked query is not re-normalized; documents
and the index are never changed); documents are ranked by exact inner product over the full
corpus. Budgets are k = 10%, 20%, ..., 90% of 1024 (100% is the full query). The headline
summary is the per-query mean of nDCG@10 over the nine budgets; selector seeds are averaged per
query before every test.
"""

STRATEGY_ROWS = (
    ("strategy", "score of dimension d for query q", "learned", "trained on", "starts from"),
    ("L2S (E0)", "`(W q + b)_d`", "1024 × 1024 map + bias",
     "KL to the oracle importance distribution", "random init"),
    ("variance (P1)", "`q_d² · Var(x_d)`", "nothing",
     "— (corpus variance from document embeddings only)", "—"),
    ("S1 retrieval", "`(W q + b)_d`", "same map",
     "softmax over the masked query's positives, hard and in-batch negatives, through a relaxed "
     "top-k mask", "scratch or L2S"),
    ("S2 geometric", "`log(q_d² · Var(x_d) + ε) + (W q + b)_d`", "same map", "same as S1",
     "the `variance` method (map at zero)"),
    ("S3 weighted", "`a · log(q_d² · Var(x_d) + ε) + (W q + b)_d`", "same map + scalar `a`",
     "same as S1", "the `variance` method (`a` = 1)"),
)  # fmt: skip

STRATEGY_TEXT = """\
## The selector strategies

{strategy_table}

- **Why a retrieval loss (S1).** L2S is trained to match a distribution, but only its top-k
  ranking is used; S1 trains the masked retrieval it is used for.
- **Why a geometric prior (S2).** Dropping dimension d removes `q_d · x_d` from every score; if
  that term barely varies across documents, the ranking barely changes. `q_d² · Var(x_d)`
  measures that variation without labels, and it approximates the full query well.
- **Why a learned weight (S3).** The prior helps where approximating the full query matters and
  hurts where the ranking must be reshaped toward relevance; one learned scalar lets each
  environment decide.
- **Selection rules, fixed before each run.** Cells are chosen on validation nDCG@10 (full
  corpus, hard mask, mean over the nine budgets) among cells whose relaxed mask ends close to the
  hard one; checkpoints by validation reranking. A selector whose kept checkpoint is its start is
  named as such in its report.
"""


def render(root: Path) -> tuple[str, dict[str, str]]:
    results = s3.load(root)
    tables = {
        "leaders": leaders(results),
        "test": results_table(results, {"test"}),
        "trec": results_table(results, {k[1] for k in results.results} - {"test"}),
    }
    lines = [
        "# HyperDIME selector track: pipeline, strategies, and results",
        "",
        "Generated by `hyperdime report overview` from the latest S3 run, which scores every",
        "strategy below on the same queries. Protocol: in-domain. Scoring: exact.",
        "",
        f"- compare-selectors `{results.run}` (commit `{results.commit[:12]}`)",
        "",
        PIPELINE,
        STRATEGY_TEXT.format(strategy_table="\n".join(_markdown(STRATEGY_ROWS))),
        "## Results",
        "",
        "Best masked strategy per test set (mean nDCG@10 over 10%-90%):",
        "",
        *_markdown(tables["leaders"]),
        "",
        "Every strategy on the test sets:",
        "",
        *_markdown(tables["test"]),
        "",
        "TREC DL sets (MS MARCO queries; context):",
        "",
        *_markdown(tables["trec"]),
        "",
        "Significance tests, per-budget curves, and diagnostics are in each experiment's report.",
        "",
        "## Reports and decisions",
        "",
        "| experiment | question | decision |",
        "|---|---|---|",
        *[f"| [{e}](../{e}/README.md) | {q} | {a} |" for e, q, a in REPORTS],
    ]
    return "\n".join(lines) + "\n", {name: _csv(table) for name, table in tables.items()}


def write(root: Path, output: Path) -> Path:
    text, csvs = render(root)
    (output / "tables").mkdir(parents=True, exist_ok=True)
    (output / "README.md").write_text(text, encoding="utf-8")
    for name, content in csvs.items():
        (output / "tables" / f"{name}.csv").write_text(content, encoding="utf-8")
    return output / "README.md"


def _mean(values: dict[str, float]) -> float:
    return sum(values.values()) / len(values)
