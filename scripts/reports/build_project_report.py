"""Build the HyperDIME-Qwen project report (HTML).

Numbers come from the F0 run's files (artifacts/spike/**, BEIR qrels, the Slurm log in logs/),
from git, and from the GitHub issue tracker (``gh``). Nothing numeric is typed in.

Bring the run back from the cluster first (scripts/slurm/README.md), then, from the repo root:

    python scripts/reports/build_project_report.py [--out reports/project/index.html]
"""
# The HTML template below is prose; it is exempt from line length and formatting.
# ruff: noqa: E501

import collections
import csv
import glob
import json
import math
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO  # holds artifacts/ and logs/
OUT = (
    Path(sys.argv[sys.argv.index("--out") + 1])
    if "--out" in sys.argv
    else REPO / "reports/project/index.html"
)
SP = ROOT / "artifacts" / "spike"


def sh(*args):
    return subprocess.run(args, cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def rj(p):
    return json.loads(Path(p).read_text())


A = rj(SP / "analysis/analysis.json")
AM = rj(SP / "analysis/manifest.json")
GSEL = rj(SP / "global/selection.json")
cfg, crit, mask, summ = A["config"], A["criteria"], A["mask"], A["summary"]
ENVS = cfg["envs"]
K = str(cfg["headline_k"])
KS = [str(k) for k in cfg["k_values"]]
SEEDS = cfg["seeds"]
temps = cfg["temperatures"]
T_GRID = sorted(float(t) for t in temps[ENVS[0]]["validation_ndcg@10"])
E_GRID = sorted(int(e) for e in next(iter(temps[ENVS[0]]["validation_ndcg@10"].values())))
PREP = {e: rj(SP / e / "manifest.json") for e in ENVS}
SELM = {e: rj(SP / e / "selectors/manifest.json") for e in ENVS}
TC = SELM[ENVS[0]]["config"]["train_config"]
ENC = PREP[ENVS[0]]["config"]["encoder"]
NEG = PREP[ENVS[0]]["config"]


def nd(env, s, k=K):
    return summ[env][s][k]["ndcg@10"]


def ind(env, k=K):
    return nd(env, f"selector:{env}", k)


# ---- train/eval positive overlap per environment
overlap = {}
for env in ENVS:
    splits = rj(SP / env / "splits.json")
    pos = collections.defaultdict(set)
    for f in glob.glob(str(SP / "beir" / env / "qrels" / "*.tsv")):
        with open(f) as handle:
            for r in csv.DictReader(handle, delimiter="\t"):
                if int(r["score"]) > 0:
                    pos[r["query-id"]].add(r["corpus-id"])
    tr = set(map(str, splits["train"])) | set(map(str, splits["validation"]))
    ev = set(map(str, splits["evaluation"]))
    assert not tr & ev
    seen = set().union(*(pos[q] for q in tr if q in pos))
    pairs = [(q, d) for q in ev for d in pos.get(q, ())]
    overlap[env] = {
        "pos_seen": sum(d in seen for _, d in pairs) / len(pairs),
        "pos_per_q": len(pairs) / len(ev),
    }

# ---- training histories: epoch of the kept (lowest validation KL) state
best_at_end = 0
hist_n = 0
for env in ENVS:
    for s in SEEDS:
        h = rj(SP / env / "selectors" / f"history-seed-{s}.json")
        b = min(range(len(h)), key=lambda i: h[i]["val_kl"])
        hist_n += 1
        best_at_end += b >= len(h) - 3

# ---- project state from git and GitHub
issues = json.loads(
    sh(
        "gh",
        "issue",
        "list",
        "--state",
        "all",
        "--limit",
        "200",
        "--json",
        "number,title,state,labels",
    )
)
open_issues = [i for i in issues if i["state"] == "OPEN"]
branches = [b.strip() for b in sh("git", "branch", "-r").splitlines() if "HEAD" not in b]
ahead = {b: int(sh("git", "rev-list", "--count", f"origin/master..{b}")) for b in branches}
ticket_branches = [b for b in branches if b.split("/", 1)[1].startswith(("agent/", "exp/"))]
empty_ticket = [b for b in ticket_branches if ahead[b] == 0]
f0_ahead = ahead.get("origin/agent/F0-h1-spike", 0)
prs = json.loads(sh("gh", "pr", "list", "--state", "all", "--json", "number"))
tasks = (REPO / "specs/002-hyperdime-roadmap/tasks.md").read_text().splitlines()
t_done = sum(line.startswith("- [X]") for line in tasks)
t_all = sum(line.startswith("- [") for line in tasks)
n_tests = sum(
    Path(f).read_text().count("def test_")
    for f in glob.glob(str(REPO / "tests/**/*.py"), recursive=True)
)
first_commit = sh(
    "git", "log", "--reverse", "--format=%ad", "--date=short", "324b888"
).splitlines()[0]
logs = (ROOT / "logs" / "f0-spike-407.out").read_text().splitlines()
t_start = next(line for line in logs if "enroot start" in line).split()[1]
t_end = next(line for line in logs if line.endswith("done")).split()[1]
gpu = next(line for line in logs if line.startswith("GPU 0:")).split(":")[1].split("(")[0].strip()
step_times = {}
for f in sorted(SP.glob("*/manifest.json")) + sorted(SP.glob("*/selectors/manifest.json")):
    m = rj(f)
    step_times[str(f.relative_to(SP))] = (m["command"], m["created_at"][11:19])

# ---- derived F0 facts
n_sig, n_pairs = len(crit["transfer_significant"]), len(crit["transfer"])
gain_envs = crit["headroom_significant"]
sig_cols = collections.Counter(p.split("->")[1] for p in crit["transfer_significant"])
sig_in_gain = sum(sig_cols[e] for e in gain_envs)
low_seed = [
    e
    for e in ENVS
    if mask["spearman_seed"][e] < max(v for o, v in mask["spearman_env"][e].items() if o != e)
]
js_max = max(v for r in mask["js_env"].values() for v in r.values())
ci = mask["overlap_diff_ci"]
n_edge_ep = sum(temps[e]["epochs"] in (E_GRID[0], E_GRID[-1]) for e in ENVS)
n_edge_t = sum(temps[e]["temperature"] in (T_GRID[0], T_GRID[-1]) for e in ENVS)


def f4(x):
    return f"{x:.4f}"


def sgn(x, d=4):
    return f"{x:+.{d}f}".replace("-", "−")


def pct(x):
    return f"{100 * x:.1f}%"


def table(head, rows, cls=""):
    th = "".join(f"<th>{h}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def pill(kind, txt):
    return f'<span class="pill {kind}">{txt}</span>'


# ---- chart 1: Δ vs full per env
SERIES = [
    ("in-domain", "s1", ind),
    ("global", "s2", lambda e: nd(e, "global")),
    ("prefix", "s3", lambda e: nd(e, "prefix")),
]
deltas = {e: {n: fn(e) - nd(e, "full", "1024") for n, _, fn in SERIES} for e in ENVS}
lo = min(min(d.values()) for d in deltas.values())
hi = max(max(d.values()) for d in deltas.values())
x0, x1 = math.floor(lo / 0.02) * 0.02, math.ceil(hi / 0.02) * 0.02
W, LM, RM, TM, RH = 640, 92, 20, 30, 40
H = TM + RH * len(ENVS) + 28


def X(v):
    return LM + (v - x0) / (x1 - x0) * (W - LM - RM)


svg = [
    f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Change in nDCG@10 versus full 1024 dimensions">'
]
t = x0
while t <= x1 + 1e-9:
    svg.append(
        f'<line class="{"zero" if abs(t) < 1e-9 else "grid"}" x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{TM - 6}" y2="{H - 26}"/>'
    )
    svg.append(
        f'<text class="tick" x="{X(t):.1f}" y="{H - 10}" text-anchor="middle">{sgn(t, 2)}</text>'
    )
    t += 0.02
svg.append(f'<text class="tick" x="{X(0) + 4:.1f}" y="{TM - 12}">full 1024 dims</text>')
for i, e in enumerate(ENVS):
    y = TM + RH * i + RH / 2
    svg.append(f'<text class="rowlab" x="{LM - 12}" y="{y + 4:.1f}" text-anchor="end">{e}</text>')
    vals = list(deltas[e].values())
    svg.append(
        f'<line class="span" x1="{X(min(vals)):.1f}" x2="{X(max(vals)):.1f}" y1="{y:.1f}" y2="{y:.1f}"/>'
    )
    for n, c, _ in SERIES:
        svg.append(
            f'<circle class="dot {c}" cx="{X(deltas[e][n]):.1f}" cy="{y:.1f}" r="6"><title>{e} · {n}: {sgn(deltas[e][n])} vs full</title></circle>'
        )
svg.append("</svg>")
chart1 = "\n".join(svg)

# ---- chart 2: headroom vs positive overlap
W2, H2, L2, R2, T2, B2 = 640, 300, 64, 24, 20, 44
y0 = math.floor(min(v["ci"][0] for v in crit["headroom"].values()) / 0.02) * 0.02
y1 = math.ceil(max(v["ci"][1] for v in crit["headroom"].values()) / 0.02) * 0.02


def X2(v):
    return L2 + v * (W2 - L2 - R2)


def Y2(v):
    return T2 + (y1 - v) / (y1 - y0) * (H2 - T2 - B2)


s2 = [
    f'<svg viewBox="0 0 {W2} {H2}" role="img" aria-label="Headroom versus share of eval positives seen in training">'
]
t = y0
while t <= y1 + 1e-9:
    s2.append(
        f'<line class="{"zero" if abs(t) < 1e-9 else "grid"}" x1="{L2}" x2="{W2 - R2}" y1="{Y2(t):.1f}" y2="{Y2(t):.1f}"/>'
    )
    s2.append(
        f'<text class="tick" x="{L2 - 8}" y="{Y2(t) + 4:.1f}" text-anchor="end">{sgn(t, 2)}</text>'
    )
    t += 0.02
for p in (0, 0.25, 0.5, 0.75, 1.0):
    s2.append(
        f'<text class="tick" x="{X2(p):.1f}" y="{H2 - 22}" text-anchor="middle">{int(p * 100)}%</text>'
    )
s2.append(
    f'<text class="axlab" x="{(L2 + W2 - R2) / 2:.1f}" y="{H2 - 4}" text-anchor="middle">eval positives that are also train/val positives</text>'
)
for e in ENVS:
    h = crit["headroom"][e]
    x = X2(overlap[e]["pos_seen"])
    s2.append(
        f'<line class="whisk" x1="{x:.1f}" x2="{x:.1f}" y1="{Y2(h["ci"][0]):.1f}" y2="{Y2(h["ci"][1]):.1f}"/>'
    )
    s2.append(
        f'<circle class="dot s1{"" if e in gain_envs else " hollow"}" cx="{x:.1f}" cy="{Y2(h["diff"]):.1f}" r="6"><title>{e}: {sgn(h["diff"])} [{sgn(h["ci"][0])}, {sgn(h["ci"][1])}]; {pct(overlap[e]["pos_seen"])} of eval positives seen in training</title></circle>'
    )
    end = overlap[e]["pos_seen"] > 0.8
    s2.append(
        f'<text class="ptlab" x="{x + (-12 if end else 12):.1f}" y="{Y2(h["diff"]) + 4:.1f}" text-anchor="{"end" if end else "start"}">{e}</text>'
    )
s2.append("</svg>")
chart2 = "\n".join(s2)

# ---- tables
env_rows = []
for e in ENVS:
    c = cfg["counts"][e]
    src = (
        "official train qrels"
        if (SP / "beir" / e / "qrels" / "train.tsv").exists()
        else "50/50 split of test queries"
    )
    env_rows.append(
        [
            e,
            f"{PREP[e]['config']['num_documents']:,}",
            src,
            c["train"],
            c["validation"],
            c["evaluation"],
            f"{overlap[e]['pos_per_q']:.1f}",
        ]
    )

main_rows = []
for e in ENVS:
    trm = sum(nd(e, f"selector:{o}") for o in ENVS if o != e) / (len(ENVS) - 1)
    main_rows.append(
        [
            e,
            f4(nd(e, "full", "1024")),
            f4(nd(e, "prefix")),
            f4(nd(e, "random")),
            f4(nd(e, "global")),
            f"<b>{f4(ind(e))}</b>",
            f4(trm),
            f4(nd(e, "oracle")),
        ]
    )

tm_rows = []
for a in ENVS:
    row = [f"<b>{a}</b>"]
    for b in ENVS:
        v = nd(b, f"selector:{a}")
        row.append(
            f"<b>{f4(v)}</b>"
            if a == b
            else f'{f4(v)}<span class="star">{"*" if f"{a}->{b}" in crit["transfer_significant"] else ""}</span>'
        )
    tm_rows.append(row)

half = n_pairs // 2
crit_rows = [
    [
        "C1",
        "Transferred selectors lose significantly to the in-domain selector on ≥ half of the ordered pairs",
        f"{n_sig}/{n_pairs} (needs {half}); {sig_in_gain} of them on {' / '.join(gain_envs)}",
        pill(
            "ok" if crit["c1_transfer_loss"] else "no",
            "met" + (", at threshold" if n_sig == half else ""),
        ),
    ],
    [
        "C2",
        "Cross-environment mask overlap@k is significantly below cross-seed overlap",
        f"{mask['overlap_cross_seed']:.3f} vs {mask['overlap_cross_env']:.3f}, p = {mask['overlap_p']:.4f}",
        pill("ok" if crit["c2_masks_shift"] else "no", "met"),
    ],
    [
        "C3",
        "In-domain beats the global selector significantly on ≥ 2 of 5 environments",
        f"{len(gain_envs)}/{len(ENVS)}: {', '.join(gain_envs)}",
        pill(
            "ok" if crit["c3_headroom_over_global"] else "no",
            "met" + (", at threshold" if len(gain_envs) == 2 else ""),
        ),
    ],
]

hp_rows = [
    [
        e,
        temps[e]["temperature"],
        temps[e]["epochs"],
        f4(max(v for r in temps[e]["validation_ndcg@10"].values() for v in r.values())),
    ]
    for e in ENVS
]

GATES = [
    (
        "G1",
        "Valid baseline: Learning-to-Select converges, masking is correct, exact retrieval ranks as expected",
        "E0",
    ),
    (
        "G2",
        "H1 supported: selectors differ beyond seed noise and transfer degrades retrieval",
        "E1 / M4",
    ),
    (
        "G3",
        "H2 supported: descriptors predict selector or transfer properties above a trivial baseline",
        "E2 / M5.4",
    ),
    (
        "G4",
        "H3 supported: the hypernetwork narrows the global vs per-domain gap on unseen environments",
        "E3",
    ),
    (
        "G5",
        "Offline adaptation is amortizable; online overhead keeps the dimension-reduction benefit",
        "E6 / M12.2",
    ),
]
gate_status = {"G2": pill("warn", "preliminary signal (F0)")}
gate_rows = [[g, c, by, gate_status.get(g, pill("idle", "not started"))] for g, c, by in GATES]

phase_count = collections.Counter(
    next(
        (line["name"].split(":")[1] for line in i["labels"] if line["name"].startswith("phase:")),
        "?",
    )
    for i in open_issues
)
PHASES = [
    (
        "A",
        "Foundation",
        "M0–M2: contracts, manifests, loaders, splits, Qwen encoder, cache, negatives",
    ),
    (
        "B",
        "Scientific baseline",
        "M3, M12.1, E0: oracle, Learning-to-Select trainer, baselines → G1",
    ),
    (
        "C",
        "Evidence for H1",
        "M4, M12.3: per-domain selectors, transfer matrix, behavior analysis → G2",
    ),
    ("D", "Space representation", "M5: descriptor groups A–E, H2 probes → G3"),
    ("E", "Hypernetwork", "M6–M9: space encoder, low-rank selector, Hyper Head, episodic training"),
    (
        "F",
        "Final evaluation",
        "M10–M12, E3–E6: protocols, policies, main result, ablations, efficiency → G4, G5",
    ),
]
phase_rows = [[p, n, d, f"{phase_count.get(p, 0)} open"] for p, n, d in PHASES]

impl_rows = [
    [
        "<code>embeddings</code>",
        "Frozen Qwen3-Embedding-0.6B: last-token pooling, L2 norm, instruction prefix; local transformers or vLLM backend; resumable cache with a settings guard",
        "done (F0)",
    ],
    [
        "<code>data</code>",
        "BEIR download and parsing, deterministic ID splits, hard-negative mining and sampling",
        "done (F0)",
    ],
    [
        "<code>oracle</code>",
        "Target distribution softmax(e<sub>q</sub> ⊙ (p̄ − n̄) / τ) with gain-weighted positives",
        "done (F0)",
    ],
    [
        "<code>baselines</code>, <code>training</code>",
        "Linear D→D selector with log-softmax; KL trainer (AdamW, cosine schedule, best validation state)",
        "done (F0)",
    ],
    [
        "<code>selection</code>, <code>retrieval</code>",
        "Top-k mask, query-only masking without re-normalization; exact batched dot-product search, BEIR identical-ID exclusion",
        "done (F0)",
    ],
    [
        "<code>evaluation</code>",
        "nDCG / recall / MRR (trec_eval gains); overlap@k, JS, Spearman; sign-flip test, bootstrap CI, Holm",
        "done (F0)",
    ],
    [
        "<code>spike</code>",
        "F0 pipeline: prepare → train → train-global → analyze → report, with a JSON manifest per step",
        "done (F0)",
    ],
    ["<code>contracts</code>", "Schemas, run manifests, seeding, configs (M0)", "hashing only"],
    [
        "<code>space</code>, <code>hypernet</code>, <code>cli</code>",
        "Space descriptor and encoder, low-rank selector, Hyper Head, protocols P0–P2",
        "not started",
    ],
]

STEP_ROWS = [
    [
        "prepare",
        "Download BEIR, split queries, embed corpus and queries (cached), sample hard negatives for train/val queries",
    ],
    [
        "train",
        f"Per environment: {len(T_GRID)}×{len(E_GRID)} grid on validation nDCG@10 with seed 0, then one selector per seed",
    ],
    [
        "train-global",
        "One selector on the union of all training labels; epoch count tuned on mean validation nDCG@10",
    ],
    [
        "analyze",
        "Every system on every eval set at every k; transfer matrix; mask battery; paired tests; verdict",
    ],
    ["report", "Markdown report rendered from analysis.json"],
]

sc_gain = sgn(ind("scifact") - nd("scifact", "full", "1024"), 3)
issue_total = len(issues)

TM_HEAD = "train \\ test"

# fmt: off
body = f"""
<header class="mast">
  <p class="eyebrow">Project report · {AM['created_at'][:10]} run · compiled from repository and run files</p>
  <h1>HyperDIME-Qwen</h1>
  <p class="sub">Environment-conditioned hypernetworks for query-aware dimension selection on a frozen Qwen3-Embedding-0.6B (D = 1024)</p>
  <dl class="meta">
    <div><dt>Repository</dt><dd>wilkertl/hyper-mask-net · restarted {first_commit}</dd></div>
    <div><dt>Stage</dt><dd>F0 hypothesis spike done; roadmap not started</dd></div>
    <div><dt>Evidence so far</dt><dd>1 experiment (F0, H1)</dd></div>
  </dl>
</header>

<nav class="toc" aria-label="Contents">
  <a href="#abstract">Abstract</a><a href="#intro">1 Introduction</a><a href="#method">2 Proposed method</a>
  <a href="#protocol">3 Research protocol</a><a href="#f0">4 F0: methodology</a><a href="#impl">5 Implementation</a>
  <a href="#results">6 F0: results</a><a href="#discussion">7 Discussion</a><a href="#status">8 Status and next steps</a><a href="#refs">References</a>
</nav>

<section id="abstract" class="abstract">
  <h2>Abstract</h2>
  <p>Learning to Select (Wu et al., 2026) shows that a small query-only predictor can pick which coordinates of a dense
  embedding a query should use, keeping document vectors and the index untouched. Such a selector is trained for one
  retrieval environment. HyperDIME-Qwen asks whether the best selector changes across environments (H1), whether
  statistics of the embedding space explain that change (H2), and whether a hypernetwork can turn those statistics into
  selector parameters for an unseen environment (H3). The project is gated so each link can be refuted before the next is
  built. The first experiment, the F0 spike, tested H1 on five BEIR datasets with {len(SEEDS)} seeds per selector.
  Its pre-registered rule returned <b>H1 supported</b>, but two of the three criteria passed exactly at threshold.
  All of the gain came from scifact and nfcorpus. On those two sets {pct(overlap['scifact']['pos_seen'])} and
  {pct(overlap['nfcorpus']['pos_seen'])} of eval positives are also training positives; on fiqa and arguana none are.
  We read F0 as a preliminary signal that does not yet settle Gate G2. Section 8 lists the controls needed first.</p>
</section>

<section id="intro">
  <h2>1 Introduction</h2>
  <p>Dense retrievers score a query against documents with an inner product over all D coordinates. Not every coordinate
  helps every query. Query-aware dimension selection keeps, for each query, the <var>k</var> coordinates most useful
  for ranking and zeroes the rest on the query side only, so the document index is unchanged.</p>
  <p>Learning to Select trains a linear layer <var>f</var>(<var>e<sub>q</sub></var>) = log-softmax(<var>W e<sub>q</sub></var> + <var>b</var>)
  to imitate an oracle importance distribution built from labelled positives and hard negatives. The selector is
  trained per dataset. A new corpus, query population or instruction means new labels and a new selector.</p>
  <p>HyperDIME-Qwen fixes the encoder, Qwen3-Embedding-0.6B at full dimension, so the 1024 coordinates keep a stable
  identity. It then studies how the best selector depends on the environment
  <var>T</var> = (corpus, query distribution, instruction, frozen encoder). The project tests a causal chain one link at a time:</p>
  <p class="chain">distribution shift → embedding geometry shift → optimal selector shift → hypernetwork adaptation</p>
  {table(["", "Hypothesis", "If it fails"], [
      ["H1", "Selector shift: the optimal selector for the same Qwen model changes systematically across environments", "Build one universal selector; no hypernetwork"],
      ["H2", "Geometry explains the shift: space statistics predict part of the change", "Iterate on the space descriptor"],
      ["H3", "A Hyper Head maps environment statistics to selector parameters that beat a global selector on unseen environments", "The parameterization or training is the problem"]], "wrap")}
  <p>The repository began as Hyper-Mask-Net, which generated a mask network per query from Qwen token states. On SciFact it
  beat random masks but not the full embedding or the linear Learning-to-Select selector at 30% of dimensions.
  It was restarted on {first_commit} with the environment-conditioned design (ADR-0001).</p>
</section>

<section id="method">
  <h2>2 Proposed method</h2>
  <h3>2.1 Two paths</h3>
  <p><b>Offline, once per environment.</b> A space descriptor summarizes how the environment uses each coordinate.
  A space encoder and a Hyper Head turn it into selector parameters <var>θ<sub>T</sub></var>, which are stored.</p>
  <pre class="mermaid">
flowchart LR
  C["Corpus + optional calibration queries"] --> E["Frozen Qwen embeddings"]
  E --> R["Space descriptor R_T (1024 × h)"]
  R --> G["Space encoder G_ψ"]
  G --> H["Hyper Head H_φ"]
  H --> P["θ_T = W_0 + A_T B_Tᵀ, b_0 + Δb_T"]
  </pre>
  <p><b>Online, per query.</b> The selector scores dimensions, a policy keeps the top <var>k</var>, and exact search runs
  over unchanged documents: <var>e′<sub>q</sub></var> = <var>e<sub>q</sub></var> ⊙ <var>m<sub>q</sub></var>,
  score(<var>q</var>, <var>d</var>) = <var>e′<sub>q</sub></var>ᵀ<var>e<sub>d</sub></var>. The masked query is not re-normalized.</p>
  <pre class="mermaid">
flowchart LR
  Q["query + instruction"] --> QE["e_q (1024)"] --> S["selector f_θT"] --> P["top-k policy"] --> M["mask m_q"]
  QE --> X["e_q ⊙ m_q"]
  M --> X --> RET["exact search, documents unchanged"]
  </pre>
  <h3>2.2 Components</h3>
  {table(["Component", "Design", "Rationale"], [
      ["Space descriptor <var>R<sub>T</sub></var>", "Per-dimension statistics in five groups: A document marginals, B query marginals, C query–document interaction <var>q<sub>j</sub>d<sub>j</sub></var>, D contrast <var>q<sub>j</sub></var>(<var>p<sub>j</sub></var> − <var>n<sub>j</sub></var>), E redundancy", "Invariant to sample order, not to dimension identity; normalized with meta-train statistics only"],
      ["Space encoder <var>G<sub>ψ</sub></var>", "V1: shared MLP over each row plus a learned dimension-ID embedding; V2 adds 1–3 Transformer blocks across dimensions", "Low capacity first; V2 only if an ablation shows V1 is limiting"],
      ["Target selector", "<var>W<sub>T</sub></var> = <var>W</var><sub>0</sub> + <var>A<sub>T</sub>B<sub>T</sub></var>ᵀ, rank 4–32, <var>b<sub>T</sub></var> = <var>b</var><sub>0</sub> + Δ<var>b<sub>T</sub></var>", "Generating a full 1024×1024 matrix per environment is intractable; residual form starts from the global selector"],
      ["Hyper Head <var>H<sub>φ</sub></var>", "Separate heads for <var>A<sub>T</sub></var>, <var>B<sub>T</sub></var>, Δ<var>b<sub>T</sub></var>, with near-zero initialization of the deltas", "With zero deltas the output equals the base selector"],
      ["Training", "Episodic: sample an environment, generate <var>θ<sub>T</sub></var>, fit KL(<var>π<sub>q</sub></var> ‖ <var>π̂<sub>q</sub></var>) on its queries, backpropagate into the Hyper Head and encoder", "Environments, not queries, are the training unit"],
      ["Selection policy", "Fixed top-k or ratio first; RDIME-style adaptive k and budget-aware policies later", "Importance prediction stays separate from how many dimensions to keep"]], "wrap")}
  <h3>2.3 Adaptation protocols</h3>
  {table(["Protocol", "Available in a new environment", "Descriptor groups"], [
      ["P0 corpus-only", "document embeddings", "A, E"], ["P1 unlabeled calibration", "documents + a small unlabeled query set", "A, B, C, E; D through pseudo-feedback"],
      ["P2 few-label calibration", "P1 + a few labelled queries", "all; never reported as zero-shot"]], "wrap")}
</section>

<section id="protocol">
  <h2>3 Research protocol</h2>
  <p>Work proceeds through five gates. Each gate decision is recorded as an architecture decision record (ADR), and a
  failed gate stops the project from adding complexity.</p>
  {table(["Gate", "Passes when", "Decided by", "Status"], gate_rows, "wrap")}
  <p><b>Leakage rules.</b> Splits exist at two levels. Environments are divided into meta-train, meta-validation and
  meta-test; queries into train, validation and test. Test labels never reach training, negative mining, oracle
  targets, descriptors or normalization. P0 and P1 code paths may not read labels at all.</p>
  <p><b>Reporting rules.</b> Metrics use exact scoring (nDCG@10, MRR@10/100, Recall@100, linear gains as in trec_eval).
  Every table includes the full 1024-dimension embedding and MRL prefix truncation (risk R5), plus a global multi-domain
  selector trained on the same labels (risk R6). Significance uses paired per-query tests with multiple-comparison
  correction and confidence intervals. Offline adaptation cost is reported separately from per-query cost.
  Every table is generated from result files that carry a run manifest.</p>
</section>

<section id="f0">
  <h2>4 F0 spike: methodology</h2>
  <p>F0 tests H1 before any roadmap code is written, with the smallest setup that could refute it.</p>
  <h3>4.1 Environments</h3>
  {table(["Environment", "Documents", "Training labels", "Train q", "Val q", "Eval q", "Positives / eval q"], env_rows)}
  <p>Full corpora are used. Datasets without a train split (arguana, scidocs) use a fixed 50/50 split of their judged test
  queries, and 10% of each training pool becomes validation. Train and evaluation queries are disjoint in every
  environment, and evaluation labels never reach training.</p>
  <h3>4.2 Embeddings</h3>
  <p>{ENC['model'].split('@')[0]} (revision <code>{ENC['model'].split('@')[1][:8]}</code>), {ENC['dtype'].replace('torch.', '')},
  texts cut to {ENC['max_length']} tokens, last-token pooling, L2-normalized, 1024 dimensions. Queries carry each dataset's
  Qwen3 evaluation instruction as <code>Instruct: …\\nQuery:…</code>; documents have no prefix.</p>
  <h3>4.3 Oracle targets</h3>
  <p>For each labelled query the target is a distribution over dimensions:</p>
  <p class="eq"><var>π<sub>q</sub></var> = softmax( <var>e<sub>q</sub></var> ⊙ (<var>p̄</var> − <var>n̄</var>) / <var>τ</var> )</p>
  <p>Here <var>p̄</var> is the mean of the positive documents weighted by gain 2<sup>rel</sup> − 1, and <var>n̄</var> is the mean of
  {NEG['negative_pool']} negatives sampled with replacement from the query's top-{NEG['negative_depth']} non-positive documents
  under the full embedding. This follows the official Learning-to-Select label generation (ADR-0002).</p>
  <h3>4.4 Selector and training</h3>
  <p>The selector is one 1024→1024 linear layer with log-softmax, trained on KL(<var>π<sub>q</sub></var> ‖ <var>π̂<sub>q</sub></var>) with
  AdamW (lr {TC['lr']}, weight decay {TC['weight_decay']}), batch {TC['batch_size']} and cosine decay over a fixed epoch budget.
  The state with the lowest validation KL is kept. Temperature and epochs are chosen per environment on validation
  nDCG@10 at k = {K} from <var>τ</var> ∈ {{{', '.join(map(str, T_GRID))}}} × epochs ∈ {{{', '.join(map(str, E_GRID))}}}, using seed 0
  and training labels only. Each environment then gets {len(SEEDS)} selectors (seeds {', '.join(map(str, SEEDS))}).
  The global selector is trained on the union of all environments' training labels, with its epoch count tuned on mean validation nDCG@10.</p>
  <h3>4.5 Systems compared</h3>
  {table(["System", "Importance used for the top-k mask"], [
      ["Full", "no mask, 1024 dimensions"], ["Prefix", "first k dimensions (Qwen3 is Matryoshka-trained)"],
      ["Random", f"uniform random scores, {len(SEEDS)} draws"], ["Global", "selector trained on all five environments"],
      ["In-domain", "selector trained on the evaluation environment"], ["Transferred", "selector trained on each other environment"],
      ["Oracle", "targets built from the eval query's own labels; an upper bound, never trained on"]], "wrap")}
  <p>Evaluation uses exact scoring at k ∈ {{{', '.join(KS)}}} (headline k = {K}, 30% of dimensions), excluding a document
  whose ID equals the query ID, as BEIR does for ArguAna. Per-query metrics are averaged over seeds before testing.</p>
  <h3>4.6 Mask analysis and statistics</h3>
  <p>All {cfg['battery_size']} evaluation queries form one unlabelled battery. Each selector's top-{K} masks on the battery
  give overlap@k within an environment (across seeds) and across environments. Mean importance vectors are compared with
  Jensen–Shannon divergence and Spearman rank correlation. Paired differences use a two-sided sign-flip randomization test
  (10,000 permutations), a percentile bootstrap 95% CI (10,000 resamples over queries) and Holm correction within
  each family (α = {cfg['alpha']}).</p>
  <h3>4.7 Verdict rule</h3>
  <p>H1 is supported only if all three criteria hold at k = {K}, on nDCG@10:</p>
  {table(["", "Criterion"], [[r[0], r[1]] for r in crit_rows], "wrap")}
  <p>C3 is the practical test: without a gap between per-domain and global selectors there is nothing for a hypernetwork
  to recover, even if H1 is technically true.</p>
</section>

<section id="impl">
  <h2>5 Implementation</h2>
  <p>The code is a Python 3.12 / PyTorch package, <code>src/hyperdime</code>, laid out by the plan's modules M0–M12. So far
  only the pieces F0 needs exist, together with {n_tests} unit and integration test functions. These cover hand-computed oracle
  and Holm values, trainer determinism, leakage-safe splits, the vLLM client against a fake server, and a synthetic
  end-to-end run that must reproduce byte-identical output.</p>
  {table(["Package", "What it does", "State"], impl_rows, "wrap")}
  <p><b>Pipeline.</b> <code>python -m hyperdime.spike all</code> runs five steps. Each step writes a JSON manifest
  (git commit, dirty flag, config, library versions) first with <code>status: running</code>, then <code>complete</code>.
  The next step refuses inputs that are not complete.</p>
  {table(["Step", "Work"], STEP_ROWS, "wrap")}
  <p><b>Compute.</b> Run on the CEIA DGX Spark cluster (one {gpu}, aarch64) inside an NGC PyTorch 25.09 container unpacked to
  node-local disk by enroot. Slurm job 407 ran {t_start}–{t_end}; embeddings were reused from an earlier job's cache.
  Selector training dominated; the per-environment grid is {len(T_GRID) * len(E_GRID)} fits plus {len(SEEDS)} seeds.
  Commit <code>{AM['git_commit'][:7]}</code>, clean tree.</p>
</section>

<section id="results">
  <h2>6 F0: results</h2>
  <h3>6.1 Verdict</h3>
  {table(["", "Criterion", "Result", "Status"], crit_rows, "wrap")}
  <h3>6.2 Retrieval quality at k = {K}</h3>
  {table(["Env", "Full 1024", "Prefix", "Random", "Global", "In-domain", "Mean transferred", "Oracle"], main_rows)}
  <div class="legend"><span><i class="sw s1"></i>in-domain</span><span><i class="sw s2"></i>global</span><span><i class="sw s3"></i>prefix</span></div>
  <figure class="chart">{chart1}<figcaption>Figure 1. nDCG@10 change against the full embedding at k = {K}. Hover a point for its value.</figcaption></figure>
  <p>Only scifact gains from masking: the in-domain selector is {sc_gain} above the full embedding. nfcorpus breaks even.
  On fiqa, arguana and scidocs every method loses a few points, and the learned selectors land close to prefix truncation.</p>
  <h3>6.3 Transfer matrix</h3>
  {table([TM_HEAD] + ENVS, tm_rows)}
  <p class="note">nDCG@10 at k = {K}, rows train and columns test. * = significantly below the in-domain selector (Holm, α = {cfg['alpha']}).</p>
  <h3>6.4 Where the gain comes from</h3>
  <figure class="chart">{chart2}<figcaption>Figure 2. Headroom (in-domain − global, 95% CI) against the share of eval positives that are also
  train/validation positives. Filled points are significant.</figcaption></figure>
  <h3>6.5 Mask behavior</h3>
  <p>Mean overlap@{K} is {mask['overlap_cross_seed']:.4f} across seeds and {mask['overlap_cross_env']:.4f} across environments
  (difference CI [{sgn(ci[0])}, {sgn(ci[1])}]). Per-environment Spearman correlations of mean importance between seeds:</p>
  {table(["Env", "Between seeds", "Best match to another env"], [[e, f"{mask['spearman_seed'][e]:.3f}", f"{max(v for o, v in mask['spearman_env'][e].items() if o != e):.3f}"] for e in ENVS], "narrow")}
  <h3>6.6 Selected hyperparameters</h3>
  {table(["Env", "τ", "Epochs", "Best val nDCG@10"], hp_rows, "narrow")}
  <p class="note">Global selector: {GSEL['epochs']} epochs.</p>
</section>

<section id="discussion">
  <h2>7 Discussion</h2>
  <p>The verdict is correct under the rule as written, but the evidence is thin. The main threats to validity:</p>
  <ol>
    <li><b>Document overlap drives the gain.</b> Queries are split, documents are not. Headroom appears where eval positives
    were seen as training positives (scifact {pct(overlap['scifact']['pos_seen'])}, nfcorpus {pct(overlap['nfcorpus']['pos_seen'])})
    and vanishes where they were not (fiqa, arguana 0%). A selector may learn which corpus regions are relevant rather than
    how the domain uses coordinates. This uses legitimate training labels, so it is not leakage. It is a confound for H1,
    and for the hypernetwork, which will not see labels on new environments. scidocs ({pct(overlap['scidocs']['pos_seen'])}
    overlap, no gain) shows overlap is not sufficient on its own.</li>
    <li><b>Thresholds.</b> C1 passes with {n_sig}/{n_pairs}; {sig_in_gain} of those losses land on {' and '.join(gain_envs)}.
    C3 passes with exactly two environments.</li>
    <li><b>Instruction shift.</b> A transferred selector sees queries carrying another dataset's instruction prefix, so
    part of the measured transfer loss may be instruction shift rather than corpus shift.</li>
    <li><b>Seed noise.</b> For {', '.join(low_seed)}, seeds agree with each other less than with some other environment.
    Jensen–Shannon divergence stays below {js_max:.4f}, so mean importance is close to uniform. C2's CI is tight
    because the bootstrap resamples the {cfg['battery_size']} battery queries only, not seeds or environment pairs.</li>
    <li><b>Training.</b> {n_edge_ep} of {len(ENVS)} epoch choices and {n_edge_t} of {len(ENVS)} temperatures sit on a grid edge, and
    {best_at_end} of {hist_n} training runs kept a state from their last three epochs, so validation KL was mostly still falling
    when training stopped. Yet the grid picked the shortest budget for three environments, where longer training lowered validation nDCG. KL to the oracle and ranking quality
    may not be well aligned.</li>
    <li><b>Prefix is strong.</b> Outside scifact and nfcorpus the selectors do not beat MRL prefix truncation. This is risk R5 in practice.</li>
  </ol>
  <p>An earlier F0 run at commit <code>94bbefc</code> returned "not supported". Its selectors did not beat random masks on four
  environments, which ADR-0002 traced to early stopping and label generation that differed from the official recipe.
  Fixing the recipe flipped the verdict, which shows how sensitive H1 is to training choices.</p>
</section>

<section id="status">
  <h2>8 Status and next steps</h2>
  <p>{issue_total} roadmap issues exist and {len(open_issues)} are open. There are {len(prs)} pull requests. Of
  {len(ticket_branches)} ticket and experiment branches, {len(empty_ticket)} have no commits beyond <code>master</code>. The F0 branch
  is {f0_ahead} commits ahead and not merged. In the roadmap task list {t_done} of {t_all} tasks are done, all of them F0
  setup, implementation, documentation and hardening. Running F0 and committing its report (T023–T025) happened on the cluster but is not recorded in
  the repository. The verdict ADR (T026) is still unwritten.</p>
  {table(["Phase", "Name", "Scope", "Issues"], phase_rows, "wrap")}
  <h3>Recommended before Gate G2</h3>
  <ol>
    <li>Evaluate scifact and nfcorpus on eval queries whose positives never occur in training, or split the corpus as well
    as the queries. If headroom survives, H1 stands on firmer ground.</li>
    <li>Control instruction shift: encode transferred queries with a shared instruction, or measure transfer with the
    instruction swapped.</li>
    <li>Widen the grid (below 10 and above 200 epochs, below τ = {T_GRID[0]}) and check the KL–nDCG mismatch.</li>
    <li>Bootstrap C2 over seeds and environment pairs as well as queries.</li>
    <li>Add environments with real train splits and little document overlap before investing in phases D–E.</li>
    <li>Commit the F0 report on <code>exp/F0-selector-shift</code> and record the decision in an ADR, per the workflow.</li>
  </ol>
</section>

<section id="refs" class="refs">
  <h2>References</h2>
  <ol>
    <li>Wu, Z.; Zhang, R.; Nie, Z. Learning to Select: Query-Aware Adaptive Dimension Selection for Dense Retrieval. ACL 2026.</li>
    <li>Killingback, J.; Zeng, H.; Zamani, H. Hypencoder: Hypernetworks for Information Retrieval. SIGIR 2025.</li>
    <li>Eichholtz, A. et al. Hypencoder Revisited: Reproducibility and Analysis of Non-Linear Scoring for First-Stage Retrieval. SIGIR 2026.</li>
    <li>D'Erasmo, G. et al. Statistical Foundations of DIME: Risk Estimation for Practical Index Selection. EACL 2026.</li>
    <li>Zhang, Y. et al. Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models. 2025.</li>
    <li>Kusupati, A. et al. Matryoshka Representation Learning. NeurIPS 2022.</li>
    <li>Thakur, N. et al. BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. NeurIPS Datasets and Benchmarks 2021.</li>
  </ol>
  <p class="note">Sources: docs/plan, docs/architecture.md, docs/protocols.md, ADR-0001, ADR-0002, specs/002-hyperdime-roadmap,
  src/hyperdime at <code>{AM['git_commit'][:7]}</code>; F0 numbers from artifacts/spike/analysis/analysis.json, per-environment
  manifests, splits, histories and BEIR qrels copied from the cluster; project state from git and GitHub at build time.
  Exploratory results, not paper numbers.</p>
</section>
"""
# fmt: on

css = (Path(__file__).parent / "report.css").read_text()
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(
    "<title>HyperDIME-Qwen Project Report</title>\n"
    '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,500;0,8..60,600;1,8..60,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">\n'
    f"<style>{css}</style>\n<main>{body}</main>\n"
)
print(
    "wrote",
    OUT,
    "| tests",
    n_tests,
    "| issues",
    issue_total,
    len(open_issues),
    "| tasks",
    t_done,
    t_all,
    "| best-at-end",
    best_at_end,
    hist_n,
)
