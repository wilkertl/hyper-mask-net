"""Synthetic `compare-selectors --experiment S4` result files for the S4 report tests."""

from collections.abc import Mapping, Sequence
from pathlib import Path

from synthetic_s1 import BUDGETS, ENVS, SEEDS
from synthetic_s2 import _write, build_track

CANDIDATE, CONTROL = "budget", "weighted-aligned"
TRACKS = (CONTROL, CANDIDATE)
OTHERS = ("retrieval", "geometric", "weighted", CONTROL)
FAMILIES = ("multi-budget-budget",)
CELL = "multi-budget-budget-lr0.001-ts0.05-b0.1"


def _seeded(track: str, seed: int) -> dict[str, object]:
    """A seeded selection record: kept step 1000 while the rerank score preferred step 0."""
    weights = [0.2 * (i + 1) + 0.1 * seed for i in range(len(BUDGETS))]
    return {
        "recipe": f"{track}-seeded",
        "seed": seed,
        "validation": {
            "kept_step": 1000,
            "best_step": 0.0,
            "soft_mass": 0.95,
            "candidate_scores": {
                "0": {"ndcg@10": 0.40, "val_rerank_ndcg10": 0.50},
                "1000": {"ndcg@10": 0.45, "val_rerank_ndcg10": 0.48},
            },
            "prior_weights": weights if track == CANDIDATE else [0.5 + 0.1 * seed],
        },
    }


def build(
    root: Path,
    over: Mapping[str, Mapping[str, float]],
    *,
    seeds: Mapping[str, Sequence[int]] | None = None,
) -> Path:
    """The budget selector scores ``over[reference][env]`` above each reference per query
    (``weighted-aligned`` is the aligned control); ``seeds`` per track narrows the seeded records
    named in the sources. Returns the evaluation directory."""
    seeds = seeds or {}
    extra_sources: dict[str, dict[str, object]] = {}
    for env in ENVS:
        grid = root / "selectors" / f"{CONTROL}-grid" / env / f"controlgrid-{env}"
        _write(grid, "selection.json", {"selected": f"multi-budget-{CONTROL}-lr0.001-ts0.05-b0.1"})
        named: dict[str, object] = {f"{CONTROL}_grid": f"controlgrid-{env}"}
        for track in TRACKS:
            runs = {}
            for seed in seeds.get(track, SEEDS):
                run = f"{track}-{env}-{seed}"
                directory = root / "selectors" / f"{track}-seeded" / env / f"seed-{seed}" / run
                _write(directory, "selection.json", _seeded(track, seed))
                runs[f"seed-{seed}"] = run
            named[track] = runs
        extra_sources[env] = named
    diagnostics = {
        track: {
            env: {
                f"seed-{seed}": {
                    str(k): {
                        "prior_share": 0.6 - 0.1 * i,
                        "overlap_variance": 0.8,
                        "overlap_correction": 0.3,
                        "prior_weight": 0.2 * (i + 1),
                    }
                    for i, k in enumerate(BUDGETS)
                }  # fmt: skip
                for seed in SEEDS
            }
            for env in ENVS
        }
        for track in TRACKS
    }
    ablation = {
        track: {
            env: {
                f"seed-{seed}": {str(k): {"kept": 0.5, "no-prior": 0.3} for k in BUDGETS}
                for seed in SEEDS
            }
            for env in ENVS
        }
        for track in TRACKS
    }
    return build_track(
        root, "S4", CANDIDATE, CANDIDATE, FAMILIES, over, others=OTHERS,
        extra_sources=extra_sources,
        files={"diagnostics.json": diagnostics, "ablation.json": ablation},
    )  # fmt: skip
