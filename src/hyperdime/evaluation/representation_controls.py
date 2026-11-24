"""P3 at every cardinality, with descriptive controls for what relevance is confounded with.

Three things can make the masks of a query's relevant documents look alike without any relevance
structure: large coordinate magnitudes, the query vector they all share, and how many documents the
query happens to have. This module measures the same quantities on relevant documents and on
corpus-reference documents paired with the same query, in a matched number, at every ``k`` from 1
to ``D`` (ADR-0014).

A reference document is unjudged, not negative: nothing here calls one nonrelevant, ranks one, or
scores the corpus. It is only a document drawn without looking at its content.

Draws are planned per query on the CPU, from that query's own streams, so they do not depend on the
batch size or the device; the kernels in ``representation_engine`` then run on whole queries.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from hyperdime.evaluation.representation import (
    POSITIVE_STRATA,
    _save_arrays,
    _stratum,
    average_ranks,
    chance_jaccard,
    distribution,
    split_assignments,
    stratum_label,
)
from hyperdime.evaluation.representation_engine import (
    CHUNK_PAIRS,
    Clock,
    Profiles,
    finite_mean,
    finite_totals,
    gather,
    group_jaccard,
    group_measures,
    jaccard_curves,
    pair_curves,
    ranks_of,
    stable_order,
)

CURVE_DRAWS = 5
GROUP_SIZES = (2, 5, 20)
GROUP_DRAWS = 8
SCALE_ROWS = 200_000
FREQUENCY_DECIMALS = 10  # sums of 1/n differ in the last bits; rounding keeps true ties tied
PAIR_CURVES = ("similarity", "retained_positive_share", "removed_negative_share")
AGREEMENT_CURVES = ("jaccard", "scaled_jaccard")
GROUPS = ("positive", "reference")
VARIANTS = ("raw", "scaled")
SCALARS = ("score", "document_cosine", "sign_conflict_share", "cancelled_share")
POSITIVE_GROUP = -1


def pair_indices(count: int, limit: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Every unordered pair of ``count`` rows, or ``limit`` uniform draws when there are more."""
    if count * (count - 1) // 2 <= limit:
        first, second = np.triu_indices(count, 1)
        return first, second
    first = rng.integers(0, count, limit)
    second = (first + rng.integers(1, count, limit)) % count
    return first, second


def chance_curve(dim: int) -> np.ndarray:
    """``chance_jaccard`` at every ``k``; a reference, not a null model of these data."""
    return np.array([chance_jaccard(k, dim) for k in range(1, dim + 1)])


def floor_curve(dim: int) -> np.ndarray:
    """The smallest Jaccard possible: two ``k``-subsets of ``D`` must share ``2k - D`` dimensions
    once ``k > D / 2``, so some agreement is forced and reaches 1 at ``k = D``."""
    k = np.arange(1, dim + 1)
    return np.clip(2 * k - dim, 0, None) / dim


def chance_corrected(values: np.ndarray, chance: np.ndarray) -> np.ndarray:
    """``(value - chance) / (1 - chance)``; undefined (NaN) where chance is already 1."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result: np.ndarray = (values - chance) / (1.0 - chance)
    return result


def reference_rows(
    documents: int, excluded: set[int], count: int, rng: np.random.Generator
) -> np.ndarray:
    """``count`` distinct corpus rows avoiding ``excluded``, sorted; fewer only if the corpus has
    fewer. Rejection sampling keeps this cheap on a corpus of millions."""
    available = documents - len(excluded)
    if available <= count:
        return np.array(sorted(set(range(documents)) - excluded), dtype=np.int64)
    if count * 4 > available:
        free = np.array(sorted(set(range(documents)) - excluded), dtype=np.int64)
        return np.sort(rng.permutation(free)[:count])
    chosen: set[int] = set()
    while len(chosen) < count:
        for row in rng.integers(0, documents, count - len(chosen)).tolist():
            if row not in excluded:
                chosen.add(row)
    return np.array(sorted(chosen), dtype=np.int64)


def query_stream(seed: int, purpose: int, query_id: str) -> np.random.Generator:
    """A generator of its own per query, so a query's draws do not depend on which others ran."""
    digest = hashlib.sha256(f"{seed}:{purpose}:{query_id}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:16], "big"))


def coordinate_scale(
    query_vectors: np.ndarray, documents: np.ndarray, rows: np.ndarray
) -> np.ndarray:
    """``mean|q_j| * mean|d_j|``: a dimension's typical contribution size, from unlabeled vectors
    only. A dimension with no magnitude keeps scale 1 rather than dividing by zero."""
    mean_query = np.abs(np.asarray(query_vectors, dtype=np.float64)).mean(axis=0)
    total = np.zeros(documents.shape[1])
    for start in range(0, rows.size, 8192):
        block = np.asarray(documents[rows[start : start + 8192]], dtype=np.float64)
        total += np.abs(block).sum(axis=0)
    scale = mean_query * (total / rows.size)
    result: np.ndarray = np.where(scale > 0, scale, 1.0)
    return result


def row_spearman(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Spearman of each row pair, tied values sharing their average rank; NaN for a constant row,
    as at ``k = D`` where every dimension is always selected."""
    a = average_ranks(np.round(left, FREQUENCY_DECIMALS))
    b = average_ranks(np.round(right, FREQUENCY_DECIMALS))
    a -= a.mean(axis=1, keepdims=True)
    b -= b.mean(axis=1, keepdims=True)
    denominator = np.sqrt((a**2).sum(axis=1) * (b**2).sum(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        found: np.ndarray = (a * b).sum(axis=1) / denominator
    return np.where(denominator > 0, found, np.nan)


def total_variation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Share of selection, per row, that two frequency profiles place differently: 0 when equal, 1
    when disjoint. Each row of a query-balanced profile sums to ``k``."""
    k = np.arange(1, left.shape[0] + 1)
    result: np.ndarray = 0.5 * np.abs(left - right).sum(axis=1) / k
    return result


@dataclass(frozen=True)
class _Group:
    """One query's documents of one kind, with every draw made on them."""

    query: int
    repeat: int  # POSITIVE_GROUP, or which reference sample
    rows: np.ndarray
    first: np.ndarray  # pairs of positions in ``rows`` for agreement
    second: np.ndarray
    subsets: tuple[tuple[int, np.ndarray], ...]  # (group size, positions in ``rows``)
    probe: int | None = None  # position whose ranks stand for this query across queries


def _plan_pairs(size: int, limit: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if size < 2:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return pair_indices(size, limit, rng)


def _plan_query(
    index: int,
    query_id: str,
    positive_rows: np.ndarray,
    judged_rows: set[int],
    corpus: int,
    *,
    seed: int,
    reference_samples: int,
    max_query_pairs: int,
) -> list[_Group]:
    """Every draw of one query, from its own streams, in a fixed order."""
    count = positive_rows.size
    pair_rng = query_stream(seed, 5, query_id)
    first, second = _plan_pairs(count, max_query_pairs, pair_rng)
    probe = int(pair_rng.integers(0, count))
    subsets = tuple(
        (size, pair_rng.permutation(count)[:size])
        for size in GROUP_SIZES
        if count > 1 and count >= size
        for _ in range(GROUP_DRAWS)
    )
    groups = [_Group(index, POSITIVE_GROUP, positive_rows, first, second, subsets, probe)]
    for repeat in range(reference_samples):
        rows = reference_rows(
            corpus, judged_rows, count, query_stream(seed, 100 + repeat, query_id)
        )
        if rows.size == 0:
            raise ValueError(f"the corpus has no reference document for query {query_id}")
        first, second = _plan_pairs(rows.size, max_query_pairs, pair_rng)
        # Reference rows are already a uniform draw, so any ``size`` of them are.
        subsets = tuple(
            (size, pair_rng.permutation(rows.size)[:size])
            for size in GROUP_SIZES
            if rows.size > 1 and count >= size and rows.size >= size
        )
        groups.append(
            _Group(index, repeat, rows, first, second, subsets, 0 if repeat == 0 else None)
        )
    return groups


class _ControlPass:
    """Runs batches of whole queries and files what each gives; holds no per-pair state."""

    def __init__(
        self,
        query_ids: Sequence[str],
        query_vector: Callable[[str], np.ndarray],
        documents: np.ndarray,
        positive_order: np.ndarray,
        counts: np.ndarray,
        scale: np.ndarray,
        assignments: np.ndarray,
        reference_samples: int,
        *,
        device: str,
        clock: Clock,
        path: Callable[[str], Path],
    ) -> None:
        self.query_ids, self.query_vector, self.documents = query_ids, query_vector, documents
        self.positive_order = positive_order
        self.counts = counts
        self.offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])
        self.multi_index = np.cumsum(counts > 1) - 1
        self.strata = np.array([_stratum(int(c)) for c in counts], dtype=np.int64)
        self.reference_samples = reference_samples
        self.device, self.clock = device, clock
        queries, dim = counts.size, documents.shape[1]
        self.dim = dim
        multi = int((counts > 1).sum())
        self.scale = torch.from_numpy(scale).to(device)
        self.pair_files = {
            (name, group): np.lib.format.open_memmap(
                path(f"curve-{name}-{group}.npy"), mode="w+", dtype=np.float32, shape=(queries, dim)
            )
            for name in PAIR_CURVES
            for group in GROUPS
        }
        self.agreement_files = {
            (name, group): np.lib.format.open_memmap(
                path(f"curve-{name}-{group}.npy"), mode="w+", dtype=np.float32, shape=(multi, dim)
            )
            for name in AGREEMENT_CURVES
            for group in GROUPS
        }
        names = (*PAIR_CURVES, *AGREEMENT_CURVES)
        self.repeat_totals = {name: np.zeros((reference_samples, dim)) for name in names}
        self.repeat_counts = {name: np.zeros((reference_samples, dim)) for name in names}
        sizes = np.bincount(self.strata, minlength=len(POSITIVE_STRATA))
        self.profiles = {
            (group, variant): Profiles(dim, assignments, sizes, device)
            for group in GROUPS
            for variant in VARIANTS
        }
        self.probes = {
            (group, variant): np.empty((queries, dim), dtype=np.uint16)
            for group in GROUPS
            for variant in VARIANTS
        }
        self.scalars = {
            f"{name}_{group}": np.full(queries, np.nan) for name in SCALARS for group in GROUPS
        }
        self.sized = {
            f"{name}_m{size}_{group}": np.full(queries, np.nan)
            for name in ("sign_conflict_share", "cancelled_share")
            for size in GROUP_SIZES
            for group in GROUPS
        }
        self.reference_store: dict[str, list[np.ndarray]] = {
            "document_row": [], "query_index": [], "repeat": [],
        }  # fmt: skip
        self.errors = dict.fromkeys((*PAIR_CURVES, "jaccard"), 0.0)

    def _tensor(self, values: np.ndarray) -> Tensor:
        return torch.from_numpy(values).to(self.device)

    def run(self, groups: Sequence[_Group]) -> None:
        sizes = np.array([g.rows.size for g in groups])
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        rows = np.concatenate([g.rows for g in groups])
        queries = np.array([g.query for g in groups])
        repeats = np.array([g.repeat for g in groups])
        batch_queries = np.unique(queries)
        positive_groups = np.flatnonzero(repeats == POSITIVE_GROUP)
        with self.clock.span("read"):
            docs = gather(self.documents, rows)
            query_rows = np.stack(
                [
                    np.asarray(self.query_vector(self.query_ids[q]), dtype=np.float64)
                    for q in batch_queries
                ]
            )
            known = np.concatenate(
                [
                    np.arange(self.offsets[q], self.offsets[q] + self.counts[q])
                    for q in queries[positive_groups]
                ]
            )
            stored_order = np.asarray(self.positive_order[known], dtype=np.int64)
        with self.clock.span("transfer"):
            docs_on = self._tensor(docs)
            query_on = self._tensor(query_rows)[
                self._tensor(np.repeat(np.searchsorted(batch_queries, queries), sizes))
            ]
            stored_on = self._tensor(stored_order)
            group_of = self._tensor(np.repeat(np.arange(len(groups)), sizes))
            positive_row = self._tensor(np.repeat(repeats == POSITIVE_GROUP, sizes))
        with self.clock.span("compute"):
            values = query_on * docs_on
            order = torch.empty_like(values, dtype=torch.int64)
            order[positive_row] = stored_on
            order[~positive_row] = stable_order(values[~positive_row])
            ranks = ranks_of(order)
            scaled_ranks = ranks_of(stable_order(values / self.scale))
            curves = pair_curves(values, order)
            full = values.sum(dim=1)
            self._check(curves, full)
            scores = finite_mean(full, group_of, len(groups))
            found = {name: finite_mean(curves[name], group_of, len(groups)) for name in PAIR_CURVES}
            del curves, order
            self._profile(groups, sizes, group_of, ranks, scaled_ranks)
            agreement = self._agreement(groups, starts, docs_on, values, ranks, scaled_ranks)
            subsets = self._subsets(groups, starts, values)
            probes = self._probes(groups, starts, ranks, scaled_ranks)
        self._file(groups, batch_queries, found, scores, agreement, subsets, probes)

    def _check(self, curves: Mapping[str, Tensor], full: Tensor) -> None:
        errors = self.errors
        gap = (curves["similarity"][:, -1] - full).abs().max().item()
        errors["similarity"] = max(errors["similarity"], gap)
        for name, target in (("retained_positive_share", 1.0), ("removed_negative_share", 0.0)):
            last = curves[name][:, -1]
            last = last[last.isfinite()]  # a pair with no such mass has no share to check
            if last.numel():
                errors[name] = max(errors[name], (last - target).abs().max().item())

    def _profile(
        self,
        groups: Sequence[_Group],
        sizes: np.ndarray,
        group_of: Tensor,
        ranks: Tensor,
        scaled_ranks: Tensor,
    ) -> None:
        queries = self._tensor(np.array([g.query for g in groups]))[group_of]
        strata = self._tensor(self.strata)[queries]
        repeats = np.array(
            [1 if g.repeat == POSITIVE_GROUP else self.reference_samples for g in groups]
        )
        weight = self._tensor(1.0 / (sizes * repeats))[group_of]
        positive = self._tensor(np.array([g.repeat == POSITIVE_GROUP for g in groups]))[group_of]
        for name, chosen in (("positive", positive), ("reference", ~positive)):
            for variant, table in (("raw", ranks), ("scaled", scaled_ranks)):
                self.profiles[name, variant].add(
                    queries[chosen], strata[chosen], table[chosen], weight[chosen]
                )

    def _agreement(
        self,
        groups: Sequence[_Group],
        starts: np.ndarray,
        docs: Tensor,
        values: Tensor,
        ranks: Tensor,
        scaled_ranks: Tensor,
    ) -> dict[str, Any]:
        """Agreement of each group with at least two documents, as ``[such groups, ...]``."""
        members = [i for i, g in enumerate(groups) if g.rows.size > 1]
        if not members:
            return {}
        first = self._tensor(np.concatenate([groups[i].first + starts[i] for i in members]))
        second = self._tensor(np.concatenate([groups[i].second + starts[i] for i in members]))
        pair_group = self._tensor(
            np.repeat(np.arange(len(members)), [groups[i].first.size for i in members])
        )
        slots = len(members)
        jaccard = group_jaccard(ranks, first, second, pair_group, slots)
        self.errors["jaccard"] = max(
            self.errors["jaccard"], (jaccard[:, -1] - 1.0).abs().max().item()
        )
        units = docs / docs.norm(dim=1, keepdim=True)
        cosine = torch.zeros(first.numel(), device=docs.device, dtype=torch.float64)
        for start in range(0, first.numel(), CHUNK_PAIRS):
            stop = start + CHUNK_PAIRS
            cosine[start:stop] = (units[first[start:stop]] * units[second[start:stop]]).sum(dim=1)
        whole_rows = np.concatenate([starts[i] + np.arange(groups[i].rows.size) for i in members])
        whole_group = np.repeat(np.arange(slots), [groups[i].rows.size for i in members])
        conflict, cancelled = group_measures(
            values, self._tensor(whole_rows), self._tensor(whole_group), slots
        )
        return {
            "members": members,
            "jaccard": jaccard,
            "scaled_jaccard": group_jaccard(scaled_ranks, first, second, pair_group, slots),
            "document_cosine": finite_mean(cosine, pair_group, slots),
            "sign_conflict_share": conflict,
            "cancelled_share": cancelled,
        }

    def _subsets(
        self, groups: Sequence[_Group], starts: np.ndarray, values: Tensor
    ) -> dict[str, Any]:
        """Conflict and cancellation of the fixed-size draws."""
        members: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        sizes: list[int] = []
        owners: list[int] = []
        for index, group in enumerate(groups):
            for size, positions in group.subsets:
                labels.append(np.full(size, len(sizes)))
                members.append(positions + starts[index])
                sizes.append(size)
                owners.append(index)
        if not sizes:
            return {}
        conflict, cancelled = group_measures(
            values,
            self._tensor(np.concatenate(members)),
            self._tensor(np.concatenate(labels)),
            len(sizes),
        )
        return {
            "conflict": conflict,
            "cancelled": cancelled,
            "sizes": np.array(sizes),
            "owners": np.array(owners),
        }

    def _probes(
        self, groups: Sequence[_Group], starts: np.ndarray, ranks: Tensor, scaled_ranks: Tensor
    ) -> dict[tuple[str, str], tuple[np.ndarray, Tensor]]:
        chosen = [(i, g) for i, g in enumerate(groups) if g.probe is not None]
        found = {}
        for name, repeat in (("positive", POSITIVE_GROUP), ("reference", 0)):
            own = [(i, g) for i, g in chosen if g.repeat == repeat]
            rows = self._tensor(np.array([starts[i] + g.probe for i, g in own]))
            queries = np.array([g.query for _, g in own])
            found[name, "raw"] = (queries, ranks[rows])
            found[name, "scaled"] = (queries, scaled_ranks[rows])
        return found

    def _file(
        self,
        groups: Sequence[_Group],
        batch_queries: np.ndarray,
        found: Mapping[str, Tensor],
        scores: Tensor,
        agreement: Mapping[str, Any],
        subsets: Mapping[str, Any],
        probes: Mapping[tuple[str, str], tuple[np.ndarray, Tensor]],
    ) -> None:
        """Reduce group values to query values and write them; the device work ends here."""
        queries = np.array([g.query for g in groups])
        repeats = np.array([g.repeat for g in groups])
        slots = np.searchsorted(batch_queries, queries)
        positive = np.flatnonzero(repeats == POSITIVE_GROUP)
        reference = np.flatnonzero(repeats != POSITIVE_GROUP)
        with self.clock.span("compute"):
            outputs = self._pair_outputs(
                found, positive, reference, slots, repeats, batch_queries.size
            )
            score_out = self._scalar_output(scores, positive, reference, slots, batch_queries.size)
            agreement_out = self._agreement_outputs(agreement, queries, repeats, batch_queries)
            size_out = self._size_outputs(subsets, repeats, slots, batch_queries.size)
        with self.clock.span("transfer"):
            pair_out = {
                key: value.to(torch.float32).cpu().numpy() for key, value in outputs.items()
            }
            score_values = {key: value.cpu().numpy() for key, value in score_out.items()}
            agreement_out = {
                key: value.cpu().numpy() if isinstance(value, Tensor) else value
                for key, value in agreement_out.items()
            }
            size_values = {key: value.cpu().numpy() for key, value in size_out.items()}
            probe_out = {
                key: (query, rows.to(torch.int32).cpu().numpy())
                for key, (query, rows) in probes.items()
            }
        with self.clock.span("write"):
            for (name, group), values in pair_out.items():
                self.pair_files[name, group][batch_queries] = values
            for group in GROUPS:
                self.scalars[f"score_{group}"][batch_queries] = score_values[group]
            for key, value in agreement_out.items():
                if key[0] == "curve":
                    _, name, group = key
                    self.agreement_files[name, group][agreement_out["multi_rows"]] = value
                elif key[0] == "scalar":
                    _, name, group = key
                    self.scalars[f"{name}_{group}"][agreement_out["multi_queries"]] = value
            for key, value in size_values.items():
                self.sized[key][batch_queries] = value
            for (name, variant), (query, rows) in probe_out.items():
                self.probes[name, variant][query] = rows
            ref_rows = [g.rows for g in groups if g.repeat != POSITIVE_GROUP]
            self.reference_store["document_row"].append(np.concatenate(ref_rows))
            self.reference_store["query_index"].append(
                np.repeat(queries[reference], [r.size for r in ref_rows])
            )
            self.reference_store["repeat"].append(
                np.repeat(repeats[reference], [r.size for r in ref_rows])
            )

    def _accumulate_repeats(self, name: str, values: Tensor, repeats: np.ndarray) -> None:
        total, counts = finite_totals(values, self._tensor(repeats), self.reference_samples)
        self.repeat_totals[name] += total.cpu().numpy()
        self.repeat_counts[name] += counts.cpu().numpy()

    def _pair_outputs(
        self,
        found: Mapping[str, Tensor],
        positive: np.ndarray,
        reference: np.ndarray,
        slots: np.ndarray,
        repeats: np.ndarray,
        queries: int,
    ) -> dict[tuple[str, str], Tensor]:
        outputs: dict[tuple[str, str], Tensor] = {}
        for name, curve in found.items():
            outputs[name, "positive"] = curve[self._tensor(positive)]
            own = curve[self._tensor(reference)]
            outputs[name, "reference"] = finite_mean(own, self._tensor(slots[reference]), queries)
            self._accumulate_repeats(name, own, repeats[reference])
        return outputs

    def _scalar_output(
        self,
        values: Tensor,
        positive: np.ndarray,
        reference: np.ndarray,
        slots: np.ndarray,
        queries: int,
    ) -> dict[str, Tensor]:
        return {
            "positive": values[self._tensor(positive)],
            "reference": finite_mean(
                values[self._tensor(reference)], self._tensor(slots[reference]), queries
            ),
        }

    def _agreement_outputs(
        self,
        agreement: Mapping[str, Any],
        queries: np.ndarray,
        repeats: np.ndarray,
        batch_queries: np.ndarray,
    ) -> dict[Any, Any]:
        multi_queries = batch_queries[self.counts[batch_queries] > 1]
        if not multi_queries.size or not agreement:
            return {}
        members = np.array(agreement["members"])
        slot = np.searchsorted(multi_queries, queries[members])
        own = repeats[members]
        positive = own == POSITIVE_GROUP
        result: dict[Any, Any] = {
            "multi_rows": self.multi_index[multi_queries],
            "multi_queries": multi_queries,
        }
        chosen = self._tensor(np.flatnonzero(positive))
        others = self._tensor(np.flatnonzero(~positive))
        other_slot = self._tensor(slot[~positive])
        for name in AGREEMENT_CURVES:
            curve = agreement[name]
            result["curve", name, "positive"] = curve[chosen].to(torch.float32)
            result["curve", name, "reference"] = finite_mean(
                curve[others], other_slot, multi_queries.size
            ).to(torch.float32)
            self._accumulate_repeats(name, curve[others], own[~positive])
        for name in ("document_cosine", "sign_conflict_share", "cancelled_share"):
            values = agreement[name]
            result["scalar", name, "positive"] = values[chosen]
            result["scalar", name, "reference"] = finite_mean(
                values[others], other_slot, multi_queries.size
            )
        return result

    def _size_outputs(
        self, subsets: Mapping[str, Any], repeats: np.ndarray, slots: np.ndarray, queries: int
    ) -> dict[str, Tensor]:
        if not subsets:
            return {}
        result: dict[str, Tensor] = {}
        owners = subsets["owners"]
        for size in GROUP_SIZES:
            for group, wanted in (
                ("positive", repeats[owners] == POSITIVE_GROUP),
                ("reference", repeats[owners] != POSITIVE_GROUP),
            ):
                chosen = np.flatnonzero((subsets["sizes"] == size) & wanted)
                if not chosen.size:
                    continue
                index = self._tensor(chosen)
                slot = self._tensor(slots[owners[chosen]])
                result[f"sign_conflict_share_m{size}_{group}"] = finite_mean(
                    subsets["conflict"][index], slot, queries
                )
                result[f"cancelled_share_m{size}_{group}"] = finite_mean(
                    subsets["cancelled"][index], slot, queries
                )
        return result


def analyze_controls(
    query_ids: Sequence[str],
    query_vector: Callable[[str], np.ndarray],
    documents: np.ndarray,
    doc_rows: Mapping[str, int],
    judgments: Mapping[str, Mapping[str, int]],
    budgets: Sequence[int],
    positive_order: np.ndarray,
    *,
    seed: int,
    reference_samples: int,
    max_query_pairs: int,
    stability_draws: int,
    across_pairs: int,
    path: Callable[[str], Path],
    device: str = "cpu",
    batch_pairs: int = 8192,
    clock: Clock | None = None,
) -> dict[str, Any]:
    """Curves at every ``k`` for relevant and matched reference documents of the same queries.

    ``positive_order`` holds the ordering already computed for each relevant pair, in query order
    and by document row within a query; only reference orderings are computed here.
    """
    started = time.perf_counter()
    clock = clock or Clock(device)
    corpus, dim = documents.shape
    queries = len(query_ids)
    judged = {qid: {doc_rows[d] for d in judgments[qid] if d in doc_rows} for qid in query_ids}
    positives = {
        qid: np.array(
            sorted(doc_rows[d] for d, g in judgments[qid].items() if d in doc_rows and g > 0),
            dtype=np.int64,
        )
        for qid in query_ids
    }
    counts = np.array([positives[qid].size for qid in query_ids], dtype=np.int64)
    multi_queries = int((counts > 1).sum())
    scale_rng = np.random.default_rng([seed, 4])
    scale_rows = np.sort(scale_rng.permutation(corpus)[: min(SCALE_ROWS, corpus)])
    with clock.span("read"):
        scale = coordinate_scale(
            np.stack([np.asarray(query_vector(qid), dtype=np.float32) for qid in query_ids]),
            documents,
            scale_rows,
        )
    np.save(path("coordinate-scale.npy"), scale)
    assignments = split_assignments(queries, seed, stability_draws)
    curve_assignments = assignments[: min(CURVE_DRAWS, stability_draws)]
    state = _ControlPass(
        query_ids, query_vector, documents, positive_order, counts, scale, curve_assignments,
        reference_samples, device=device, clock=clock, path=path,
    )  # fmt: skip
    capped_queries = sum(int(c) * (int(c) - 1) // 2 > max_query_pairs for c in counts)
    short_groups = 0
    pending: list[_Group] = []
    held = 0
    for index, qid in enumerate(query_ids):
        with clock.span("plan"):
            planned = _plan_query(
                index, qid, positives[qid], judged[qid], corpus, seed=seed,
                reference_samples=reference_samples, max_query_pairs=max_query_pairs,
            )  # fmt: skip
        short_groups += sum(g.rows.size < counts[index] for g in planned)
        pending += planned
        held += sum(g.rows.size for g in planned)
        if held >= batch_pairs:
            state.run(pending)
            pending, held = [], 0
    if pending:
        state.run(pending)
    with clock.span("write"):
        for mapped in (*state.pair_files.values(), *state.agreement_files.values()):
            mapped.flush()
        _save_arrays(
            path("control-stats.npz"), {"positives": counts, **state.scalars, **state.sized}
        )
        reference = {key: np.concatenate(items) for key, items in state.reference_store.items()}
        _save_arrays(path("reference-documents.npz"), reference)
    pass_seconds = time.perf_counter() - started

    summary_started = time.perf_counter()
    pair_files, agreement_files = state.pair_files, state.agreement_files
    subsets = _subsets(counts)
    multi = counts > 1
    multi_subsets = {
        name: mask[multi] for name, mask in subsets.items() if name not in ("all", "single")
    }
    chance = chance_curve(dim)
    jaccard_keys = {
        name: _paired(agreement_files, name, multi_subsets) for name in AGREEMENT_CURVES
    }
    half_labels = assignments[:, multi] if multi_queries else assignments[:, :0]
    summary: dict[str, Any] = {
        "k": list(range(1, dim + 1)),
        "empty_mask_reference": {
            "k": 0,
            "similarity": 0.0,
            "retained_positive_share": 0.0,
            "removed_negative_share": 1.0,
            "note": "the empty mask scores 0 by definition; a reference, not a measurement",
        },
        "settings": {
            "reference_samples": reference_samples,
            "max_query_pairs": max_query_pairs,
            "scale_documents": int(scale_rows.size),
            "group_sizes": list(GROUP_SIZES),
            "group_draws": GROUP_DRAWS,
            "profile_draws": int(curve_assignments.shape[0]),
            "stability_draws": stability_draws,
            "across_pairs": across_pairs,
        },
        "coverage": {
            "queries": queries,
            "multi_positive_queries": multi_queries,
            "queries_with_sampled_pairs": capped_queries,
            "reference_groups_short_of_documents": int(short_groups),
            "reference_documents_drawn": int(reference["document_row"].size),
        },
        "consistency_at_full": state.errors,
        "chance_jaccard": chance.tolist(),
        "floor_jaccard": floor_curve(dim).tolist(),
        "pair_curves": {name: _paired(pair_files, name, subsets) for name in PAIR_CURVES},
        "agreement": jaccard_keys,
        "agreement_chance_corrected": _chance_corrected_means(
            agreement_files, multi_subsets, chance
        ),
        "delta_stability": {
            name: _half_range(
                np.asarray(agreement_files[name, "positive"], dtype=np.float64)
                - np.asarray(agreement_files[name, "reference"], dtype=np.float64),
                half_labels,
            )
            for name in AGREEMENT_CURVES
        },
        "reference_repeats": {
            name: _repeat_band(state.repeat_totals[name], state.repeat_counts[name])
            for name in (*PAIR_CURVES, *AGREEMENT_CURVES)
        },
        "across_query": _across(state.probes, seed, across_pairs, device),
        "frequency": _frequency(state.profiles, scale, queries, path),
        "scalars": _scalar_tables(state.scalars, subsets),
        "group_size": _size_tables(state.sized, counts),
    }
    summary["checkpoints"] = list(budgets)
    summary["cost"] = {
        "pass_seconds": pass_seconds,
        "summary_seconds": time.perf_counter() - summary_started,
    }
    del state, pair_files, agreement_files
    return summary


def _nanmean(matrix: np.ndarray) -> np.ndarray:
    """Column means of the finite entries; a column with none gives NaN, without a warning."""
    seen = np.isfinite(matrix)
    counts = seen.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        result: np.ndarray = np.where(seen, matrix, 0.0).sum(axis=0) / counts
    return result


def _subsets(counts: np.ndarray) -> dict[str, np.ndarray]:
    found = {"all": np.ones(counts.size, dtype=bool), "single": counts == 1, "multi": counts > 1}
    for low, high in POSITIVE_STRATA[1:]:
        mask = counts >= low
        if high is not None:
            mask = mask & (counts <= high)
        found[f"positives {stratum_label(low, high)}"] = mask
    return found


def _standard_error(matrix: np.ndarray) -> np.ndarray:
    """Standard error of each column's finite mean, over queries; NaN with fewer than two."""
    seen = np.isfinite(matrix)
    counts = seen.sum(axis=0)
    squares = np.where(seen, (matrix - _nanmean(matrix)) ** 2, 0.0).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        result: np.ndarray = np.sqrt(squares / (counts - 1) / counts)
    return np.where(counts > 1, result, np.nan)


def _curve_distribution(matrix: np.ndarray) -> dict[str, Any]:
    """Per column: mean, its standard error over queries, finite query count, and 10th/50th/90th
    percentiles; ``None`` for an empty subset. Every entry is a list of ``D`` values."""
    dim = matrix.shape[1]
    if matrix.shape[0] == 0:
        empty: list[None] = [None] * dim
        return {"n": 0, "queries": [0] * dim, "mean": empty, "se": empty, "p10": empty,
                "p50": empty, "p90": empty}  # fmt: skip
    p10 = np.empty(dim)
    p50 = np.empty(dim)
    p90 = np.empty(dim)
    if np.isfinite(matrix).all():
        p10, p50, p90 = np.quantile(matrix, [0.1, 0.5, 0.9], axis=0)
    else:
        for column in range(dim):
            finite = matrix[:, column]
            finite = finite[np.isfinite(finite)]
            if finite.size:
                p10[column], p50[column], p90[column] = np.quantile(finite, [0.1, 0.5, 0.9])
            else:
                p10[column] = p50[column] = p90[column] = np.nan
    return {
        "n": int(matrix.shape[0]),
        "queries": np.isfinite(matrix).sum(axis=0).tolist(),
        "mean": _nanmean(matrix).tolist(),
        "se": _standard_error(matrix).tolist(),
        "p10": p10.tolist(),
        "p50": p50.tolist(),
        "p90": p90.tolist(),
    }


def _paired(
    files: Mapping[tuple[str, str], np.ndarray],
    name: str,
    subsets: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Relevant, reference, and their within-query difference, for each subset of queries."""
    found: dict[str, Any] = {}
    positive = np.asarray(files[name, "positive"], dtype=np.float64)
    reference = np.asarray(files[name, "reference"], dtype=np.float64)
    for subset, mask in subsets.items():
        found[subset] = {
            "positive": _curve_distribution(positive[mask]),
            "reference": _curve_distribution(reference[mask]),
            "difference": _curve_distribution(positive[mask] - reference[mask]),
        }
    return found


def _chance_corrected_means(
    files: Mapping[tuple[str, str], np.ndarray],
    subsets: Mapping[str, np.ndarray],
    chance: np.ndarray,
) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for name in AGREEMENT_CURVES:
        found[name] = {
            group: chance_corrected(
                _nanmean(np.asarray(files[name, group], dtype=np.float64)), chance
            ).tolist()
            if subsets["multi"].any()
            else [None] * chance.size
            for group in GROUPS
        }
    return found


def _half_range(difference: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Lowest and highest mean difference at each ``k`` over disjoint query halves."""
    dim = difference.shape[1]
    if difference.shape[0] < 2 or labels.shape[0] == 0:
        return {"min": [None] * dim, "max": [None] * dim, "draws": 0}
    means = []
    for draw in labels:
        for half in (0, 1):
            chosen = difference[draw == half]
            if chosen.shape[0]:
                means.append(_nanmean(chosen))
    stacked = np.stack(means)
    return {"min": stacked.min(axis=0).tolist(), "max": stacked.max(axis=0).tolist(),
            "draws": int(labels.shape[0])}  # fmt: skip


def _repeat_band(totals: np.ndarray, counts: np.ndarray) -> dict[str, Any]:
    """How far the reference mean moves between repeated samples, at each ``k``."""
    with np.errstate(divide="ignore", invalid="ignore"):
        means = totals / counts
    if not np.isfinite(means).any():
        empty: list[None] = [None] * totals.shape[1]
        return {"min": empty, "max": empty}
    return {"min": np.nanmin(means, axis=0).tolist(), "max": np.nanmax(means, axis=0).tolist()}


def _across(
    probes: Mapping[tuple[str, str], np.ndarray], seed: int, pairs: int, device: str
) -> dict[str, Any]:
    """Masks of two different queries, one document each, so no query vector is shared. What is
    left is what coordinate magnitudes and common structure give two unrelated pairs."""
    any_probe = next(iter(probes.values()))
    queries = any_probe.shape[0]
    if queries < 2 or pairs < 1:
        return {}
    rng = np.random.default_rng([seed, 6])
    first = rng.integers(0, queries, pairs)
    second = (first + rng.integers(1, queries, pairs)) % queries
    found: dict[str, Any] = {}
    for (group, variant), probe in probes.items():
        table = torch.from_numpy(probe.astype(np.int32)).to(device)
        rows = []
        for start in range(0, pairs, CHUNK_PAIRS):
            stop = start + CHUNK_PAIRS
            curve = jaccard_curves(
                table[torch.from_numpy(first[start:stop]).to(device)],
                table[torch.from_numpy(second[start:stop]).to(device)],
            )
            rows.append(curve.cpu().numpy())
        found[f"{group}_{variant}"] = _curve_distribution(np.concatenate(rows))
    return found


def _frequency(
    profiles: Mapping[tuple[str, str], Profiles],
    scale: np.ndarray,
    queries: int,
    path: Callable[[str], Path],
) -> dict[str, Any]:
    """Whether recurrence belongs to relevant documents or to the coordinates, at every ``k``."""
    dim = scale.size
    arrays: dict[str, np.ndarray] = {}
    found: dict[str, Any] = {}
    marginal = np.broadcast_to(scale, (dim, dim))
    for variant in VARIANTS:
        positive, reference = profiles["positive", variant], profiles["reference", variant]
        f_positive = positive.frequency(queries)
        f_reference = reference.frequency(queries)
        arrays[f"positive_{variant}"] = f_positive.astype(np.float32)
        arrays[f"reference_{variant}"] = f_reference.astype(np.float32)
        excess = f_positive - f_reference
        half_positive, half_reference = positive.half_frequency(), reference.half_frequency()
        half_excess = half_positive - half_reference
        found[variant] = {
            "max_positive": f_positive.max(axis=1).tolist(),
            "max_reference": f_reference.max(axis=1).tolist(),
            "positive_vs_magnitude": row_spearman(f_positive, marginal).tolist(),
            "reference_vs_magnitude": row_spearman(f_reference, marginal).tolist(),
            "positive_vs_reference": row_spearman(f_positive, f_reference).tolist(),
            "excess_vs_magnitude": row_spearman(excess, marginal).tolist(),
            "total_variation": total_variation(f_positive, f_reference).tolist(),
            "split_half_positive": _half_spearman(half_positive),
            "split_half_reference": _half_spearman(half_reference),
            "split_half_excess": _half_spearman(half_excess),
        }
    raw_positive = profiles["positive", "raw"]
    raw_reference = profiles["reference", "raw"]
    overall = raw_positive.frequency(queries)
    strata = []
    for index, (low, high) in enumerate(POSITIVE_STRATA):
        size = round(raw_positive.stratum_queries[index])
        row: dict[str, Any] = {"positives": stratum_label(low, high), "queries": size}
        if size:
            own = raw_positive.stratum_frequency(index)
            other = raw_reference.stratum_frequency(index)
            row["positive_vs_corpus"] = row_spearman(own, overall).tolist()
            row["positive_vs_own_reference"] = row_spearman(own, other).tolist()
        strata.append(row)
    found["strata"] = strata
    _save_arrays(path("frequency-curves.npz"), arrays)
    return found


def _half_spearman(halves: np.ndarray) -> dict[str, Any]:
    """Split-half Spearman at each ``k``: mean and lowest over draws."""
    values = np.stack([row_spearman(draw[0], draw[1]) for draw in halves])
    with np.errstate(invalid="ignore"):
        defined = np.isfinite(values)
        counts = defined.sum(axis=0)
        mean = np.where(defined, values, 0.0).sum(axis=0) / np.where(counts, counts, 1)
        lowest = np.where(defined, values, np.inf).min(axis=0)
    return {
        "mean": np.where(counts > 0, mean, np.nan).tolist(),
        "min": np.where(counts > 0, lowest, np.nan).tolist(),
        "draws": int(halves.shape[0]),
    }


def _scalar_tables(
    scalars: Mapping[str, np.ndarray], subsets: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for name in SCALARS:
        positive, reference = scalars[f"{name}_positive"], scalars[f"{name}_reference"]
        found[name] = {
            subset: {
                "positive": distribution(positive[mask]),
                "reference": distribution(reference[mask]),
                "difference": distribution(positive[mask] - reference[mask]),
            }
            for subset, mask in subsets.items()
        }
    return found


def _size_tables(sized: Mapping[str, np.ndarray], counts: np.ndarray) -> list[dict[str, Any]]:
    """Conflict and cancellation at a fixed group size, so the count of documents is the same on
    every row. Only queries with at least that many positives enter, and that selection remains."""
    rows = []
    for size in GROUP_SIZES:
        row: dict[str, Any] = {"size": size, "queries": int((counts >= size).sum())}
        for name in ("sign_conflict_share", "cancelled_share"):
            positive = sized[f"{name}_m{size}_positive"]
            reference = sized[f"{name}_m{size}_reference"]
            row[name] = {
                "positive": distribution(positive),
                "reference": distribution(reference),
                "difference": distribution(positive - reference),
            }
        rows.append(row)
    return rows
