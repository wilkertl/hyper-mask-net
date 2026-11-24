# Independent council review after round 2

**7/10.** I would fund the P0 comparator and a one-output pilot. The decisive question is whether an unlabeled corpus tells us which adjustment will improve retrieval beyond a selector that already uses that corpus's variance. E1, S4 and E2 each measure a different part of that question. None measures the incremental gain.

## Verification

- **Confirmed.** At 307 dimensions the specialist minus E1 global difference was +0.1116 on SciFact, +0.0171 on NFCorpus and −0.0019 on MS MARCO. The global includes each recipient's training supervision. Epoch selection averages environment validation scores equally. This is evidence about pooling labeled environments, with a second difference between the training and checkpoint weighting. See `reports/E1/tables/payoff.csv:2–4` and `src/hyperdime/cli/train_global.py:47–60,70–84`.
- **Confirmed at the input-query level.** The F2 counts give 45,000/(45,000+2,331+728) = 93.63%. The three corpora differ sharply in both document count and positives per training query (1.06, 43.26 and 1.13 respectively for MS MARCO, NFCorpus and SciFact). Query share alone cannot quantify influence on the fitted weights. See `reports/P3/README.md:44–48,60–64`.
- **Confirmed with a qualification.** The S4 prior shares at 307 were 0.9512, 0.4281 and 0.3580, a 2.66-fold spread. NFCorpus reaches zero at 716 and SciFact at 819. The statistic averages a ratio of within-query logit spreads. It measures neither optimal mixing weight nor the fraction of ranking gain attributable to the prior. Standardizing each term separately and fitting a constant mixture changes the model. See `reports/S4/tables/prior_share.csv:13,31–37,49–55` and `src/hyperdime/selection/diagnostics.py:23–30`.
- **Refuted as worded.** The MS MARCO S3 weight moved in every run, ending at 0.0161–0.0162. Validation retained step zero in all three runs. An estimate at the selected checkpoint and an optimization trajectory answer different questions. See `reports/S3/tables/weights.csv:2–4`.
- **Confirmed.** E2 failed its decisive shift criteria. On MS MARCO, variance-only direct prediction reached 0.7864, while the shift probe reached 0.5657 against B2's 0.6687. The direct result gives no license for a row-conditioned residual. See `docs/decisions/ADR-0013-g3-space-signal.md:20–29,33–41`.
- **Confirmed.** The frozen encoder has disclosed supervised fine-tuning exposure to MS MARCO, NQ and HotpotQA. Exact benchmark query overlap is **unverifiable** from the paper. Its weakly supervised pretraining row lists synthetic data. See `/home/wilker/Papers/Qwen3 Embedding.pdf`, p. 13, Table 6, rows 1–2 (lines 31–39 of that page with layout-preserving text extraction).
- **Confirmed.** A held-family supervised reference on meta-test violates the label rule even if its training and scoring queries are distinct. Test qrels remain reserved for final scoring under the query split rule. See `AGENTS.md:14–18` and `docs/protocols.md:19–22,36–42`.
- **Unverifiable as a study-power claim.** Round 2 assumes an SD and independent families (`round2-opus-m.md:469–490` in this folder). As a cross-check, for independent normal family differences, one-sided one-sample t at 0.05, power is the upper noncentral-t tail at the central 95th percentile, df = K−1, noncentrality = √K·gain/SD. For +0.010 and SD 0.010, power at K = 8/10/12 is 0.815/0.898/0.945. At SD 0.015 it is 0.522/0.618/0.698. For +0.005 the corresponding 80%-power counts are 27 and 58. These calculations concern a single test against zero. Family independence, actual variance, and power for a joint decision remain unverified.

## Problems that change the decision

**Critical**

1. Freeze the family and label-access map before the payoff census. Shared source documents, near-duplicate queries, related releases and instruction variants can cross nominal dataset boundaries. Test-only qrels used to rank architectures on meta-validation would already violate the query split. Reserve test-only families for the final scorer. Use native non-test labels on separate meta-validation families or explicitly stay exploratory. Enforce this at the loader boundary, including negatives, cached specialists, normalization and checkpoint selection.
2. A scalar optimum is defined relative to a fitted base. Multiply W, b and all nine weights by the same positive constant and every hard mask stays identical, yet Astra's optimal *additive* offset scales by that constant. Cross-fitting raw offsets against different bases can manufacture apparent corpus shift. Separately, the hard top-k metric has plateaus. Arbitrary argmax tie breaking and ICC can reject interchangeable optima. Use a dimensionless adjustment and score the cost of choosing it on independent queries.
3. Count independent families before adopting a significance gate. With three, even the smallest one-sided sign-flip p is 1/8. Moreover, shared training sets make leave-family-out predictions dependent. Flipping their observed signs does not yield an exact test of the training pipeline. On untouched families, sign flips still require independent symmetric null differences or a justified exchangeability design. A failure to detect harm in many small corpora is weak evidence of safety.

**Important**

- Freeze G+V and match the second-stage optimization budget. A donor descriptor with the correct recipient variance isolates learned conditioning. A global refit isolates extra training. A zero centered input can switch off every generator gradient, making the proposed “matched” template control vacuous. Give the global alternative an actual trainable constant. Include a size-only predictor since log document count can identify MS MARCO almost perfectly among the first three.
- Qwen exposure, instructions, document count, and judgment density can all track the apparent payoff. Human labels distinguish many positives per NFCorpus query and typically one on MS MARCO. The same negative sampler and nDCG definition thereby reward different behaviors. Record these observables without using target qrels in adaptation. Freeze instructions and exposure tags from published provenance, report effects by family and a sensitivity excluding known exposed families. Use an “undisclosed” exposure category. A corpus absent from Table 6 has unknown exposure. The defensible population is the eligible benchmark families with frozen Qwen.
- Three historical environments allow two training families per outer fold. Even three geometry slopes plus an intercept are underidentified there. Nine budgets, 1,024 coordinates, many seeds and synthetic shards increase within-family information only. A supervised specialist is a comparison with a particular training recipe, not a mathematical upper bound on label-free performance. Ratios of gains with near-zero or negative denominators can gate a promising method on measurement noise.

**Minor**

- The corpus-dependent predictor must specify what happens at zero variance, missing calibration queries, and out-of-range descriptor values. Fix epsilon, omit unsupported protocols prospectively and clip using train-only limits.
- Report training-seed spread and query uncertainty separately. Verify k=1024 reproduces the full ranking with the same tie rule. State precision, exact-search implementation, identical-ID policy and the source of each query split in result manifests.

## Five rulings

1. **Generate one prior adjustment first.** Retain the nine shared budget coefficients. A single corpus output tests whether moving that existing profile helps. Nine free outputs or a budget slope require separate evidence of incremental utility.
2. **A second failed G3 closes the coordinate-conditioned bias and rank-four proposals in this program.** Keep scalar calibration eligible on its own retrieval evidence. G3 probes coordinate-frequency shift, so this is a resource decision with a specific scope. A passing G3 still needs an incremental retrieval result.
3. **Test-only corpora are final P0 score-only families.** Evaluate P1 only where a disjoint unlabeled calibration set was identified in advance. Keep P2 and every target-label specialist out of this study. If these rules exhaust the inventory, narrow the claim.
4. **At 8–12 families, plan estimation by default.** Permit a confirmatory mean-gain claim only after prospective calibration on an audited design shows adequate size and power for its actual joint rule. A mean advantage can coexist with appreciable harm on a new family.
5. **Primary endpoint is equal-family mean paired exact nDCG@10 across the nine deciles, against frozen family-balanced G+V.** Use k=307 for continuity. Compare each more complex rung to its accepted predecessor during selection. Use correct-versus-donor performance for attribution.

## Proposal I would sign

**Data and training.** Predeclare parent families using source provenance and content hashes, label regime, P0 eligibility, and final membership before examining any new effects. Keep the historical three in development. Freeze the encoder revision, D=1024 and instructions. Draw family, then corpus, then training query uniformly within level, and draw one of the nine budgets uniformly per batch. Fit a pooled S4 base with its multi-budget relaxed-mask retrieval loss, train positives and train-query negatives. Use identical batches, seed schedule, step allowance and checkpoint criterion across paired arms. Choose checkpoints by decile-mean exact retrieval on permitted validation queries. Within meta-train, repeat *all* supervised fitting, descriptor scaling and donor selection inside each outer held-family fold. Within the training side, build scalar training examples using inner held families and fold-specific bases, so a response profile always measures an adjustment to a base that has never seen that family. Reserve native validation families to accept or reject the locked candidate. Final families enter the pipeline once.

**Nested models.** Write π = log(q² ⊙ v_T + 10⁻¹²), where v_T is the recipient document variance, and u₀ = Wq+b+a_k π, with a_k ≥ 0. Every rung returns m = hard-top-k(u) and exact inner products (q ⊙ m)ᵀd using the original documents and unnormalized masked query. Freeze the accepted base including b and a. Specify deterministic ties.

- **R0, G+V.** Learn W of shape 1024 × 1024, b of length 1024 and nine shared weights a. No learned target adjustment.
- **R1, HyperPrior.** Replace a_k by max(0, a_k+s·t_T). Take s as the mean of the nine fitted a values, a train-only reference that rescales with the logits. If s=0, this parameterization is degenerate and the rung stops. Use one P0 feature x, log covariance participation ratio standardized over the training families and clipped to ±3. Set t_T = clip(c+βx,−2,2). Initialization c=β=0 exactly recovers R0. Two fitted numbers suffice. Evaluate a fixed grid of adjustments from −2 to +2 in 0.25 steps on meta-train inner held families. Fit c and β by maximizing the equal-family mean of linearly interpolated retrieval profiles on that grid, with ties resolved toward the smallest slope then intercept magnitude. Validate the resulting hard masks by actual retrieval. Fit the constant-only and log-size-only alternatives on identical profiles and bounds. This preserves the raw S4 logits and avoids trying to regress a noisy argmax.
- **R2, conditional and currently closed.** Add X_T h, where the 1024 × 18 P0 descriptor is centered per coordinate on training families and scaled per column with a frozen training RMS and epsilon. The shared vector h has 18 parameters, initialized at zero and fitted by the same masked retrieval objective with R1 frozen. This is exactly a linear feature bias whether applied online or generated offline.
- **R3, deferred.** Add (X_T H)(Bq), with H of shape 18 × 4 initialized at zero and B of shape 4 × 1024 initialized with orthonormal rows. The extra 4,168 trainable parameters include the shared query basis. Defer until there are at least 20 independent meta-train families, separately reserved validation/test families, and evidence for query-dependent gain over bias alone. Twenty is a resource floor, not a power calculation.

**Prospective gates.** Use absolute gains in decile-mean nDCG@10. The proposed 0.005 is a practical threshold for pursuing the experiment. Verify its operational value before registration. On development families report specialist minus G+V as context. For the actual scalar screen, select a grid adjustment on a development family's non-test calibration queries and score on disjoint queries. Require mean attainable gain ≥0.005 above the best global constant, with positive gains on at least half the families. If the upper one-sided 95% bound falls below 0.005 on independent families, retire this adjustment family for this objective. With an imprecise estimate below threshold, hold investment and report uncertainty. Avoid treating overlapping development folds as independent for that bound.

Advance R1 only for outer held-family mean gain ≥0.005 over G+V, positive mean gain over the constant and size-only fits, and positive mean over a donor control using the average prediction outcome across preassigned training-family donors while retaining v_T. Require all three directional controls to replicate on reserved validation families before the final lock. A size-only tie supports a simpler calibration rule. The three-corpus exercise cannot authorize promotion. R2 additionally requires a fresh development G3 replication meeting registered SC-001/002 for P0 and ≥0.005 incremental development gain from an attainable tied-bias oracle over R1. Fit the oracle on one set of non-test queries and score on another. R3 requires the same incremental oracle threshold over R2 with B learned off the measured family. For either learned rung require ≥0.005 incremental held-family gain over its predecessor and positive gains over both a trainable corpus-independent correction with matched updates and its donor control. Failed gates retain the simpler model. Keep a fresh validation set for any later reopening.

**Evaluation and inference.** For each query take the paired difference averaged over seeds and k ∈ {102,204,307,409,512,614,716,819,921}. Average queries within corpus, corpora within family, then families equally. Report family effects and intervals, a leave-one-family-out mean, and separate query-level paired bootstrap intervals with Holm-adjusted secondary tests. Include full, prefix, variance-only, balanced no-prior, matched global and G+V. Report pooled-variance sensitivity in development. For a practical efficiency claim also compare at equal k to a frozen per-query feedback baseline such as PRF-DIME and charge its extra retrieval pass. Keep historical E1 descriptive. Report offline descriptor/generation cost and model bytes separately from per-query encoding, selector, masking and search.

For a confirmatory run freeze one model, one protocol (P0), and a one-sided family-level mean test at 0.05 against zero. Prespecify a studentized family mean and t-based one-sided bounds. Before opening final qrels, simulate correlated model–G+V and correct–donor effects at the planned family and query counts, including observed query noise, skew and rare harmful families. Require ≤5% false positives at each null boundary within Monte Carlo error and ≥80% joint success at a prespecified +0.010 G+V alternative **and** a defensible donor alternative estimated on development alone. Success requires both lower 95% bounds above zero and the G+V point estimate ≥0.005. A claim that gain exceeds 0.005 requires its lower bound above 0.005. If calibration, inventory or donor-effect planning falls short, publish estimates for the observed families only. Report actual harmful family estimates and uncertainty. No universal non-inferiority or deployment-safety claim follows.

## First step available now

Recompute the kept-versus-removed prior validation contrast from the published S4 ablation on all three development corpora. This uses no test labels, fits no model, and checks whether the prior actually changes retrieval under the intended endpoint. Run from the repository root:

```python
import csv
from collections import defaultdict
from statistics import mean

by_corpus = defaultdict(dict)
with open("reports/S4/tables/ablation.csv", newline="") as source:
    for row in csv.DictReader(source):
        if row["track"] == "budget":
            by_corpus[row["environment"]][int(row["budget"])] = (
                float(row["kept"]) - float(row["prior removed"])
            )
for corpus, gains in sorted(by_corpus.items()):
    assert set(gains) == {102, 204, 307, 409, 512, 614, 716, 819, 921}
    print(corpus, round(mean(gains.values()), 4), round(gains[307], 4))
```

I ran this read-only calculation. Mean gains from retaining the prior are approximately +0.0388 MS MARCO, +0.0106 NFCorpus and +0.0064 SciFact, computed from the rounded validation rows. The corresponding 307 contrasts are +0.0390, +0.0114 and +0.0041. These specialist ablations establish that removing the prior costs retrieval quality. They leave the value of corpus conditioning open. The first new fit should therefore be three two-source G+V fits, each scored on the third development corpus, with matched no-prior and constant-offset controls and split-query offset profiles. The existing concatenating E1 command implements a different objective and sampling scheme. No three-family p-value or new architecture decision is due at this stage.

## Five changes required for 9/10

1. Publish an audited family, label-access and exposure manifest, including eligible validation families.
2. Freeze a balanced G+V and compare on paired data to a refitted constant, size-only and correct-variance donor.
3. Make scalar calibration transportable across base fits and judge it by held-query retrieval regret.
4. Register the independent-family estimand, attainable gates, joint power and exploratory fallback.
5. Cap initial capacity at one output and require incremental retrieval evidence plus fresh G3 evidence for coordinate generation.
