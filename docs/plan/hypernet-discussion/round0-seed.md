# Round 0 seed — orchestrator's initial proposal (unverified against PDFs)

Written before the discussion from ADRs 0007–0020, E1/E2/S1–S4/P3 reports and
docs/references.md. Paper claims come from catalog one-liners and must be checked.

## Thesis
Start with a tiny output head (per-corpus, per-budget prior weight), grow capacity only when
each rung wins on unseen corpora. Three environments cannot train or test a corpus→selector map;
more environments come before any architecture.

## Findings → implications
- E1 payoff/transfer (k=307): in-domain − global = +0.112 SciFact, +0.017 NFCorpus, −0.002 MS
  MARCO; transferring to SciFact costs 0.16–0.19. → Gain ceiling is mostly SciFact; the model must
  start equal to the global selector.
- ADR-0011/0012: S3 prior weight ≈1.0 MS MARCO, ≈0.002–0.02 NFCorpus/SciFact; S4 weight varies
  with k (NFCorpus 0.065→0). → First output: per-budget prior weight. Caveat: MS MARCO weight
  never moved off init (training dynamics, not corpus).
- ADR-0013 (G3 failed): per-dimension stats ≈ B1 (SciFact 0.438 vs 0.435). E2 secondary check:
  predicting per-dimension selection rate beats B1 (0.54/0.68/0.79 vs 0.44/0.56/0.54), seed ceiling
  0.82–0.998. → Signal on which dims are selected exists; 2 training corpora may be too few to
  separate "no signal" from "too few corpora".
- P3-mask-ranking/bench: positive-only masks rank worse than L2S; contrastive teacher
  (p̄ − n̄) ⊙ q closest. → Corpus summary needs neighborhood contrast (label-free: DIME/Eclipse PRF).
- ADR-0020 (+A1): under fair LR tuning no target/loss/per-budget/capacity/ranking effect. →
  Per-corpus training signal exhausted; cross-corpus sharing is the untried lever.

## Model
s_T(q,k) = W₀q + b₀ + α_T(k)·(q⊙q⊙v_T) + U_T(V_Tᵀq);  mask = top-k(s_T); score = <q⊙mask, d>
- W₀,b₀ global selector; v_T = Var_T(x) label-free corpus variance.
- (α_T,U_T,V_T) = H_φ(E_ψ(R_T)); R_T 1024×F per-dimension table (doc mean/var/tails, dim-ID
  embedding; under P1 query stats, PRF top-10 mean q_j d_j, Eclipse contrast
  E_q[q_j(μ_top−μ_bottom)_j]) + corpus scalars (eff. dim, PRF margin, log size).
- E_ψ: shared per-dimension MLP + attention pooling → Z_T. U_T,V_T generated per dimension with
  shared weights from (row_j, Z_T). Zero-init final layers (Chang et al. 2020).
- Ladder: H-0 global + fixed prior weight; H-1 α_T(k) (9 outputs); H-2 + per-dimension gain
  (≡ feature-augmented global selector — must be a comparator); H-3 + low-rank r∈{4,8}.
- Training: episodic leave-corpus-out, S1 multi-budget relaxed top-k loss, equal corpus weights,
  norm penalty on generated correction; pseudo-corpora (clusters/subsamples/mixtures) as extra
  training episodes only, checked to reproduce the real spread of α.

## Gates before building
1. Headroom (in-domain S4 − global, CIs) on all corpora. 2. Oracle α*_T(k) recovers what share
of headroom. 3. Corpus-level predictability: LOCO ridge from ~5 corpus descriptors to α* and
headroom vs predict-the-mean, permutation test. 4. Rerun F10/E2 on new corpora.

## Evaluation (E3/G4)
~10 train + ≥5 untouched held-out corpora (test-only BEIR sets usable as held-out under P0/P1).
Comparators: full query, MRL prefix, variance, DIME, Eclipse, global, H-0, feature-augmented
global, P2 refs (few-label α, Search-Adaptor), in-domain S4 as ceiling. Endpoint: mean nDCG@10
over 10–90% budgets; corpus = unit; paired bootstrap + Holm; non-inferior to global on every
held-out corpus (δ=0.005), better on ≥m, recover stated share of headroom. Controls: shuffled
Z_T, dataset-identity check, corpus/query-sample size 0/50/500, hyperparameters by LOCO only.

## Not doing / risks
No Hypencoder-style per-query scorer (breaks masked exact IP); no full 1024² generation. L2S not
reproduced on MS MARCO (ADR-0006); NFCorpus targets poorly calibrated; α partly training
artifact. Optional framing: Leblanc et al. 2025 PAC-Bayes/sample-compression hypernetworks.
