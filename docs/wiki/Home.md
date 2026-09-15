# HyperDIME-Qwen

Environment-conditioned hypernetwork for query-aware dimension selection on a frozen
Qwen3-Embedding-0.6B (D = 1024).

For each retrieval environment, per-dimension statistics of the embedding space are turned into a
low-rank modification of a Learning-to-Select dimension selector. Queries then keep only their most
important dimensions and search an unchanged document index.

> This wiki is generated from [`docs/wiki/`](https://github.com/wilkertl/hyper-mask-net/tree/master/docs/wiki)
> in the repository. Edits made in the web editor are overwritten on the next publish.

## Current status

**Milestone 1 — Reproduction, Phase A (foundation).** Nothing past the carried-over math utilities
is implemented. Next tickets: M0.1, M0.2.

## Pages

- [Architecture](Architecture): components, offline and online paths, design choices
- [Hypotheses and Gates](Hypotheses-and-Gates): what is being tested and when work stops
- [Roadmap](Roadmap): milestones, phases, and the ticket dependency graph
- [Experiment Protocols](Experiment-Protocols): environments, splits, P0–P2, leakage rules
- [Development Workflow](Development-Workflow): branches, issues, labels, pull requests, agents

## Links

- [Implementation plan](https://github.com/wilkertl/hyper-mask-net/blob/master/docs/plan/Qwen3_HyperDIME_Implementation_Plan.en.md)
- [Issues](https://github.com/wilkertl/hyper-mask-net/issues) · [Milestones](https://github.com/wilkertl/hyper-mask-net/milestones)
- [Experiment registry](https://github.com/wilkertl/hyper-mask-net/blob/master/docs/experiment_registry.md)
- [Decisions (ADRs)](https://github.com/wilkertl/hyper-mask-net/tree/master/docs/decisions)
