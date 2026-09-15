# Roadmap

Issues carry the ticket IDs below in their titles, e.g. `[M3.1] Oracle importance targets`.

## Milestones and phases

| Milestone | Phase | Tickets | Exit |
|---|---|---|---|
| **1 — Reproduction** | A Foundation | M0.1, M0.2, M0.3, M0.4, M1.1, M1.2, M1.3, M2.1, M2.2, M1.4 | — |
| | B Scientific baseline | M12.1, M3.1, M3.2, M3.3, E0 | Gate G1 |
| **2 — Scientific premise** | C Evidence for H1 | M12.3, M4.1, M4.2, M4.3, G2 | Gate G2 (Go/No-Go) |
| **3 — Space representation** | D Space representation | M5.1, M5.2, M5.3, M5.4, M5.5 | Gate G3 |
| **4 — HyperDIME prototype** | E Hypernetwork | M7.1, M6.1, M8.1, M9.1, M9.2 | Meta-validation vs global selector |
| **5 — Paper-grade evaluation** | F Final evaluation | M10.1, M11.1, E3, E4/E5, M12.2 | Gates G4 and G5 |

M12.3 (significance tests) is scheduled earlier than in the plan because Gate G2 needs it.

## Ticket dependencies

```mermaid
flowchart TD
  subgraph MS1["Milestone 1 — Reproduction"]
    M0_1["M0.1 Tooling"]
    M0_2["M0.2 Schemas"] --> M0_3["M0.3 Manifests"] --> M0_4["M0.4 Configs + CLI"]
    M0_2 --> M1_1["M1.1 BEIR loaders"] --> M1_2["M1.2 MS MARCO"]
    M1_1 --> M1_3["M1.3 Splits"]
    M0_3 --> M1_3
    M0_2 --> M2_1["M2.1 Qwen wrapper"] --> M2_2["M2.2 Embedding cache"]
    M1_3 --> M2_2
    M0_4 --> M2_2
    M1_3 --> M1_4["M1.4 Negatives"]
    M2_2 --> M1_4
    M2_2 --> M12_1["M12.1 Exact evaluation"]
    M0_3 --> M12_1
    M1_4 --> M3_1["M3.1 Oracle"]
    M2_2 --> M3_1
    M3_1 --> M3_2["M3.2 L2S trainer"]
    M3_1 --> M3_3["M3.3 Baselines"]
    M12_1 --> M3_3
    M3_2 --> E0["E0 Report + G1"]
    M3_3 --> E0
    M12_1 --> E0
  end
  subgraph MS2["Milestone 2 — Scientific premise"]
    M12_3["M12.3 Significance"]
    M4_1["M4.1 Per-domain selectors"] --> M4_2["M4.2 Transfer matrix"]
    M4_1 --> M4_3["M4.3 Behavior analysis"]
    M4_2 --> G2["G2 Go/No-Go"]
    M4_3 --> G2
  end
  subgraph MS3["Milestone 3 — Space representation"]
    M5_1["M5.1 Document stats"] --> M5_2["M5.2 Query + interaction"] --> M5_3["M5.3 Contrast + redundancy"] --> M5_4["M5.4 H2 probes + G3"] --> M5_5["M5.5 Freeze descriptor"]
  end
  subgraph MS4["Milestone 4 — Prototype"]
    M7_1["M7.1 Low-rank selector"] --> M8_1["M8.1 Hyper Head"]
    M6_1["M6.1 Space Encoder"] --> M8_1
    M8_1 --> M9_1["M9.1 Episodic trainer"] --> M9_2["M9.2 Meta-validation"]
  end
  subgraph MS5["Milestone 5 — Evaluation"]
    M10_1["M10.1 Protocols"] --> E3["E3 Main result + G4"]
    M11_1["M11.1 Policies"] --> E3
    E3 --> E45["E4/E5 Ablations"]
    M10_1 --> M12_2["M12.2 Efficiency + G5"]
  end
  M12_1 --> M12_3
  E0 --> M4_1
  M1_2 --> M4_1
  M12_3 --> M4_2
  G2 --> M5_1
  M4_3 --> M5_4
  M5_5 --> M7_1
  M5_5 --> M6_1
  M9_2 --> M10_1
  M9_2 --> M11_1
```
