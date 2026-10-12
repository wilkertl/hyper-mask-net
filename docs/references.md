# Research references

This catalog records the 22 research papers supplied for HyperDIME-Qwen. Titles, lead authors,
and dates were checked against the PDFs; links use publisher or preprint identifiers. The PDFs
remain outside Git. A paper listed here is background, not evidence that HyperDIME reproduces
its findings. Cite specific papers where their claims inform a design or protocol.

## Core method and encoder

- Wu, Zhang, and Nie (2026). [Learning to Select: Query-Aware Adaptive Dimension Selection for Dense Retrieval](https://doi.org/10.18653/v1/2026.acl-long.849). ACL. Query-aware selector and primary reproduction baseline.
- Zhang et al. (2025). [Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models](https://arxiv.org/abs/2506.05176). Technical report. Frozen embedding model.
- Kusupati et al. (2022). [Matryoshka Representation Learning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/c32319f4868da7613d78af9993100e42-Abstract-Conference.html). NeurIPS. Prefix-dimension baseline.
- Faggioli, Ferro, Perego, and Tonellotto (2024). [Dimension Importance Estimation for Dense Information Retrieval](https://doi.org/10.1145/3626772.3657691). SIGIR. Dimension importance and masking background.
- Killingback, Zeng, and Zamani (2025). [Hypencoder: Hypernetworks for Information Retrieval](https://doi.org/10.1145/3726302.3729983). SIGIR. Hypernetwork-based retrieval comparison.
- D'Erasmo et al. (2026). [Statistical Foundations of DIME: Risk Estimation for Practical Index Selection](https://doi.org/10.18653/v1/2026.eacl-short.51). EACL. Statistical view of dimension selection.

## Dimension selection and retrieval efficiency

- Faggioli, Ferro, Perego, and Tonellotto (2025). [CoDIME: A Counterfactual Approach for Dimension Importance Estimation through Click Logs](https://doi.org/10.1145/3726302.3729926). SIGIR. Dimension importance from implicit feedback.
- D'Erasmo, Trappolini, and Silvestri (2025). [Eclipse: Contrastive Dimension Importance Estimation with Pseudo-Irrelevance Feedback for Dense Retrieval](https://doi.org/10.1145/3731120.3744579). ICTIR. Pseudo-feedback alternative.
- Faggioli, Ferro, Perego, and Tonellotto (2025). [Getting off the DIME: Dimension Pruning via Dimension Importance Estimation for Dense Information Retrieval](https://doi.org/10.1145/3765619). Dimension pruning background.
- Campagnano, Mallia, and Silvestri (2025). [Unveiling DIME: Reproducibility, Generalizability, and Formal Analysis of Dimension Importance Estimation for Dense Retrieval](https://doi.org/10.1145/3726302.3730318). SIGIR. Reproducibility and generalization of DIME.
- Calugaru et al. (2026). [Load-sensitive Selective Pruning in Dense Retrieval](https://doi.org/10.1145/3805712.3809875). SIGIR. Serving-time efficiency context.
- Pasin, Faggioli, Ferro, Perego, and Tonellotto. *When Reducing Representations Improves Performance*. Supplied manuscript; the PDF did not establish a publication year or public identifier. Dimension-subset selection and transfer.
- Saxena et al. (2026). [IMRNNs: An Efficient Method for Interpretable Dense Retrieval via Embedding Modulation](https://arxiv.org/abs/2601.20084). Preprint. Embedding modulation comparison.

## Hypernetworks and adaptation

- Ha, Dai, and Le (2016). [HyperNetworks](https://arxiv.org/abs/1609.09106). Preprint. Foundational hypernetwork design.
- Chang, Flokas, and Lipson (2020). [Principled Weight Initialization for Hypernetworks](https://arxiv.org/abs/2312.08399). ICLR. Initialization of generated weights.
- Eichholtz et al. (2026). [Hypencoder Revisited: Reproducibility and Analysis of Non-Linear Scoring for First-Stage Retrieval](https://doi.org/10.1145/3805712.3808549). SIGIR. Independent Hypencoder reproduction and analysis.
- Leblanc et al. (2025). [Generalization Bounds via Meta-Learned Model Representations: PAC-Bayes and Sample Compression Hypernetworks](https://proceedings.mlr.press/v267/leblanc25a.html). ICML. Dataset-to-model meta-learning background.
- Maekawa et al. (2026). [Align then Train: Efficient Retrieval Adapter Learning](https://arxiv.org/abs/2604.03403). Preprint, under review in the supplied PDF. Retrieval adaptation comparison.
- Yoon, Chen, Arik, and Pfister (2024). [Search-Adaptor: Embedding Customization for Information Retrieval](https://aclanthology.org/2024.acl-long.661/). ACL. Retrieval adaptation comparison.

## Broader and mathematical background

- Shen et al. (2025). [A Survey of Controllable Learning: Methods and Applications in Information Retrieval](https://arxiv.org/abs/2407.06083). Survey preprint. Positioning of controllable retrieval.
- Jiao et al. (2024). [SPIN: Sparsifying and Integrating Internal Neurons in Large Language Models for Text Classification](https://aclanthology.org/2024.findings-acl.277/). Findings of ACL. Related feature-selection idea in a different task.
- Barata and Hussein (2011). [The Moore-Penrose Pseudoinverse: A Tutorial Review of the Theory](https://arxiv.org/abs/1110.6882). Preprint. Linear-algebra background.
