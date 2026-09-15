# Plano de Implementação Completa — Hypernetwork para Seleção Adaptativa de Dimensões no Qwen3-Embedding-0.6B

**Objetivo:** implementar e validar uma arquitetura em que um *Embedding-Space Encoder* representa o ambiente de retrieval produzido pelo **Qwen3-Embedding-0.6B**, uma **Hyper Head** converte essa representação em parâmetros de um *Dimension Selector*, e o seletor gera importância por dimensão para cada query, de forma análoga ao predictor do *Learning to Select*, porém adaptável ao ambiente/corpus.

**Escopo fixo da primeira versão:** somente **Qwen3-Embedding-0.6B**, dimensão máxima **D = 1024**, encoder congelado, seleção no lado da query, documentos/index sem modificação durante a seleção.

---

## 1. Base científica e separação entre evidência e proposta

### 1.1 O que vem diretamente dos trabalhos de referência

1. **Learning to Select (Wu et al., ACL 2026).** O método constrói uma distribuição alvo de importância por dimensão usando positivos e hard negatives supervisionados e treina um predictor query-only para aproximá-la. Na inferência, o predictor recebe apenas o embedding da query, seleciona Top-k dimensões, zera as demais e mantém embeddings dos documentos e o índice ANN inalterados. No paper, o predictor é uma única camada fully connected D→D seguida de (log-)softmax. O Qwen-Embedding-0.6B é avaliado com D=1024.
2. **Hypencoder (Killingback et al., SIGIR 2025).** O trabalho usa hypernetwork/hyperhead para transformar representação da query em pesos e biases de uma pequena rede específica da query (*q-net*). A contribuição que reutilizamos é o padrão arquitetural “contexto → gerador de parâmetros → rede alvo”, e não a função de scoring do Hypencoder.
3. **Statistical Foundations of DIME / RDIME (D’Erasmo et al., EACL 2026).** O trabalho formaliza seleção query-dependent e propõe um critério para determinar adaptativamente quantas dimensões reter por query, evitando um único k global definido por grid search. RDIME é uma extensão opcional para a etapa final de seleção.
4. **Qwen3-Embedding-0.6B.** O modelo produz embeddings de até 1024 dimensões, suporta dimensionalidades customizadas/MRL e é instruction-aware. Nesta implementação, a dimensionalidade completa de 1024 será preservada na geração de embeddings para permitir seleção arbitrária de coordenadas.

### 1.2 O que é hipótese/proposta deste projeto

A hipótese central não está estabelecida pelos trabalhos acima. Ela será testada experimentalmente:

> **H1 — Selector shift:** o seletor ótimo de dimensões para o mesmo Qwen3-Embedding-0.6B muda de forma sistemática entre ambientes de retrieval.
>
> **H2 — Geometry explains selector shift:** estatísticas não supervisionadas/semi-supervisionadas do espaço induzido pelo Qwen contêm informação suficiente para prever parte dessa mudança.
>
> **H3 — Hypernetwork adaptation:** uma Hyper Head consegue mapear a representação do ambiente para parâmetros de um seletor query-aware e melhorar generalização cross-domain comparada a um seletor global fixo.

A cadeia causal a investigar é:

`distribution shift → embedding/retrieval geometry shift → optimal selector shift → hypernetwork adaptation`

A implementação deve ser construída para permitir **refutar** cada elo separadamente. Se H1 falhar, a hypernetwork é desnecessária. Se H1 for verdadeira e H2 falhar, a representação do espaço é insuficiente. Se H1/H2 forem sustentadas e H3 falhar, o problema está na parametrização/treinamento da Hyper Head.

---

## 2. Arquitetura conceitual final

### 2.1 Caminho offline, por ambiente

Um ambiente de retrieval é definido como:

`T = (corpus C, distribuição de queries Q, instrução I, Qwen3-Embedding-0.6B fixo)`

Pipeline offline:

`C (+ Q_calib opcional) → Qwen embeddings → Space Statistics R_T → Space Encoder Gψ → environment representation Z_T → Hyper Head Hφ → selector parameters θ_T`

Os pesos `θ_T` são persistidos e reutilizados para todas as queries daquele ambiente.

### 2.2 Caminho online, por query

`query → Qwen3-Embedding-0.6B → e_q ∈ R^1024 → Dimension Selector f_{θ_T} → importance scores s_q ∈ R^1024 → selection policy → mask m_q → masked query → retrieval`

A versão mínima mantém a estratégia do *Learning to Select*:

`e'_q = e_q ⊙ m_q`

`score(q,d) = (e'_q)^T e_d`

Os embeddings dos documentos permanecem inalterados.

### 2.3 Representação recomendada do ambiente

Como o encoder é sempre o mesmo, a identidade das 1024 coordenadas é estável. Logo, a representação deve ser:

- invariável à ordem dos documentos/queries amostrados;
- **não invariável à identidade das dimensões**;
- construída como uma matriz por dimensão:

`R_T ∈ R^(1024 × h_stats)`

Cada linha `r_j` representa a dimensão `j` e agrega quatro famílias de sinais:

1. **Marginal/document:** média, desvio, quantis, energia e sparsity-like statistics de `d_j`.
2. **Query:** mesmas estatísticas para `q_j`, quando queries de calibração estiverem disponíveis.
3. **Interaction:** estatísticas de `q_j d_j`, aproximando a contribuição da dimensão para o produto interno.
4. **Contrast/redundancy:** estatísticas de `q_j(p_j-n_j)` com pseudo/true positives e negatives, além de informação de correlação/redundância interdimensional.

A primeira versão deve começar simples e adicionar grupos por ablação.

---

# 3. Princípios de implementação

1. **Encoder congelado e determinístico.** A implementação nunca deve alterar pesos do Qwen na fase inicial.
2. **Contratos antes de modelos.** Todo módulo deve ter entradas/saídas serializáveis e versionadas antes da implementação neural.
3. **Baselines primeiro.** A Hyper Head só entra após a reprodução do *Learning to Select* e do experimento de selector shift.
4. **Cada hipótese tem um teste falsificável.** Métricas e critérios de aceitação devem ser definidos antes do treino principal.
5. **Nada de “magic pipeline”.** Space statistics, space encoder, hyperhead, target selector e policy são módulos independentes e substituíveis.
6. **Separar treino de inferência.** Labels podem ser usados para criar targets no meta-training; a inferência em novo ambiente deve usar somente os sinais permitidos pelo protocolo experimental.
7. **Reprodutibilidade integral.** Configuração, seed, dataset version, checkpoint hash, model revision e resultados devem ser registrados para toda execução.

---

# 4. Estrutura do repositório

```text
hyperdime-qwen/
├── pyproject.toml
├── README.md
├── AGENTS.md
├── .gitignore
├── .pre-commit-config.yaml
├── configs/
│   ├── model/qwen3_0_6b.yaml
│   ├── data/{scifact,nfcorpus,msmarco}.yaml
│   ├── space_rep/*.yaml
│   ├── selector/*.yaml
│   ├── hyperhead/*.yaml
│   └── experiment/*.yaml
├── src/hyperdime/
│   ├── contracts/
│   │   ├── schemas.py
│   │   ├── manifests.py
│   │   └── validation.py
│   ├── data/
│   │   ├── loaders.py
│   │   ├── splits.py
│   │   ├── sampling.py
│   │   └── negatives.py
│   ├── embeddings/
│   │   ├── qwen.py
│   │   ├── cache.py
│   │   └── normalize.py
│   ├── oracle/
│   │   ├── importance.py
│   │   └── targets.py
│   ├── baselines/
│   │   ├── learning_to_select.py
│   │   ├── global_selector.py
│   │   ├── mrl_prefix.py
│   │   └── random_mask.py
│   ├── space/
│   │   ├── marginal.py
│   │   ├── query_stats.py
│   │   ├── interactions.py
│   │   ├── contrast.py
│   │   ├── redundancy.py
│   │   ├── descriptor.py
│   │   └── encoder.py
│   ├── hypernet/
│   │   ├── hyperhead.py
│   │   ├── modulation.py
│   │   └── generated_selector.py
│   ├── selection/
│   │   ├── topk.py
│   │   ├── rdime.py
│   │   └── masks.py
│   ├── retrieval/
│   │   ├── exact.py
│   │   ├── faiss_backend.py
│   │   └── scoring.py
│   ├── training/
│   │   ├── selector_trainer.py
│   │   ├── meta_trainer.py
│   │   ├── episodes.py
│   │   └── losses.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── selector_shift.py
│   │   ├── cross_domain.py
│   │   ├── ablations.py
│   │   └── statistics.py
│   └── cli/
│       ├── embed.py
│       ├── build_space.py
│       ├── train_selector.py
│       ├── train_hypernet.py
│       └── evaluate.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── regression/
│   └── fixtures/
├── scripts/
│   ├── smoke_pipeline.sh
│   └── reproduce_paper.sh
├── experiments/
│   ├── 00_baseline_reproduction/
│   ├── 01_selector_shift/
│   ├── 02_space_signal/
│   ├── 03_hypernetwork/
│   └── 04_ablations/
├── artifacts/              # gitignored; manifests are versioned
├── reports/
│   ├── tables/
│   └── figures/
└── docs/
    ├── architecture.md
    ├── protocols.md
    ├── decisions/
    └── experiment_registry.md
```

---

# 5. Módulos encadeados de implementação

## M0 — Bootstrap, contratos e reprodutibilidade

**Objetivo científico:** impedir que diferenças de pipeline sejam confundidas com diferenças de modelo.

**Implementar:**
- `pyproject.toml` com versões fixadas para `torch`, `transformers`, `datasets`, `faiss`, `numpy`, `scipy`, `scikit-learn`, `ir-measures`, `pytest`, `ruff`, `mypy`, `hydra-core`/`omegaconf`.
- seed manager único;
- `RunManifest` contendo `git_commit`, `config_hash`, `dataset_hash`, `model_revision`, `seed`, device, dtype e timestamps;
- schemas para `QueryRecord`, `DocumentRecord`, `QrelsRecord`, `EmbeddingArtifact`, `SpaceDescriptor`, `SelectorCheckpoint`, `EvaluationResult`.

**Saída:** CLI de diagnóstico que carrega uma config e gera manifest sem treinar nada.

**Testes:** serialização round-trip; hashes determinísticos; config inválida falha explicitamente.

**Gate:** nenhum módulo posterior pode gravar artefato sem manifest.

---

## M1 — Camada de dados e splits

**Objetivo científico:** garantir que selector shift não seja causado por split inconsistente ou leakage.

**Implementar:** loaders normalizados para SciFact, NFCorpus e MS MARCO; IDs canônicos; qrels; train/validation/test; amostragem reproduzível de documentos e queries; hard-negative pools separados do conjunto de avaliação.

**Contrato principal:**

```python
DatasetBundle(
    corpus: dict[str, str],
    queries: dict[str, str],
    qrels: list[Qrel],
    split: str,
    instruction: str,
)
```

**Testes:** IDs de qrels existem; nenhum test label entra no treino; split hashes estáveis; fixtures minúsculas executam em CPU.

**Gate:** produzir `dataset_manifest.json` para cada ambiente.

---

## M2 — Qwen3-Embedding-0.6B congelado

**Objetivo científico:** fixar completamente o sistema de coordenadas de 1024 dimensões.

**Implementar:** wrapper único para Qwen; dimensão=1024; pooling conforme implementação oficial; instrução configurável para queries; documentos sem transformação adicional além da recomendada pelo modelo; L2-normalização configurável e, no protocolo principal, habilitada.

**Artefatos:** `documents.f16.npy`, `queries.f16.npy`, maps ID→row, metadata com revision/hash.

**Invariantes:** shape `(N,1024)`; norma ≈1 quando normalizado; mesmo texto/config/revision produz mesmo vetor dentro da tolerância numérica.

**Testes:** shape; norma; batching; cache hit/miss; comparação CPU/GPU com tolerância; smoke de 10 documentos.

**Gate:** embeddings congelados e reutilizados nos experimentos seguintes.

---

## M3 — Oracle de importância e reprodução do Learning to Select

**Objetivo científico:** possuir um baseline validado antes de introduzir hypernetwork.

Para cada query `q`, formar embedding de positivos agregados `p` e negativos agregados `n`. Implementar o score oracle por dimensão:

`r_q,j = e_q,j (p_j - n_j)`

Transformar em distribuição alvo com temperatura:

`π_q = softmax(r_q / τ)`

Implementar predictor baseline:

`f_θ : R^1024 → R^1024`

Inicialmente uma única camada linear 1024→1024 seguida de log-softmax, reproduzindo a configuração descrita no paper.

Loss principal:

`L_selector = KL(π_q || π̂_q)`

Na inferência: `Top-k(π̂_q)`, máscara query-only, documentos inalterados.

**Baselines adicionais obrigatórios:** full 1024; MRL prefix-k; random-k; global learned importance; oracle top-k (upper bound); predictor per-domain.

**Testes:** oracle manual em tensor pequeno; Top-k exato; máscara não altera documento; ranking equivalente a score calculado apenas nas coordenadas selecionadas; overfit em dataset sintético de 20 queries.

**Gate científico:** reproduzir a tendência central do paper: o predictor aprende target e o pipeline de máscara funciona sem reindexar documentos. Não exigir números idênticos antes de alinhar datasets/hyperparameters.

---

## M4 — Experimento 0: provar ou refutar selector shift

**Este módulo é obrigatório antes da Hyper Head.**

Treinar seletores independentes:

`θ_SciFact*`, `θ_NFCorpus*`, `θ_MSMARCO*`

Construir matriz de transferência:

`Train domain A → Test domain B`

Medir:
- nDCG@10, MRR@10/100, Recall@100;
- KL/Jensen-Shannon entre distribuições de importância produzidas para uma mesma bateria de queries;
- overlap@k entre dimensões selecionadas;
- correlação Spearman dos rankings de dimensão;
- degradação cross-domain vs in-domain.

**Critério para H1:** deve haver evidência consistente de que seletores especializados diferem em comportamento e que pelo menos parte dessas diferenças importa para retrieval. Distância em pesos isoladamente não é evidência suficiente devido a simetrias de parametrização.

**Decisão:**
- se não houver selector shift funcional, interromper Hyper Head e trabalhar num selector universal;
- se houver, avançar para M5.

---

## M5 — Space Descriptor v1: estatísticas observáveis por dimensão

**Objetivo científico:** testar H2 sem ainda treinar hypernetwork.

Construir `R_T ∈ R^(1024×h)` com features versionadas.

### Grupo A — Document marginal
Por dimensão `j`: mean, std, mean_abs, RMS, min/max robustos, q05/q10/q25/q50/q75/q90/q95, skewness opcional, kurtosis opcional.

### Grupo B — Query marginal
Mesmas features usando `Q_calib`, sem labels.

### Grupo C — Interaction
Para amostras controladas de pares query-documento: média/std/quantis de `q_j*d_j`; decompor random pairs e top-retrieved pairs em canais separados.

### Grupo D — Contrast
Quando permitido no protocolo:
- supervisionado em meta-training: `Δ_q,j = q_j(p_j-n_j)`;
- não supervisionado em novo domínio: pseudo-positive top-K e random/lower-ranked negatives.

### Grupo E — Redundancy
Evitar entregar inicialmente covariance completa à rede. Usar por dimensão: mean absolute correlation, max absolute correlation, top-r correlation energy e participação em componentes principais. Guardar covariance completa apenas como artefato diagnóstico.

**Normalização:** statistics devem ser normalizadas entre ambientes usando parâmetros calculados **somente nos meta-train environments**.

**Teste científico de H2 — antes da Hyper Head:** avaliar se distância/similaridade entre `R_T` se relaciona com distância funcional entre seletores de M4. Usar regressões simples/CCA/probes e permutation tests. Se nenhum sinal existir, iterar o descriptor.

**Gate:** pelo menos um conjunto de features deve prever acima de baseline trivial uma propriedade do selector, como importância média por dimensão ou degradação ao transferir seletor entre ambientes.

---

## M6 — Space Encoder

**Objetivo:** transformar `R_T` em representação aprendida preservando identidade dimensional.

### V1 recomendada — per-dimension MLP + dimension ID embedding

Para cada dimensão:

`t_j = MLP_stats(r_j) + p_j`

onde `p_j` é embedding aprendido da identidade `j ∈ [1,1024]`.

Saída: `Z_T ∈ R^(1024×h_z)`.

### V2 — contextualização entre dimensões

Adicionar 1–3 blocos Transformer sobre os 1024 tokens de dimensão para modelar redundância/interação. Não implementar V2 antes da ablação mostrar limitação da V1.

**Testes:** permutation de documentos não muda `R_T`; permutation das dimensões muda a representação; gradientes fluem; output shape fixo; memória controlada.

---

## M7 — Target Dimension Selector parametrizável

A rede alvo deve permanecer pequena para que a Hyper Head tenha problema tratável.

### Baseline de destino

`f_θ(e_q) = W e_q + b`, com `W ∈ R^(1024×1024)`.

Gerar W inteiro exigiria >1M parâmetros por ambiente. Portanto, a primeira hypernetwork **não deve gerar W diretamente**.

### Parametrização recomendada: residual low-rank modulation

Manter um seletor-base global:

`W_T = W_0 + A_T B_T^T`

com rank `r << 1024`, por exemplo `r ∈ {4,8,16,32}`.

Bias:

`b_T = b_0 + Δb_T`

A Hyper Head gera apenas `A_T`, `B_T` e opcionalmente `Δb_T`.

Alternativa ainda mais simples para primeira prova:

`W_T = diag(γ_T) W_0 + diag(β_T)` ou modulação FiLM de hidden state.

**Princípio:** primeiro provar adaptação com baixa capacidade; aumentar capacidade só se houver underfitting demonstrado.

**Testes:** com delta=0, saída idêntica ao selector global; shapes; rank efetivo; checkpoint determinístico.

---

## M8 — Hyper Head

**Entrada:** `Z_T` do Space Encoder.

**Saída:** parâmetros/modulações do seletor alvo.

### V1 recomendada

1. contextualização por dimensão em `Z_T`;
2. pooling global opcional `g_T` para contexto do ambiente;
3. heads separados para gerar fatores low-rank;
4. controle explícito de escala para que `ΔW` comece próximo de zero.

Esquema:

`Z_T → HyperHead_A → A_T`

`Z_T → HyperHead_B → B_T`

`Pool(Z_T) → HyperHead_b → Δb_T`

Inicialização recomendada: último estágio que gera deltas próximo de zero para iniciar `θ_T ≈ θ_0`, evitando que o primeiro passo destrua o selector-base.

**Não copiar literalmente o hyperhead do Hypencoder.** O Hypencoder condiciona a rede em tokens da query e produz uma q-net por query. Aqui o contexto é um descriptor de ambiente e a rede produzida/modulada é persistida por ambiente.

---

## M9 — Treinamento episódico/meta-learning

A unidade de treinamento é um **environment episode**, não uma query isolada.

Para cada episódio:

1. escolher ambiente `T_i`;
2. carregar `R_Ti` calculado apenas com dados permitidos;
3. gerar `θ_Ti = Hφ(Gψ(R_Ti))`;
4. amostrar minibatch de queries do ambiente;
5. obter `π_q` oracle;
6. calcular `π̂_q = f_{θ_Ti}(e_q)`;
7. calcular loss;
8. retropropagar por selector gerado → Hyper Head → Space Encoder.

Loss inicial:

`L = KL(π_q || π̂_q)`

Adicionar somente depois:

`L_total = L_KL + λ_rank L_rank + λ_delta ||Δθ_T||²`

onde `L_rank` pode ser um surrogate sobre diferença positive-negative depois de máscara.

### Splits de meta-learning

É necessário separar ambientes, não somente queries:
- **meta-train environments**: usados para aprender Gψ/Hφ/θ0;
- **meta-validation environments**: seleção de arquitetura/hyperparameters;
- **meta-test environments**: nunca usados para gradientes nem normalização de descriptors.

Com poucos datasets, criar ambientes adicionais por domínio/subdomínio, instrução ou shard semanticamente coerente, mas documentar que shards do mesmo dataset não equivalem a domínio realmente não visto.

**Gate:** Hypernetwork deve bater ou igualar selector global no meta-validation antes de avaliar meta-test.

---

## M10 — Protocolos de inferência

Implementar três níveis explícitos; nunca misturá-los em uma única tabela sem rotular.

### P0 — Corpus-only zero-query

Entrada na adaptação: somente embeddings dos documentos. Usa Grupo A/E do descriptor. É o protocolo mais forte de zero-shot, porém com menos sinal.

### P1 — Unlabeled calibration queries

Entrada: corpus + um conjunto pequeno de queries sem qrels. Permite Grupo B/C e pseudo-feedback. É provavelmente o protocolo principal mais realista.

### P2 — Few-label calibration

Permite pequeno conjunto rotulado para descriptor/adaptação. Deve ser tratado como cenário separado, não como zero-shot.

Online, após `θ_T` ser gerado, cada query exige somente:

`encode query → selector forward → mask → retrieval`

Registrar separadamente latência de Qwen, selector, mask/scoring e busca.

---

## M11 — Selection Policy

A arquitetura deve separar **importance prediction** de **number-of-dimensions policy**.

### Policy A — Fixed Top-k
Obrigatória para comparação direta com *Learning to Select*.

### Policy B — Fixed ratio
`k = floor(ρD)` para curvas de custo-qualidade.

### Policy C — RDIME-inspired adaptive k
Implementar depois que a reprodução do critério estiver validada; usar importance score produzido pelo novo selector como entrada apenas se a formulação teórica continuar válida. Se a hipótese estatística de RDIME não for diretamente compatível com scores aprendidos, reportar como heurística inspirada em RDIME, não como RDIME canônico.

### Policy D — Budget-aware
Opcional: escolher k sujeito a orçamento de latência, separando a contribuição deste projeto da estratégia load-sensitive.

---

## M12 — Retrieval e avaliação

Começar com **exact dot product** para eliminar efeitos de ANN. Só então adicionar FAISS.

Métricas primárias: nDCG@10, MRR@10/100, Recall@100.

Métricas da seleção: número de dimensões retidas; overlap@k; entropy de `π̂_q`; KL para oracle; distribuição de dimensões usadas; frequência por dimensão.

Métricas de eficiência: tempo de selector; tempo total da query; memória; FLOPs/estimativa; custo de adaptação offline; tamanho de `θ_T`.

Métricas de generalização: in-domain; selector transfer A→B; hypernetwork zero-shot/few-query; gap para selector oracle/per-domain.

**Significância:** testes pareados por query, correção para múltiplas comparações quando houver famílias de hipóteses. Reportar intervalos de confiança, não apenas p-values.

---

# 6. Plano experimental científico

## E0 — Reproduction sanity
Objetivo: validar embeddings, oracle e selector baseline.

## E1 — Selector shift
Pergunta: um selector treinado em A degrada em B? As máscaras/importance distributions mudam sistematicamente?

## E2 — Space signal
Pergunta: descriptors do ambiente predizem propriedades do selector especializado?

## E3 — Hypernetwork main result
Comparar:
- full 1024;
- MRL prefix;
- Learning-to-Select per-domain;
- selector global multi-domain;
- selector treinado em source e transferido;
- HyperDIME P0;
- HyperDIME P1;
- oracle/per-domain upper bound.

## E4 — Ablations
Remover um grupo por vez: document stats; query stats; interaction; contrast; redundancy; dimension-ID embeddings; Transformer inter-dimension; low-rank rank; número de calibration queries.

## E5 — Sensitivity
Seeds, sample size do corpus, sample size de Q_calib, temperature τ, hard-negative K/M, rank r, k de seleção.

## E6 — Efficiency
Separar custo offline de adaptação e custo online por query.

---

# 7. Matriz de ablação mínima

| Experimento | Space input | Hyper Head | Selector | Objetivo |
|---|---|---|---|---|
| B0 | nenhum | não | global fixo | baseline universal |
| B1 | nenhum | não | per-domain | upper bound de especialização |
| H0 | document stats | sim | low-rank | testar corpus-only |
| H1 | + query stats | sim | low-rank | efeito da distribuição de queries |
| H2 | + interactions | sim | low-rank | contribuição do mecanismo de scoring |
| H3 | + contrast | sim | low-rank | sinal próximo ao oracle |
| H4 | H3 + redundancy | sim | low-rank | complementaridade entre dimensões |
| H5 | H4 | sim | richer target | necessidade de maior capacidade |

---

# 8. Estratégia de testes

## 8.1 Unit tests
Cada função matemática crítica deve ter teste em tensores pequenos com resultado calculável manualmente: L2 normalization, oracle `q*(p-n)`, KL target, masks, Top-k, descriptor stats, correlation summaries, low-rank reconstruction.

## 8.2 Property tests
- permutation dos documentos não muda descriptor;
- troca de dimensão deve trocar o canal correspondente, não desaparecer;
- `k=1024` é equivalente ao full-dimensional scoring;
- zero modulation é equivalente ao selector-base;
- nenhuma função de meta-test pode acessar qrels no protocolo P0/P1.

## 8.3 Integration tests
Pipeline sintético de ponta a ponta com 50 docs/10 queries em CPU: embed fixture mockada → oracle → train selector → build descriptor → hyperhead → inference → metric.

## 8.4 Regression tests
Salvar resultados de fixtures determinísticas e verificar tolerâncias. Não usar métricas grandes de datasets externos como unit test.

## 8.5 Scientific tests
Scripts que geram automaticamente tabelas E1–E6 a partir de manifests, sem edição manual de números.

---

# 9. Controle de repositório para trabalho com agentes em IDE

## 9.1 Branching

- `main`: somente estados reproduzíveis e protegidos.
- cada tarefa de agente: `agent/<ticket>-<modulo>-<descricao>`.
- nunca permitir dois agentes alterando o mesmo módulo sem contrato explícito.

Exemplos:

`agent/M3-oracle-targets`

`agent/M5-document-stats`

`agent/M8-lowrank-hyperhead`

## 9.2 Commits
Usar Conventional Commits:

`feat(space): add per-dimension marginal descriptor`

`test(selector): verify masked dot-product equivalence`

`fix(data): prevent qrel leakage into meta-test descriptor`

Cada commit deve passar testes relacionados ao módulo.

## 9.3 Pull requests
Toda PR deve declarar:
1. contrato alterado;
2. arquivos alterados;
3. testes adicionados;
4. comandos executados;
5. artefatos/regressões esperados;
6. se muda protocolo científico;
7. se invalida resultados anteriores.

## 9.4 AGENTS.md
Criar um arquivo com regras obrigatórias para agentes:
- ler `docs/architecture.md` e contrato do módulo antes de editar;
- não alterar APIs públicas fora do ticket;
- nunca usar test qrels em adaptação;
- não baixar/commitar datasets ou checkpoints no Git;
- adicionar teste para cada comportamento novo;
- executar `ruff`, `mypy` e subset de `pytest` antes do commit;
- registrar decisões arquiteturais não triviais em `docs/decisions/ADR-XXXX.md`;
- não “corrigir” números de experimentos manualmente;
- toda tabela deve ser derivada de arquivos de resultados.

## 9.5 Artefatos grandes
Git deve versionar somente manifests/configs. Embeddings/checkpoints ficam em `artifacts/` ou storage externo. Cada artefato recebe SHA256 e provenance. Se necessário, adotar DVC depois da primeira reprodução; não tornar DVC dependência crítica no bootstrap.

---

# 10. Sequência de execução para agentes

## Fase A — Fundação
**A1:** M0 contracts/reproducibility.
**A2:** M1 datasets.
**A3:** M2 Qwen embeddings/cache.

Bloqueio: A2/A3 não podem definir schemas paralelos; usam M0.

## Fase B — Baseline científico
**B1:** M3 oracle.
**B2:** M3 predictor.
**B3:** retrieval exact + metrics.
**B4:** reprodução e relatório.

Bloqueio: nenhuma hypernetwork antes de B4.

## Fase C — Evidência para a hipótese
**C1:** M4 selectors per-domain.
**C2:** cross-domain transfer matrix.
**C3:** selector behavior analysis.
**C4:** decisão Go/No-Go para H1.

## Fase D — Representação do espaço
**D1:** marginal stats.
**D2:** query/interaction stats.
**D3:** contrast/redundancy.
**D4:** H2 probes e permutation tests.
**D5:** congelar `SpaceDescriptorV1`.

## Fase E — Hypernetwork
**E1:** target selector low-rank.
**E2:** space encoder V1.
**E3:** hyperhead V1.
**E4:** episodic trainer.
**E5:** meta-validation.

## Fase F — Avaliação final
**F1:** P0.
**F2:** P1.
**F3:** ablations.
**F4:** efficiency.
**F5:** significance + tables.

---

# 11. Tickets agent-ready

### Ticket M3.1 — Oracle importance
**Entradas:** query embeddings, relevant doc IDs, hard-negative IDs.
**Saída:** `TargetImportance[query_id,1024]`.
**DoD:** unit tests contra cálculo manual; distribuição soma 1; sem NaN; CLI gera artifact + manifest.

### Ticket M4.2 — Selector transfer matrix
**Entradas:** checkpoints per-domain + test bundles.
**Saída:** CSV/Parquet A×B com métricas e mask similarity.
**DoD:** nenhum treino executado durante avaliação; tabela reproduzível por um comando.

### Ticket M5.1 — Document descriptor
**Entradas:** matriz N×1024.
**Saída:** 1024×h e schema de feature names.
**DoD:** invariância à permutação de linhas; tolerância a sampling; sem usar qrels.

### Ticket M7.1 — Low-rank selector
**Entradas:** base W0,b0 + A,B,Δb.
**Saída:** logits 1024.
**DoD:** delta zero == base; autograd validado; rank configurável.

### Ticket M8.1 — Hyperhead
**Entradas:** `SpaceDescriptorV1`/`Z_T`.
**Saída:** modulation package.
**DoD:** parâmetros gerados respeitam shapes; inicialização near-zero; gradient test.

### Ticket M9.1 — Episodic trainer
**Entradas:** lista de environments meta-train.
**Saída:** checkpoint `Gψ,Hφ,θ0`, logs e manifest.
**DoD:** batch alterna environments; meta-test impossível de carregar no trainer por validação de config.

---

# 12. Template de prompt para agentes na IDE

```text
Você está implementando o ticket <ID> do projeto HyperDIME-Qwen.

Antes de editar:
1. leia AGENTS.md;
2. leia docs/architecture.md;
3. leia o contrato em src/hyperdime/contracts;
4. liste os arquivos que pretende alterar;
5. não altere APIs fora do ticket sem justificar.

Objetivo do ticket:
<objetivo>

Entradas/saídas obrigatórias:
<contrato>

Critérios de aceitação:
<DoD>

Restrições científicas:
- Qwen3-Embedding-0.6B congelado;
- D=1024;
- não usar qrels de meta-test em adaptação;
- separar custo offline e online;
- não modificar resultados manualmente.

Ao terminar:
1. execute testes unitários do módulo;
2. execute ruff/mypy;
3. execute smoke test relevante;
4. mostre git diff --stat;
5. resuma decisões e riscos;
6. proponha a mensagem Conventional Commit.
```

---

# 13. Critérios de Go/No-Go científicos

### Gate G1 — Baseline válido
Learning-to-Select baseline converge, máscara é matematicamente correta e retrieval exact reproduz ranking esperado.

### Gate G2 — H1 sustentada
Há selector shift funcional entre ambientes, observado em métricas de retrieval e comportamento das máscaras.

### Gate G3 — H2 sustentada
O descriptor do ambiente prediz propriedades do selector/transfer acima de baseline e de forma robusta a seeds/sampling.

### Gate G4 — H3 sustentada
Hypernetwork reduz o gap entre selector global e selector per-domain em ambientes não usados para meta-training.

### Gate G5 — Custo aceitável
A adaptação offline é amortizável e o overhead online do selector não elimina a vantagem da redução dimensional.

Se qualquer gate falhar, registrar o resultado como evidência e não avançar aumentando complexidade indiscriminadamente.

---

# 14. Riscos principais e mitigação

**R1 — Selector shift pequeno.** Mitigação: provar primeiro em M4; se pequeno, abandonar hypernetwork.

**R2 — Descriptor aprende identidade do dataset, não geometria útil.** Mitigação: meta-test em ambientes realmente separados; permutation/probe controls; reduzir features semânticas explícitas.

**R3 — Pseudo-feedback cria circularidade.** Mitigação: P0 corpus-only como controle; P1 claramente rotulado; nunca usar qrels no descriptor de meta-test.

**R4 — Hyper Head gera parâmetros instáveis.** Mitigação: residual low-rank, deltas near-zero, clipping, regularização de Δθ.

**R5 — MRL já resolve o problema.** Mitigação: sempre comparar contra prefix truncation do Qwen; a contribuição só é válida se subset query-aware acrescentar valor além de MRL.

**R6 — Ganho vem de supervisionar o selector, não da hypernetwork.** Mitigação: comparar com selector global multi-domain com mesma quantidade de labels e capacidade semelhante.

**R7 — ANN mascara o efeito.** Mitigação: resultados científicos primários em exact scoring; ANN apenas na análise de deployment.

---

# 15. Primeira versão mínima que deve ser implementada

Não comece pelo Transformer de 1024 dimension-tokens. A versão mínima deve conter:

1. Qwen3-Embedding-0.6B → embeddings 1024 normalizados;
2. Learning-to-Select linear reproduzido;
3. selectors independentes em ≥3 ambientes;
4. prova de selector shift;
5. descriptor `1024 × h` com document/query marginals + interaction simples;
6. `MLP_stats(r_j)+dimension_id_embedding` como Space Encoder;
7. selector-base linear + modulação low-rank;
8. Hyper Head simples que gera low-rank deltas por ambiente;
9. treino episódico;
10. avaliação meta-test P0/P1;
11. apenas então contrast, redundancy, Transformer e adaptive-k.

Essa ordem reduz o risco de construir uma arquitetura complexa antes de provar que existe sinal para ela explorar.

---

# 16. Entregáveis por milestone

**Milestone 1 — Reproduction:** código M0–M3, tests, relatório baseline.

**Milestone 2 — Scientific premise:** M4 + relatório selector shift + decisão Go/No-Go.

**Milestone 3 — Space representation:** M5 + probes de H2 + `SpaceDescriptorV1` congelado.

**Milestone 4 — HyperDIME prototype:** M6–M9 + checkpoint meta-train + avaliação meta-validation.

**Milestone 5 — Paper-grade evaluation:** M10–M12 + ablations + significance + efficiency + tabelas geradas automaticamente.

---

# 17. Referências centrais

- Wu, Z.; Zhang, R.; Nie, Z. **Learning to Select: Query-Aware Adaptive Dimension Selection for Dense Retrieval.** ACL 2026.
- Killingback, J.; Zeng, H.; Zamani, H. **Hypencoder: Hypernetworks for Information Retrieval.** SIGIR 2025.
- Eichholtz, A. et al. **Hypencoder Revisited: Reproducibility and Analysis of Non-Linear Scoring for First-Stage Retrieval.** SIGIR 2026.
- D’Erasmo, G. et al. **Statistical Foundations of DIME: Risk Estimation for Practical Index Selection.** EACL 2026.
- Zhang, Y. et al. **Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models.** 2025.
- Kusupati, A. et al. **Matryoshka Representation Learning.** NeurIPS 2022.

---

# 18. Resultado arquitetural esperado

Ao final, o sistema deve materializar explicitamente três níveis de adaptação:

`Qwen3-Embedding-0.6B (fixed coordinate system)`

`↓`

`Space Descriptor + Space Encoder (environment representation)`

`↓`

`Hyper Head (environment → selector parameters)`

`↓`

`Dimension Selector (query → dimension importance)`

`↓`

`Selection Policy (importance → subset/k)`

`↓`

`Query-only masked dense retrieval`

O sucesso científico não será “a hypernetwork treinou”, mas demonstrar que **mudanças mensuráveis no ambiente de retrieval do mesmo Qwen explicam mudanças no seletor ótimo e podem ser convertidas, sem retreinar um selector completo no novo domínio, em parâmetros úteis para seleção query-aware de dimensões**.
