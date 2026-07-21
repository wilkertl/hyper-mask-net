# Hyper-Mask-Net

Hyper-Mask-Net learns a query-dependent subset of dense-embedding dimensions
without changing document vectors or rebuilding a FAISS index. A frozen
`Qwen/Qwen3-Embedding-0.6B` encoder produces contextual query tokens; a
cross-attention Hyperhead generates per-query, low-rank weights for a transient
Mask-Net that predicts dimension importance.

This is an extension of *Learning to Select: Query-Aware Adaptive Dimension
Selection for Dense Retrieval* (arXiv:2602.03306). The repository also includes
the paper-faithful single linear predictor as a baseline.

## Design

1. Freeze Qwen and encode queries with last-token pooling plus L2 normalization.
2. Build the oracle target `softmax(e_q * (p - n) / target_temperature)` from a
   positive and a mined hard negative.
3. Train the Hyperhead with `KL(oracle || predicted_importance)`.
4. At search time, retain only a query's top-k dimensions, zero the rest, and
   perform unchanged inner-product FAISS retrieval over frozen document vectors.

The Mask-Net has topology `D → D/4 → D` with GELU. Its two per-query matrices
are generated as rank-32 products, avoiding an impractically large dense
hypernetwork output.

> [!IMPORTANT]
> Query, positive, negative, and indexed document vectors must use the same
> Qwen checkpoint, last-token pooling, query instruction convention, and L2
> normalization. The masked query is deliberately not re-normalized.

## Install

```bash
cd /home/wilker/hyper-mask-net
python -m pip install -e ".[dev]"
```

`faiss-cpu` powers the portable exact-search demo. PyTorch automatically uses
CUDA and bfloat16 during training when available. A compatible local FAISS GPU
build can be substituted for CPU FAISS.

## Data preparation

Provide JSONL training and validation files whose rows contain one query, one
relevant document, and one already-mined hard negative:

```json
{"query":"what is contrastive learning?","positive":"...","negative":"..."}
```

Create frozen tensors separately for each split:

```bash
python prepare_embeddings.py \
  --input-jsonl train.jsonl \
  --output-data train.pt \
  --output-config qwen_config.json \
  --query-instruction "Given a web search query, retrieve relevant passages."

python prepare_embeddings.py \
  --input-jsonl validation.jsonl \
  --output-data validation.pt \
  --output-config qwen_config.json \
  --query-instruction "Given a web search query, retrieve relevant passages."
```

The `.pt` output stores `token_embeddings`, `attention_mask`,
`query_embeddings`, `positive_embeddings`, and `negative_embeddings`.
Contextual query tokens are precomputed so training optimizes only the
Hyperhead. `prepare_embeddings.py` uses `trust_remote_code=True` for Qwen.

## BEIR SciFact experiment

The repository includes a reproducible in-domain SciFact setup. It downloads
BEIR SciFact, uses its provided `train` qrels only, reserves 15% of those
training queries for validation, and keeps the BEIR `test` qrels untouched for
the final evaluation. For every training query, it selects one judged relevant
abstract and mines a Qwen hard negative from the fixed SciFact corpus.

Run the complete Hyperhead and linear-baseline experiment on a CUDA machine:

```bash
bash scripts/run_scifact_4090.sh
```

The script writes all derived data to `data/scifact/` and results to
`runs/scifact/`. It evaluates the following retrieval conditions over the same
fixed document index:

- `unmasked`: frozen Qwen query embeddings;
- `random_k=*`: a random dimension subset at the same sparsity;
- `learned_k=*`: the selected dimensions from the trained predictor.

`runs/scifact/{hyper,linear}/test_metrics.json` reports `nDCG`, recall, and
MRR at the selected cutoff. Change `--k-values` in
`scripts/run_scifact_4090.sh` to sweep a different set of retained-dimension
ratios.

### Remote RTX 4090 setup

Clone the repository directly on the SSH host so all model and dataset files
remain on its local disk:

```bash
ssh user@remote-host
git clone <repository-url> hyper-mask-net
cd hyper-mask-net
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(), torch.cuda.is_available())"
```

`torch.cuda.is_available()` must print `True` before starting the experiment.
For a long SSH session, run it under `tmux`:

```bash
tmux new -s scifact
source .venv/bin/activate
bash scripts/run_scifact_4090.sh 2>&1 | tee runs/scifact/run.log
```

Detach with `Ctrl-b d`, then later reconnect with `tmux attach -t scifact`.
Set `PROJECT_DIR`, `DATA_DIR`, `RUN_DIR`, or `PYTHON` before invoking the
script when the default paths or interpreter are unsuitable.

## Train

```bash
python train.py \
  --train-data train.pt \
  --validation-data validation.pt \
  --model-config qwen_config.json \
  --output-dir checkpoints/hyper \
  --architecture hyper
```

The best validation-KL checkpoint is saved as `model.safetensors` plus
`config.json`. To train the paper baseline instead, pass
`--architecture linear`.

## Retrieve

With a float32 `[num_documents, hidden_dim]` NumPy matrix made with the same
embedding convention:

```bash
python inference.py \
  --checkpoint-dir checkpoints/hyper \
  --query "what is contrastive learning?" \
  --query-instruction "Given a web search query, retrieve relevant passages." \
  --document-embeddings documents.npy \
  --k 0.30 \
  --top-n 10
```

`--k` accepts either a ratio in `(0, 1]` or an integer dimension count. Omit
`--document-embeddings` to run a deterministic dummy FAISS demonstration.

## Verification

```bash
pytest
```

The tests validate dynamic Mask-Net tensor behavior, top-k masking, and that
the KL loss backpropagates into the Hyperhead but not the frozen encoder.
