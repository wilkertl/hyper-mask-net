#!/usr/bin/env bash
# Run from the repository root, or set PROJECT_DIR to its absolute path.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data/scifact}"
RUN_DIR="${RUN_DIR:-$PROJECT_DIR/runs/scifact}"
QUERY_INSTRUCTION="Given a scientific claim, retrieve relevant scientific abstracts."

cd "$PROJECT_DIR"
mkdir -p "$DATA_DIR" "$RUN_DIR"

# Downloads BEIR SciFact, mines Qwen hard negatives, and reserves 15% of BEIR
# train queries for validation.  The BEIR test qrels are never used here.
"$PYTHON" prepare_scifact.py \
  --output-dir "$DATA_DIR" \
  --query-instruction "$QUERY_INSTRUCTION" \
  --batch-size 32 \
  --device cuda

for split in train validation; do
  "$PYTHON" prepare_embeddings.py \
    --input-jsonl "$DATA_DIR/$split.jsonl" \
    --output-data "$DATA_DIR/$split.pt" \
    --output-config "$DATA_DIR/qwen_config.json" \
    --query-instruction "$QUERY_INSTRUCTION" \
    --batch-size 32 \
    --device cuda
done

for architecture in hyper linear; do
  checkpoint_dir="$RUN_DIR/$architecture"
  "$PYTHON" train.py \
    --train-data "$DATA_DIR/train.pt" \
    --validation-data "$DATA_DIR/validation.pt" \
    --model-config "$DATA_DIR/qwen_config.json" \
    --output-dir "$checkpoint_dir" \
    --architecture "$architecture" \
    --epochs 30 \
    --batch-size 128 \
    --learning-rate 1e-4 \
    --device cuda

  "$PYTHON" evaluate_scifact.py \
    --checkpoint-dir "$checkpoint_dir" \
    --data-dir "$DATA_DIR" \
    --query-instruction "$QUERY_INSTRUCTION" \
    --batch-size 128 \
    --k-values "0.1,0.2,0.3,0.5" \
    --output "$checkpoint_dir/test_metrics.json" \
    --device cuda
done
