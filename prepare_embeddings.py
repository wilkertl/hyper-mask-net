"""Create frozen Qwen representations for Hyper-Mask-Net training.

Input JSONL rows require string fields: ``query``, ``positive``, and
``negative``.  The output `.pt` file contains only frozen tensors consumed by
`train.py`; no transformer is run during predictor training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from models import HyperMaskConfig


def last_token_pool(token_embeddings: Tensor, attention_mask: Tensor) -> Tensor:
    """Pool the last non-padding token, the Qwen embedding convention."""
    positions = attention_mask.long().sum(dim=1).sub(1).clamp_min(0)
    return token_embeddings[
        torch.arange(token_embeddings.shape[0], device=token_embeddings.device), positions
    ]


def format_query(query: str, instruction: str | None) -> str:
    return f"Instruct: {instruction}\nQuery:{query}" if instruction else query


def encode_texts(
    model: torch.nn.Module,
    tokenizer: object,
    texts: list[str],
    max_length: int,
    device: torch.device,
    return_tokens: bool = False,
) -> tuple[Tensor, Tensor] | Tensor:
    encoded = tokenizer(  # type: ignore[operator]
        texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt"
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    outputs = model(**encoded)
    tokens = outputs.last_hidden_state
    embedding = F.normalize(last_token_pool(tokens, encoded["attention_mask"]), dim=-1)
    if return_tokens:
        return tokens.cpu(), encoded["attention_mask"].cpu()
    return embedding.cpu()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-data", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--query-instruction")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive.")
    rows = [json.loads(line) for line in Path(args.input_jsonl).read_text().splitlines() if line]
    if not rows:
        raise ValueError("input-jsonl is empty.")
    required = {"query", "positive", "negative"}
    if any(required - row.keys() for row in rows):
        raise ValueError("Every row must define string query, positive, and negative fields.")
    if not all(all(isinstance(row[key], str) for key in required) for row in rows):
        raise ValueError("query, positive, and negative fields must be strings.")

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True).to(device).eval()
    token_parts: list[Tensor] = []
    mask_parts: list[Tensor] = []
    query_parts: list[Tensor] = []
    positive_parts: list[Tensor] = []
    negative_parts: list[Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            queries = [format_query(row["query"], args.query_instruction) for row in batch]
            tokens, masks = encode_texts(
                model, tokenizer, queries, args.max_length, device, return_tokens=True
            )
            token_parts.append(tokens.float())
            mask_parts.append(masks.bool())
            query_parts.append(F.normalize(last_token_pool(tokens, masks), dim=-1).float())
            positive_parts.append(
                encode_texts(
                    model, tokenizer, [row["positive"] for row in batch], args.max_length, device
                ).float()
            )
            negative_parts.append(
                encode_texts(
                    model, tokenizer, [row["negative"] for row in batch], args.max_length, device
                ).float()
            )
    torch.save(
        {
            "token_embeddings": torch.cat(token_parts),
            "attention_mask": torch.cat(mask_parts),
            "query_embeddings": torch.cat(query_parts),
            "positive_embeddings": torch.cat(positive_parts),
            "negative_embeddings": torch.cat(negative_parts),
        },
        args.output_data,
    )
    config = HyperMaskConfig(
        model_name=args.model_name,
        hidden_dim=model.config.hidden_size,
        pooling="last_token",
        trust_remote_code=True,
    )
    Path(args.output_config).write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    print(f"Wrote {len(rows)} examples to {args.output_data}")


if __name__ == "__main__":
    main()
