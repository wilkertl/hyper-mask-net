"""Train Hyper-Mask-Net from precomputed frozen Qwen representations."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from models import HyperMaskConfig, Hyperhead, LinearMaskPredictor


def oracle_importance(
    query_embeddings: Tensor,
    positive_embeddings: Tensor,
    negative_embeddings: Tensor,
    temperature: float,
) -> Tensor:
    """Build π_q = softmax(e_q * (p - n) / τ) without target gradients."""
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if not (
        query_embeddings.shape == positive_embeddings.shape == negative_embeddings.shape
    ):
        raise ValueError("query, positive, and negative embeddings must share a shape.")
    scores = query_embeddings * (positive_embeddings - negative_embeddings)
    return F.softmax(scores.float() / temperature, dim=-1)


class FrozenEmbeddingDataset(Dataset[dict[str, Tensor]]):
    """Validated on-disk frozen contextual representations and relevance triples."""

    required = {
        "token_embeddings",
        "attention_mask",
        "query_embeddings",
        "positive_embeddings",
        "negative_embeddings",
    }

    def __init__(self, path: str | Path) -> None:
        payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
        missing = self.required - payload.keys()
        if missing:
            raise ValueError(f"{path} is missing required tensors: {sorted(missing)}")
        self.tensors = {name: payload[name] for name in self.required}
        length = self.tensors["query_embeddings"].shape[0]
        if any(tensor.shape[0] != length for tensor in self.tensors.values()):
            raise ValueError("All tensors must have the same first dimension.")
        if self.tensors["token_embeddings"].ndim != 3:
            raise ValueError("token_embeddings must have shape [N, sequence, hidden].")
        if self.tensors["attention_mask"].shape != self.tensors["token_embeddings"].shape[:2]:
            raise ValueError("attention_mask shape must match token_embeddings[:2].")
        for name in ("query_embeddings", "positive_embeddings", "negative_embeddings"):
            if self.tensors[name].ndim != 2:
                raise ValueError(f"{name} must have shape [N, hidden].")

    def __len__(self) -> int:
        return self.tensors["query_embeddings"].shape[0]

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {name: tensor[index] for name, tensor in self.tensors.items()}


def predicted_importance(
    model: nn.Module,
    batch: dict[str, Tensor],
    prediction_temperature: float,
) -> Tensor:
    """Run either the Hyperhead extension or the paper's linear baseline."""
    if isinstance(model, Hyperhead):
        parameters = model(batch["token_embeddings"], batch["attention_mask"])
        logits = model.mask_net(batch["query_embeddings"], parameters)
        return F.softmax(logits.float() / prediction_temperature, dim=-1)
    return model(batch["query_embeddings"])


def kl_loss(target: Tensor, prediction: Tensor) -> Tensor:
    """KL(π_q || π̂_q), reduced as a batch mean."""
    return F.kl_div(prediction.clamp_min(1e-12).log(), target, reduction="batchmean")


def load_config(path: str | Path) -> HyperMaskConfig:
    with Path(path).open() as file:
        return HyperMaskConfig(**json.load(file))


def save_checkpoint(
    output_dir: Path, model: nn.Module, config: HyperMaskConfig, architecture: str
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), output_dir / "model.safetensors")
    metadata = {"architecture": architecture, "model_config": asdict(config)}
    (output_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")


def evaluate(
    model: nn.Module,
    loader: DataLoader[dict[str, Tensor]],
    device: torch.device,
    target_temperature: float,
    prediction_temperature: float,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = {name: value.to(device) for name, value in batch.items()}
            target = oracle_importance(
                batch["query_embeddings"],
                batch["positive_embeddings"],
                batch["negative_embeddings"],
                target_temperature,
            )
            prediction = predicted_importance(model, batch, prediction_temperature)
            losses.append(kl_loss(target, prediction).item())
    return sum(losses) / max(len(losses), 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--model-config", required=True, help="JSON from prepare_embeddings.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--architecture", choices=("hyper", "linear"), default="hyper")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--target-temperature", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive.")
    config = load_config(args.model_config)
    train_data, validation_data = FrozenEmbeddingDataset(args.train_data), FrozenEmbeddingDataset(
        args.validation_data
    )
    if train_data.tensors["query_embeddings"].shape[-1] != config.hidden_dim:
        raise ValueError("The embedding dimension does not match model_config.hidden_dim.")
    device = torch.device(args.device)
    model: nn.Module
    model = (
        Hyperhead(config)
        if args.architecture == "hyper"
        else LinearMaskPredictor(config.hidden_dim, config.prediction_temperature)
    )
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, pin_memory=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs * len(train_loader), 1)
    )
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    best_loss = math.inf

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            batch = {name: value.to(device, non_blocking=True) for name, value in batch.items()}
            target = oracle_importance(
                batch["query_embeddings"],
                batch["positive_embeddings"],
                batch["negative_embeddings"],
                args.target_temperature,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                prediction = predicted_importance(model, batch, config.prediction_temperature)
                loss = kl_loss(target, prediction)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        validation_loss = evaluate(
            model,
            validation_loader,
            device,
            args.target_temperature,
            config.prediction_temperature,
        )
        print(f"epoch={epoch} validation_kl={validation_loss:.6f}")
        if validation_loss < best_loss:
            best_loss = validation_loss
            save_checkpoint(Path(args.output_dir), model, config, args.architecture)
    print(f"best_validation_kl={best_loss:.6f}; checkpoint={args.output_dir}")


if __name__ == "__main__":
    main()
