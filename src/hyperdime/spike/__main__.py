"""``python -m hyperdime.spike <step>``: run the F0 selector-shift spike."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from hyperdime.spike.environments import ENVIRONMENTS
from hyperdime.spike.pipeline import Encoder

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def main(argv: list[str] | None = None) -> None:
    from hyperdime.embeddings.remote import BASE_URL_ENV

    parser = argparse.ArgumentParser(prog="python -m hyperdime.spike", description=__doc__)
    parser.add_argument(
        "step", choices=["prepare", "train", "train-global", "analyze", "report", "all"]
    )
    parser.add_argument(
        "--env", action="append", choices=sorted(ENVIRONMENTS), help="repeatable; default: all five"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--root", type=Path, default=Path("artifacts/spike"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/F0"))
    embedding = parser.add_argument_group("embedding (prepare step only)")
    embedding.add_argument(
        "--backend",
        choices=["auto", "vllm", "local"],
        default="auto",
        help=f"auto: vllm when --base-url or {BASE_URL_ENV} is set, else local transformers",
    )
    embedding.add_argument(
        "--base-url", default=os.environ.get(BASE_URL_ENV), help=f"default: ${BASE_URL_ENV}"
    )
    embedding.add_argument("--timeout", type=float, default=120.0, help="seconds per vLLM request")
    embedding.add_argument("--batch-size", type=int, help="default: 64 (16 for local CPU)")
    embedding.add_argument("--max-length", type=int, default=512, help="token limit per text")
    embedding.add_argument(
        "--corpus-cap",
        type=int,
        default=0,
        help="documents per corpus (judged + seeded fill); 0 = full corpus",
    )
    embedding.add_argument("--device", default="auto", help="local backend only")
    embedding.add_argument(
        "--dtype", choices=["auto", *DTYPES], default="auto", help="local backend only"
    )
    args = parser.parse_args(argv)

    from hyperdime.spike import pipeline, report

    envs: list[str] = args.env or list(ENVIRONMENTS)
    if args.step in ("prepare", "all"):
        encoder = _encoder(args, parser)
        for env in envs:
            pipeline.prepare(env, args.root, encoder, args.corpus_cap)
    if args.step in ("train", "all"):
        for env in envs:
            pipeline.train(env, args.root, args.seeds)
    if args.step in ("train-global", "all"):
        pipeline.train_global(envs, args.root, args.seeds)
    if args.step in ("analyze", "all"):
        verdict = pipeline.analyze(envs, args.root, args.seeds)["criteria"]["verdict"]
        print(f"H1 {verdict}")
    if args.step in ("report", "all"):
        print(report.write_report(args.root, args.report_dir))


def _encoder(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Encoder:
    from hyperdime.embeddings.remote import BASE_URL_ENV, VllmEncoder

    backend = args.backend
    if backend == "auto":
        backend = "vllm" if args.base_url else "local"
    if backend == "vllm":
        if not args.base_url:
            parser.error(f"--base-url is required for the vllm backend (or set {BASE_URL_ENV})")
        return VllmEncoder(
            args.base_url,
            max_length=args.max_length,
            batch_size=args.batch_size or 64,
            timeout=args.timeout,
        )
    from hyperdime.embeddings.qwen import QwenEncoder

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    on_gpu = device.startswith("cuda")
    dtype_name = ("bfloat16" if on_gpu else "float32") if args.dtype == "auto" else args.dtype
    return QwenEncoder(
        device=device,
        max_length=args.max_length,
        batch_size=args.batch_size or (64 if on_gpu else 16),
        dtype=DTYPES[dtype_name],
    )


if __name__ == "__main__":
    main()
