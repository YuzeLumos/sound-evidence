"""Command line tools for the bundled core and explicitly supplied datasets."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from . import _core
from .runtime import evaluate_formal, load_model, predict_manifest, read_manifest, validate_formal
from .outputs import new_directory, require_new_directory, require_new_prediction, write_predictions


def make_demo(output: str | Path) -> Path:
    """Two deterministic signals; no dataset recordings or codec-label claims."""
    output = Path(output)
    with new_directory(output) as staging:
        _generate_demo_files(staging)
    return output / "samples.jsonl"


def _generate_demo_files(output: Path) -> None:
    rate = 24000
    time = np.arange(round(rate * 4.25), dtype=np.float64) / rate
    wave = 0.2 * np.sin(2 * np.pi * 220 * time) + 0.07 * np.sin(2 * np.pi * 440 * time)
    rows = []
    for index, values in enumerate((wave, 0.8 * wave + 0.02 * np.sin(2 * np.pi * 900 * time))):
        name = f"signal_{index}.wav"
        sf.write(output / name, values, rate, subtype="PCM_16")
        rows.append({"sample_id": f"synthetic-{index}", "group_id": "synthetic-pair", "path": name})
    manifest = output / "samples.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (output / "ABOUT.json").write_text(json.dumps({
        "synthetic": True, "sample_rate": rate,
        "scope": "runtime smoke example; no authenticity, codec or task-performance ground truth",
    }, indent=2) + "\n", encoding="utf-8")


def _model_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "auto", "cuda"), default="cpu")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--threads", type=int, default=2, help="CPU intra-op threads")
    commands = result.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("make-demo", help="generate deterministic synthetic audio")
    generate.add_argument("--output", type=Path, required=True)
    demo = commands.add_parser("demo", help="generate signals and run the actual checkpoint")
    _model_options(demo)
    demo.add_argument("--output", type=Path, required=True)
    predict = commands.add_parser("predict", help="export embeddings and native M/C head probabilities")
    _model_options(predict)
    predict.add_argument("--manifest", type=Path, required=True)
    predict.add_argument("--data-root", type=Path)
    predict.add_argument("--output", type=Path, required=True, help="output NPZ file")
    predict.add_argument("--crops", type=int, default=3)
    evaluate = commands.add_parser("evaluate", help="full-grid frozen export plus training-only M/C/R linear probes")
    _model_options(evaluate)
    evaluate.add_argument("--samples", type=Path, required=True)
    evaluate.add_argument("--pairs", type=Path, required=True)
    evaluate.add_argument("--data-root", type=Path)
    evaluate.add_argument("--output", type=Path, required=True)
    train = commands.add_parser("train", help="unchanged core training with portable input paths")
    train.add_argument("--samples", type=Path, required=True)
    train.add_argument("--pairs", type=Path, required=True)
    train.add_argument("--data-root", type=Path)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--variant", choices=_core.VARIANTS, default="far_full")
    train.add_argument("core_args", nargs=argparse.REMAINDER, help="after --: original core options, e.g. --device cpu --validate-only")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    torch.set_num_threads(args.threads)
    if args.command == "make-demo":
        print(json.dumps({"manifest": str(make_demo(args.output))}))
        return 0
    if args.command == "train":
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError("training output must be empty; use a fresh directory")
        rows = read_manifest(args.samples, args.data_root)
        pairs = _core.read_jsonl(args.pairs)
        validate_formal(rows, pairs)
        args.output.mkdir(parents=True, exist_ok=True)
        prepared = args.output / "inputs"
        prepared.mkdir()
        samples_path = prepared / "samples.jsonl"
        pairs_path = prepared / "pairs.jsonl"
        samples_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        pairs_path.write_text("".join(json.dumps(row) + "\n" for row in pairs), encoding="utf-8")
        extra = args.core_args[1:] if args.core_args[:1] == ["--"] else args.core_args
        arguments = ["sound-evidence-core", "--sample-manifest", str(samples_path),
                     "--pair-manifest", str(pairs_path), "--output-dir", str(args.output),
                     "--variant", args.variant, "--overwrite", *extra]
        previous = sys.argv
        try:
            sys.argv = arguments
            return _core.main()
        finally:
            sys.argv = previous
    if args.command in ("demo", "evaluate"):
        require_new_directory(args.output)
    else:
        require_new_prediction(args.output)
    loaded = load_model(args.checkpoint, args.config, device=args.device)
    if args.command == "evaluate":
        print(json.dumps(evaluate_formal(loaded, args.samples, args.pairs, args.output, data_root=args.data_root)))
        return 0
    if args.command == "demo":
        with new_directory(args.output) as staging:
            manifest = make_demo(staging / "audio")
            rows = read_manifest(manifest)
            result = _predict_and_save(loaded, rows, staging / "predictions.npz", 3,
                                       reported_path=args.output / "predictions.npz")
    else:
        rows = read_manifest(args.manifest, args.data_root)
        result = _predict_and_save(loaded, rows, args.output, args.crops)
    print(json.dumps(result))
    return 0


def _predict_and_save(loaded, rows, destination, crops, *, reported_path=None):
    arrays = predict_manifest(loaded, rows, crops=crops)
    result = {
        "status": "complete", "output": str(reported_path or destination), "samples": len(rows),
        "embedding_shape": list(arrays["embeddings"].shape),
        "class_contract": loaded.class_contract, "checkpoint_sha256": loaded.checkpoint_sha256,
        "scope": "native M/C head probabilities and features; not frozen-probe CPS or Path-Chg accuracy",
    }
    write_predictions(destination, arrays, result)
    return result
