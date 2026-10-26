"""I/O wrappers; the registered network, losses and crop math live in _core."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import _core
from .outputs import new_directory, require_new_directory


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_json(value: Any) -> Any:
    """Represent undefined segment metrics as JSON null, without changing math."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    return value


def read_manifest(path: str | Path, data_root: str | Path | None = None) -> list[dict[str, Any]]:
    """Resolve paths relative to --data-root, otherwise the manifest directory."""
    path = Path(path).resolve()
    root = Path(data_root).resolve() if data_root is not None else path.parent
    rows = _core.read_jsonl(path)
    seen = set()
    for row in rows:
        identifier = str(row.get("sample_id", ""))
        if not identifier or identifier in seen:
            raise ValueError("sample_id must be nonempty and unique")
        seen.add(identifier)
        audio = Path(str(row.get("path", "")))
        if not str(row.get("path", "")):
            raise ValueError(f"missing path for {identifier}")
        audio = audio if audio.is_absolute() else root / audio
        if not audio.is_file():
            raise FileNotFoundError(f"audio for {identifier} is unavailable: {audio}")
        row["path"] = str(audio.resolve())
    if not rows:
        raise ValueError("manifest is empty")
    return rows


def validate_formal(rows: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """Preserve core schema validation and reject cross-split source reuse."""
    split_by_source: dict[str, str] = {}
    for row in rows:
        source, split = str(row["source_key"]), str(row["split"])
        previous = split_by_source.setdefault(source, split)
        if previous != split:
            raise ValueError(f"source {source!r} occurs in multiple splits")
    return _core.validate_manifests(rows, pairs)


@dataclass
class LoadedModel:
    model: _core.FactorialAudioCnn
    config: dict[str, Any]
    class_contract: dict[str, list[str]]
    checkpoint_sha256: str


def load_model(checkpoint: str | Path, config: str | Path, *, device: str = "cpu") -> LoadedModel:
    """Load the actual FactorialAudioCnn schema, never the older AudioCnn."""
    config_data = json.loads(Path(config).read_text(encoding="utf-8"))
    required = ("sample_rate", "clip_seconds", "n_mels", "n_fft", "hop_length", "embedding_dim", "dropout")
    missing = [key for key in required if key not in config_data]
    if missing:
        raise ValueError(f"model config lacks {missing}")
    expected = config_data.get("checkpoint_sha256")
    actual = sha256(checkpoint)
    if expected and actual != expected:
        raise ValueError("checkpoint SHA256 does not match its config")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("variant") != "far_full":
        raise ValueError("this loader accepts the bundled core far_full checkpoints only")
    classes = payload.get("class_contract")
    if not isinstance(classes, dict) or not {"M", "C"}.issubset(classes):
        raise ValueError("not a FactorialAudioCnn checkpoint: missing class_contract")
    bandwidths = payload.get("structural_latent", {}).get("B")
    if not bandwidths:
        raise ValueError("checkpoint lacks its bandwidth contract")
    levels = {
        "O": tuple(classes.get("O", ["original"])),
        "B": tuple(bandwidths), "M": tuple(classes["M"]), "C": tuple(classes["C"]),
    }
    _core.configure_factor_levels(levels)
    model = _core.build_model(argparse.Namespace(**config_data), "far_full")
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(_core.resolve_device(device)).eval()
    return LoadedModel(model, config_data, classes, actual)


def predict_manifest(loaded: LoadedModel, rows: list[dict[str, Any]], *, crops: int = 3) -> dict[str, np.ndarray]:
    """Export embeddings and native M/C head outputs with synchronous group crops.

    group_id identifies variants sharing one normalization/crop reference set.
    Missing group_id means one independent clip. This convenience input is not
    a claim of complete factorial coverage or a published CPS/Path-Chg score.
    """
    if crops < 1:
        raise ValueError("crops must be positive")
    groups = defaultdict(list)
    for row in rows:
        groups[str(row.get("group_id", row["sample_id"]))].append(row)
    config = loaded.config
    device = next(loaded.model.parameters()).device
    clip_samples = round(config["sample_rate"] * config["clip_seconds"])
    accum = {}
    with torch.inference_mode():
        for group in groups.values():
            waves = [_core.load_audio(row["path"], config["sample_rate"]) for row in group]
            max_start = max(0, min(w.numel() for w in waves) - clip_samples)
            for crop_index in range(crops):
                start = max_start // 2 if crops == 1 else round(max_start * crop_index / (crops - 1))
                batch, _ = _core.synchronous_crop_and_group_gain(waves, [True] * len(waves), clip_samples, start=start)
                outputs = loaded.model(batch.to(device))
                for index, row in enumerate(group):
                    item = accum.setdefault(row["sample_id"], {})
                    for name, values in outputs.items():
                        value = values[index].cpu().numpy().astype(np.float64)
                        item[name] = item.get(name, np.zeros_like(value)) + value / crops
                        if name.endswith("_logits"):
                            prob = torch.softmax(values[index], dim=-1).cpu().numpy().astype(np.float64)
                            key = name.replace("_logits", "_probs")
                            item[key] = item.get(key, np.zeros_like(prob)) + prob / crops
    identifiers = [row["sample_id"] for row in rows]
    arrays = {key: np.stack([accum[sid][key] for sid in identifiers]).astype(np.float32) for key in accum[identifiers[0]]}
    arrays["embeddings"] = arrays.pop("embedding")
    norm = np.maximum(np.linalg.norm(arrays["embeddings"], axis=1, keepdims=True), 1e-12)
    arrays["embeddings_normalized"] = arrays["embeddings"] / norm
    arrays["sample_ids"] = np.asarray(identifiers)
    arrays["checkpoint_sha256"] = np.asarray(loaded.checkpoint_sha256)
    arrays["crops"] = np.asarray(crops)
    return arrays


def evaluate_formal(loaded: LoadedModel, samples: str | Path, pairs: str | Path, output: str | Path,
                    *, data_root: str | Path | None = None) -> dict[str, Any]:
    """Use the unchanged full-grid exporter and training-only fixed probes."""
    output = require_new_directory(output)
    rows = read_manifest(samples, data_root)
    pair_rows = _core.read_jsonl(Path(pairs))
    audit = validate_formal(rows, pair_rows)
    if list(_core.DECODERS) != loaded.class_contract["M"] or list(_core.CHANNELS) != loaded.class_contract["C"]:
        raise ValueError("manifest factor levels disagree with checkpoint")
    if ("O" in _core.ACTIVE_SINGLE_CLIP_QUERIES) != ("O" in loaded.class_contract):
        raise ValueError("manifest origin-query role disagrees with checkpoint")
    config = dict(loaded.config)
    config.update(sample_manifest=Path(samples), pair_manifest=Path(pairs), num_workers=0)
    args = argparse.Namespace(**config)
    groups = {split: _core.build_source_groups(rows, pair_rows, split) for split in _core.SPLITS}
    if any(not group for group in groups.values()):
        raise ValueError("formal evaluation requires train/dev/test source groups")
    with new_directory(output) as staging:
        embeddings = staging / "frozen_embeddings.npz"
        export = _core.export_frozen_embeddings(loaded.model, groups, rows, args, next(loaded.model.parameters()).device, embeddings)
        metrics, parameters = _core.run_fixed_linear_probes(embeddings, pair_rows, loaded.model, args)
        metrics = finite_json(metrics)
        parameters = finite_json(parameters)
        _core.write_json_atomic(staging / "probe_results.json", metrics)
        _core.write_json_atomic(staging / "probe_parameters.json", parameters)
        _core.write_probe_csv(staging / "probe_results.csv", metrics)
        export["path"] = str(output / "frozen_embeddings.npz")
        result = {"status": "complete", "manifest_audit": audit, "export": export,
                  "checkpoint_sha256": loaded.checkpoint_sha256,
                  "scope": "M/C/R fixed linear probes, not the separate Path-Chg endpoint"}
        _core.write_json_atomic(staging / "evaluation.json", result)
    return result
