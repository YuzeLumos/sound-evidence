#!/usr/bin/env python3
"""Reviewer-stable source-matched factorial pilot for CPS/FAR.

This runner consumes the sample and relation-pair manifests emitted by
``generate_cps_factorial_audio.py``.  Bandwidth is a structural latent: it is
never exposed through a single-clip prediction head.  It is used only to
define exact registered-factor states and the two transition families required by
FAR.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


ORIGINS = ("original", "bigvgan_resynthesis")
BANDWIDTHS = (6.0, 12.0)
DECODERS = ("encodec_native", "vocos_encodec")
CHANNELS = ("none", "opus_24kbps", "aac_64kbps")
SPLITS = ("train", "dev", "test")
SEEN_PARTITION = "seen_mxc"
HELDOUT_PARTITION = "heldout_mxc"
VARIANTS = (
    "ce_omc",
    "ce_capacity_matched",
    "ce_M_only",
    "ce_C_only",
    "pair_R_only",
    "ce_omc_supcon",
    "ce_omc_pair_bce",
    "far_full",
    "far_random_grouping",
    "far_without_offset",
    "far_without_B_transition",
)
FAR_STRUCTURAL_VARIANTS = (
    "far_full",
    "far_random_grouping",
    "far_without_offset",
    "far_without_B_transition",
)
FAR_FULL_WEIGHT_VARIANTS = ("far_full", "far_random_grouping")
CELL_POLICIES = ("full", SEEN_PARTITION)

ALL_STATES = tuple(product(ORIGINS, BANDWIDTHS, DECODERS, CHANNELS))
STATE_TO_ID = {state: index for index, state in enumerate(ALL_STATES)}
SEEN_STATES = tuple(state for state in ALL_STATES if not (state[2] == "vocos_encodec" and state[3] == "aac_64kbps"))
ORIGIN_TO_ID = {value: index for index, value in enumerate(ORIGINS)}
BANDWIDTH_TO_ID = {value: index for index, value in enumerate(BANDWIDTHS)}
DECODER_TO_ID = {value: index for index, value in enumerate(DECODERS)}
CHANNEL_TO_ID = {value: index for index, value in enumerate(CHANNELS)}
ACTIVE_SINGLE_CLIP_QUERIES = ("O", "M", "C")


def infer_factor_levels(sample_rows: list[dict[str, Any]]) -> dict[str, tuple[Any, ...]]:
    if not sample_rows:
        raise ValueError("sample manifest is empty")
    origins: set[str] = set()
    origin_labels: dict[str, int] = {}
    origin_label_rows = 0
    decoder_labels: dict[str, int] = {}
    channel_labels: dict[str, int] = {}
    bandwidths: set[float] = set()
    for row in sample_rows:
        origin, bandwidth, decoder, channel = state_tuple(row)
        labels = row["labels"]
        origins.add(origin)
        if "ad_origin" in labels:
            origin_label_rows += 1
            label = int(labels["ad_origin"])
            previous = origin_labels.setdefault(origin, label)
            if previous != label:
                raise ValueError(f"inconsistent label mapping for {origin!r}")
        for mapping, value, label_key in (
            (decoder_labels, decoder, "sa_decoder"),
            (channel_labels, channel, "pi_channel"),
        ):
            label = int(labels[label_key])
            previous = mapping.setdefault(value, label)
            if previous != label:
                raise ValueError(f"inconsistent label mapping for {value!r}")
        bandwidths.add(bandwidth)
    if len(origins) == 2 and origin_label_rows != len(sample_rows):
        raise ValueError("a two-level O query requires ad_origin on every sample row")
    if origin_label_rows not in (0, len(sample_rows)):
        raise ValueError("ad_origin is only present on part of the sample manifest")

    registered_contracts = {
        json.dumps(
            row.get("cps_factorial", {}).get("factor_contract"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for row in sample_rows
        if row.get("cps_factorial", {}).get("factor_contract") is not None
    }
    if len(registered_contracts) > 1:
        raise ValueError("sample manifest mixes incompatible factor_contract registries")
    registered_origins: tuple[str, ...] = ()
    if registered_contracts:
        contract = json.loads(next(iter(registered_contracts)))
        registered_origins = tuple(
            str(value) for value in contract.get("axis_levels", {}).get("origin", [])
        )
        if set(registered_origins) != origins:
            raise ValueError("observed origin levels disagree with factor_contract")

    if registered_origins:
        ordered_origins = registered_origins
    elif origin_labels:
        ordered_origins = tuple(
            value for value, _label in sorted(origin_labels.items(), key=lambda item: item[1])
        )
    else:
        ordered_origins = tuple(sorted(origins))
    levels = {
        "O": ordered_origins,
        "B": tuple(sorted(bandwidths)),
        "M": tuple(value for value, _label in sorted(decoder_labels.items(), key=lambda item: item[1])),
        "C": tuple(value for value, _label in sorted(channel_labels.items(), key=lambda item: item[1])),
    }
    wrong: dict[str, list[Any]] = {}
    if len(levels["O"]) not in (1, 2):
        wrong["O"] = list(levels["O"])
    for axis, size in {"B": 2, "M": 2, "C": 3}.items():
        if len(levels[axis]) != size:
            wrong[axis] = list(levels[axis])
    if wrong:
        raise ValueError(f"factorial axis cardinality mismatch: {wrong}")
    for name, mapping in (
        ("O", origin_labels),
        ("M", decoder_labels),
        ("C", channel_labels),
    ):
        if mapping and set(mapping.values()) != set(range(len(mapping))):
            raise ValueError(f"{name} label IDs must be contiguous from zero: {mapping}")
    return levels


def configure_factor_levels(levels: dict[str, tuple[Any, ...]]) -> None:
    global ORIGINS, BANDWIDTHS, DECODERS, CHANNELS
    global ALL_STATES, STATE_TO_ID, SEEN_STATES
    global ORIGIN_TO_ID, BANDWIDTH_TO_ID, DECODER_TO_ID, CHANNEL_TO_ID
    global ACTIVE_SINGLE_CLIP_QUERIES
    ORIGINS = tuple(str(value) for value in levels["O"])
    BANDWIDTHS = tuple(float(value) for value in levels["B"])
    DECODERS = tuple(str(value) for value in levels["M"])
    CHANNELS = tuple(str(value) for value in levels["C"])
    ALL_STATES = tuple(product(ORIGINS, BANDWIDTHS, DECODERS, CHANNELS))
    STATE_TO_ID = {state: index for index, state in enumerate(ALL_STATES)}
    SEEN_STATES = tuple(
        state for state in ALL_STATES
        if not (state[2] == "vocos_encodec" and state[3] == "aac_64kbps")
    )
    ORIGIN_TO_ID = {value: index for index, value in enumerate(ORIGINS)}
    BANDWIDTH_TO_ID = {value: index for index, value in enumerate(BANDWIDTHS)}
    DECODER_TO_ID = {value: index for index, value in enumerate(DECODERS)}
    CHANNEL_TO_ID = {value: index for index, value in enumerate(CHANNELS)}
    ACTIVE_SINGLE_CLIP_QUERIES = (
        ("O", "M", "C") if len(ORIGINS) > 1 else ("M", "C")
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def state_tuple(row: dict[str, Any]) -> tuple[str, float, str, str]:
    factors = row["cps_factorial"]
    return (
        str(factors["origin"]),
        float(factors["bandwidth_kbps"]),
        str(factors["decoder"]),
        str(factors["channel"]),
    )


def expected_partition(state: tuple[str, float, str, str]) -> str:
    return HELDOUT_PARTITION if state[2] == "vocos_encodec" and state[3] == "aac_64kbps" else SEEN_PARTITION


def _single_identity(values: set[str], name: str) -> str | None:
    nonempty = values - {""}
    if len(nonempty) > 1 or (nonempty and "" in values):
        raise ValueError(f"manifest has partial or mixed {name} IDs: {sorted(values)}")
    return next(iter(nonempty), None)


def validate_manifests(sample_rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not sample_rows:
        raise ValueError("sample manifest is empty")
    if not pair_rows:
        raise ValueError("pair manifest is empty")
    configure_factor_levels(infer_factor_levels(sample_rows))
    sample_protocol = _single_identity(
        {str(row.get("cps_factorial", {}).get("protocol") or "") for row in sample_rows},
        "sample protocol",
    )
    sample_dataset = _single_identity(
        {str(row.get("dataset") or "") for row in sample_rows}, "sample dataset"
    )
    nested_dataset = _single_identity(
        {
            str(row.get("cps_factorial", {}).get("dataset_id") or "")
            for row in sample_rows
        },
        "nested sample dataset",
    )
    pair_protocol = _single_identity(
        {str(row.get("protocol") or "") for row in pair_rows}, "pair protocol"
    )
    pair_dataset = _single_identity(
        {str(row.get("dataset") or "") for row in pair_rows}, "pair dataset"
    )
    if sample_dataset and nested_dataset and sample_dataset != nested_dataset:
        raise ValueError("top-level and nested sample dataset IDs disagree")
    if pair_protocol and pair_protocol != sample_protocol:
        raise ValueError("sample and pair protocol IDs disagree")
    if pair_dataset and pair_dataset != sample_dataset:
        raise ValueError("sample and pair dataset IDs disagree")

    sample_ids: set[str] = set()
    sample_by_id: dict[str, dict[str, Any]] = {}
    group_states: dict[tuple[str, str], set[tuple[str, float, str, str]]] = defaultdict(set)
    for row in sample_rows:
        sample_id = str(row["sample_id"])
        if sample_id in sample_ids:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        sample_ids.add(sample_id)
        sample_by_id[sample_id] = row
        split = str(row["split"])
        if split not in SPLITS:
            raise ValueError(f"unsupported split {split!r}")
        state = state_tuple(row)
        if state not in STATE_TO_ID:
            raise ValueError(f"unregistered factorial state {state!r}")
        partition = str(row["cps_factorial"]["combination_partition"])
        if partition != expected_partition(state):
            raise ValueError(f"partition mismatch for {sample_id}: {partition}")
        labels = row["labels"]
        expected_labels = {
            "sa_decoder": DECODER_TO_ID[state[2]],
            "pi_channel": CHANNEL_TO_ID[state[3]],
        }
        if "O" in ACTIVE_SINGLE_CLIP_QUERIES:
            expected_labels["ad_origin"] = ORIGIN_TO_ID[state[0]]
        actual_labels = {key: int(labels[key]) for key in expected_labels}
        if actual_labels != expected_labels:
            raise ValueError(f"label/factor disagreement for {sample_id}")
        key = (split, str(row["source_key"]))
        if state in group_states[key]:
            raise ValueError(f"duplicate factorial state {state!r} in source group {key!r}")
        group_states[key].add(state)
    incomplete = {key: len(states) for key, states in group_states.items() if states != set(ALL_STATES)}
    if incomplete:
        preview = list(sorted(incomplete.items()))[:5]
        raise ValueError(f"incomplete registered factorial source groups: {preview}")

    pair_ids: set[str] = set()
    pair_counts: Counter[tuple[str, str]] = Counter()
    pair_patterns: dict[tuple[str, str, str, str], set[tuple[float, float, int]]] = defaultdict(set)
    for row in pair_rows:
        pair_id = str(row["pair_id"])
        if pair_id in pair_ids:
            raise ValueError(f"duplicate pair_id: {pair_id}")
        pair_ids.add(pair_id)
        sample_id_a = str(row["sample_id_a"])
        sample_id_b = str(row["sample_id_b"])
        if sample_id_a not in sample_ids or sample_id_b not in sample_ids:
            raise ValueError(f"pair endpoint missing from sample manifest: {pair_id}")
        left = sample_by_id[sample_id_a]
        right = sample_by_id[sample_id_b]
        left_state = state_tuple(left)
        right_state = state_tuple(right)
        label = int(row["same_token_label"])
        if label not in (0, 1):
            raise ValueError(f"non-binary relation label for {pair_id}")
        same_bandwidth = float(row["bandwidth_a_kbps"]) == float(row["bandwidth_b_kbps"])
        if same_bandwidth != bool(label):
            raise ValueError(f"bandwidth/relation-label disagreement for {pair_id}")
        if (
            str(row["source_key_a"]) != str(row["source_key_b"])
            or str(row["source_key_a"]) != str(left["source_key"])
            or str(row["source_key_b"]) != str(right["source_key"])
            or str(left["split"]) != str(right["split"])
            or str(left["split"]) != str(row["split"])
            or left_state[0] != right_state[0]
            or left_state[3] != right_state[3]
            or left_state[2] != "encodec_native"
            or right_state[2] != "vocos_encodec"
            or float(row["bandwidth_a_kbps"]) != left_state[1]
            or float(row["bandwidth_b_kbps"]) != right_state[1]
        ):
            raise ValueError(f"R pair violates the source/O/C-matched native-to-Vocos contract: {pair_id}")
        partition = str(row["combination_partition"])
        expected = HELDOUT_PARTITION if str(row["channel_a"]) == "aac_64kbps" else SEEN_PARTITION
        if partition != expected:
            raise ValueError(f"pair partition mismatch for {pair_id}")
        pair_counts[(str(row["split"]), str(row["source_key_a"]))] += 1
        pair_patterns[
            (
                str(row["split"]),
                str(row["source_key_a"]),
                left_state[0],
                left_state[3],
            )
        ].add((left_state[1], right_state[1], label))
    expected_group_keys = set(group_states)
    expected_pairs_per_source = len(ORIGINS) * len(CHANNELS) * 4
    if set(pair_counts) != expected_group_keys or any(
        count != expected_pairs_per_source for count in pair_counts.values()
    ):
        raise ValueError(
            f"each source group must contain the complete {expected_pairs_per_source}-row R pair registry"
        )
    low, high = BANDWIDTHS
    expected_patterns = {
        (low, low, 1), (high, high, 1), (low, high, 0), (high, low, 0)
    }
    expected_pattern_keys = {
        (split, source, origin, channel)
        for split, source in expected_group_keys
        for origin in ORIGINS
        for channel in CHANNELS
    }
    if set(pair_patterns) != expected_pattern_keys or any(
        patterns != expected_patterns for patterns in pair_patterns.values()
    ):
        raise ValueError("R pair registry is missing or duplicates a registered B endpoint pattern")
    heldout_pairs = 4 * len(ORIGINS)
    seen_pairs = expected_pairs_per_source - heldout_pairs
    return {
        "samples": len(sample_rows),
        "pairs": len(pair_rows),
        "source_groups": len(group_states),
        "samples_by_split": dict(sorted(Counter(str(row["split"]) for row in sample_rows).items())),
        "pairs_by_split": dict(sorted(Counter(str(row["split"]) for row in pair_rows).items())),
        "seen_states_per_source": len(SEEN_STATES),
        "heldout_states_per_source": len(ALL_STATES) - len(SEEN_STATES),
        "seen_pairs_per_source": seen_pairs,
        "heldout_pairs_per_source": heldout_pairs,
        "cells_per_source": len(ALL_STATES),
        "pairs_per_source": expected_pairs_per_source,
        "factor_levels": {
            "O": list(ORIGINS),
            "B": list(BANDWIDTHS),
            "M": list(DECODERS),
            "C": list(CHANNELS),
        },
        "active_single_clip_queries": list(ACTIVE_SINGLE_CLIP_QUERIES),
        "origin_role": "query" if "O" in ACTIVE_SINGLE_CLIP_QUERIES else "fixed_nuisance",
        "protocol_id": sample_protocol,
        "dataset_id": sample_dataset or nested_dataset,
    }


def build_source_groups(
    sample_rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]], split: str
) -> list[dict[str, Any]]:
    samples_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pairs_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        if row["split"] == split:
            samples_by_source[str(row["source_key"])].append(row)
    for row in pair_rows:
        if row["split"] == split:
            source_a = str(row["source_key_a"])
            source_b = str(row["source_key_b"])
            if source_a != source_b:
                raise ValueError(f"R pair is not source matched: {row['pair_id']}")
            pairs_by_source[source_a].append(row)
    if not samples_by_source:
        raise ValueError(f"no sample rows for required split {split!r}")
    groups: list[dict[str, Any]] = []
    for source_key in sorted(samples_by_source):
        rows = sorted(samples_by_source[source_key], key=lambda row: STATE_TO_ID[state_tuple(row)])
        pairs = sorted(pairs_by_source[source_key], key=lambda row: str(row["pair_id"]))
        expected_pairs = len(ORIGINS) * len(CHANNELS) * 4
        if len(rows) != len(ALL_STATES) or len(pairs) != expected_pairs:
            raise ValueError(f"incomplete source group {split}/{source_key}")
        groups.append({"split": split, "source_key": source_key, "rows": rows, "pairs": pairs})
    return groups


def load_audio(path: str, sample_rate: int) -> torch.Tensor:
    audio, native_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio.T.copy())
    waveform = waveform.mean(dim=0)
    if int(native_rate) != sample_rate:
        waveform = torchaudio.functional.resample(waveform.unsqueeze(0), int(native_rate), sample_rate).squeeze(0)
    return waveform.to(torch.float32)


def stable_train_start(seed: int, epoch: int, source_key: str, max_start: int) -> int:
    if max_start <= 0:
        return 0
    token = f"{seed}:{epoch}:{source_key}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(token).digest()[:8], "big")
    return value % (max_start + 1)


def synchronous_crop_and_group_gain(
    waveforms: list[torch.Tensor],
    reference_mask: list[bool],
    clip_samples: int,
    *,
    start: int,
) -> tuple[torch.Tensor, float]:
    """Common-trim, synchronous-crop, then apply one reference-only gain.

    The caller supplies the registered gain-reference set: all registered cells for the
    primary full-grid policy, or seen-MxC cells for the exploratory holdout
    policy.  All source variants define one common shortest support before
    cropping; this removes raw duration as a condition-specific cue.
    """
    if len(waveforms) != len(reference_mask) or not any(reference_mask):
        raise ValueError("waveforms/reference_mask must be aligned and contain a reference cell")
    if clip_samples <= 0 or start < 0:
        raise ValueError("clip_samples must be positive and start must be non-negative")
    common_samples = min(waveform.numel() for waveform in waveforms)
    max_start = max(0, common_samples - clip_samples)
    if start > max_start:
        raise ValueError(f"start {start} exceeds common-support maximum {max_start}")
    crops: list[torch.Tensor] = []
    stop = start + clip_samples
    for waveform in waveforms:
        if waveform.ndim != 1:
            raise ValueError("synchronous crop expects mono 1-D waveforms")
        waveform = waveform[:common_samples]
        if common_samples < clip_samples:
            waveform = F.pad(waveform, (0, clip_samples - common_samples))
        crops.append(waveform[start:stop])
    stacked = torch.stack(crops)
    reference = stacked[torch.tensor(reference_mask, dtype=torch.bool)]
    peak = float(reference.abs().max().clamp_min(1e-6))
    return stacked / peak, 1.0 / peak


class SourceGroupDataset(Dataset):
    def __init__(
        self,
        groups: list[dict[str, Any]],
        sample_rate: int,
        clip_seconds: float,
        *,
        train: bool,
        cell_policy: str,
        gain_policy: str,
        seed: int,
        crop_index: int = 0,
        num_crops: int = 1,
    ) -> None:
        self.groups = groups
        registered_rows = [row for group in groups for row in group["rows"]]
        levels = infer_factor_levels(registered_rows)
        self.origin_to_id = {
            str(value): index for index, value in enumerate(levels["O"])
        }
        self.bandwidth_to_id = {
            float(value): index for index, value in enumerate(levels["B"])
        }
        registered_states = tuple(
            product(levels["O"], levels["B"], levels["M"], levels["C"])
        )
        self.state_to_id = {
            (str(origin), float(bandwidth), str(decoder), str(channel)): index
            for index, (origin, bandwidth, decoder, channel) in enumerate(
                registered_states
            )
        }
        self.sample_rate = sample_rate
        self.clip_samples = int(round(sample_rate * clip_seconds))
        self.train = train
        if cell_policy not in CELL_POLICIES or gain_policy not in CELL_POLICIES:
            raise ValueError(f"cell/gain policy must be one of {CELL_POLICIES}")
        self.cell_policy = cell_policy
        self.gain_policy = gain_policy
        self.seed = seed
        self.crop_index = crop_index
        self.num_crops = max(1, num_crops)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> dict[str, Any]:
        group = self.groups[index]
        all_rows = group["rows"]
        all_waveforms = [load_audio(str(row["path"]), self.sample_rate) for row in all_rows]
        reference_mask = [
            self.gain_policy == "full"
            or row["cps_factorial"]["combination_partition"] == SEEN_PARTITION
            for row in all_rows
        ]
        common_samples = min(waveform.numel() for waveform in all_waveforms)
        max_start = max(0, common_samples - self.clip_samples)
        if self.train:
            start = stable_train_start(self.seed, self.epoch, str(group["source_key"]), max_start)
        elif self.num_crops == 1:
            start = max_start // 2
        else:
            start = int(round(max_start * self.crop_index / (self.num_crops - 1)))
        all_crops, gain = synchronous_crop_and_group_gain(
            all_waveforms, reference_mask, self.clip_samples, start=start
        )
        output_indices = [
            idx for idx, row in enumerate(all_rows)
            if self.cell_policy == "full"
            or row["cps_factorial"]["combination_partition"] == SEEN_PARTITION
        ]
        rows = [all_rows[idx] for idx in output_indices]
        waveform = all_crops[output_indices]
        sample_to_index = {str(row["sample_id"]): idx for idx, row in enumerate(rows)}
        pairs = [
            pair for pair in group["pairs"]
            if str(pair["sample_id_a"]) in sample_to_index and str(pair["sample_id_b"]) in sample_to_index
        ]
        pair_index_a = torch.tensor([sample_to_index[str(pair["sample_id_a"])] for pair in pairs], dtype=torch.long)
        pair_index_b = torch.tensor([sample_to_index[str(pair["sample_id_b"])] for pair in pairs], dtype=torch.long)
        return {
            "waveform": waveform,
            "O": torch.tensor(
                [self.origin_to_id[state_tuple(row)[0]] for row in rows], dtype=torch.long
            ),
            "M": torch.tensor([int(row["labels"]["sa_decoder"]) for row in rows], dtype=torch.long),
            "C": torch.tensor([int(row["labels"]["pi_channel"]) for row in rows], dtype=torch.long),
            "B": torch.tensor(
                [self.bandwidth_to_id[state_tuple(row)[1]] for row in rows],
                dtype=torch.long,
            ),
            "state": torch.tensor(
                [self.state_to_id[state_tuple(row)] for row in rows], dtype=torch.long
            ),
            "pair_index_a": pair_index_a,
            "pair_index_b": pair_index_b,
            "pair_label": torch.tensor([int(pair["same_token_label"]) for pair in pairs], dtype=torch.float32),
            "pair_bandwidth_a": torch.tensor(
                [self.bandwidth_to_id[float(pair["bandwidth_a_kbps"])] for pair in pairs],
                dtype=torch.long,
            ),
            "source_key": str(group["source_key"]),
            "sample_ids": [str(row["sample_id"]) for row in rows],
            "gain": gain,
            "start": start,
            "common_samples": common_samples,
        }


def collate_source_groups(batch: list[dict[str, Any]]) -> dict[str, Any]:
    tensor_keys = (
        "waveform", "O", "M", "C", "B", "state", "pair_index_a", "pair_index_b", "pair_label",
        "pair_bandwidth_a",
    )
    return {
        **{key: torch.stack([item[key] for item in batch]) for key in tensor_keys},
        "source_key": [item["source_key"] for item in batch],
        "sample_ids": [item["sample_ids"] for item in batch],
        "gain": [item["gain"] for item in batch],
        "start": [item["start"] for item in batch],
        "common_samples": [item["common_samples"] for item in batch],
    }


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple[int, int] = (1, 1), dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout2d(dropout)
        if in_channels != out_channels or stride != (1, 1):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.drop(x)
        x = self.bn2(self.conv2(x))
        return F.gelu(x + residual)


def symmetric_pair_features_torch(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = F.normalize(left, dim=-1)
    right = F.normalize(right, dim=-1)
    cosine = (left * right).sum(dim=-1, keepdim=True)
    return torch.cat([(left - right).abs(), left * right, cosine], dim=-1)


class FactorialAudioCnn(nn.Module):
    """One shared encoder and registered query heads; deliberately no B head."""

    def __init__(
        self,
        sample_rate: int,
        n_mels: int,
        n_fft: int,
        hop_length: int,
        embedding_dim: int,
        dropout: float,
        *,
        with_pair_head: bool = False,
    ) -> None:
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=20.0,
            f_max=float(sample_rate // 2),
            power=2.0,
        )
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(32, 32, dropout=dropout),
            ResidualBlock(32, 64, stride=(2, 2), dropout=dropout),
            ResidualBlock(64, 64, dropout=dropout),
            ResidualBlock(64, 128, stride=(2, 2), dropout=dropout),
            ResidualBlock(128, 128, dropout=dropout),
            ResidualBlock(128, 256, stride=(2, 2), dropout=dropout),
            ResidualBlock(256, 256, dropout=dropout),
        )
        self.projection = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.o_head = (
            nn.Linear(embedding_dim, len(ORIGINS))
            if "O" in ACTIVE_SINGLE_CLIP_QUERIES
            else None
        )
        self.m_head = nn.Linear(embedding_dim, len(DECODERS))
        self.c_head = nn.Linear(embedding_dim, len(CHANNELS))
        self.pair_head = nn.Linear(2 * embedding_dim + 1, 1) if with_pair_head else None

    def forward(self, waveform: torch.Tensor) -> dict[str, torch.Tensor]:
        mel = self.mel(waveform)
        log_mel = torch.log(mel.clamp_min(1e-7))
        mean = log_mel.mean(dim=(-2, -1), keepdim=True)
        std = log_mel.std(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
        x = ((log_mel - mean) / std).unsqueeze(1)
        x = self.blocks(self.stem(x))
        average = F.adaptive_avg_pool2d(x, 1).flatten(1)
        maximum = F.adaptive_max_pool2d(x, 1).flatten(1)
        embedding = self.projection(torch.cat([average, maximum], dim=1))
        output = {
            "embedding": embedding,
            "M_logits": self.m_head(embedding),
            "C_logits": self.c_head(embedding),
        }
        if self.o_head is not None:
            output["O_logits"] = self.o_head(embedding)
        return output

    def relation_logits(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if self.pair_head is None:
            raise RuntimeError("this model variant has no matched-information pair head")
        return self.pair_head(symmetric_pair_features_torch(left, right)).squeeze(-1)


LAYER_EMBEDDING_NAMES = (
    "stem",
    "residual_1",
    "residual_2",
    "residual_3",
    "residual_4",
    "residual_5",
    "residual_6",
    "residual_7",
    "projection",
)


def pooled_feature_map(feature_map: torch.Tensor) -> torch.Tensor:
    """Return the fixed global average-plus-maximum representation."""
    if feature_map.ndim != 4:
        raise ValueError("feature_map must have shape [batch, channels, frequency, time]")
    average = F.adaptive_avg_pool2d(feature_map, 1).flatten(1)
    maximum = F.adaptive_max_pool2d(feature_map, 1).flatten(1)
    return torch.cat([average, maximum], dim=1)


def forward_layer_embeddings(
    model: FactorialAudioCnn, waveform: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Expose fixed pooled encoder stages without changing ``forward`` output.

    Stem and residual-stage tensors use global average-plus-maximum pooling.  The
    projection entry is the model's ordinary final embedding.  This helper owns
    no parameters and is called only by the optional post-training export.
    """
    mel = model.mel(waveform)
    log_mel = torch.log(mel.clamp_min(1e-7))
    mean = log_mel.mean(dim=(-2, -1), keepdim=True)
    std = log_mel.std(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
    feature_map = ((log_mel - mean) / std).unsqueeze(1)
    feature_map = model.stem(feature_map)
    layers: dict[str, torch.Tensor] = {"stem": pooled_feature_map(feature_map)}
    for index, block in enumerate(model.blocks, start=1):
        feature_map = block(feature_map)
        layers[f"residual_{index}"] = pooled_feature_map(feature_map)
    layers["projection"] = model.projection(layers["residual_7"])
    if tuple(layers) != LAYER_EMBEDDING_NAMES:
        raise AssertionError(f"unexpected layer registry: {tuple(layers)}")
    return layers


def variant_uses_pair_head(variant: str) -> bool:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    return variant in {"ce_omc_pair_bce", "pair_R_only"}


def single_clip_training_queries(variant: str) -> tuple[str, ...]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    if variant == "ce_M_only":
        return ("M",)
    if variant == "ce_C_only":
        return ("C",)
    if variant == "pair_R_only":
        return ()
    return tuple(ACTIVE_SINGLE_CLIP_QUERIES)


def native_training_queries(variant: str) -> tuple[str, ...]:
    if variant == "pair_R_only":
        return ("R",)
    return single_clip_training_queries(variant)


def build_model(args: argparse.Namespace, variant: str) -> FactorialAudioCnn:
    """Build the registered architecture without variant-specific hidden capacity."""
    return FactorialAudioCnn(
        args.sample_rate,
        args.n_mels,
        args.n_fft,
        args.hop_length,
        args.embedding_dim,
        args.dropout,
        with_pair_head=variant_uses_pair_head(variant),
    )


def parameter_schema(model: nn.Module) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "shape": list(parameter.shape),
            "numel": parameter.numel(),
            "requires_grad": bool(parameter.requires_grad),
        }
        for name, parameter in model.named_parameters()
    ]


def parameter_schema_sha256(schema: list[dict[str, Any]]) -> str:
    serialized = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _schema_subset(schema: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    return [row for row in schema if str(row["name"]).startswith(prefix)]


def _schema_names(schema: list[dict[str, Any]]) -> set[str]:
    return {str(row["name"]) for row in schema}


def _representation_schema(schema: list[dict[str, Any]]) -> list[dict[str, Any]]:
    readout_prefixes = ("o_head.", "m_head.", "c_head.", "pair_head.")
    return [
        row
        for row in schema
        if not str(row["name"]).startswith(readout_prefixes)
    ]


def variant_capacity_audit(
    model: FactorialAudioCnn,
    variant: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Prove architectural capacity against a freshly constructed FAR reference.

    FAR's offset and transition objectives are parameter-free.  In particular,
    ``ce_capacity_matched`` must therefore reuse the exact FAR architecture
    instead of adding a dummy module that could silently change optimization.
    Model construction for this audit is isolated from the training RNG stream.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    current_schema = parameter_schema(model)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        far_reference = build_model(args, "far_full")
    far_schema = parameter_schema(far_reference)
    current_names = _schema_names(current_schema)
    far_names = _schema_names(far_schema)
    current_trainable = sum(
        int(row["numel"]) for row in current_schema if bool(row["requires_grad"])
    )
    far_trainable = sum(
        int(row["numel"]) for row in far_schema if bool(row["requires_grad"])
    )
    projection_schema = _schema_subset(current_schema, "projection.")
    far_projection_schema = _schema_subset(far_schema, "projection.")
    representation_schema = _representation_schema(current_schema)
    far_representation_schema = _representation_schema(far_schema)
    exact_schema_match = current_schema == far_schema
    exact_trainable_match = current_trainable == far_trainable
    projection_match = projection_schema == far_projection_schema
    representation_match = representation_schema == far_representation_schema
    weights = variant_weights(variant, args)
    far_weights = variant_weights("far_full", args)
    if variant in {"ce_capacity_matched", "ce_M_only", "ce_C_only"} and not (
        exact_schema_match and exact_trainable_match and projection_match
    ):
        raise AssertionError(f"{variant} is not exactly capacity matched to far_full")
    if variant == "pair_R_only" and not (projection_match and representation_match):
        raise AssertionError("pair_R_only changed the registered encoder/projection capacity")
    if variant == "far_random_grouping" and weights != far_weights:
        raise AssertionError("far_random_grouping must use the exact far_full loss coefficients")
    structural_labels = [
        label
        for label, active in (
            ("registered_exact_state", weights["offset"] > 0.0),
            ("registered_relation_pair", weights["transition"] > 0.0),
            ("registered_B_transition_family", weights["transition"] > 0.0),
        )
        if active
    ]
    training_queries = native_training_queries(variant)
    non_training_clip_heads = [
        query for query in ACTIVE_SINGLE_CLIP_QUERIES if query not in training_queries
    ]
    return {
        "audit_version": "cps_variant_capacity_audit_v1",
        "variant": variant,
        "capacity_match_target": "far_full",
        "embedding_dim": int(args.embedding_dim),
        "projection_parameter_count": sum(int(row["numel"]) for row in projection_schema),
        "far_full_projection_parameter_count": sum(
            int(row["numel"]) for row in far_projection_schema
        ),
        "trainable_parameter_count": current_trainable,
        "far_full_trainable_parameter_count": far_trainable,
        "parameter_schema_sha256": parameter_schema_sha256(current_schema),
        "far_full_parameter_schema_sha256": parameter_schema_sha256(far_schema),
        "exact_parameter_schema_match_to_far_full": exact_schema_match,
        "exact_trainable_parameter_count_match_to_far_full": exact_trainable_match,
        "exact_projection_schema_match_to_far_full": projection_match,
        "exact_encoder_projection_schema_match_to_far_full": representation_match,
        "encoder_projection_parameter_count": sum(
            int(row["numel"]) for row in representation_schema
        ),
        "far_full_encoder_projection_parameter_count": sum(
            int(row["numel"]) for row in far_representation_schema
        ),
        "encoder_projection_schema_sha256": parameter_schema_sha256(
            representation_schema
        ),
        "far_full_encoder_projection_schema_sha256": parameter_schema_sha256(
            far_representation_schema
        ),
        "extra_parameter_names_vs_far_full": sorted(current_names - far_names),
        "missing_parameter_names_vs_far_full": sorted(far_names - current_names),
        "pair_head_present": model.pair_head is not None,
        "far_objectives_add_trainable_parameters": False,
        "loss_weights": weights,
        "native_training_queries": list(training_queries),
        "native_query": training_queries[0] if len(training_queries) == 1 else None,
        "optimization_labels": list(training_queries),
        "single_clip_heads_present": list(ACTIVE_SINGLE_CLIP_QUERIES),
        "single_clip_heads_excluded_from_training_loss": non_training_clip_heads,
        "excluded_heads_contribute_gradient": False,
        "frozen_probe_queries": ["M", "C", "R"],
        "structural_labels_consumed_by_training_loss": structural_labels,
        "additional_labels_consumed_by_ce_capacity_matched": [],
        "ce_capacity_matched_is_explicit_ce_only_alias": variant == "ce_capacity_matched",
        "pair_R_only_uses_source_metadata": False,
        "pair_R_only_additional_parameters_are_readout_only": (
            variant == "pair_R_only"
            and representation_match
            and not exact_schema_match
        ),
        "far_full_loss_weight_match": weights == far_weights,
        "structural_loss_graph_match_to_far_full": variant in FAR_FULL_WEIGHT_VARIANTS,
        "random_grouping": {
            "active": variant == "far_random_grouping",
            "scope": "training structural embeddings only",
            "head_ce_and_supcon_use_true_unshuffled_examples": True,
            "dev_selection_and_frozen_probes_unchanged": True,
            "source_identifiers_or_metadata_used_by_randomizer": False,
            "same_structural_loss_functions_shapes_and_weights_as_far_full": (
                variant == "far_random_grouping"
            ),
        },
    }


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    if features.shape[0] < 2:
        return features.sum() * 0.0
    features = F.normalize(features, dim=1)
    logits = features @ features.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    eye = torch.eye(features.shape[0], dtype=torch.bool, device=features.device)
    positive = labels[:, None].eq(labels[None, :]) & ~eye
    exp_logits = torch.exp(logits) * ~eye
    log_probability = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positive_count = positive.sum(dim=1)
    valid = positive_count > 0
    if not valid.any():
        return features.sum() * 0.0
    mean_log_probability = (log_probability * positive).sum(dim=1) / positive_count.clamp_min(1)
    return -mean_log_probability[valid].mean()


def exact_state_offset_loss(
    embeddings: torch.Tensor, state_ids: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Prototype contrast over exact registered factorial residual states."""
    normalized = F.normalize(embeddings, dim=-1)
    center = normalized.mean(dim=1, keepdim=True).detach()
    residual = F.normalize(normalized - center, dim=-1)
    flat_residual = residual.flatten(0, 1)
    flat_state = state_ids.flatten()
    unique_states = torch.unique(flat_state, sorted=True)
    prototypes = torch.stack(
        [F.normalize(flat_residual[flat_state == state].mean(dim=0), dim=0) for state in unique_states]
    ).detach()
    target_map = torch.full(
        (len(ALL_STATES),), -1, dtype=torch.long, device=embeddings.device
    )
    target_map[unique_states] = torch.arange(unique_states.numel(), device=embeddings.device)
    logits = flat_residual @ prototypes.T / temperature
    return F.cross_entropy(logits, target_map[flat_state])


def gather_pairs(embeddings: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    expanded = indices.unsqueeze(-1).expand(-1, -1, embeddings.shape[-1])
    return torch.gather(embeddings, dim=1, index=expanded)


def b_transition_prototype_loss(
    embeddings: torch.Tensor,
    pair_index_a: torch.Tensor,
    pair_index_b: torch.Tensor,
    pair_labels: torch.Tensor,
    pair_bandwidth_a: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Align native->Vocos transitions within B and contrast the two registered B levels.

    The positive rows in the fixed R registry are the two valid transition
    families.  Cross-B R rows are represented as the opposing family in the
    prototype denominator rather than as a third transition family.
    """
    left = F.normalize(gather_pairs(embeddings, pair_index_a), dim=-1)
    right = F.normalize(gather_pairs(embeddings, pair_index_b), dim=-1)
    positive = pair_labels > 0.5
    transitions = F.normalize(right - left, dim=-1)[positive]
    families = pair_bandwidth_a[positive]
    if transitions.numel() == 0 or torch.unique(families).numel() != len(BANDWIDTHS):
        raise ValueError(
            f"both registered B transition families are required: {list(BANDWIDTHS)}"
        )
    prototypes = torch.stack(
        [F.normalize(transitions[families == family].mean(dim=0), dim=0) for family in range(len(BANDWIDTHS))]
    ).detach()
    logits = transitions @ prototypes.T / temperature
    return F.cross_entropy(logits, families)


def variant_weights(variant: str, args: argparse.Namespace) -> dict[str, float]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    return {
        "supcon": args.supcon_weight if variant in {
            "ce_omc_supcon", *FAR_STRUCTURAL_VARIANTS
        } else 0.0,
        "offset": args.offset_weight if variant in {
            "far_full", "far_random_grouping", "far_without_B_transition"
        } else 0.0,
        "transition": args.transition_weight if variant in {
            "far_full", "far_random_grouping", "far_without_offset"
        } else 0.0,
        "pair_bce": (
            args.pair_bce_weight
            if variant in {"ce_omc_pair_bce", "pair_R_only"}
            else 0.0
        ),
    }


def _random_grouping_seed(seed: int, epoch: int, batch_index: int) -> int:
    token = f"far_random_grouping:{seed}:{epoch}:{batch_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big") % (2**63 - 1)


def deterministic_cross_source_structural_grouping(
    embeddings: torch.Tensor,
    decoder_ids: torch.Tensor,
    pair_index_a: torch.Tensor,
    pair_index_b: torch.Tensor,
    *,
    seed: int,
    epoch: int,
    batch_index: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Create deterministic mismatched formation groups without reading metadata.

    Each synthetic structural group receives all native states from one randomly
    permuted source and all Vocos states from a different, deranged source.  Thus
    every registered native-to-Vocos transition is cross-source while tensor
    shapes, state marginals, relation patterns, loss calls, and weights remain
    identical to FAR.  Head CE and SupCon never consume this returned tensor.
    """
    if embeddings.ndim != 3 or decoder_ids.shape != embeddings.shape[:2]:
        raise ValueError("embeddings and decoder_ids must be aligned [groups, states, ...]")
    batch_size, state_count, _ = embeddings.shape
    if batch_size < 2:
        raise ValueError(
            "far_random_grouping requires at least two source groups in every training batch"
        )
    if pair_index_a.shape != pair_index_b.shape or pair_index_a.shape[0] != batch_size:
        raise ValueError("pair endpoint indices must be aligned by source-group batch")
    if not torch.equal(decoder_ids, decoder_ids[:1].expand_as(decoder_ids)):
        raise ValueError("registered decoder-state layout differs across source groups")
    try:
        native_id = DECODER_TO_ID["encodec_native"]
        vocos_id = DECODER_TO_ID["vocos_encodec"]
    except KeyError as error:
        raise ValueError("random grouping requires the registered native/Vocos decoder pair") from error
    if set(torch.unique(decoder_ids).detach().cpu().tolist()) != {native_id, vocos_id}:
        raise ValueError("random grouping saw an unregistered decoder ID")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(_random_grouping_seed(seed, epoch, batch_index))
    native_permutation_cpu = torch.randperm(batch_size, generator=generator)
    shift = int(torch.randint(1, batch_size, (1,), generator=generator).item())
    vocos_permutation_cpu = torch.roll(native_permutation_cpu, shifts=shift)
    if torch.any(native_permutation_cpu == vocos_permutation_cpu):
        raise AssertionError("cyclic derangement unexpectedly retained a source match")

    native_permutation = native_permutation_cpu.to(embeddings.device)
    vocos_permutation = vocos_permutation_cpu.to(embeddings.device)
    native_mask = decoder_ids.eq(native_id)
    assignment = torch.where(
        native_mask,
        native_permutation[:, None],
        vocos_permutation[:, None],
    )
    state_index = torch.arange(state_count, device=embeddings.device)[None, :].expand(
        batch_size, -1
    )
    randomized = embeddings[assignment, state_index]
    left_sources = torch.gather(assignment, 1, pair_index_a)
    right_sources = torch.gather(assignment, 1, pair_index_b)
    endpoint_matches = left_sources.eq(right_sources)
    unique_sources_per_group = torch.tensor(
        [torch.unique(row).numel() for row in assignment],
        dtype=torch.float32,
        device=embeddings.device,
    )
    audit = {
        "seed_derivation": "sha256(far_random_grouping:seed:epoch:batch_index)",
        "epoch": int(epoch),
        "batch_index": int(batch_index),
        "native_source_permutation": native_permutation_cpu.tolist(),
        "vocos_source_permutation": vocos_permutation_cpu.tolist(),
        "derangement_shift": shift,
        "transition_endpoint_same_source_count": int(endpoint_matches.sum().item()),
        "transition_endpoint_count": int(endpoint_matches.numel()),
        "cross_source_transition_fraction": float((~endpoint_matches).float().mean().item()),
        "mixed_structural_group_fraction": float(
            unique_sources_per_group.gt(1).float().mean().item()
        ),
        "mean_unique_sources_per_structural_group": float(
            unique_sources_per_group.mean().item()
        ),
        "source_metadata_used": False,
    }
    if audit["transition_endpoint_same_source_count"] != 0:
        raise AssertionError("random grouping failed to break a registered transition source match")
    return randomized, audit


def random_grouping_batch_contract(source_groups: int, group_batch_size: int) -> dict[str, Any]:
    if source_groups <= 0 or group_batch_size <= 0:
        raise ValueError("source_groups and group_batch_size must be positive")
    full_batches, remainder = divmod(source_groups, group_batch_size)
    batch_sizes = [group_batch_size] * full_batches
    if remainder:
        batch_sizes.append(remainder)
    if not batch_sizes:
        batch_sizes = [source_groups]
    singleton_batches = sum(size < 2 for size in batch_sizes)
    audit = {
        "train_source_groups": int(source_groups),
        "configured_group_batch_size": int(group_batch_size),
        "planned_batch_sizes": batch_sizes,
        "singleton_batch_count": singleton_batches,
        "every_training_batch_supports_cross_source_derangement": singleton_batches == 0,
        "drop_last": False,
    }
    if singleton_batches:
        raise ValueError(
            "far_random_grouping cannot preserve all examples and cross-source mismatch with a "
            f"singleton training batch; planned sizes are {batch_sizes}. Adjust --group-batch-size."
        )
    return audit


def forward_group(model: FactorialAudioCnn, waveform: torch.Tensor) -> dict[str, torch.Tensor]:
    batch_size, states, samples = waveform.shape
    output = model(waveform.reshape(batch_size * states, samples))
    output["embedding_grouped"] = output["embedding"].reshape(batch_size, states, -1)
    return output


def move_training_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def compute_training_loss(
    model: FactorialAudioCnn,
    batch: dict[str, Any],
    args: argparse.Namespace,
    weights: dict[str, float],
    *,
    epoch: int = 0,
    batch_index: int = 0,
) -> tuple[torch.Tensor, dict[str, float]]:
    output = forward_group(model, batch["waveform"])
    embedding = output["embedding_grouped"]
    zero = embedding.sum() * 0.0
    training_queries = single_clip_training_queries(args.variant)
    head_losses = {
        query: F.cross_entropy(output[f"{query}_logits"], batch[query].flatten())
        for query in training_queries
    }
    head_ce = (
        torch.stack(list(head_losses.values())).mean()
        if head_losses
        else zero
    )
    if weights["supcon"] > 0.0:
        joint_query = torch.zeros_like(batch[ACTIVE_SINGLE_CLIP_QUERIES[0]].flatten())
        class_counts = {"O": len(ORIGINS), "M": len(DECODERS), "C": len(CHANNELS)}
        for query in ACTIVE_SINGLE_CLIP_QUERIES:
            joint_query = joint_query * class_counts[query] + batch[query].flatten()
        supcon = supervised_contrastive_loss(
            output["embedding"], joint_query, args.supcon_temperature
        )
    else:
        supcon = zero
    random_grouping_audit: dict[str, Any] | None = None
    structural_embedding = embedding
    if args.variant == "far_random_grouping":
        structural_embedding, random_grouping_audit = (
            deterministic_cross_source_structural_grouping(
                embedding,
                batch["M"],
                batch["pair_index_a"],
                batch["pair_index_b"],
                seed=args.seed,
                epoch=epoch,
                batch_index=batch_index,
            )
        )
    offset = (
        exact_state_offset_loss(
            structural_embedding, batch["state"], args.offset_temperature
        )
        if weights["offset"] > 0.0 else zero
    )
    transition = (
        b_transition_prototype_loss(
            structural_embedding,
            batch["pair_index_a"],
            batch["pair_index_b"],
            batch["pair_label"],
            batch["pair_bandwidth_a"],
            args.transition_temperature,
        )
        if weights["transition"] > 0.0 else zero
    )
    if model.pair_head is not None and weights["pair_bce"] > 0.0:
        left = gather_pairs(embedding, batch["pair_index_a"])
        right = gather_pairs(embedding, batch["pair_index_b"])
        pair_logits = model.relation_logits(left, right)
        pair_bce = F.binary_cross_entropy_with_logits(pair_logits, batch["pair_label"])
    else:
        pair_bce = zero
    total = (
        head_ce
        + weights["supcon"] * supcon
        + weights["offset"] * offset
        + weights["transition"] * transition
        + weights["pair_bce"] * pair_bce
    )
    components = {
        "total": float(total.detach().cpu()),
        "head_ce": float(head_ce.detach().cpu()),
        **{
            f"{query}_ce": float(loss.detach().cpu())
            for query, loss in head_losses.items()
        },
        "supcon": float(supcon.detach().cpu()),
        "offset": float(offset.detach().cpu()),
        "B_transition": float(transition.detach().cpu()),
        "pair_bce": float(pair_bce.detach().cpu()),
        "random_grouping_active": float(random_grouping_audit is not None),
        "random_grouping_cross_source_transition_fraction": (
            float(random_grouping_audit["cross_source_transition_fraction"])
            if random_grouping_audit is not None else 0.0
        ),
        "random_grouping_mixed_structural_group_fraction": (
            float(random_grouping_audit["mixed_structural_group_fraction"])
            if random_grouping_audit is not None else 0.0
        ),
        "random_grouping_mean_unique_sources_per_structural_group": (
            float(random_grouping_audit["mean_unique_sources_per_structural_group"])
            if random_grouping_audit is not None else 0.0
        ),
    }
    return total, components


def train_one_epoch(
    model: FactorialAudioCnn,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    weights: dict[str, float],
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    model.train()
    totals: Counter[str] = Counter()
    batches = 0
    for batch_index, raw_batch in enumerate(loader):
        batch = move_training_batch(raw_batch, device)
        loss, components = compute_training_loss(
            model,
            batch,
            args,
            weights,
            epoch=epoch,
            batch_index=batch_index,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        totals.update(components)
        batches += 1
    if batches == 0:
        raise ValueError("training loader produced no batches")
    return {key: value / batches for key, value in totals.items()}


def evaluate_dev_head_ce(
    model: FactorialAudioCnn,
    groups: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    sums: dict[str, dict[str, Any]] = {}
    relation_sums: dict[tuple[str, int], dict[str, float | int]] = {}
    model.eval()
    with torch.no_grad():
        for crop_index in range(args.dev_crops):
            dataset = SourceGroupDataset(
                groups,
                args.sample_rate,
                args.clip_seconds,
                train=False,
                cell_policy=args.train_cell_policy,
                gain_policy=args.train_cell_policy,
                seed=args.seed,
                crop_index=crop_index,
                num_crops=args.dev_crops,
            )
            loader = DataLoader(
                dataset,
                batch_size=args.eval_group_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
                collate_fn=collate_source_groups,
            )
            for raw_batch in loader:
                batch = move_training_batch(raw_batch, device)
                output = forward_group(model, batch["waveform"])
                states = batch["waveform"].shape[1]
                class_counts = {
                    "O": len(ORIGINS), "M": len(DECODERS), "C": len(CHANNELS)
                }
                probabilities = {
                    query: torch.softmax(
                        output[f"{query}_logits"].reshape(
                            -1, states, class_counts[query]
                        ),
                        dim=-1,
                    ).cpu().numpy()
                    for query in ACTIVE_SINGLE_CLIP_QUERIES
                }
                if model.pair_head is not None:
                    left = gather_pairs(output["embedding_grouped"], batch["pair_index_a"])
                    right = gather_pairs(output["embedding_grouped"], batch["pair_index_b"])
                    relation_probabilities = torch.sigmoid(
                        model.relation_logits(left, right)
                    ).cpu().numpy()
                    relation_labels = raw_batch["pair_label"].cpu().numpy()
                    for group_index, source_key in enumerate(raw_batch["source_key"]):
                        for pair_index in range(relation_probabilities.shape[1]):
                            key = (str(source_key), pair_index)
                            item = relation_sums.setdefault(
                                key,
                                {
                                    "probability": 0.0,
                                    "label": int(relation_labels[group_index, pair_index]),
                                },
                            )
                            if int(item["label"]) != int(
                                relation_labels[group_index, pair_index]
                            ):
                                raise AssertionError("relation label changed across dev crops")
                            item["probability"] = float(item["probability"]) + float(
                                relation_probabilities[group_index, pair_index]
                            )
                for group_index, sample_ids in enumerate(raw_batch["sample_ids"]):
                    for state_index, sample_id in enumerate(sample_ids):
                        item = sums.setdefault(
                            sample_id,
                            {
                                **{
                                    f"{query}_probs": np.zeros(
                                        class_counts[query], dtype=np.float64
                                    )
                                    for query in ACTIVE_SINGLE_CLIP_QUERIES
                                },
                                **{
                                    query: int(
                                        raw_batch[query][group_index, state_index]
                                    )
                                    for query in ACTIVE_SINGLE_CLIP_QUERIES
                                },
                            },
                        )
                        for query in ACTIVE_SINGLE_CLIP_QUERIES:
                            item[f"{query}_probs"] += probabilities[query][
                                group_index, state_index
                            ]
    if not sums:
        raise ValueError("development set produced no predictions")
    ce: dict[str, float] = {}
    class_counts = {"O": len(ORIGINS), "M": len(DECODERS), "C": len(CHANNELS)}
    for query in ACTIVE_SINGLE_CLIP_QUERIES:
        classes = class_counts[query]
        probabilities = np.stack(
            [item[f"{query}_probs"] / args.dev_crops for item in sums.values()]
        )
        labels = np.asarray([item[query] for item in sums.values()], dtype=np.int64)
        if probabilities.shape[1] != classes:
            raise AssertionError("head class count changed unexpectedly")
        ce[f"{query}_ce"] = float(
            -np.log(np.maximum(probabilities[np.arange(labels.size), labels], 1e-12)).mean()
        )
    metric_name = "mean_OMC_ce" if "O" in ACTIVE_SINGLE_CLIP_QUERIES else "mean_MC_ce"
    ce[metric_name] = float(
        np.mean([ce[f"{query}_ce"] for query in ACTIVE_SINGLE_CLIP_QUERIES])
    )
    ce["mean_active_head_ce"] = ce[metric_name]
    if relation_sums:
        relation_probability = np.asarray(
            [float(item["probability"]) / args.dev_crops for item in relation_sums.values()],
            dtype=np.float64,
        )
        relation_label = np.asarray(
            [int(item["label"]) for item in relation_sums.values()], dtype=np.float64
        )
        ce["R_ce"] = float(
            -np.mean(
                relation_label * np.log(np.maximum(relation_probability, 1e-12))
                + (1.0 - relation_label)
                * np.log(np.maximum(1.0 - relation_probability, 1e-12))
            )
        )
    selection_queries = native_training_queries(args.variant)
    missing_selection = [query for query in selection_queries if f"{query}_ce" not in ce]
    if missing_selection:
        raise AssertionError(f"missing development selection CE for {missing_selection}")
    ce["selection_queries"] = list(selection_queries)
    ce["selection_head_ce"] = float(
        np.mean([ce[f"{query}_ce"] for query in selection_queries])
    )
    ce["active_single_clip_queries"] = list(ACTIVE_SINGLE_CLIP_QUERIES)
    ce["selection_scope"] = args.train_cell_policy
    ce["calibration"] = "none"
    ce["crop_aggregation"] = "mean probability over fixed start/middle/end fractions"
    return ce


def train_and_select(
    model: FactorialAudioCnn,
    train_groups: list[dict[str, Any]],
    dev_groups: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if args.variant == "far_random_grouping":
        random_grouping_batch_contract(len(train_groups), args.group_batch_size)
    dataset = SourceGroupDataset(
        train_groups,
        args.sample_rate,
        args.clip_seconds,
        train=True,
        cell_policy=args.train_cell_policy,
        gain_policy=args.train_cell_policy,
        seed=args.seed,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.group_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_source_groups,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    weights = variant_weights(args.variant, args)
    history: list[dict[str, Any]] = []
    best_ce = math.inf
    best_epoch = -1
    stale_epochs = 0
    checkpoint_path = output_dir / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        dataset.set_epoch(epoch)
        training = train_one_epoch(
            model, loader, optimizer, args, weights, device, epoch
        )
        development = evaluate_dev_head_ce(model, dev_groups, args, device)
        improved = development["selection_head_ce"] < best_ce - args.min_delta
        if improved:
            best_ce = float(development["selection_head_ce"])
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "variant": args.variant,
                    "epoch": epoch,
                    "dev_selection": development,
                    "native_training_queries": list(
                        native_training_queries(args.variant)
                    ),
                    "single_clip_training_queries": list(
                        single_clip_training_queries(args.variant)
                    ),
                    "class_contract": {
                        query: list(
                            {"O": ORIGINS, "M": DECODERS, "C": CHANNELS}[query]
                        )
                        for query in ACTIVE_SINGLE_CLIP_QUERIES
                    },
                    "structural_latent": {"B": list(BANDWIDTHS), "has_single_clip_head": False},
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
        row = {
            "epoch": epoch,
            "train": training,
            "dev": development,
            "selected": improved,
            "stale_epochs": stale_epochs,
        }
        history.append(row)
        print(json.dumps({"event": "epoch", **row}, ensure_ascii=False), flush=True)
        if stale_epochs >= args.patience:
            break
    if best_epoch < 0:
        raise RuntimeError("no checkpoint was selected")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    selection = {
        "criterion": (
            "uncalibrated mean "
            + "/".join(native_training_queries(args.variant))
            + " cross-entropy"
        ),
        "scope": f"dev/{args.train_cell_policy}",
        "direction": "minimize",
        "best_epoch": best_epoch,
        "best_value": best_ce,
        "exploratory_mxc_segment_included": args.train_cell_policy == "full",
        "sealed_unseen_composition_used_for_selection": False,
    }
    return history, selection


def export_frozen_embeddings(
    model: FactorialAudioCnn,
    groups_by_split: dict[str, list[dict[str, Any]]],
    sample_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
    output_path: Path,
) -> dict[str, Any]:
    accumulators: dict[str, dict[str, Any]] = {}
    model.eval()
    with torch.no_grad():
        for split in SPLITS:
            groups = groups_by_split[split]
            for crop_index in range(args.final_eval_crops):
                dataset = SourceGroupDataset(
                    groups,
                    args.sample_rate,
                    args.clip_seconds,
                    train=False,
                    cell_policy="full",
                    gain_policy=args.train_cell_policy,
                    seed=args.seed,
                    crop_index=crop_index,
                    num_crops=args.final_eval_crops,
                )
                loader = DataLoader(
                    dataset,
                    batch_size=args.eval_group_batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=device.type == "cuda",
                    collate_fn=collate_source_groups,
                )
                for raw_batch in loader:
                    batch = move_training_batch(raw_batch, device)
                    output = forward_group(model, batch["waveform"])
                    states = batch["waveform"].shape[1]
                    embedding = output["embedding_grouped"].cpu().numpy()
                    class_counts = {
                        "O": len(ORIGINS), "M": len(DECODERS), "C": len(CHANNELS)
                    }
                    logits = {
                        query: output[f"{query}_logits"].reshape(
                            -1, states, class_counts[query]
                        ).cpu().numpy()
                        for query in ACTIVE_SINGLE_CLIP_QUERIES
                    }
                    probabilities = {
                        query: torch.softmax(torch.from_numpy(value), dim=-1).numpy()
                        for query, value in logits.items()
                    }
                    for group_index, sample_ids in enumerate(raw_batch["sample_ids"]):
                        for state_index, sample_id in enumerate(sample_ids):
                            item = accumulators.setdefault(
                                sample_id,
                                {
                                    "embedding": np.zeros(embedding.shape[-1], dtype=np.float64),
                                    **{
                                        f"{query}_logits": np.zeros(
                                            class_counts[query], dtype=np.float64
                                        )
                                        for query in ACTIVE_SINGLE_CLIP_QUERIES
                                    },
                                    **{
                                        f"{query}_probs": np.zeros(
                                            class_counts[query], dtype=np.float64
                                        )
                                        for query in ACTIVE_SINGLE_CLIP_QUERIES
                                    },
                                },
                            )
                            item["embedding"] += embedding[group_index, state_index]
                            for query in ACTIVE_SINGLE_CLIP_QUERIES:
                                item[f"{query}_logits"] += logits[query][group_index, state_index]
                                item[f"{query}_probs"] += probabilities[query][group_index, state_index]
    row_by_id = {str(row["sample_id"]): row for row in sample_rows}
    ordered_ids = sorted(accumulators, key=lambda sample_id: (SPLITS.index(str(row_by_id[sample_id]["split"])), sample_id))
    embeddings = np.stack(
        [accumulators[sample_id]["embedding"] / args.final_eval_crops for sample_id in ordered_ids]
    ).astype(np.float32)
    embedding_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(embedding_norms, 1e-12)
    rows = [row_by_id[sample_id] for sample_id in ordered_ids]
    arrays: dict[str, np.ndarray] = {
        "sample_ids": np.asarray(ordered_ids),
        "embeddings": embeddings,
        "embeddings_normalized": normalized.astype(np.float32),
        "splits": np.asarray([row["split"] for row in rows]),
        "partitions": np.asarray([row["cps_factorial"]["combination_partition"] for row in rows]),
        "training_membership": np.asarray([
            "included"
            if args.train_cell_policy == "full"
            or row["cps_factorial"]["combination_partition"] == SEEN_PARTITION
            else "excluded_exploratory_holdout"
            for row in rows
        ]),
        "train_cell_policy": np.asarray(args.train_cell_policy),
        "source_keys": np.asarray([row["source_key"] for row in rows]),
        "origins": np.asarray([row["cps_factorial"]["origin"] for row in rows]),
        "bandwidths_kbps": np.asarray([row["cps_factorial"]["bandwidth_kbps"] for row in rows], dtype=np.float32),
        "decoders": np.asarray([row["cps_factorial"]["decoder"] for row in rows]),
        "channels": np.asarray([row["cps_factorial"]["channel"] for row in rows]),
        "M_labels": np.asarray([row["labels"]["sa_decoder"] for row in rows], dtype=np.int64),
        "C_labels": np.asarray([row["labels"]["pi_channel"] for row in rows], dtype=np.int64),
    }
    if "O" in ACTIVE_SINGLE_CLIP_QUERIES:
        arrays["O_labels"] = np.asarray(
            [row["labels"]["ad_origin"] for row in rows], dtype=np.int64
        )
    for query in ACTIVE_SINGLE_CLIP_QUERIES:
        arrays[f"{query}_logits"] = np.stack(
            [accumulators[sample_id][f"{query}_logits"] / args.final_eval_crops for sample_id in ordered_ids]
        ).astype(np.float32)
        arrays[f"{query}_probs"] = np.stack(
            [accumulators[sample_id][f"{query}_probs"] / args.final_eval_crops for sample_id in ordered_ids]
        ).astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    return {
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "samples": len(ordered_ids),
        "embedding_dim": int(embeddings.shape[1]),
        "splits": dict(sorted(Counter(arrays["splits"].tolist()).items())),
        "partitions": dict(sorted(Counter(arrays["partitions"].tolist()).items())),
        "crops": args.final_eval_crops,
        "duration_control": (
            f"all {len(ALL_STATES)} cells trimmed to the source-common shortest support before crop"
        ),
        "active_single_clip_queries": list(ACTIVE_SINGLE_CLIP_QUERIES),
        "gain_reference": f"{args.train_cell_policy} cells; one gain per source/crop",
        "contains_heldout": True,
    }


def export_frozen_layer_embeddings(
    model: FactorialAudioCnn,
    groups_by_split: dict[str, list[dict[str, Any]]],
    sample_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
    output_path: Path,
) -> dict[str, Any]:
    """Export stem/block/projection representations into a separate NPZ.

    This is deliberately independent of ``frozen_embeddings_all_splits.npz``:
    enabling the audit neither changes the ordinary model forward contract nor
    replaces any registered training or final-probe artifact.
    """
    accumulators: dict[str, dict[str, np.ndarray]] = {}
    model.eval()
    with torch.no_grad():
        for split in SPLITS:
            for crop_index in range(args.final_eval_crops):
                dataset = SourceGroupDataset(
                    groups_by_split[split],
                    args.sample_rate,
                    args.clip_seconds,
                    train=False,
                    cell_policy="full",
                    gain_policy=args.train_cell_policy,
                    seed=args.seed,
                    crop_index=crop_index,
                    num_crops=args.final_eval_crops,
                )
                loader = DataLoader(
                    dataset,
                    batch_size=args.eval_group_batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=device.type == "cuda",
                    collate_fn=collate_source_groups,
                )
                for raw_batch in loader:
                    batch = move_training_batch(raw_batch, device)
                    batch_size, states = batch["waveform"].shape[:2]
                    flat_waveform = batch["waveform"].flatten(0, 1)
                    layer_batch = {
                        name: values.reshape(batch_size, states, -1).cpu().numpy()
                        for name, values in forward_layer_embeddings(
                            model, flat_waveform
                        ).items()
                    }
                    for group_index, sample_ids in enumerate(raw_batch["sample_ids"]):
                        for state_index, sample_id in enumerate(sample_ids):
                            item = accumulators.setdefault(sample_id, {})
                            for name, values in layer_batch.items():
                                if name not in item:
                                    item[name] = np.zeros(
                                        values.shape[-1], dtype=np.float64
                                    )
                                item[name] += values[group_index, state_index]

    row_by_id = {str(row["sample_id"]): row for row in sample_rows}
    missing = sorted(set(row_by_id) - set(accumulators))
    unexpected = sorted(set(accumulators) - set(row_by_id))
    if missing or unexpected:
        raise AssertionError(
            f"layer export sample mismatch: missing={missing[:3]}, "
            f"unexpected={unexpected[:3]}"
        )
    ordered_ids = sorted(
        accumulators,
        key=lambda sample_id: (
            SPLITS.index(str(row_by_id[sample_id]["split"])),
            sample_id,
        ),
    )
    rows = [row_by_id[sample_id] for sample_id in ordered_ids]
    layer_registry: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for index, name in enumerate(LAYER_EMBEDDING_NAMES):
        array_key = f"layer_{index:02d}"
        features = np.stack(
            [
                accumulators[sample_id][name] / args.final_eval_crops
                for sample_id in ordered_ids
            ]
        ).astype(np.float32)
        arrays[array_key] = features
        layer_registry.append(
            {
                "index": index,
                "name": name,
                "array_key": array_key,
                "dimension": int(features.shape[1]),
                "pooling": (
                    "model projection output"
                    if name == "projection"
                    else "global adaptive average concatenated with global adaptive maximum"
                ),
            }
        )

    arrays.update(
        {
            "format_version": np.asarray("cps_frozen_layer_embeddings_v1"),
            "sample_ids": np.asarray(ordered_ids),
            "splits": np.asarray([row["split"] for row in rows]),
            "partitions": np.asarray(
                [
                    row["cps_factorial"]["combination_partition"]
                    for row in rows
                ]
            ),
            "source_keys": np.asarray([row["source_key"] for row in rows]),
            "speaker_ids": np.asarray(
                [str(row.get("speaker_id", "")) for row in rows]
            ),
            "origins": np.asarray(
                [row["cps_factorial"]["origin"] for row in rows]
            ),
            "bandwidths_kbps": np.asarray(
                [row["cps_factorial"]["bandwidth_kbps"] for row in rows],
                dtype=np.float32,
            ),
            "decoders": np.asarray(
                [row["cps_factorial"]["decoder"] for row in rows]
            ),
            "channels": np.asarray(
                [row["cps_factorial"]["channel"] for row in rows]
            ),
            "M_labels": np.asarray(
                [row["labels"]["sa_decoder"] for row in rows], dtype=np.int64
            ),
            "C_labels": np.asarray(
                [row["labels"]["pi_channel"] for row in rows], dtype=np.int64
            ),
            "variant": np.asarray(args.variant),
            "native_training_queries_json": np.asarray(
                json.dumps(list(native_training_queries(args.variant)))
            ),
            "train_cell_policy": np.asarray(args.train_cell_policy),
            "layer_registry_json": np.asarray(
                json.dumps(layer_registry, sort_keys=True, separators=(",", ":"))
            ),
            "pooling_contract": np.asarray(
                "stem/residual: global_avg+global_max; projection: ordinary embedding"
            ),
            "sample_manifest_sha256": np.asarray(
                sha256_file(args.sample_manifest)
            ),
            "pair_manifest_sha256": np.asarray(sha256_file(args.pair_manifest)),
            "parameter_schema_sha256": np.asarray(
                parameter_schema_sha256(parameter_schema(model))
            ),
            "final_eval_crops": np.asarray(args.final_eval_crops, dtype=np.int64),
            "seed": np.asarray(args.seed, dtype=np.int64),
        }
    )
    if "O" in ACTIVE_SINGLE_CLIP_QUERIES:
        arrays["O_labels"] = np.asarray(
            [row["labels"]["ad_origin"] for row in rows], dtype=np.int64
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    return {
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "format_version": "cps_frozen_layer_embeddings_v1",
        "samples": len(ordered_ids),
        "layers": layer_registry,
        "crops": args.final_eval_crops,
        "pooling_contract": arrays["pooling_contract"].item(),
        "variant": args.variant,
        "native_training_queries": list(native_training_queries(args.variant)),
        "sample_manifest_sha256": arrays["sample_manifest_sha256"].item(),
        "pair_manifest_sha256": arrays["pair_manifest_sha256"].item(),
        "default_forward_contract_changed": False,
        "training_or_checkpoint_selection_changed": False,
    }


def symmetric_pair_features_numpy(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left / np.maximum(np.linalg.norm(left, axis=-1, keepdims=True), 1e-12)
    right = right / np.maximum(np.linalg.norm(right, axis=-1, keepdims=True), 1e-12)
    cosine = np.sum(left * right, axis=-1, keepdims=True)
    return np.concatenate([np.abs(left - right), left * right, cosine], axis=-1)


def fit_fixed_linear_probe(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    regularization_c: float,
    max_iter: int,
) -> tuple[StandardScaler, LogisticRegression]:
    if np.unique(labels).size < 2:
        raise ValueError("a linear probe requires at least two training classes")
    scaler = StandardScaler().fit(features)
    classifier = LogisticRegression(
        C=regularization_c,
        max_iter=max_iter,
        solver="lbfgs",
        random_state=seed,
    ).fit(scaler.transform(features), labels)
    return scaler, classifier


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> dict[str, Any]:
    labels = np.asarray(labels)
    classes = np.asarray(classes)
    prediction = classes[np.argmax(probabilities, axis=1)]
    metrics: dict[str, Any] = {
        "n": int(labels.size),
        "support": {str(value): int(np.sum(labels == value)) for value in classes},
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, labels=classes, average="macro", zero_division=0)),
        "cross_entropy": float(log_loss(labels, probabilities, labels=classes)),
    }
    try:
        if classes.size == 2:
            positive_index = int(np.flatnonzero(classes == 1)[0]) if 1 in classes else 1
            metrics["auroc"] = float(roc_auc_score(labels, probabilities[:, positive_index]))
        else:
            metrics["auroc"] = float(
                roc_auc_score(labels, probabilities, labels=classes, multi_class="ovr", average="macro")
            )
    except ValueError:
        metrics["auroc"] = None
    return metrics


def probe_parameters(scaler: StandardScaler, classifier: LogisticRegression) -> dict[str, Any]:
    return {
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "classes": classifier.classes_.tolist(),
        "coefficient": classifier.coef_.tolist(),
        "intercept": classifier.intercept_.tolist(),
    }


def evaluate_probe_segments(
    features: np.ndarray,
    labels: np.ndarray,
    splits: np.ndarray,
    partitions: np.ndarray,
    scaler: StandardScaler,
    classifier: LogisticRegression,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in SPLITS:
        result[split] = {}
        for partition in ("all_in_grid", SEEN_PARTITION, HELDOUT_PARTITION):
            mask = splits == split
            if partition != "all_in_grid":
                mask &= partitions == partition
            if not mask.any():
                continue
            probability = classifier.predict_proba(scaler.transform(features[mask]))
            result[split][partition] = classification_metrics(labels[mask], probability, classifier.classes_)
    return result


def pair_head_segments(
    features: np.ndarray,
    labels: np.ndarray,
    splits: np.ndarray,
    partitions: np.ndarray,
    model: FactorialAudioCnn,
) -> dict[str, Any] | None:
    if model.pair_head is None:
        return None
    weight = model.pair_head.weight.detach().cpu().numpy().reshape(-1)
    bias = float(model.pair_head.bias.detach().cpu())
    probability_one = 1.0 / (1.0 + np.exp(-(features @ weight + bias)))
    probabilities = np.column_stack([1.0 - probability_one, probability_one])
    result: dict[str, Any] = {}
    for split in SPLITS:
        result[split] = {}
        for partition in ("all_in_grid", SEEN_PARTITION, HELDOUT_PARTITION):
            mask = splits == split
            if partition != "all_in_grid":
                mask &= partitions == partition
            if mask.any():
                result[split][partition] = classification_metrics(
                    labels[mask], probabilities[mask], np.asarray([0, 1])
                )
    return result


def run_fixed_linear_probes(
    embedding_path: Path,
    pair_rows: list[dict[str, Any]],
    model: FactorialAudioCnn,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with np.load(embedding_path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    embeddings = data["embeddings"].astype(np.float64)
    splits = data["splits"]
    partitions = data["partitions"]
    probe_fit_scope = "train/all_in_grid" if args.train_cell_policy == "full" else "train/seen_mxc"
    probe_fit_mask = splits == "train"
    if args.train_cell_policy == SEEN_PARTITION:
        probe_fit_mask &= partitions == SEEN_PARTITION
    results: dict[str, Any] = {
        "protocol": {
            "probe": "fixed linear logistic regression",
            "regularization_C": args.probe_c,
            "max_iter": args.probe_max_iter,
            "scaler": f"StandardScaler fit on {probe_fit_scope} only",
            "hyperparameter_tuning": False,
            "exploratory_mxc_segment_included_for_fit": args.train_cell_policy == "full",
            "sealed_unseen_composition_used_for_fit_or_tuning": False,
        },
        "queries": {},
    }
    parameters: dict[str, Any] = {}
    label_keys = {
        query: f"{query}_labels" for query in ACTIVE_SINGLE_CLIP_QUERIES
    }
    for query, label_key in label_keys.items():
        labels = data[label_key].astype(np.int64)
        scaler, classifier = fit_fixed_linear_probe(
            embeddings[probe_fit_mask],
            labels[probe_fit_mask],
            seed=args.seed,
            regularization_c=args.probe_c,
            max_iter=args.probe_max_iter,
        )
        results["queries"][query] = {
            "fit_n": int(probe_fit_mask.sum()),
            "fit_scope": probe_fit_scope,
            "feature": "frozen single-clip embedding",
            "segments": evaluate_probe_segments(
                embeddings, labels, splits, partitions, scaler, classifier
            ),
        }
        parameters[query] = probe_parameters(scaler, classifier)

    embedding_by_id = {
        str(sample_id): embeddings[index]
        for index, sample_id in enumerate(data["sample_ids"].tolist())
    }
    usable_pairs = [
        row for row in pair_rows
        if str(row["sample_id_a"]) in embedding_by_id and str(row["sample_id_b"]) in embedding_by_id
    ]
    pair_features = symmetric_pair_features_numpy(
        np.stack([embedding_by_id[str(row["sample_id_a"])] for row in usable_pairs]),
        np.stack([embedding_by_id[str(row["sample_id_b"])] for row in usable_pairs]),
    )
    pair_labels = np.asarray([int(row["same_token_label"]) for row in usable_pairs], dtype=np.int64)
    pair_splits = np.asarray([row["split"] for row in usable_pairs])
    pair_partitions = np.asarray([row["combination_partition"] for row in usable_pairs])
    pair_fit_mask = pair_splits == "train"
    if args.train_cell_policy == SEEN_PARTITION:
        pair_fit_mask &= pair_partitions == SEEN_PARTITION
    scaler, classifier = fit_fixed_linear_probe(
        pair_features[pair_fit_mask],
        pair_labels[pair_fit_mask],
        seed=args.seed,
        regularization_c=args.probe_c,
        max_iter=args.probe_max_iter,
    )
    results["queries"]["R"] = {
        "fit_n": int(pair_fit_mask.sum()),
        "fit_scope": probe_fit_scope,
        "feature": "[abs(l2(z_a)-l2(z_b)), l2(z_a)*l2(z_b), cosine]",
        "symmetric": True,
        "segments": evaluate_probe_segments(
            pair_features,
            pair_labels,
            pair_splits,
            pair_partitions,
            scaler,
            classifier,
        ),
    }
    direct_pair_head = pair_head_segments(
        pair_features, pair_labels, pair_splits, pair_partitions, model
    )
    if direct_pair_head is not None:
        results["matched_information_pair_head"] = {
            "feature": results["queries"]["R"]["feature"],
            "segments": direct_pair_head,
            "checkpoint_selection_used_pair_metric": False,
        }
    parameters["R"] = probe_parameters(scaler, classifier)
    return results, parameters


def write_probe_csv(path: Path, probe_results: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "query", "split", "partition", "n", "accuracy", "balanced_accuracy", "macro_f1", "cross_entropy", "auroc"
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for query, query_payload in probe_results["queries"].items():
            for split, split_payload in query_payload["segments"].items():
                for partition, metrics in split_payload.items():
                    writer.writerow(
                        {
                            "query": query,
                            "split": split,
                            "partition": partition,
                            **{key: metrics.get(key) for key in fields if key not in {"query", "split", "partition"}},
                        }
                    )


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return torch.device(name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument(
        "--train-cell-policy",
        choices=CELL_POLICIES,
        default="full",
        help="full is the registered primary; seen_mxc is an exploratory combination holdout",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--clip-seconds", type=float, default=4.0)
    parser.add_argument("--n-mels", type=int, default=96)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--group-batch-size", type=int, default=4)
    parser.add_argument("--eval-group-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dev-crops", type=int, default=3)
    parser.add_argument("--final-eval-crops", type=int, default=3)
    parser.add_argument("--supcon-weight", type=float, default=0.1)
    parser.add_argument("--offset-weight", type=float, default=0.2)
    parser.add_argument("--transition-weight", type=float, default=0.2)
    parser.add_argument("--pair-bce-weight", type=float, default=1.0)
    parser.add_argument("--supcon-temperature", type=float, default=0.1)
    parser.add_argument("--offset-temperature", type=float, default=0.1)
    parser.add_argument("--transition-temperature", type=float, default=0.1)
    parser.add_argument("--probe-c", type=float, default=1.0)
    parser.add_argument("--probe-max-iter", type=int, default=2000)
    parser.add_argument(
        "--export-layer-embeddings",
        action="store_true",
        help=(
            "after checkpoint selection, additionally export fixed pooled stem, "
            "seven residual-stage, and projection representations to a separate NPZ"
        ),
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {args.output_dir}; pass --overwrite to reuse it")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    sample_rows = read_jsonl(args.sample_manifest)
    pair_rows = read_jsonl(args.pair_manifest)
    audit = validate_manifests(sample_rows, pair_rows)
    write_json_atomic(args.output_dir / "manifest_audit.json", audit)
    write_json_atomic(args.output_dir / "run_config.json", serializable_args(args))
    model = build_model(args, args.variant)
    variant_audit = variant_capacity_audit(model, args.variant, args)
    if args.variant == "far_random_grouping":
        train_source_groups = len(
            {
                str(row["source_key"])
                for row in sample_rows
                if str(row["split"]) == "train"
            }
        )
        variant_audit["random_grouping"]["training_batch_contract"] = (
            random_grouping_batch_contract(train_source_groups, args.group_batch_size)
        )
    write_json_atomic(args.output_dir / "variant_audit.json", variant_audit)
    if args.validate_only:
        print(
            json.dumps(
                {"event": "validated", **audit, "variant_audit": variant_audit},
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return 0

    device = resolve_device(args.device)
    groups_by_split = {
        split: build_source_groups(sample_rows, pair_rows, split)
        for split in SPLITS
    }
    model = model.to(device)
    history, selection = train_and_select(
        model,
        groups_by_split["train"],
        groups_by_split["dev"],
        args,
        device,
        args.output_dir,
    )
    write_json_atomic(args.output_dir / "training_history.json", history)
    embedding_path = args.output_dir / "frozen_embeddings_all_splits.npz"
    export = export_frozen_embeddings(
        model,
        groups_by_split,
        sample_rows,
        args,
        device,
        embedding_path,
    )
    layer_export = None
    if args.export_layer_embeddings:
        layer_export = export_frozen_layer_embeddings(
            model,
            groups_by_split,
            sample_rows,
            args,
            device,
            args.output_dir / "frozen_layer_embeddings_all_splits.npz",
        )
    probe_results, probe_parameters_payload = run_fixed_linear_probes(
        embedding_path, pair_rows, model, args
    )
    write_json_atomic(args.output_dir / "probe_results.json", probe_results)
    write_json_atomic(args.output_dir / "probe_parameters.json", probe_parameters_payload)
    write_probe_csv(args.output_dir / "probe_results.csv", probe_results)
    summary = {
        "status": "complete",
        "variant": args.variant,
        "device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "variant_audit": variant_audit,
        "manifest_audit": audit,
        "selection": selection,
        "embedding_export": export,
        "probe_results": str(args.output_dir / "probe_results.json"),
        "contracts": {
            "single_clip_heads": list(ACTIVE_SINGLE_CLIP_QUERIES),
            "origin_role": (
                "query" if "O" in ACTIVE_SINGLE_CLIP_QUERIES else "fixed_nonquery"
            ),
            "B_single_clip_head": False,
            "train_cell_policy": args.train_cell_policy,
            "primary_policy": "full",
            "seen_mxc_policy_role": "exploratory combination holdout only",
            "sealed_unseen_composition_used_for_training_or_selection": False,
            "training_variant": args.variant,
            "native_training_queries": list(native_training_queries(args.variant)),
            "single_clip_training_queries": list(
                single_clip_training_queries(args.variant)
            ),
            "all_frozen_probe_queries": ["M", "C", "R"],
            "capacity_match_target": "far_full",
            "variant_audit": str(args.output_dir / "variant_audit.json"),
            "random_grouping_applied_to_dev_or_test": False,
            "R_feature": "[abs(l2 difference), product, cosine]",
            "probe_scaler_fit": (
                "train/all_in_grid only" if args.train_cell_policy == "full" else "train/seen_mxc only"
            ),
        },
    }
    if layer_export is not None:
        summary["layer_embedding_export"] = layer_export
    write_json_atomic(args.output_dir / "results.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
