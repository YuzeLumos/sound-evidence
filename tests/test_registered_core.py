from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import torch


from sound_evidence import _core as pilot
SCRIPT = Path(pilot.__file__)
REPO_ROOT = SCRIPT.parent


def make_complete_manifests() -> tuple[list[dict], list[dict]]:
    split = "train"
    source_key = "train:source-1"
    rows: list[dict] = []
    index: dict[tuple[str, float, str, str], dict] = {}
    for origin, bandwidth, decoder, channel in product(
        pilot.ORIGINS, pilot.BANDWIDTHS, pilot.DECODERS, pilot.CHANNELS
    ):
        state = (origin, bandwidth, decoder, channel)
        sample_id = f"sample:{origin}:{bandwidth:g}:{decoder}:{channel}"
        partition = pilot.expected_partition(state)
        row = {
            "sample_id": sample_id,
            "split": split,
            "source_key": source_key,
            "path": f"/{sample_id}.flac",
            "labels": {
                "ad_origin": pilot.ORIGIN_TO_ID[origin],
                "sa_decoder": pilot.DECODER_TO_ID[decoder],
                "pi_channel": pilot.CHANNEL_TO_ID[channel],
            },
            "cps_factorial": {
                "origin": origin,
                "bandwidth_kbps": bandwidth,
                "decoder": decoder,
                "channel": channel,
                "combination_partition": partition,
            },
        }
        rows.append(row)
        index[state] = row
    pairs: list[dict] = []
    patterns = ((6.0, 6.0, 1), (12.0, 12.0, 1), (6.0, 12.0, 0), (12.0, 6.0, 0))
    for origin, channel, (bandwidth_a, bandwidth_b, label) in product(
        pilot.ORIGINS, pilot.CHANNELS, patterns
    ):
        left = index[(origin, bandwidth_a, "encodec_native", channel)]
        right = index[(origin, bandwidth_b, "vocos_encodec", channel)]
        pairs.append(
            {
                "pair_id": f"pair:{origin}:{channel}:{bandwidth_a:g}:{bandwidth_b:g}",
                "split": split,
                "source_key_a": source_key,
                "source_key_b": source_key,
                "sample_id_a": left["sample_id"],
                "sample_id_b": right["sample_id"],
                "same_token_label": label,
                "origin_a": origin,
                "origin_b": origin,
                "channel_a": channel,
                "channel_b": channel,
                "decoder_a": "encodec_native",
                "decoder_b": "vocos_encodec",
                "bandwidth_a_kbps": bandwidth_a,
                "bandwidth_b_kbps": bandwidth_b,
                "combination_partition": (
                    pilot.HELDOUT_PARTITION if channel == "aac_64kbps" else pilot.SEEN_PARTITION
                ),
            }
        )
    return rows, pairs


def test_manifest_contract_has_20_seen_states_and_no_B_query() -> None:
    rows, pairs = make_complete_manifests()
    audit = pilot.validate_manifests(rows, pairs)
    groups = pilot.build_source_groups(rows, pairs, "train")
    assert audit["seen_states_per_source"] == 20
    assert audit["heldout_states_per_source"] == 4
    assert audit["seen_pairs_per_source"] == 16
    assert audit["heldout_pairs_per_source"] == 8
    assert len(groups) == 1


def test_common_trim_then_synchronous_crop_and_seen_only_gain() -> None:
    base = torch.arange(1.0, 11.0)
    waveforms = [base, 2.0 * base[:8], 100.0 * base[:9]]
    cropped, gain = pilot.synchronous_crop_and_group_gain(
        waveforms,
        [True, True, False],
        clip_samples=4,
        start=4,
    )
    # Common support is eight frames, so start four is the final legal crop.
    assert torch.equal(cropped[0] / gain, base[4:8])
    assert torch.equal(cropped[1] / gain, 2.0 * base[4:8])
    # The very loud held-out cell cannot determine shared gain.
    assert np.isclose(gain, 1.0 / 16.0)
    assert cropped[2].abs().max() > 1.0

    full_cropped, full_gain = pilot.synchronous_crop_and_group_gain(
        waveforms,
        [True, True, True],
        clip_samples=4,
        start=4,
    )
    assert np.isclose(full_gain, 1.0 / 800.0)
    assert np.isclose(float(full_cropped.abs().max()), 1.0)


def test_transition_prototypes_align_within_B_and_separate_between_B() -> None:
    # State order per group: B6/native, B6/Vocos, B12/native, B12/Vocos.
    separated = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        ],
        requires_grad=True,
    )
    collapsed = separated.detach().clone()
    collapsed[:, 3] = collapsed[:, 1]
    pair_a = torch.tensor([[0, 2], [0, 2]])
    pair_b = torch.tensor([[1, 3], [1, 3]])
    labels = torch.ones((2, 2))
    families = torch.tensor([[0, 1], [0, 1]])
    separated_loss = pilot.b_transition_prototype_loss(
        separated, pair_a, pair_b, labels, families, temperature=0.1
    )
    collapsed_loss = pilot.b_transition_prototype_loss(
        collapsed, pair_a, pair_b, labels, families, temperature=0.1
    )
    assert separated_loss < collapsed_loss
    separated_loss.backward()
    assert separated.grad is not None


def test_pair_features_are_symmetric_and_have_fixed_dimension() -> None:
    left = np.asarray([[1.0, 2.0, 3.0], [3.0, 1.0, -1.0]])
    right = np.asarray([[3.0, 2.0, 1.0], [1.0, -1.0, 2.0]])
    forward = pilot.symmetric_pair_features_numpy(left, right)
    reverse = pilot.symmetric_pair_features_numpy(right, left)
    assert forward.shape == (2, 2 * left.shape[1] + 1)
    np.testing.assert_allclose(forward, reverse)


def test_fixed_probe_scaler_is_fit_only_on_supplied_train_seen_rows() -> None:
    train_seen = np.asarray([[0.0, 10.0], [2.0, 12.0], [4.0, 14.0], [6.0, 16.0]])
    labels = np.asarray([0, 0, 1, 1])
    scaler, classifier = pilot.fit_fixed_linear_probe(
        train_seen,
        labels,
        seed=7,
        regularization_c=1.0,
        max_iter=200,
    )
    np.testing.assert_allclose(scaler.mean_, train_seen.mean(axis=0))
    assert set(classifier.classes_) == {0, 1}


def test_model_exposes_only_O_M_C_single_clip_heads_and_optional_pair_head() -> None:
    model = pilot.FactorialAudioCnn(
        sample_rate=8000,
        n_mels=16,
        n_fft=256,
        hop_length=64,
        embedding_dim=12,
        dropout=0.0,
        with_pair_head=False,
    ).eval()
    waveform = torch.randn(2, 2048)
    schema_before = pilot.parameter_schema(model)
    with torch.no_grad():
        output = model(waveform)
        layers = pilot.forward_layer_embeddings(model, waveform)
    assert set(output) == {"embedding", "O_logits", "M_logits", "C_logits"}
    assert tuple(layers) == pilot.LAYER_EMBEDDING_NAMES
    assert len(layers) == 9
    assert layers["stem"].shape == (2, 64)
    assert layers["residual_7"].shape == (2, 512)
    assert layers["projection"].shape == (2, 12)
    torch.testing.assert_close(layers["projection"], output["embedding"])
    assert pilot.parameter_schema(model) == schema_before
    assert model.pair_head is None
    assert not any("b_head" in name.lower() or "bandwidth" in name.lower() for name, _ in model.named_parameters())

    matched = pilot.FactorialAudioCnn(
        sample_rate=8000,
        n_mels=16,
        n_fft=256,
        hop_length=64,
        embedding_dim=12,
        dropout=0.0,
        with_pair_head=True,
    )
    left = torch.randn(5, 12)
    right = torch.randn(5, 12)
    torch.testing.assert_close(matched.relation_logits(left, right), matched.relation_logits(right, left))


def test_optional_layer_export_is_separate_and_keeps_projection_exact(
    tmp_path: Path, monkeypatch: object
) -> None:
    class StubSourceGroupDataset(torch.utils.data.Dataset):
        def __init__(self, groups: list[dict], *_args: object, **_kwargs: object) -> None:
            self.groups = groups

        def __len__(self) -> int:
            return len(self.groups)

        def __getitem__(self, index: int) -> dict:
            sample_ids = self.groups[index]["sample_ids"]
            time = torch.linspace(0.0, 1.0, 1024)
            waveform = torch.stack(
                [torch.sin((state + 1) * 7.0 * time) for state in range(len(sample_ids))]
            )
            state_count = len(sample_ids)
            empty_long = torch.empty(0, dtype=torch.long)
            return {
                "waveform": waveform,
                "O": torch.zeros(state_count, dtype=torch.long),
                "M": torch.arange(state_count, dtype=torch.long) % 2,
                "C": torch.arange(state_count, dtype=torch.long) % 3,
                "B": torch.zeros(state_count, dtype=torch.long),
                "state": torch.arange(state_count, dtype=torch.long),
                "pair_index_a": empty_long,
                "pair_index_b": empty_long,
                "pair_label": torch.empty(0, dtype=torch.float32),
                "pair_bandwidth_a": empty_long,
                "source_key": self.groups[index]["source_key"],
                "sample_ids": sample_ids,
                "gain": 1.0,
                "start": 0,
                "common_samples": 1024,
            }

    monkeypatch.setattr(pilot, "SourceGroupDataset", StubSourceGroupDataset)
    sample_rows: list[dict] = []
    groups_by_split: dict[str, list[dict]] = {}
    for split in pilot.SPLITS:
        sample_ids = [f"{split}:sample-{index}" for index in range(2)]
        groups_by_split[split] = [
            {"source_key": f"{split}:source", "sample_ids": sample_ids}
        ]
        for index, sample_id in enumerate(sample_ids):
            sample_rows.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "source_key": f"{split}:source",
                    "speaker_id": f"{split}:speaker",
                    "labels": {
                        "ad_origin": 0,
                        "sa_decoder": index % 2,
                        "pi_channel": index % 3,
                    },
                    "cps_factorial": {
                        "origin": "original",
                        "bandwidth_kbps": 6.0,
                        "decoder": "encodec_native",
                        "channel": "none",
                        "combination_partition": pilot.SEEN_PARTITION,
                    },
                }
            )
    sample_manifest = tmp_path / "samples.jsonl"
    pair_manifest = tmp_path / "pairs.jsonl"
    sample_manifest.write_text("sample registry\n", encoding="utf-8")
    pair_manifest.write_text("pair registry\n", encoding="utf-8")
    args = argparse.Namespace(
        sample_rate=8000,
        clip_seconds=0.128,
        train_cell_policy="full",
        seed=7,
        final_eval_crops=1,
        eval_group_batch_size=1,
        num_workers=0,
        variant="ce_M_only",
        sample_manifest=sample_manifest,
        pair_manifest=pair_manifest,
    )
    model = pilot.FactorialAudioCnn(
        sample_rate=8000,
        n_mels=16,
        n_fft=128,
        hop_length=32,
        embedding_dim=12,
        dropout=0.0,
    ).eval()
    schema_before = pilot.parameter_schema(model)
    output_path = tmp_path / "frozen_layer_embeddings_all_splits.npz"
    summary = pilot.export_frozen_layer_embeddings(
        model,
        groups_by_split,
        sample_rows,
        args,
        torch.device("cpu"),
        output_path,
    )
    assert output_path.is_file()
    assert summary["default_forward_contract_changed"] is False
    assert summary["training_or_checkpoint_selection_changed"] is False
    assert len(summary["layers"]) == 9
    assert pilot.parameter_schema(model) == schema_before
    time = torch.linspace(0.0, 1.0, 1024)
    waveform = torch.stack([torch.sin((state + 1) * 7.0 * time) for state in range(2)])
    with torch.no_grad():
        expected_projection = pilot.forward_layer_embeddings(model, waveform)[
            "projection"
        ].numpy()
    with np.load(output_path, allow_pickle=False) as archive:
        registry = json.loads(str(archive["layer_registry_json"].item()))
        assert [row["name"] for row in registry] == list(pilot.LAYER_EMBEDDING_NAMES)
        assert archive["sample_ids"].shape == (6,)
        projection_key = registry[-1]["array_key"]
        assert archive[projection_key].shape == (6, 12)
        np.testing.assert_allclose(
            archive[projection_key],
            np.concatenate([expected_projection] * 3),
            atol=1e-6,
            rtol=1e-6,
        )


def test_variant_registry_includes_matched_information_control() -> None:
    args = argparse.Namespace(
        supcon_weight=0.1,
        offset_weight=0.2,
        transition_weight=0.3,
        pair_bce_weight=1.0,
    )
    weights = pilot.variant_weights("ce_omc_pair_bce", args)
    assert weights == {"supcon": 0.0, "offset": 0.0, "transition": 0.0, "pair_bce": 1.0}
    assert "ce_omc_pair_bce" in pilot.VARIANTS


def variant_args(variant: str) -> argparse.Namespace:
    return argparse.Namespace(
        variant=variant,
        sample_rate=8000,
        n_mels=16,
        n_fft=256,
        hop_length=64,
        embedding_dim=12,
        dropout=0.0,
        supcon_weight=0.1,
        offset_weight=0.2,
        transition_weight=0.3,
        pair_bce_weight=1.0,
        supcon_temperature=0.1,
        offset_temperature=0.1,
        transition_temperature=0.1,
        seed=20260714,
    )


def test_capacity_matched_ce_is_exact_far_architecture_with_no_extra_labels() -> None:
    args = variant_args("ce_capacity_matched")
    model = pilot.build_model(args, args.variant)
    audit = pilot.variant_capacity_audit(model, args.variant, args)
    assert pilot.variant_weights(args.variant, args) == {
        "supcon": 0.0,
        "offset": 0.0,
        "transition": 0.0,
        "pair_bce": 0.0,
    }
    assert audit["exact_parameter_schema_match_to_far_full"] is True
    assert audit["exact_trainable_parameter_count_match_to_far_full"] is True
    assert audit["exact_projection_schema_match_to_far_full"] is True
    assert audit["parameter_schema_sha256"] == audit["far_full_parameter_schema_sha256"]
    assert audit["extra_parameter_names_vs_far_full"] == []
    assert audit["missing_parameter_names_vs_far_full"] == []
    assert audit["structural_labels_consumed_by_training_loss"] == []
    assert audit["additional_labels_consumed_by_ce_capacity_matched"] == []
    assert model.pair_head is None


def test_capacity_matched_ce_loss_does_not_read_structural_labels() -> None:
    args = variant_args("ce_capacity_matched")
    model = pilot.build_model(args, args.variant)
    states = 3
    batch = {
        "waveform": torch.randn(2, states, 2048),
        "O": torch.arange(states).remainder(len(pilot.ORIGINS)).repeat(2, 1),
        "M": torch.arange(states).remainder(len(pilot.DECODERS)).repeat(2, 1),
        "C": torch.arange(states).remainder(len(pilot.CHANNELS)).repeat(2, 1),
    }
    total, components = pilot.compute_training_loss(
        model,
        batch,
        args,
        pilot.variant_weights(args.variant, args),
    )
    assert torch.isfinite(total)
    assert components["supcon"] == 0.0
    assert components["offset"] == 0.0
    assert components["B_transition"] == 0.0
    assert components["pair_bce"] == 0.0


def test_task_specific_clip_variants_only_backpropagate_the_native_head() -> None:
    for variant, native_query, excluded_query in (
        ("ce_M_only", "M", "C"),
        ("ce_C_only", "C", "M"),
    ):
        args = variant_args(variant)
        model = pilot.build_model(args, variant)
        batch = {
            "waveform": torch.randn(2, 3, 2048),
            native_query: torch.arange(3).remainder(
                {"M": len(pilot.DECODERS), "C": len(pilot.CHANNELS)}[native_query]
            ).repeat(2, 1),
        }
        total, _components = pilot.compute_training_loss(
            model, batch, args, pilot.variant_weights(variant, args)
        )
        total.backward()
        native_head = {"M": model.m_head, "C": model.c_head}[native_query]
        excluded_head = {"M": model.m_head, "C": model.c_head}[excluded_query]
        assert native_head.weight.grad is not None
        assert excluded_head.weight.grad is None
        if model.o_head is not None:
            assert model.o_head.weight.grad is None
        assert model.projection[0].weight.grad is not None

        audit = pilot.variant_capacity_audit(model, variant, args)
        assert audit["native_query"] == native_query
        assert audit["optimization_labels"] == [native_query]
        assert audit["exact_parameter_schema_match_to_far_full"] is True
        assert audit["excluded_heads_contribute_gradient"] is False
        assert audit["frozen_probe_queries"] == ["M", "C", "R"]


def test_pair_R_only_backpropagates_pair_head_not_clip_heads() -> None:
    args = variant_args("pair_R_only")
    model = pilot.build_model(args, args.variant)
    batch = {
        "waveform": torch.randn(2, 4, 2048),
        "pair_index_a": torch.tensor([[0, 1], [0, 1]]),
        "pair_index_b": torch.tensor([[2, 3], [2, 3]]),
        "pair_label": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    }
    total, components = pilot.compute_training_loss(
        model, batch, args, pilot.variant_weights(args.variant, args)
    )
    total.backward()
    assert components["head_ce"] == 0.0
    assert components["pair_bce"] > 0.0
    assert model.pair_head is not None
    assert model.pair_head.weight.grad is not None
    assert model.m_head.weight.grad is None
    assert model.c_head.weight.grad is None
    if model.o_head is not None:
        assert model.o_head.weight.grad is None
    assert model.projection[0].weight.grad is not None

    audit = pilot.variant_capacity_audit(model, args.variant, args)
    assert audit["native_query"] == "R"
    assert audit["optimization_labels"] == ["R"]
    assert audit["exact_encoder_projection_schema_match_to_far_full"] is True
    assert audit["exact_parameter_schema_match_to_far_full"] is False
    assert audit["pair_R_only_additional_parameters_are_readout_only"] is True
    assert audit["pair_R_only_uses_source_metadata"] is False
    assert set(audit["extra_parameter_names_vs_far_full"]) == {
        "pair_head.bias",
        "pair_head.weight",
    }


def test_random_grouping_is_deterministic_cross_source_and_far_weight_matched() -> None:
    args = variant_args("far_random_grouping")
    assert pilot.variant_weights(args.variant, args) == pilot.variant_weights("far_full", args)
    model = pilot.build_model(args, args.variant)
    audit = pilot.variant_capacity_audit(model, args.variant, args)
    assert audit["structural_loss_graph_match_to_far_full"] is True
    assert audit["far_full_loss_weight_match"] is True
    assert audit["random_grouping"]["source_identifiers_or_metadata_used_by_randomizer"] is False

    groups, states, dimensions = 4, 4, 2
    embeddings = torch.empty(groups, states, dimensions)
    for source in range(groups):
        for state in range(states):
            embeddings[source, state] = torch.tensor([source, state])
    native_id = pilot.DECODER_TO_ID["encodec_native"]
    vocos_id = pilot.DECODER_TO_ID["vocos_encodec"]
    decoder_ids = torch.tensor(
        [[native_id, native_id, vocos_id, vocos_id]] * groups,
        dtype=torch.long,
    )
    pair_a = torch.tensor([[0, 1]] * groups)
    pair_b = torch.tensor([[2, 3]] * groups)
    first, first_audit = pilot.deterministic_cross_source_structural_grouping(
        embeddings,
        decoder_ids,
        pair_a,
        pair_b,
        seed=17,
        epoch=3,
        batch_index=2,
    )
    second, second_audit = pilot.deterministic_cross_source_structural_grouping(
        embeddings,
        decoder_ids,
        pair_a,
        pair_b,
        seed=17,
        epoch=3,
        batch_index=2,
    )
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert first_audit == second_audit
    assert first_audit["transition_endpoint_same_source_count"] == 0
    assert first_audit["cross_source_transition_fraction"] == 1.0
    assert first_audit["mixed_structural_group_fraction"] == 1.0
    assert first_audit["source_metadata_used"] is False
    for group in range(groups):
        native_source = int(first[group, 0, 0].item())
        vocos_source = int(first[group, 2, 0].item())
        assert native_source != vocos_source


def test_random_grouping_rejects_a_singleton_without_silent_fallback() -> None:
    with np.testing.assert_raises_regex(ValueError, "singleton training batch"):
        pilot.random_grouping_batch_contract(source_groups=5, group_batch_size=2)
    audit = pilot.random_grouping_batch_contract(source_groups=6, group_batch_size=2)
    assert audit["planned_batch_sizes"] == [2, 2, 2]
    assert audit["drop_last"] is False


def duplicate_source_group(rows: list[dict], pairs: list[dict]) -> tuple[list[dict], list[dict]]:
    cloned_rows = copy.deepcopy(rows)
    cloned_pairs = copy.deepcopy(pairs)
    sample_map: dict[str, str] = {}
    for row in cloned_rows:
        old_id = row["sample_id"]
        row["source_key"] = "train:source-2"
        row["sample_id"] = f"clone:{old_id}"
        row["path"] = f"/clone:{old_id}.flac"
        sample_map[old_id] = row["sample_id"]
    for pair in cloned_pairs:
        pair["pair_id"] = f"clone:{pair['pair_id']}"
        pair["source_key_a"] = "train:source-2"
        pair["source_key_b"] = "train:source-2"
        pair["sample_id_a"] = sample_map[pair["sample_id_a"]]
        pair["sample_id_b"] = sample_map[pair["sample_id_b"]]
    return rows + cloned_rows, pairs + cloned_pairs


def test_validate_only_writes_explicit_variant_audits(
    tmp_path: Path, monkeypatch: object
) -> None:
    rows, pairs = duplicate_source_group(*make_complete_manifests())
    sample_manifest = tmp_path / "samples.jsonl"
    pair_manifest = tmp_path / "pairs.jsonl"
    sample_manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    pair_manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in pairs), encoding="utf-8"
    )
    for variant in (
        "ce_capacity_matched",
        "ce_M_only",
        "ce_C_only",
        "pair_R_only",
        "far_random_grouping",
    ):
        output_dir = tmp_path / variant
        monkeypatch.setattr(
            sys,
            "argv",
            [
                str(pilot.__file__),
                "--sample-manifest",
                str(sample_manifest),
                "--pair-manifest",
                str(pair_manifest),
                "--output-dir",
                str(output_dir),
                "--variant",
                variant,
                "--group-batch-size",
                "2",
                "--validate-only",
            ],
        )
        assert pilot.main() == 0
        payload = json.loads((output_dir / "variant_audit.json").read_text(encoding="utf-8"))
        assert payload["variant"] == variant
        if variant != "pair_R_only":
            assert payload["exact_parameter_schema_match_to_far_full"] is True
        else:
            assert payload["exact_encoder_projection_schema_match_to_far_full"] is True
        if variant == "ce_capacity_matched":
            assert payload["structural_labels_consumed_by_training_loss"] == []
        elif variant in {"ce_M_only", "ce_C_only", "pair_R_only"}:
            assert len(payload["native_training_queries"]) == 1
            assert payload["frozen_probe_queries"] == ["M", "C", "R"]
        else:
            contract = payload["random_grouping"]["training_batch_contract"]
            assert contract["every_training_batch_supports_cross_source_derangement"] is True


def test_registered_defaults_use_full_24_cell_primary_protocol() -> None:
    parser = pilot.build_parser()
    args = parser.parse_args(
        [
            "--sample-manifest", "samples.jsonl",
            "--pair-manifest", "pairs.jsonl",
            "--output-dir", "out",
            "--variant", "far_full",
        ]
    )
    assert args.train_cell_policy == "full"
    assert args.clip_seconds == 4.0
    assert args.n_mels == 96
    assert args.hop_length == 256
    assert args.dropout == 0.1
    assert args.dev_crops == 3
    assert args.final_eval_crops == 3
    assert args.export_layer_embeddings is False
