from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from sound_evidence import _core
from sound_evidence import cli as release_cli
from sound_evidence import outputs as release_outputs
from sound_evidence.cli import make_demo, main
from sound_evidence.runtime import load_model, predict_manifest, read_manifest, sha256, validate_formal

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models" / "far_full_seed20260713.pt"
CONFIG = CHECKPOINT.with_suffix(".json")


@pytest.fixture(scope="module")
def loaded():
    torch.set_num_threads(2)
    return load_model(CHECKPOINT, CONFIG)


def test_actual_core_checkpoint_is_loaded_strictly(loaded):
    assert isinstance(loaded.model, _core.FactorialAudioCnn)
    assert loaded.model.o_head is None
    assert loaded.model.pair_head is None
    assert loaded.class_contract["M"] == ["encodec_native", "vocos_encodec"]
    assert loaded.model.projection[0].out_features == 256


def test_hash_mismatch_is_rejected_before_deserialization(tmp_path):
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["checkpoint_sha256"] = "0" * 64
    config = tmp_path / "bad.json"
    config.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256"):
        load_model(CHECKPOINT, config)


def test_manifest_and_inference_survive_relocation(tmp_path, loaded, monkeypatch):
    original = tmp_path / "first"
    relocated = tmp_path / "second"
    manifest = make_demo(original)
    expected = predict_manifest(loaded, read_manifest(manifest), crops=1)
    shutil.copytree(original, relocated)
    shutil.rmtree(original)
    monkeypatch.chdir(tmp_path)
    actual = predict_manifest(loaded, read_manifest(relocated / "samples.jsonl"), crops=1)
    np.testing.assert_array_equal(actual["embeddings"], expected["embeddings"])
    # An external manifest can use the explicit data-root instead of its parent.
    external = tmp_path / "external.jsonl"
    shutil.copyfile(relocated / "samples.jsonl", external)
    rooted = read_manifest(external, relocated)
    assert Path(rooted[0]["path"]).parent == relocated


def test_crop_wrapper_matches_registered_direct_forward(tmp_path, loaded):
    rows = read_manifest(make_demo(tmp_path / "input"))
    actual = predict_manifest(loaded, rows, crops=1)
    waves = [_core.load_audio(row["path"], 24000) for row in rows]
    clip = 96000
    start = (min(w.numel() for w in waves) - clip) // 2
    batch, _ = _core.synchronous_crop_and_group_gain(waves, [True, True], clip, start=start)
    with torch.inference_mode():
        direct = loaded.model(batch)
    np.testing.assert_array_equal(actual["embeddings"], direct["embedding"].numpy())
    np.testing.assert_array_equal(actual["M_logits"], direct["M_logits"].numpy())
    np.testing.assert_array_equal(actual["C_probs"], torch.softmax(direct["C_logits"], -1).numpy())


def test_cli_uses_actual_checkpoint_and_writes_finite_features(tmp_path):
    output = tmp_path / "demo"
    assert main(["demo", "--checkpoint", str(CHECKPOINT), "--config", str(CONFIG), "--output", str(output)]) == 0
    with np.load(output / "predictions.npz", allow_pickle=False) as arrays:
        assert arrays["embeddings"].shape == (2, 256)
        assert arrays["M_probs"].shape == (2, 2)
        assert arrays["C_probs"].shape == (2, 3)
        assert np.isfinite(arrays["embeddings"]).all()
        np.testing.assert_allclose(arrays["C_probs"].sum(axis=1), 1.0, atol=1e-6)


def test_models_and_source_match_recorded_bindings():
    manifest = json.loads((ROOT / "SOURCE_BINDINGS.json").read_text(encoding="utf-8"))
    for binding in manifest["bindings"]:
        assert sha256(ROOT / binding["candidate_path"]) == binding["sha256"]
    for binding in manifest.get("wrapper_bindings", []):
        assert sha256(ROOT / binding["candidate_path"]) == binding["sha256"]


def test_formal_wrapper_rejects_cross_split_source_reuse():
    with pytest.raises(ValueError, match="multiple splits"):
        validate_formal([{"source_key": "same", "split": "train"}, {"source_key": "same", "split": "test"}], [])


@pytest.mark.parametrize("command", ["demo", "evaluate", "make-demo"])
def test_directory_commands_refuse_even_existing_empty_outputs(tmp_path, command):
    output = tmp_path / "existing"
    output.mkdir()
    arguments = [command, "--output", str(output)]
    if command != "make-demo":
        arguments += ["--checkpoint", "missing.pt", "--config", "missing.json"]
    if command == "evaluate":
        arguments += ["--samples", "missing.jsonl", "--pairs", "missing_pairs.jsonl"]
    with pytest.raises(FileExistsError, match="output already exists"):
        main(arguments)
    assert output.is_dir() and not list(output.iterdir())


@pytest.mark.parametrize("existing_suffix", [".npz", ".json"])
def test_predict_refuses_existing_result_or_sidecar_without_modification(tmp_path, existing_suffix):
    destination = tmp_path / "prediction.npz"
    protected = destination.with_suffix(existing_suffix)
    protected.write_bytes(b"completed previous result")
    with pytest.raises(FileExistsError, match="output already exists"):
        main(["predict", "--checkpoint", "missing.pt", "--config", "missing.json",
              "--manifest", "missing.jsonl", "--output", str(destination)])
    assert protected.read_bytes() == b"completed previous result"
    assert len(list(tmp_path.iterdir())) == 1


def test_demo_failure_after_audio_generation_leaves_no_partial_result(tmp_path, monkeypatch):
    def fail_prediction(*args, **kwargs):
        raise ValueError("injected inference failure")
    monkeypatch.setattr(release_cli, "predict_manifest", fail_prediction)
    output = tmp_path / "failed-demo"
    with pytest.raises(ValueError, match="injected inference"):
        main(["demo", "--checkpoint", str(CHECKPOINT), "--config", str(CONFIG), "--output", str(output)])
    assert not output.exists()
    assert not list(tmp_path.glob(".failed-demo.pending-*"))


def test_prediction_write_failure_removes_only_its_own_incomplete_file(tmp_path, monkeypatch):
    def fail_write(handle, **arrays):
        handle.write(b"incomplete")
        raise OSError("injected disk write failure")
    monkeypatch.setattr(release_outputs.np, "savez_compressed", fail_write)
    destination = tmp_path / "failed.npz"
    with pytest.raises(OSError, match="disk write"):
        release_outputs.write_predictions(destination, {"x": np.ones(2)}, {"status": "complete"})
    assert not destination.exists()
    assert not destination.with_suffix(".json").exists()


def test_invalid_evaluation_input_leaves_no_output_directory(tmp_path, loaded):
    from sound_evidence.runtime import evaluate_formal
    output = tmp_path / "invalid-evaluation"
    with pytest.raises(FileNotFoundError):
        evaluate_formal(loaded, tmp_path / "missing.jsonl", tmp_path / "missing_pairs.jsonl", output)
    assert not output.exists()
    assert not list(tmp_path.glob(".invalid-evaluation.pending-*"))
