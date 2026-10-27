# Sound Evidence — FAR factorial core candidate

## Public release candidate

`v0.1.0-rc2` publishes the tested core code and companion model/state assets.
The locally executed checks and coverage limits are recorded in `VERIFY.json`
and below; complete raw-data-to-paper-table reproduction is not claimed.
License status is recorded in `NOTICE`.


This repository packages the **FactorialAudioCnn / FAR core** of *Sound Evidence:
A Unified Evidence Framework for Audio Forensics* (AAAI 2027 submission 11704).
It includes the actual registered network, FAR losses, source-group cropping,
training loop, fixed M/C/R linear-probe evaluation, and the three original
`far_full` checkpoints (seeds 20260713–20260715).

This is a working release candidate. The authors still need to select a code
and checkpoint license; see `NOTICE`. `VERIFY.json` records what was actually
run. This core package does not include the separately trained public-endpoint
models, the later matched-semantic/rereview experiments, or the separate
Path-Chg/Chain-Ver endpoint registrations.

## Install and run a real checkpoint

Python 3.11–3.13 is supported by the package metadata; the recorded verification
used Python 3.13 on Windows and the exact versions in `requirements-tested.txt`.
Only that recorded environment was tested. Install a compatible Torch/
Torchaudio pair first if you need a particular CPU/CUDA wheel, then:

```sh
python -m pip install -e ".[test]"
sound-evidence demo --checkpoint models/far_full_seed20260713.pt --config models/far_full_seed20260713.json --output demo-output
python -m pytest -q
```

The demo creates two deterministic synthetic signals and runs CPU inference.
Expected shapes: `embeddings [2,256]`, `M_probs [2,2]`, `C_probs [2,3]`.
The synthetic signals test the executable pipeline; their predictions do not
measure forensic accuracy. Nothing is downloaded and no dataset audio is used.
The `.pt` files are external to the Python wheel; keep `models/` from this
repository and pass the model/config paths explicitly.

Every `demo`, `make-demo` and `evaluate` output directory must be new, including
when an existing directory is empty. `predict` requires a new `.npz` path and a
new matching `.json` sidecar. Existing results are refused. Directory outputs
are prepared in temporary siblings and published only on success; failed input
does not leave a partially completed result directory.

## Features and native query heads on your audio

Use JSONL with one row per clip, for example:

```json
{"sample_id":"clip-a","path":"audio/a.wav","group_id":"same-source-variants"}
```

`path` is relative to `--data-root`, or to the manifest directory if omitted.
Absolute user-supplied paths also work. `sample_id` must be unique. `group_id`
marks variants of a common source: they share the shortest support, crop
location and peak gain, matching the core's full-grid crop math. If omitted,
each clip is its own group. Do not group unrelated clips to emulate a factorial
audit. Start/middle/end crops are averaged by default; there is no per-clip
gain when multiple variants share a group.

```sh
sound-evidence predict --checkpoint models/far_full_seed20260713.pt --config models/far_full_seed20260713.json --manifest samples.jsonl --data-root my-audio --output features.npz
```

The NPZ has `sample_ids`, raw and L2-normalized `embeddings`, `M_logits`,
`C_logits`, and mean crop-wise softmax `M_probs`/`C_probs`. M classes are
`encodec_native`, `vocos_encodec`; C classes are `none`, `opus_24kbps`,
`aac_64kbps`. These are **native trained heads**, not the paper's frozen linear
probes. There is deliberately no bandwidth head or direct R head in `far_full`.
Predictions outside the registered formation conditions are unvalidated.

## Registered training and M/C/R evaluation

`src/sound_evidence/_core.py` is an unchanged copy of the research runner.
Its factor, loss, early-selection and readout behavior is preserved. The
wrapper accepts portable copies of the original sample/pair manifest schemas.
You must obtain or regenerate the controlled audio yourself; this package does
not bundle LibriTTS or codec datasets, their generators or downloaders. The
sample manifest needs `sample_id`, `path`, `split`, `source_key`, `labels`, and
the original `cps_factorial` factor contract. Pair rows need the original
source/sample identifiers, factor metadata, split, partition and relation label.
The core validates complete source groups and source-disjoint splits.

```sh
# Existing checkpoint: exact full-grid crop/export and fixed train-only probes
sound-evidence evaluate --checkpoint models/far_full_seed20260713.pt --config models/far_full_seed20260713.json --samples factorial_samples.jsonl --pairs factorial_pairs.jsonl --data-root controlled-audio --output evaluation-output

# Training: additional original runner arguments follow --
sound-evidence train --samples factorial_samples.jsonl --pairs factorial_pairs.jsonl --data-root controlled-audio --output training-output --variant far_full -- --device cpu --seed 20260713 --validate-only
```

Remove `--validate-only` to train. Training is substantial and was not rerun for
this candidate. `evaluate` exports the unchanged frozen embeddings and fits
the registered StandardScaler/logistic readouts on **train only**. It produces
M/C/R metrics and probe parameters (undefined single-class segment metrics are
JSON `null`); it does not claim to reconstruct every
paper table or the separately registered zero-training Path-Chg endpoint.
CPU smoke tests are not a full-dataset result reproduction. Source-relative
training configs and checkpoint hashes are in `models/` and
`SOURCE_BINDINGS.json`; inference configs include every construction parameter
missing from the historical checkpoint schema.

## Integrity and scope

The loader verifies a checkpoint's SHA256 from its paired JSON config and loads
the registered state dict strictly. It uses `torch.load(weights_only=True)`.
The source implementation has module-level factor configuration, so use separate
processes for concurrent datasets with different factor registries. Tests cover
the legacy scientific invariants, actual checkpoint compatibility, relocation
of relative audio paths, hash rejection, and wrapper/legacy output agreement.
No original project file or server is required at runtime.
