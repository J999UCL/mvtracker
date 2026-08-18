# MV-Tracker Modal profiling runbook

## VGGT-Omega H100 throughput profile

This profile requests exactly one H100 (`max_containers=1`) and carries the
billing tags `owner=jeet`, `project=mvtracker`, `purpose=profiling`.  It stages
the pinned DIEGESIS snapshot directly from Hugging Face/Xet onto container-local
SSD (about 90 seconds), and copies only MV-Kubric scenes `900`--`903` with
`rgba_*.png` plus `metadata.json` from the read-only Modal Volume.  Inference
and temporary profile outputs stay on local SSD; only the small report is
committed to `jeet-mvtracker-runs-v2` and metrics go to W&B.  Each dataset
tries loader workers `1`, `4`, and `8` at scene batch `1`, then sweeps the
scene-batch candidates with the fastest loader.  The measured depth, mask,
intrinsics, extrinsics, and scale arrays are overwritten in one local temp
directory so write time is included in the throughput result.

```bash
cd /Users/jeetthakwani/dev/PointTracking/mvtracker
export MVTRACKER_MODAL_COMMIT=<full-pushed-main-sha>

# Check the workspace before launching; never stop another app's container.
modal container list --json
modal run --timestamps tools/modal_vggt_omega_profile.py::profile \
  --run-name vggt-omega-h100-throughput
```

All Modal runs use the `ucl-prism` profile. Check the workspace before every
submission and do not stop containers whose app name is not `jeet-mvtracker-profile`.
The profiler requests exactly one GPU per function and attaches the billing tags
`owner=jeet`, `project=mvtracker`, `purpose=gpu-economics-profile`, plus the
experiment and GPU lane.

```bash
cd /Users/jeetthakwani/dev/PointTracking/mvtracker
export MVTRACKER_MODAL_COMMIT=<full-pushed-main-sha>

modal container list --json
modal run --timestamps tools/modal_training_profile.py::validate_image
modal run --timestamps tools/modal_training_profile.py::setup_data
modal run --timestamps tools/modal_training_profile.py::prepare_batches

# One exact lane at a time; compatibility is a GPU gate, not a fallback.
modal container list --json
modal run --timestamps tools/modal_training_profile.py::compatibility --gpu H100!
modal container list --json
modal run --timestamps tools/modal_training_profile.py::smoke --gpu H100!
modal container list --json
modal run --timestamps tools/modal_training_profile.py::run_profile --gpu H100!
```

Use `--gpu H200` or `--gpu B200` for the other exact lanes. The profile uses
one cached batch of eight scenes with 2,048 tracks for each of the four view
counts, slices scene and track prefixes for every trial, searches physical
batches 1–8 at 1× accumulation for both 1,024 and 2,048 tracks, and confirms
the selected batch with two warm-ups and three measured updates. Results and
logs are committed to the `jeet-mvtracker-runs-v2` Volume under the run name.
If batch 8 is safe, report the result as at least 8 rather than as an unbounded
maximum. Always use these local entrypoints so the app receives the exact GPU
and experiment billing tags.

## GT-depth DIEGESIS + MV-Kubric continual training

The launcher refuses a two-H100 submission unless the selected source SHA is
the pushed `origin/main` commit and at least two Prism container slots are free.
It never stops another app. Data setup verifies the existing DIEGESIS and
100-scene MV-Kubric micro pool, then materializes the pinned, checksummed
published mixed-depth checkpoint into the existing data Volume.

Validation scenes `101` and `102` use the pinned upstream
`kubric-multiview--train.full.0031-1000.tar.gz` archive. The tagged setup
entrypoint streams only the ordered prefix needed to extract those two scenes,
copies them into `datasets/kubric-multiview/train`, and rebuilds one combined
102-scene `MVTracker_index`. It is a CPU/data setup job, not training; inspect
`modal container list --json` before launching because the source archive is
very large even when streamed selectively.

```bash
cd /Users/jeetthakwani/dev/PointTracking/mvtracker
export MVTRACKER_MODAL_COMMIT=<full-pushed-origin-main-sha>
modal container list --json
modal run --timestamps tools/modal_mvkubric_validation.py
```

```bash
cd /Users/jeetthakwani/dev/PointTracking/mvtracker
export MVTRACKER_MODAL_COMMIT=<full-pushed-origin-main-sha>

# One-time CPU-only source-data verification and checkpoint materialization.
modal run --timestamps tools/modal_continual_training.py::setup_data

# One-time CPU build of the expanded immutable dataset image.
modal run --timestamps tools/modal_continual_training.py::build_dataset_image

# CPU-only first-touch and warm loader verification against the dataset image.
modal run --timestamps tools/modal_continual_training.py::profile-cpu-loader

# Required DDP/W&B resume smoke: steps 1-3, then resume to exactly step 5.
modal container list --json
modal run --timestamps tools/modal_continual_training.py::smoke

# Production-path timing smoke: one phase of exactly 10 optimizer steps.
modal container list --json
modal run --timestamps tools/modal_continual_training.py::smoke10

# Main run is deliberately gated and cannot launch without this flag.
modal container list --json
modal run --timestamps tools/modal_continual_training.py::train --confirm-main
```

Both GPU modes request exactly `H100!:2`, set `max_containers=1`, and attach
`owner=jeet`, `project=mvtracker`, and `purpose=training` billing tags. Reuse an
explicit `--run-name` to resume the same run, W&B identity, seed, and checkpoint
directory.

The dataset image expands the DIEGESIS archive and four MV-Kubric tar.zst
shards once under `/opt/mvtracker-data`. The immutable dataset layer sits above
dependencies and below the commit-specific source layer. Training mounts only
the writable results Volume and reads the dataset directly from the image.

```bash
cd /Users/jeetthakwani/dev/PointTracking/mvtracker
export MVTRACKER_MODAL_COMMIT=<full-pushed-origin-main-sha>

modal run --timestamps tools/modal_continual_training.py::profile-cpu-loader
modal run --timestamps tools/modal_continual_training.py::profile-h100-loader
```

The CPU loader profile uses four warm-ups and 32 measured samples. The H100
profile stages only the first 25-scene MV-Kubric shard, then measures 20 warm-up
and 100 production-path samples using batched nvImageCodec PNG/TIFF decoding.
Both profile entrypoints carry
`owner=jeet`, `project=mvtracker`, `purpose=profiling` tags.

To measure archive transfer, extraction, and one CPU sample per dataset without
launching training or a GPU:

```bash
MVTRACKER_MODAL_COMMIT=$(git rev-parse HEAD) \
modal run --timestamps tools/modal_data_loading_smoke.py
```

To split the pinned MV-Kubric archive into four independently extractable
`tar.zst` shards (CPU-only, one Modal container):

```bash
modal container list --json
modal run --timestamps tools/modal_mvkubric_shard.py
```

To benchmark parallel local copy and extraction of the four MV-Kubric shards:

```bash
modal container list --json
modal run --timestamps tools/modal_mvkubric_shard.py::benchmark_shards
```

## Single-T4 production loader benchmark

The reusable T4 harness uses the cached continual-training dataset image and
one tagged T4 container. It runs warm-only DIEGESIS-4-view, MV-Kubric-4/6-view,
and mixed-view-4 cases with 16 measured samples. Only the mixed case includes
the 1.25-second simulated-compute gap. Every case logs progress immediately to
stdout and W&B and commits a partial JSON report to `jeet-mvtracker-runs-v2`.

```bash
cd /Users/jeetthakwani/dev/PointTracking/mvtracker
export MVTRACKER_MODAL_COMMIT=<full-pushed-main-sha>
modal container list --json
modal run --detach --timestamps tools/modal_t4_loader_benchmark.py::profile \
  --run-name t4-loader-baseline
```
### Inspect the current sequential mixed-source sampler on Dopey

After pulling the desired commit and building the MV-Kubric metadata index,
capture five groups of eight current sampler outputs without running training:

```bash
source /media/data3/jthakwani/mvtracker-venv/bin/activate
cd /media/data3/jthakwani/mvtracker
python tools/inspect_mixed_sampling.py \
  --diegesis-root /media/data3/jthakwani/datasets/diegesis-mvtracker \
  --mvkubric-root /media/data3/jthakwani/datasets/mv3dpt-train-micro \
  --mvkubric-index-root /media/data3/jthakwani/datasets/mv3dpt-train-micro/kubric-multiview/train/MVTracker_index \
  --output-dir /media/data3/jthakwani/mvtracker-runs/mixed-sampling-baseline \
  --steps 5 --seed 72 \
  --run-name mixed-sampling-baseline-seed72 \
  --wandb-entity jeetucl-ucl --wandb-project mvtracker-modal-profiling
```
