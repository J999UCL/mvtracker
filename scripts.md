# MV-Tracker Modal profiling runbook

## UpdateFormer single-H100 autoresearch contract

These commands use one H100 and the billing tags `owner=jeet`,
`project=mvtracker`, `purpose=profiling`. Capture creates the immutable v3
golden contract once. Every candidate must stay within one BF16 ULP and pass
the tight FP32 output, input-gradient, parameter-gradient, Adam-update, and
loss checks before its fixed B1/B2/B4 benchmark runs. The end-to-end command
also compares whole-update and reusable live backends against eager run-to-run
nondeterminism, selects the fastest valid live backend, then confirms it for ten
real changing optimizer steps. Inductor kernels persist on the run Volume.

```bash
cd /Users/jeetthakwani/dev/PointTracking/mvtracker
export MVTRACKER_MODAL_COMMIT=<full-pushed-main-sha>

modal run --timestamps tools/modal_updateformer_autoresearch.py::capture
modal run --timestamps tools/modal_updateformer_autoresearch.py::verify
modal run --timestamps tools/modal_updateformer_autoresearch.py::run_benchmark \
  --warmup 3 --measured 10

# Candidate search followed by a ten-update live confirmation; no DDP.
modal run --timestamps tools/modal_updateformer_autoresearch.py::autoresearch \
  --steps 10

# Explicit live candidate confirmation when rerunning one backend.
modal run --timestamps tools/modal_updateformer_autoresearch.py::single_gpu_smoke \
  --backend graphed_bucketed --steps 10

# Matched full-model eager-versus-checkpoint H200 throughput comparison.
modal run --timestamps tools/modal_checkpoint_net_throughput.py::prepare
modal run --timestamps tools/modal_checkpoint_net_throughput.py::sweep
```

## VGGT-Omega H100 throughput profile

This profile requests exactly one H100 (`max_containers=1`) and carries the
billing tags `owner=jeet`, `project=mvtracker`, `purpose=profiling`.  It stages
the pinned DIEGESIS snapshot directly from Hugging Face/Xet onto container-local
SSD (about 90 seconds), and copies only MV-Kubric scenes `1001`--`1004` with
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

## GT-depth DIEGESIS + 2,000-scene MV-Kubric continual training

The launcher refuses a two-H100 submission unless the selected source SHA is
the pushed `origin/main` commit and at least two Prism container slots are free.
It never stops another app. Data setup verifies the existing DIEGESIS and
2,000-scene MV-Kubric pool, then materializes the pinned, checksummed
published mixed-depth checkpoint into the existing data Volume.

The tagged CPU setup downloads and checksum-verifies the unchanged official
archives `kubric-multiview--train.full.1001-2000.tar.gz` and
`kubric-multiview--train.full.2001-3000.tar.gz` at revision
`cccb9128fb95d302c662151e65a09377175c2a3a` into the versioned Volume path
`archives/mvkubric/2000-scenes-v1/`. It streams only the ordered range needed
for validation scenes `101`--`127` from
`kubric-multiview--train.full.0031-1000.tar.gz` into
`datasets/kubric-multiview/2000-scenes-v1/validation/`. No 395-GB validation
source archive is retained.

```bash
cd /Users/jeetthakwani/dev/PointTracking/mvtracker
export MVTRACKER_MODAL_COMMIT=<full-pushed-origin-main-sha>

# One-time CPU-only direct-Volume ingestion. It expands DIEGESIS and both
# MV-Kubric archives into jeet-mvtracker-data-v2, copies validation 101--127,
# rebuilds the metadata index, and prints the detached Function call ID.
modal run --detach --timestamps tools/modal_continual_data_setup.py::setup_data

# CPU-only first-touch and warm loader verification against Volume v2.
modal run --timestamps tools/modal_continual_training.py::profile_cpu_loader

# Required DDP/W&B resume smoke: steps 1-3, then resume to exactly step 5.
modal container list --json
modal run --timestamps tools/modal_continual_training.py::smoke

# Production-path timing smoke: one phase of exactly 10 optimizer steps.
modal container list --json
modal run --timestamps tools/modal_continual_training.py::smoke10

# Two-update, two-H100 memory attribution profile. Update 0 warms the live
# WIDS/DALI physical-batching path; update 1 records both ranks under
# memory_profile/ and profiler/ in the run directory.
modal container list --json
modal run --timestamps tools/modal_continual_training.py::memory_profile

# Reproduce a loader A/B with an identical seed; eager materialization is an
# explicit benchmark mode and is not the production default.
modal run --timestamps tools/modal_continual_training.py::smoke10 --run-name <unique-name> --seed <fixed-seed> --materialize-whole-step true

# Main run is deliberately gated and cannot launch without this flag.
modal container list --json
modal run --timestamps tools/modal_continual_training.py::train --confirm-main
```

Both GPU modes request exactly `H100!:2`, set `max_containers=1`, and attach
`owner=jeet`, `project=mvtracker`, and `purpose=training` billing tags. Reuse an
explicit `--run-name` to resume the same run, W&B identity, seed, and checkpoint
directory.

The CPU setup expands the DIEGESIS archive and each official 1,000-scene
MV-Kubric archive once under `/mnt/mvtracker-data`. It copies validation scenes
`101`--`127`, builds `MVTracker_index` from the observed scene inventory, and
writes `direct-volume-data-manifest.json`. Profiling and training mount
`jeet-mvtracker-data-v2` read-only and mount only the results Volume writable.

```bash
cd /Users/jeetthakwani/dev/PointTracking/mvtracker
export MVTRACKER_MODAL_COMMIT=<full-pushed-origin-main-sha>

modal run --timestamps tools/modal_continual_training.py::profile_cpu_loader
modal run --timestamps tools/modal_continual_training.py::profile_h100_loader
```

The CPU loader profile uses four warm-ups and 32 measured samples. The H100
profile stages the first 500-scene MV-Kubric archive range, then measures 20
warm-up and 100 production-path samples using batched nvImageCodec PNG/TIFF
decoding.
Both profile entrypoints carry
`owner=jeet`, `project=mvtracker`, `purpose=profiling` tags.

## Single-T4 production loader benchmark

The reusable T4 harness reads the continual-training data directly from Volume
v2 using one tagged T4 container. It runs warm-only DIEGESIS-4-view, MV-Kubric-4/6-view,
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

## Full MV-Kubric scene/view WebDataset conversion

This CPU-only Modal job processes the two pinned MV-Kubric archives one at a
time. Each archive is copied to local SSD, decompressed with 16-way
`rapidgzip`, converted into resumable indexed scene/view TAR shards, and
removed from local SSD before the next archive starts. Progress heartbeats are
emitted at least every 30 seconds and completed scene/shard events are flushed
to stdout and W&B. The job uses one 16-CPU, 32-GiB container with a 1-TiB
ephemeral disk; it does not request a GPU.

```bash
cd /Users/jeetthakwani/dev/PointTracking/mvtracker
export MVTRACKER_MODAL_COMMIT=<full-pushed-origin-main-sha>
modal container list --json
modal run --detach --timestamps tools/modal_mvkubric_full_conversion.py
```

The function resumes completed TAR plus `.inventory.json` pairs, removes only
orphan `.partial` files, and publishes train and validation outputs only after
both splits are finalized. The command prints the Modal call ID immediately;
monitor it with `modal app logs <app-id>` or resume inspection from the same
repository, commit and output paths shown in the startup event.

## MV-Kubric WebDataset pilot conversion and T4 A/B

The pilot converts exactly scenes `1001`--`1032` into four-scene uncompressed
TAR shards and creates NVIDIA DALI `.idx` files. It preserves the native
dataset and writes the derived dataset under a new Volume prefix.

```bash
cd /Users/jeetthakwani/dev/PointTracking/mvtracker
export MVTRACKER_MODAL_COMMIT=<full-pushed-origin-main-sha>
modal run --timestamps tools/modal_mvkubric_webdataset.py::convert \
  --scene-root /mnt/mvtracker-data/datasets/kubric-multiview/train \
  --output-root /mnt/mvtracker-data/datasets/kubric-multiview-webdataset/v1/train \
  --scenes 1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012,1013,1014,1015,1016,1017,1018,1019,1020,1021,1022,1023,1024,1025,1026,1027,1028,1029,1030,1031,1032 \
  --scenes-per-shard 4 --shard-workers 1 --read-workers 16
```

After the derived manifest exists, the tagged one-T4 benchmark compares the
native loader and DALI path at 1, 2, 4 and 6 selected views. It runs four
warm-ups and sixteen measured samples per path/view case, with no model or
optimizer.

```bash
modal container list --json
modal run --detach --timestamps tools/modal_mvkubric_webdataset.py::benchmark \
  --run-name mvkubric-webdataset-t4-pilot \
  --scenes 1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012,1013,1014,1015,1016,1017,1018,1019,1020,1021,1022,1023,1024,1025,1026,1027,1028,1029,1030,1031,1032 \
  --warmup 4 --measured 16 --workers 8
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

### Profile the planned mixed physical loader on Dopey

This is a bounded loader/decode profiler, not training.  It is pinned to the
physical Titan devices `1,2` and refuses any other device pair.  The first run
creates a deterministic four-step plan stream; later runs may pass the same
`plan.json` to replay exactly the same requests.  It reports cold/warm CPU
planning and materialisation, encoded-byte cache volume, CUDA decode time,
exposed wait, samples/sec, trajectories/sec, process RSS, and NVML GPU memory
and utilisation.  CUDA outputs are checked for device placement and finite
values after each physical group.

```bash
source /media/data3/jthakwani/mvtracker-venv/bin/activate
cd /media/data3/jthakwani/mvtracker
python tools/profile_mixed_physical_loader.py \
  --device-ids 1,2 \
  --steps 4 --passes 1 --workers 4 \
  --decode-image-chunk-size 8 \
  --output-dir /media/data3/jthakwani/mvtracker-runs/mixed-physical-loader
```

The output directory contains `plan.json` and `report.json`.  The optional
bounded numerical parity check uses no model or optimizer and can be run on a
3090 separately:

```bash
python tools/profile_mixed_physical_loader.py \
  --mode parity --parity-device cuda:0
```
## UpdateFormer fused-backend candidate gate

Run the single-H100 real-update gate comparing eager UpdateFormer against one
experimental backend. The command uses the saved real mixed-source crash batch,
checks one and five updates, and writes its JSON result to the Modal run Volume.
`qkv` and `fused` are research candidates; neither is approved for training.

```bash
MVTRACKER_MODAL_COMMIT="$(git rev-parse HEAD)" \
  modal run tools/modal_updateformer_autoresearch.py::fused_candidate_gate \
    --candidate-backend qkv
```
