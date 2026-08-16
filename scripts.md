# MV-Tracker Modal profiling runbook

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

```bash
cd /Users/jeetthakwani/dev/PointTracking/mvtracker
export MVTRACKER_MODAL_COMMIT=<full-pushed-origin-main-sha>

# One-time CPU-only data verification and checkpoint materialization.
modal run --timestamps tools/modal_continual_training.py::setup_data

# Required DDP/W&B resume smoke: steps 1-3, then resume to exactly step 5.
modal container list --json
modal run --timestamps tools/modal_continual_training.py::smoke

# Main run is deliberately gated and cannot launch without this flag.
modal container list --json
modal run --timestamps tools/modal_continual_training.py::train --confirm-main
```

Both GPU modes request exactly `H100!:2`, set `max_containers=1`, and attach
`owner=jeet`, `project=mvtracker`, and `purpose=training` billing tags. Reuse an
explicit `--run-name` to resume the same run, W&B identity, seed, and checkpoint
directory.

Each compute container copies the existing `datasets/`, `source/`, and
`checkpoints/` directories directly from the Modal data Volume to
`/tmp/mvtracker-continual-data`, then loads from that local SSD. Staging time and
copied bytes are logged to W&B.

```bash
cd /Users/jeetthakwani/dev/PointTracking/mvtracker
export MVTRACKER_MODAL_COMMIT=<full-pushed-origin-main-sha>

modal run --timestamps tools/modal_continual_training.py::profile-cpu-loader
modal run --timestamps tools/modal_continual_training.py::profile-h100-loader
```

The loader profiles use four warm-ups and 32 measured samples. The CPU profile
reports DIEGESIS and MV-Kubric separately; the H100 profile uses the production
CUDA prefetch/nvJPEG path for DIEGESIS. Both profile entrypoints carry
`owner=jeet`, `project=mvtracker`, `purpose=profiling` tags.
