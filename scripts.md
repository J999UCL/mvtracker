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
modal run --timestamps tools/modal_training_profile.py::validate-image
modal run --timestamps tools/modal_training_profile.py::setup-data
modal run --timestamps tools/modal_training_profile.py::prepare-batches

# One exact lane at a time; compatibility is a GPU gate, not a fallback.
modal container list --json
modal run --timestamps tools/modal_training_profile.py::compatibility --gpu H100!
modal container list --json
modal run --timestamps tools/modal_training_profile.py::run-profile --gpu H100!
```

Use `--gpu H200` or `--gpu B200` for the other exact lanes. The profile uses
one cached batch of eight scenes with 2,048 tracks for each of the four view
counts, slices scene and track prefixes for every trial, searches physical
batches 1–8 at 1× accumulation for both 1,024 and 2,048 tracks, and confirms
the selected batch with two warm-ups and three measured updates. Results and
logs are committed to the `jeet-mvtracker-runs-v2` Volume under the run name.
