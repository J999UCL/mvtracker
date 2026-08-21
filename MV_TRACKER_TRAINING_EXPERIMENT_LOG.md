# MV-Tracker Training Experiment Log

This is the primary record for MV-Tracker training experiments in this repository. It records the exact hypothesis, data, configuration, execution, artifacts, results, failures, and interpretation of each substantial run. Paths are retained so that results can be audited without reconstructing them from chat history.

## Experiment 001 — DIEGESIS-only proof-of-signal fine-tuning

### Identity and status

| Field | Value |
|---|---|
| Date | 2026-08-13 |
| Run ID | `diegesis-proof-20260813T111318Z-52ac9be` |
| Host | `dopey-prism` (`dopey`) |
| GPU | NVIDIA GeForce RTX 3090, 24 GB |
| Repository revision at launch | `52ac9be` (`Configure DIEGESIS proof-of-signal run`) |
| Intended duration | 2,000 optimizer steps |
| Completed updates | 1,804 optimizer updates; the checkpoint counter is `total_steps=1804` |
| Terminal state | Failed with CUDA OOM on the first microbatch of optimizer step 1,804 |
| W&B | Disabled by configuration |
| Primary telemetry | Local TensorBoard event file and text log |

This was the first substantial fine-tuning run on the generated DIEGESIS dataset. It was a proof-of-signal experiment, not a final training recipe. It fine-tuned the entire MV-Tracker model on DIEGESIS alone, starting from the released clean-depth checkpoint. No MV-Kubric replay or other training dataset was mixed into the run.

The run began at **2026-08-13 12:13:30 BST** and failed at **17:31:48 BST**, for approximately **5 hours 18 minutes wall time**.

### Hypothesis

The narrow question was:

> Does full-model fine-tuning on DIEGESIS provide a real optimization signal and improve performance on held-out DIEGESIS scenes?

This experiment was not designed to preserve the pretrained model's original MV-Kubric, Panoptic, or DexYCB capabilities. The later external evaluation was added specifically to measure whether DIEGESIS-only fine-tuning caused capability regression.

### Starting checkpoint and model

The run restored model weights from:

```text
/media/data3/jthakwani/mvtracker/checkpoints/mvtracker_200000_june2025_cleandepth.pth
```

This is the released MV-Tracker checkpoint trained with clean/ground-truth depth. It is a weights-only checkpoint, so the DIEGESIS run started a fresh optimizer, scheduler, and local step count. All **22,607,356 model parameters** were trainable; this was full-model fine-tuning rather than adapter training or partial freezing.

The architecture used:

- four refinement iterations;
- 12-frame sliding windows with stride 4 inside each 24-frame training sample;
- 128-dimensional image features;
- six temporal and six spatial transformer blocks;
- 64 virtual tracks;
- FlashAttention enabled;
- four correlation pyramid levels and 16 neighbours per level.

This run predates the later hybrid compiled-forward/fused-backward correlation implementation and the later training-path cleanup. Its throughput therefore includes the original indexed-correlation forward and expensive diagnostics that were subsequently optimized or throttled. The learned checkpoints remain valid, but this run's speed should not be treated as current-code throughput.

### Dataset snapshot and split

The source dataset was the private Hugging Face dataset `j99999/diegesis` at immutable revision:

```text
81389015a6d713a848a120e34850f360621bcdce
```

The verified source snapshot contains 570 payload files totaling 29,298,917,033 bytes. On Dopey it is stored at:

```text
/media/data3/jthakwani/datasets/diegesis
```

The MV-Tracker-ready tree and reusable JPEG index are stored at:

```text
/media/data3/jthakwani/datasets/diegesis-mvtracker/TAPVid3D_raw
/media/data3/jthakwani/datasets/diegesis-mvtracker/TAPVid3D_MVTracker_cache
```

The cached tree is approximately 1.2 GB and contains 21 manifests, 84 JPEG byte stores, and 84 corresponding 241-entry offset arrays. The split is fixed by `configs/diegesis_split_v1.json` using the recorded room-diverse SHA-256 procedure.

#### Training scenes — 17

```text
bathroom01  bathroom02  bathroom03
bedroom02   bedroom03   bedroom04
diningroom01 diningroom03 diningroom04
kitchen01   kitchen02   kitchen03   kitchen04
livingroom01 livingroom03 livingroom04 livingroom05
```

#### Validation scenes — 2

```text
diningroom02
bedroom01
```

#### Test scenes — 2

```text
bathroom04
livingroom02
```

The DIEGESIS test scenes were not evaluated during this experiment. Training-time evaluation used only the two validation scenes.

### Exact resolved training configuration

The authoritative resolved Hydra configuration is preserved inside the run at `.hydra/config.yaml`. The two command-line overrides were only the experiment selection and output directory:

```yaml
- +experiment=diegesis
- experiment_path=/media/data3/jthakwani/mvtracker-runs/diegesis-proof-20260813T111318Z-52ac9be
```

The important resolved values were:

| Setting | Value |
|---|---:|
| Optimizer steps | 2,000 intended |
| Physical scene batch | 1 |
| Gradient accumulation | 8 microbatches per optimizer step |
| Maximum trajectories per sample | 256 |
| Frames per sampled clip | 24 |
| Available views | 4 |
| Sampled view count | Uniform over 1, 2, 3, or 4 views |
| Input crop | 384 × 512 |
| Precision | BF16 mixed precision |
| Peak learning rate | 0.0005 |
| LR schedule | Linear annealing, `gamma=0.8` |
| Weight decay | 0.00001 |
| Gradient clipping | 1.0 |
| Visibility loss weight | 0.1 |
| Train/eval refinements | 4 / 4 |
| Validation frequency | Every 250 optimizer steps, plus startup |
| Checkpoint frequency | Every 500 optimizer steps |
| Reproducibility seed | 36 |
| Deterministic CUDA mode | Disabled |
| Training workers | 8 |
| Validation workers | 4 |

The run used ground-truth DIEGESIS depth as its depth source. `variable_depth_type` was false, so estimated depth was never substituted. Depth augmentation remained enabled: on augmented samples, the clean depth could receive the standard synthetic corruption/occlusion augmentation.

### Sampling behavior

The loader exposed 17 real training scenes through a virtual dataset of 24,000 requests. Only 14,432 requests were consumed before the failure:

```text
1,804 completed optimizer updates × 8 microbatches = 14,432 scene microbatches
```

That is approximately 849 scene draws per training scene on average. A draw was not a repeated fixed tensor. For every virtual sample, the loader deterministically derived a unique random seed and sampled:

- one of the 17 scenes;
- a contiguous 24-frame crop from the 240-frame sequence, giving 217 legal temporal starts;
- 1–4 views, with each view count assigned probability 0.25, followed by a subset of the available cameras;
- a new set of eligible trajectories and query times;
- stochastic augmentations.

The shared augmentation gate fired with probability 0.8. When active, it controlled the enabled RGB, depth, trajectory-count, crop, scene-transform, and camera-parameter augmentations. RGB augmentation included colour jitter, blur, erasing, and replacement patches. Spatial transformations updated RGB, depth, visibility, projected trajectories, and intrinsics consistently.

Trajectory sampling capped samples at 256 tracks but deliberately varied the realized count on augmented examples. Every 20th virtual sample used 32 trajectories; most other augmented samples sampled between 64 and 256, bounded by eligibility. A trajectory was eligible only when it was visible in at least two frames and was visible at the beginning or midpoint of the sampled window. One quarter of query times were selected randomly from eligible visible times; the remainder used the first eligible time after reserving the last visible observation.

The recorded 24-frame windows contained, on average:

| Sampling statistic | Mean per optimizer step |
|---|---:|
| Static tracks (`< 0.01 m`) | 29.721 |
| Dynamic tracks (`> 0.1 m`) | 136.114 |
| Very dynamic tracks (`> 2.0 m`) | 0.094 |
| Mean visible path length | 0.387 m |

The almost absent very-dynamic bucket is an empirical property of these 24-frame DIEGESIS crops; it is not evidence that the logger failed.

### Optimization behavior

The following values are trailing 50-step means from the TensorBoard event file. `live_total_loss` is the logged combined optimization loss; the components are the separately logged trajectory and visibility contributions.

| Optimizer-step window | 3D trajectory loss | Visibility loss | Total loss | Mean LR |
|---:|---:|---:|---:|---:|
| 0–49 | 0.05639 | 0.27810 | 0.33449 | 0.000138 |
| 50–99 | 0.04681 | 0.25452 | 0.30132 | 0.000368 |
| 200–249 | 0.04298 | 0.24037 | 0.28336 | 0.000470 |
| 450–499 | 0.03948 | 0.22794 | 0.26742 | 0.000407 |
| 950–999 | 0.03684 | 0.22308 | 0.25992 | 0.000282 |
| 1450–1499 | 0.03447 | 0.21381 | 0.24828 | 0.000156 |
| 1700–1749 | 0.03381 | 0.21382 | 0.24763 | 0.000094 |

The run showed a clear initial optimization signal. The 3D trajectory loss fell by roughly 40% from the first 50-step window to steps 1700–1749. Visibility loss also declined. Improvement slowed substantially after the early phase, which is consistent with the learning-rate schedule and the normal transition from easy initial gains to smaller later updates; it is not by itself evidence of overfitting.

Mean recorded optimizer-step timing across all 1,804 completed updates was:

| Component | Mean |
|---|---:|
| Total step | 10.568 s |
| Data wait | 2.031 s |
| Forward | 3.704 s |
| Backward/update | 4.734 s |
| GPU JPEG decode | 47.20 ms |
| GPU batch preparation | 256.03 ms |

This corresponds to approximately 0.757 microbatches/second or 341 optimizer steps/hour. The timing contains diagnostic overhead that is no longer representative of the optimized training path.

### Held-out DIEGESIS validation

Validation ran before training and then every 250 optimizer steps through step 1,750. These are the aggregate `any`-motion metrics across `bedroom01` and `diningroom02`:

| Step | Consumed microbatches | Average Jaccard ↑ | Delta-average ↑ | Occlusion accuracy ↑ | MTE (cm) ↓ | ATE (cm) ↓ |
|---:|---:|---:|---:|---:|---:|---:|
| Initial | 0 | 62.20 | 87.62 | 74.78 | 5.25 | 5.70 |
| 250 | 2,000 | 64.23 | 88.42 | 76.65 | 5.11 | 5.56 |
| 500 | 4,000 | 64.32 | 90.30 | 75.38 | 4.54 | 4.94 |
| 750 | 6,000 | 64.81 | 90.86 | 75.22 | 4.28 | 4.71 |
| 1,000 | 8,000 | 65.19 | 91.36 | 75.62 | 4.20 | 4.68 |
| 1,250 | 10,000 | 65.39 | 91.46 | 75.88 | 4.15 | 4.49 |
| 1,500 | 12,000 | 66.23 | 91.65 | 76.61 | 4.06 | 4.38 |
| 1,750 | 14,000 | **67.44** | **92.40** | **77.39** | **3.99** | **4.28** |

There was **no observed in-domain overfitting through step 1,750**. Average Jaccard and Delta-average improved at every recorded validation, while MTE and ATE decreased at every recorded validation. Occlusion accuracy briefly fluctuated early but finished at its best value.

This conclusion is limited by the validation set containing only two generated scenes. It establishes a signal on held-out DIEGESIS scenes; it does not establish broad generalization.

### External capability-regression evaluation

The original clean-depth checkpoint and the DIEGESIS step-1,000 and step-1,500 checkpoints were later evaluated on 30 MV-Kubric, six Panoptic, and ten DexYCB benchmark sequences. The selected hybrid correlation forward was used for these evaluations; it preserves the original PyTorch forward expression while optimizing training-side backward computation.

| Benchmark | Metric | Original clean-depth | DIEGESIS step 1,000 | DIEGESIS step 1,500 |
|---|---:|---:|---:|---:|
| MV-Kubric | AJ ↑ | 79.32 | 49.18 | 52.48 |
| MV-Kubric | Delta-average ↑ | 88.11 | 67.88 | 70.05 |
| MV-Kubric | Occlusion accuracy ↑ | 93.20 | 76.37 | 79.05 |
| MV-Kubric | MTE (cm) ↓ | 0.77 | 2.79 | 2.68 |
| Panoptic | AJ ↑ | 85.01 | 82.33 | 82.89 |
| Panoptic | Delta-average ↑ | 94.17 | 91.91 | 92.27 |
| Panoptic | Occlusion accuracy ↑ | 92.12 | 91.69 | 91.98 |
| Panoptic | MTE (cm) ↓ | 3.27 | 3.93 | 3.85 |
| DexYCB | AJ ↑ | 56.40 | 51.58 | 52.21 |
| DexYCB | Delta-average ↑ | 66.44 | 61.50 | 61.80 |
| DexYCB | Occlusion accuracy ↑ | 90.33 | 90.26 | 90.14 |
| DexYCB | MTE (cm) ↓ | 5.76 | 6.20 | 6.70 |

The run therefore demonstrated two things simultaneously:

1. DIEGESIS provided a real training signal and continuously improved held-out DIEGESIS validation.
2. DIEGESIS-only fine-tuning caused substantial forgetting, especially on MV-Kubric, and smaller regressions on the two real benchmarks.

The first external checkpoint evaluated was step 1,000. Consequently, this experiment proves that forgetting was already severe by step 1,000, but it cannot locate when forgetting began between steps 0 and 1,000. Step 1,500 was actually better than step 1,000 on all reported MV-Kubric and Panoptic metrics, so the external regression was not monotonically worsening with step count.

### Failure at step 1,804

The run completed optimizer step 1,803. The next sampled microbatch was `livingroom01` with four views, 24 frames, and the maximum 256 trajectories. Its forward pass attempted one additional 20 MiB allocation and failed:

```text
GPU capacity:                 23.56 GiB
Free at failure:              19.00 MiB
In use including non-PyTorch: 23.53 GiB
Allocated by PyTorch:         21.56 GiB
Reserved but unallocated:      1.61 GiB
Requested allocation:         20.00 MiB
```

The failure was a memory-capacity/fragmentation event on a maximum-shape sample, not a dataset-validation failure. The process saved both a resumable crash checkpoint and the exact crashing batch. No step-2,000 checkpoint exists.

Checkpoint filenames use the loop index, while the stored `total_steps` counter is one greater after completed updates:

| File | Stored `total_steps` | Size |
|---|---:|---:|
| `model_000000.pth` | 1 | 271,697,278 bytes |
| `model_000500.pth` | 501 | 271,697,278 bytes |
| `model_001000.pth` | 1,001 | 271,697,278 bytes |
| `model_001500.pth` | 1,501 | 271,697,278 bytes |
| `test_001804.pth` | 1,804 | 271,697,278 bytes |

For consistency with the evaluation reports and historical discussion, `model_001000.pth` and `model_001500.pth` are called “step 1,000” and “step 1,500” checkpoints even though their internal post-update counters are 1,001 and 1,501.

### Artifact map

#### Primary run directory on Dopey

```text
/media/data3/jthakwani/mvtracker-runs/diegesis-proof-20260813T111318Z-52ac9be
```

Important contents:

```text
.hydra/config.yaml                     exact resolved configuration
.hydra/overrides.yaml                  exact command-line overrides
train.log                              19.99 MB detailed training log
runs_0/events.out.tfevents.*           38.79 MB TensorBoard telemetry
model_000000.pth                       periodic training checkpoint
model_000500.pth                       periodic training checkpoint
model_001000.pth                       evaluated training checkpoint
model_001500.pth                       evaluated training checkpoint
test_001804.pth                        crash-time model/optimizer/scheduler state
crash_batch_step_001804.pt             exact 302.50 MB batch that triggered OOM
train_tapvid3d-multiview-training/     training visualizations
eval_tapvid3d-multiview-validation/    metrics, predictions, and validation MP4s
```

The validation directory contains per-step aggregate and per-sequence CSV files, prediction archives for both validation scenes, and MP4 visualizations for the startup evaluation and steps 250–1,750.

#### UCL launcher and dashboard logs

```text
/media/data3/jthakwani/mvtracker-runs/diegesis-proof-20260813T111318Z-52ac9be.ucl.log
/media/data3/jthakwani/mvtracker-runs/diegesis-proof-20260813T111318Z-52ac9be-dashboard.ucl.log
```

#### Later external evaluation output on Dopey

```text
/media/data3/jthakwani/mvtracker-evals/mvtracker-clean-regression-20260814T124340Z/full/original-gt
/media/data3/jthakwani/mvtracker-evals/mvtracker-clean-regression-20260814T124340Z/full/step1000
/media/data3/jthakwani/mvtracker-evals/mvtracker-clean-regression-20260814T124340Z/full/step1500
```

#### Local copies of external comparison reports

```text
artifacts/mvtracker-clean-regression-20260814T124340Z/report/
artifacts/mvtracker-clean-regression-20260814T124340Z/report-custom-triton/
```

The `report/` directory is the accepted comparison. The custom handwritten Triton forward was later rejected as the default because it altered predictions; its separate report is retained only as an implementation study.

### Final interpretation

This experiment succeeded as a proof of training signal but failed as a retention strategy.

- DIEGESIS training and validation losses improved.
- Held-out DIEGESIS metrics improved continuously through the last validation.
- There was no measurable DIEGESIS validation overfitting before termination.
- Severe MV-Kubric forgetting and smaller real-benchmark regressions were already present at step 1,000.
- The run terminated 196 intended updates early because of a single-sample CUDA OOM.
- Its performance timings should not be compared directly with the current optimized training path.

The clean follow-up experiment is therefore DIEGESIS fine-tuning with MV-Kubric replay, keeping the total update budget fixed and measuring both held-out DIEGESIS adaptation and MV-Kubric retention.

## 2026-08-15 — Modal H100 single-device capacity profile

### Question

Measure how many trajectories per scene fit for the four proposed homogeneous-view microbatch shapes on one H100, without using DDP or more than one GPU. Every shape represents four scenes per optimizer update:

| Views per scene | Physical batch | Accumulation |
|---:|---:|---:|
| 1 | 4 | 1 |
| 2 | 4 | 1 |
| 3 | 2 | 2 |
| 4 | 2 | 2 |

The search used the real clean-depth checkpoint, BF16 mixed precision, the optimized hybrid forward, real forward/loss/backward/gradient clipping/optimizer steps, 24 frames, and 384×512 MV-Kubric training samples. Candidate trajectory counts were 256 through 2,048 in increments of 256. A candidate was accepted only when observed peak GPU memory was no more than 90% of physical memory. The ceiling was probed first, followed by binary search and a confirmation run with two warm-up and three measured updates.

### Data and execution

The source revision was `5d450f267b40cfba32bb11a3e2800d592d4dccd1`. Modal used exactly one H100; no DDP or second GPU was launched. Before every GPU launch, active UCL Prism containers were counted and the launch was held unless a workspace slot was visibly free.

To prevent an H100 from waiting on repeated MV-Kubric decoding, a CPU-only job first selected exact 2,048-trajectory samples and saved one reusable batch per shape. Smaller candidate counts used a deterministic prefix of the same batch. This makes the reported timing a model-step benchmark; it intentionally excludes steady-state dataset and JPEG decoding throughput.

Artifacts:

```text
Modal data volume: jeet-mvtracker-data-v2/profile-batches/
Modal run volume:  jeet-mvtracker-runs-v2/profile-20260815T210133Z/
Summary:           jeet-mvtracker-runs-v2/profile-20260815T210133Z/summary.json
```

W&B:

- Data setup: https://wandb.ai/jeetucl-ucl/mvtracker-modal-profiling/runs/uzdjjkyk
- CPU batch preparation: https://wandb.ai/jeetucl-ucl/mvtracker-modal-profiling/runs/494gjy7f
- Final cached smoke: https://wandb.ai/jeetucl-ucl/mvtracker-modal-profiling/runs/ksrsqedm
- Final capacity sweep: https://wandb.ai/jeetucl-ucl/mvtracker-modal-profiling/runs/4dzchbit

### Results

| Views | Batch | Accum. | Selected trajectories/scene | Peak memory | Peak fraction | Median update | Scenes/s | Trajectories/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 1 | 768 | 73.87 GB | 86.89% | 1,124.6 ms | 3.56 | 2,732 |
| 2 | 4 | 1 | 512 | 71.46 GB | 84.05% | 1,199.6 ms | 3.33 | 1,707 |
| 3 | 2 | 2 | 1,280 | 72.16 GB | 84.87% | 2,444.6 ms | 1.64 | 2,094 |
| 4 | 2 | 2 | 1,024 | 70.68 GB | 83.13% | 2,539.6 ms | 1.58 | 1,613 |

Observed upper boundaries:

- 1 view: 1,024 trajectories OOM; 768 confirmed safe.
- 2 views: 768 trajectories OOM; 512 confirmed safe.
- 3 views: 1,536 trajectories reached 97.11% memory and was rejected; 1,280 confirmed safe.
- 4 views: 1,280 trajectories reached 94.28% memory and was rejected; 1,024 confirmed safe.

The selected counts are the largest safe values on the tested 256-trajectory grid, not mathematical maxima. These are per-device capacities. With two-GPU DDP, each rank should use the same local shape; DDP should not assign different view-count shapes to different ranks within one synchronized optimizer step.

## 2026-08-16 — Common-stack H100/H200/B200 economics profile

### Question

For homogeneous-view scene batching, compare the largest safe physical batch and confirmed training-update throughput on H100, H200, and B200 at exactly 1,024 and 2,048 trajectories per scene. The comparison must use one common software image, one GPU at a time, the same cached inputs, real forward/loss/backward/clipping/optimizer work, and no DDP.

### Controlled setup

The source revision was `bb4f4a18baf81b2cfc9e30c2ce2f9b456a68d8cc`. The common image used CUDA 12.8.1, Python 3.10.13, PyTorch 2.7.1+cu128, Triton 3.3.1, FlashAttention 2.8.3.post1 compiled for SM 90 and SM 100, PointOps at revision `2082918`, and SpConv 2.3.6. H100 was requested as `H100!` to prevent an automatic upgrade.

The CPU preparation job cached batch 8 with 2,048 trajectories for each of 1–4 views. Every GPU trial sliced a deterministic scene and trajectory prefix from those same tensors. Physical batches 1–8 were searched at accumulation 1 under the 90% peak-memory rule; the selected batch was confirmed with two warm-ups and three measured updates. Data decoding and host input throughput are deliberately excluded.

Only one profiler GPU ran at a time because unrelated Prism jobs occupied the other workspace slots. No unrelated container was stopped or interrupted.

### Compatibility and artifacts

All three GPUs passed import compatibility and a real 1-view, batch-1, 1,024-trajectory forward/backward/optimizer smoke.

| GPU | Compatibility W&B | Smoke W&B | Full profile W&B | Modal run volume |
|---|---|---|---|---|
| H100 | `sq7jtdbr` | `3nnmd0t2` | `hlhmrkyn` | `profile-h100-20260816T010300Z/` |
| H200 | `1punq4kx` | `63omrpoy` | `0rrq751k` | `profile-20260816T004050Z/` |
| B200 | `y9szbc5z` | `7pcfhafd` | `sf6derbt` | `profile-20260816T002252Z/` |

The paths above are under the `jeet-mvtracker-runs-v2` Modal Volume. Cached inputs are under `jeet-mvtracker-data-v2/profile-batches/`.

### Confirmed results

Each GPU cell is `selected batch / peak VRAM fraction / trajectories per second / nominal dollars per million trajectories`. Nominal economics use Modal rates observed for this run: H100 $3.95/h, H200 $4.54/h, and B200 $6.25/h.

| Views | Tracks/scene | H100 | H200 | B200 |
|---:|---:|---:|---:|---:|
| 1 | 1,024 | 3 / 81.95% / 2,943 / $0.373 | 5 / 79.98% / 3,380 / $0.373 | 7 / 88.54% / 4,148 / $0.418 |
| 1 | 2,048 | 1 / 51.58% / 2,792 / $0.393 | 3 / 85.61% / 3,941 / $0.320 | 3 / 67.19% / 4,595 / $0.378 |
| 2 | 1,024 | 2 / 65.64% / 2,219 / $0.494 | 4 / 72.68% / 2,775 / $0.454 | 6 / 85.36% / 3,413 / $0.509 |
| 2 | 2,048 | 1 / 56.69% / 2,504 / $0.438 | 2 / 62.97% / 3,276 / $0.385 | 3 / 73.86% / 4,086 / $0.425 |
| 3 | 1,024 | 2 / 75.93% / 1,955 / $0.561 | 4 / 82.60% / 2,345 / $0.538 | 5 / 80.75% / 2,773 / $0.626 |
| 3 | 2,048 | 1 / 61.61% / 2,338 / $0.469 | 2 / 68.28% / 2,917 / $0.432 | 3 / 79.81% / 3,641 / $0.477 |
| 4 | 1,024 | 2 / 82.99% / 1,695 / $0.647 | 3 / 69.64% / 1,979 / $0.637 | 4 / 74.48% / 2,336 / $0.743 |
| 4 | 2,048 | 1 / 65.32% / 2,144 / $0.512 | 2 / 72.92% / 2,650 / $0.476 | 3 / 86.87% / 3,260 / $0.533 |

### Interpretation

- B200 delivered the highest absolute throughput in every tested shape, about 38–65% above H100, and fit the largest batches.
- H200 was the best economic choice at 2,048 trajectories for every view count and was best or tied at 1,024. Its modest price premium bought useful batch capacity without B200's larger hourly premium.
- B200 is justified when elapsed time or per-device batch capacity matters more than lowest cost. H200 is the default recommendation for these short training experiments.
- At all view counts, 2,048 trajectories per scene improved trajectory throughput economics over 1,024 on the same GPU, despite reducing the number of scenes in the physical batch.

The tagged Modal billing report total for the common-image setup, cache preparation, compatibility checks, smokes, and all three full sweeps was $4.74. H100 was launched through the direct remote function during this run, so its fine-grained app tag inherited the old generic `gpu=cpu, experiment=common-stack` values; ownership/project/purpose tags and W&B metadata remained correct. The profiler was subsequently changed so direct invocations are truthfully `unclassified` and the documented local entrypoints attach exact GPU/experiment tags.

## 2026-08-16 — Five/six-view H200 and B200 frontier

### Question and setup

Measure single-GPU H200 and B200 capacity for the six-view MV-Kubric training regime that was absent from the 1–4-view economics profile. The run used real cached MV-Kubric tensors, 24 frames, 384×512 images, BF16 mixed precision, the optimized hybrid forward, and complete forward/loss/backward/clipping/optimizer updates. Physical batch was searched through 12 at 1,024 and 2,048 trajectories per scene. Single-scene trajectory capacity was searched in 512-track increments under the same 90% peak-VRAM acceptance rule.

Only the two profiler GPUs ran concurrently. The unrelated `adapt-vqa` container was left untouched. All Modal applications carried `owner=jeet`, `project=mvtracker`, and `purpose=profiling` tags.

Artifacts and W&B:

- H200: `jeet-mvtracker-runs-v2/frontier56-h200-20260816-r2/`; https://wandb.ai/jeetucl-ucl/mvtracker-modal-profiling/runs/3hiu3492
- B200: `jeet-mvtracker-runs-v2/frontier56-b200-20260816-r2/`; https://wandb.ai/jeetucl-ucl/mvtracker-modal-profiling/runs/4pj76qxi

### Confirmed batch frontier

| GPU | Views | Tracks/scene | Max safe batch | Peak VRAM | Median update | Scenes/s | Tracks/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| H200 | 5 | 1,024 | 3 | 76.91% | 1,786.6 ms | 1.679 | 1,719 |
| H200 | 5 | 2,048 | 2 | 77.10% | 1,704.0 ms | 1.174 | 2,404 |
| H200 | 6 | 1,024 | 3 | 84.96% | 2,029.1 ms | 1.479 | 1,514 |
| H200 | 6 | 2,048 | 2 | 82.21% | 1,874.0 ms | 1.067 | 2,186 |
| B200 | 5 | 1,024 | 4 | 80.25% | 2,127.2 ms | 1.880 | 1,926 |
| B200 | 5 | 2,048 | 2 | 60.53% | 1,622.7 ms | 1.233 | 2,524 |
| B200 | 6 | 1,024 | 4 | 88.57% | 2,365.6 ms | 1.691 | 1,731 |
| B200 | 6 | 2,048 | 2 | 64.47% | 1,717.9 ms | 1.164 | 2,384 |

### Single-scene trajectory frontier

| GPU | Views | Confirmed tracks | Peak VRAM | Median update | Tracks/s | Boundary status |
|---|---:|---:|---:|---:|---:|---|
| H200 | 5 | 6,144 | 88.83% | 1,814.5 ms | 3,386 | Practical tested ceiling; the next 512-track point was unavailable in the prepared real-scene cache. |
| H200 | 6 | 5,632 | 84.72% | 1,802.5 ms | 3,125 | Exact tested maximum: 6,144 used 91.58% and was rejected. |
| B200 | 5 | at least 6,144 | 69.75% | 1,707.5 ms | 3,598 | Data-limited lower bound, not a GPU maximum. |
| B200 | 6 | at least 6,144 | 71.74% | 1,835.2 ms | 3,348 | Data-limited lower bound, not a GPU maximum. |

Attempts to prepare unique 8,192–12,288-track samples were stopped after the real MV-Kubric microset could not produce the requested motion-balanced sample promptly. No duplicated or synthetic trajectories were substituted. Consequently the B200 single-scene numbers are explicitly lower bounds.

### Takeaway

B200 buys one extra scene at 1,024 tracks for both five and six views, but it does not increase the selected batch at 2,048 tracks. Its throughput gain over H200 is about 5–14% for these confirmed shapes while its hourly price is about 38% higher. H200 remains the better cost default; B200 is useful when the extra 1,024-track scene per update or higher per-device trajectory ceiling matters.

## 2026-08-16 — VGGT-Omega temporal multi-view preprocessing smoke

### Question and correction

The first preprocessor treated timestamps as independent batch elements with
input shape `[B=timestamps, S=views, 3, H, W]`. VGGT-Omega therefore exchanged
information across cameras at one timestamp but never across time, and every
timestamp received an independent reconstruction gauge and metric scale.

The corrected path uses one scene sequence with timestamp-major ordering:
`[B=1, S=timestamps×views, 3, H, W]`. It estimates one camera-centre Sim(3)
alignment per temporal chunk, scales camera-Z depth into metres, and writes
rigid world-to-camera matrices consistent with the scaled depth. Production
defaults to 24 timestamps per chunk; the Dopey run below deliberately used
smaller chunks and did not generate a complete sidecar.

### Bounded Dopey smoke

Source revision: `30dbd591bfbe74abab89267b6066c5b511ce9133`.
Checkpoint SHA-256:
`c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934`.
Both trials ran sequentially on one RTX 3090 and left all other jobs untouched.

| Dataset | Joint input | Inference | Cleaned AbsRel | Median estimated/GT | Cleaned coverage | Mean camera-centre RMSE |
|---|---:|---:|---:|---:|---:|---:|
| DIEGESIS `kitchen03` frames 0–7 | 8 timestamps × 4 views = 32 images | 3.70 s | 7.52% | 0.970 | 77.74% | 0.457 m |
| MV-Kubric `900` frames 0–3 | 4 timestamps × 10 views = 40 images | 5.31 s | 1.45% | 1.000 | 78.98% | 0.156 m |

Remote outputs are under
`/media/data3/jthakwani/smoke/vggt-omega-temporal-20260816T142515Z`.
Local JSON and contact sheets are under
`artifacts/vggt-omega-temporal-smoke/`. The robust metric scale is correct on
both datasets. DIEGESIS still has large absolute-error outliers at very distant
window/background pixels, but its median depth and relative error remain
well-scaled; those pixels were not clipped or replaced with ground truth.

## 2026-08-16 — VGGT-Omega H100 throughput profile

### Setup

The profile ran in one tagged Modal H100 container. DIEGESIS was downloaded
directly from Hugging Face to the container SSD using Xet; only MV-Kubric scenes
900–903 were copied from the network Volume. All inference and temporary depth
writes used local SSD. Inputs were 24-timestamp temporal chunks at 512-pixel
model resolution: 96 images for each four-view DIEGESIS chunk and 240 images for
each ten-view MV-Kubric chunk. Each measured point used one warm-up and three
timed repetitions.

Artifacts:

- W&B: https://wandb.ai/jeetucl-ucl/mvtracker-modal-profiling/runs/5dx19ozn
- Modal Volume: `jeet-mvtracker-runs-v2/vggt-omega-h100-throughput-20260816-xet2/report.json`

Staging took 184.60 seconds in total. The 29.30 GB DIEGESIS snapshot downloaded
in 95.25 seconds; the four staged MV-Kubric scenes occupied 587.81 MB. The whole
profile took 1,315.84 seconds (21.93 minutes), approximately $1.44 at $3.95 per
H100-hour.

### Results

Eight loader workers were fastest for both datasets. Scene batching did not
increase throughput: model cost scaled approximately linearly with batch size,
while reserved VRAM rose sharply.

| Dataset | Batch | Total/chunk | Scenes/s | Peak reserved | Status |
|---|---:|---:|---:|---:|---|
| DIEGESIS, 4 views | 1 | 4.675 s | 0.2139 | 20.62% | safe |
| DIEGESIS, 4 views | 2 | 9.516 s | 0.2102 | 36.02% | safe |
| DIEGESIS, 4 views | 4 | 18.901 s | 0.2116 | 66.37% | safe |
| DIEGESIS, 4 views | 6 | 28.462 s | 0.2108 | 96.73% | rejected by 90% rule |
| MV-Kubric, 10 views | 1 | 22.351 s | 0.04474 | 42.41% | safe |
| MV-Kubric, 10 views | 2 | 44.963 s | 0.04448 | 79.13% | safe |
| MV-Kubric, 10 views | 3 | 67.868 s | 0.04420 | 98.44% | rejected by 90% rule |

The production recommendation is batch size 1 with eight loader workers. It is
the fastest measured setting per scene, leaves substantial VRAM headroom, and
avoids gaining no throughput from larger batches. The ten-view MV-Kubric
sequence is slower per image than four-view DIEGESIS because the joint sequence
is 240 rather than 96 images and the model's cross-image processing is not
linear in sequence length.

For 17 DIEGESIS scenes of 240 frames (ten chunks each) plus 100 MV-Kubric scenes
of 24 frames (one chunk each), the measured batch-1 rates project to 50.5 minutes
of H100 inference, or about $3.32. This excludes one-time input staging and the
final upload of complete sidecars to persistent storage.

## 2026-08-17 — Cached Modal dataset image and CPU loader profile

The continual-training data is now a cached Modal image layer at
`/opt/mvtracker-data`, built from the archive Volume before the commit-specific
Git checkout layer. Training no longer mounts the data Volume or copies and
extracts archives at container startup. The dataset image is
`im-oWfJU0ZwiRztxAnVw5Tbcz`, version
`diegesis21-mvkubric100-val101-102-v1`. Its successful CPU build ran at
https://modal.com/apps/ucl-prism/main/ap-pJu2cabHTuVIgSG7nZDfei. Parallel
extraction plus filesystem capture took about ten minutes; saving the expanded
dataset layer itself took 53.78 seconds. A later source-only commit reused this
dataset layer without re-extraction.

The CPU validation loaded valid samples from both DIEGESIS and MV-Kubric and
recorded cold and warmed single-worker loader measurements. The run is at
https://wandb.ai/jeetucl-ucl/mvtracker-continual-training/runs/6qvj3tqr and the
Modal invocation is at
https://modal.com/apps/ucl-prism/main/ap-LA9zfyNtqUyRXjKbDklq3V.

| Dataset | Phase | Valid samples | Rejected samples | Samples/s | Median | p95 |
|---|---|---:|---:|---:|---:|---:|
| DIEGESIS | cold | 4 | 0 | 1.782 | 0.768 s | 0.768 s |
| DIEGESIS | warm | 32 | 2 | 0.611 | 0.630 s | 5.219 s |
| MV-Kubric | cold | 4 | 0 | 0.783 | 1.521 s | 1.521 s |
| MV-Kubric | warm | 32 | 0 | 0.815 | 1.324 s | 1.903 s |

The two DIEGESIS rejections are normal dataset-window rejections and were
resampled, matching the training and CUDA loader behavior. These measurements
are CPU-only validation of image availability and loader behavior; no H100
throughput benchmark was run.

## 2026-08-17 — Asynchronous CUDA loader profile

The production encoded loader now prepares samples in a bounded background
queue, batches compatible view counts for decode, runs RGB and depth decode on
separate CUDA streams, and uses per-group completion events. The training thread
therefore waits only for the sample it consumes. The production-safe bounds are
four queued source batches and two source batches per decode submission.

The final profile used one tagged T4, 8 loader workers, 4 warm-up samples and 16
measured samples per case. The first queue-8/decode-4 trial completed each fixed
source but exhausted the 15 GB T4 when both source queues were active; it was
replaced by the bounded configuration below.

- W&B: https://wandb.ai/jeetucl-ucl/mvtracker-modal-profiling/runs/pe2dyzla
- Modal artifact: `jeet-mvtracker-runs-v2/t4-loader-benchmark/t4-loader-async-bounded-8d7989a.json`
- Runtime: 3 minutes 46 seconds

| Case | Samples/s from exposed loader time | Wait p50 | Wait p95 |
|---|---:|---:|---:|
| DIEGESIS, 4 views | 1.556 | 0.795 s | 1.597 s |
| MV-Kubric, 4 views | 1.194 | 0.661 s | 2.296 s |
| MV-Kubric, 6 views | 0.816 | 0.971 s | 4.675 s |
| Alternating DIEGESIS/MV-Kubric, 4 views with 1.25 s compute | 7.873 | 0.0017 s | 0.984 s |

Against the valid short baseline, fixed-source throughput improved by 1.1% for
DIEGESIS-4, 5.5% for MV-Kubric-4 and 12.0% for MV-Kubric-6. More importantly,
the representative alternating schedule reduced aggregate exposed loader wait
from about 10.29 seconds to 2.03 seconds across 16 samples, an 80.3% reduction.
This benchmark measures loader exposure rather than model throughput; the
alternating case demonstrates that preparation is hidden when model compute is
available to overlap it.

## 2026-08-17 — Two-H100 asynchronous-loader training smoke

A fresh two-H100 DDP smoke completed exactly ten optimizer updates using the
mixed DIEGESIS/MV-Kubric GT-depth recipe, four local microbatches per rank and
eight scenes per global update. Evaluation was disabled. The run used the
bounded asynchronous CUDA loader after trimming per-group trajectory padding
back to each source sample's original track count.

- W&B: https://wandb.ai/jeetucl-ucl/mvtracker-continual-training/runs/bbc4bc074954
- Modal app: `ap-FGF8xz00ZUPGFKmBAENQaZ`
- Run Volume: `jeet-mvtracker-runs-v2/continual-training/loader-h100-smoke10-cbc606a/`
- Step-10 checkpoint: `model_000010.pth`
- Final checkpoint: `model_final.pth`

The Modal app started at 20:32:43 UTC. Distributed initialization began at
20:34:02, the trainer was ready at 20:34:28, and the first prepared batch
arrived at 20:35:05. This was 2 minutes 22 seconds from app creation to the
first batch, including 37.2 seconds of worker and import warmup after trainer
readiness.
The first update took 145.29 seconds because it also ran first-step compilation
and expensive diagnostics. Updates 2–10 averaged 8.95 seconds, including 1.00
second of exposed data wait.

| Step | Total | Data | Forward | Backward |
|---:|---:|---:|---:|---:|
| 1 | 145.29 s | 48.50 s | 18.16 s | 78.48 s |
| 2 | 9.25 s | 1.00 s | 4.13 s | 2.57 s |
| 3 | 4.96 s | 0.51 s | 2.37 s | 1.97 s |
| 4 | 11.10 s | 0.65 s | 4.95 s | 4.07 s |
| 5 | 12.98 s | 3.65 s | 5.61 s | 2.84 s |
| 6 | 6.41 s | 0.48 s | 2.81 s | 2.60 s |
| 7 | 8.44 s | 0.73 s | 4.71 s | 2.41 s |
| 8 | 11.20 s | 0.22 s | 7.08 s | 3.74 s |
| 9 | 5.65 s | 0.63 s | 2.39 s | 1.94 s |
| 10 | 10.58 s | 1.10 s | 5.33 s | 2.70 s |

At step 10 the global update contained 8 scenes and 8,233 trajectories,
yielding 0.756 scenes/s and 778.3 trajectories/s. The point-in-time hardware
sample reported GPU 0 at 56% utilization and 28.0/79.6 GiB, GPU 1 at 100% and
55.0/79.6 GiB, 4.34 CPU cores used, and 69.6 GiB container RAM. These are one
instant rather than sustained-utilization averages. The final exposed loader
wait was 1.10 seconds; loader-worker preparation was 1.11 seconds and reported
GPU JPEG decode time was 1.13 seconds.

Three duplicate launcher invocations occurred after successful completion.
They found `model_final.pth` at ten completed steps and performed no additional
optimizer updates. The final duplicate app was stopped manually after
confirming both checkpoints were durable.

## 2026-08-18 — Sequential mixed-sampler baseline

Before changing mixed sampling or introducing physical scene batching, the
current one-sample-at-a-time path was recorded on Dopey for five optimizer-step
groups. The run used seed 72, the real 17-scene DIEGESIS training split and
MV-Kubric scenes 900–997. It produced 40 accepted samples in about 20 seconds
on CPU and did not decode them on a GPU.

- Remote JSON: `/media/data3/jthakwani/mvtracker-runs/mixed-sampling-baseline-seed72-20260818/samples.json`
- Remote table: `/media/data3/jthakwani/mvtracker-runs/mixed-sampling-baseline-seed72-20260818/samples.md`
- Local copies: `artifacts/mixed-sampling-baseline-seed72-20260818/`

The current paired-rank retry behavior fired twice. DIEGESIS virtual samples
6/7 were discarded and replaced by 8/9 at step 1, microbatch 2; virtual samples
18/19 were discarded and replaced by 20/21 at step 4, microbatch 0. The other
three groups had no retries. W&B logging was requested but unavailable because
Dopey had no configured W&B API key; the JSON and Markdown artifacts completed
before that logging step.

## 2026-08-18 — Direct-dataset whole-step sampling parity

The sampler was extended with one whole-step call that selects all eight
ordinary requests for an optimizer update: four DIEGESIS and four MV-Kubric
across the two ranks. It does not batch tensors or alter either dataset's
sampling. Paired failures still advance both ranks together and refill only the
failed source.

The normal committed CLI was run CPU-only on Dopey for five steps with seed 72
and compared directly with the sequential baseline above. It completed in
18.40 seconds and produced 40/40 identical accepted sample records, including
scene, virtual index, seed, frame window, views, trajectory count, RGB/depth
augmentation flags, and retry attempts. The same two paired DIEGESIS retries
occurred: 6/7 to 8/9 and 18/19 to 20/21.

- Remote output: `/media/data3/jthakwani/mvtracker-runs/mixed-sampling-whole-step-seed72-20260818-direct/`
- Local copy: `artifacts/mixed-sampling-whole-step-seed72-20260818-direct/`
- Comparison: `identical=true`, expected 40, actual 40
- W&B was not used because no W&B credential is configured on Dopey.

This was a dataset-direct parity check. It did not pass through the production
DataLoaders, CUDA prefetch wrapper, or GPU decoders, so its 18.40-second runtime
must not be used as a training-loader throughput measurement.

## 2026-08-18 — Live training-loader whole-step sampling parity

Stage 2 was repeated through the actual mixed-training input path on Dopey:
two Fabric ranks, the production scheduled source samplers, eight workers per
source per rank, CUDA prefetch, nvImageCodec RGB/depth decoding, and the same
cross-rank failure reduction used by training. The exact pinned
`nvidia-nvimgcodec-cu12[nvtiff]==0.9.0.20` package first had to be installed in
the Dopey venv; the previous CPU-direct check had hidden that missing runtime
dependency.

The live run produced 40/40 records identical to the saved sequential baseline,
including the same four retry rows. Cold startup plus the first step took
46.682 seconds. Subsequent loader-only steps took 4.149, 2.332, 1.747, and
5.331 seconds (3.390-second mean). These timings expose loader work without
model compute overlap; they are authoritative for loader-path behavior but not
for final end-to-end step throughput.

- Remote output: `/media/data3/jthakwani/mvtracker-runs/mixed-sampling-live-stage2-seed72-20260818-r3/`
- Local copy: `artifacts/mixed-sampling-live-stage2-seed72-20260818/`
- Comparison: `identical=true`, expected 40, actual 40
- Step timings: `46.682, 4.149, 2.332, 1.747, 5.331` seconds
- W&B was not used because this was a bounded loader verifier and Dopey has no
  configured W&B credential.

## 2026-08-18 — Concurrent whole-step live-loader benchmark

The mixed trainer was changed to materialize both source streams for a complete
optimizer step concurrently. Each rank prepares its four local samples in the
existing DIEGESIS/MV-Kubric/DIEGESIS/MV-Kubric order; with two DDP ranks this is
the same eight-scene global update as before. Forward, backward, gradient
accumulation, loss scaling and optimizer behavior remain serial and unchanged.

The exact live-loader verifier was repeated on Dopey at commit `9b03850` using
an RTX 3090 and Titan X. All 40 accepted sample records matched the sequential
baseline exactly. The resulting `samples.json` SHA256 was
`47cab67a0ef4efd6c9f052134a672acdbc33358585c161048ed79dadc91ae0de`.

Cold startup plus the first step took 45.218 seconds. Warm loader-only steps
took 3.729, 1.048, 2.845 and 4.104 seconds, a 2.932-second mean. The prior live
path measured 3.390 seconds on the same five-step sequence, so concurrent
whole-step materialization reduced this bounded loader-only mean by 13.5%.
Because this check deliberately has no model compute, a two-H100 training smoke
is still required to determine whether the change improves end-to-end update
time or merely moves the exposed wait to the beginning of each update.

- Remote output: `/media/data3/jthakwani/mvtracker-runs/mixed-sampling-live-wholestep-seed72-20260818-r4/`
- Comparison: `identical=true`, expected 40, actual 40
- Step timings: `45.218, 3.729, 1.048, 2.845, 4.104` seconds
- W&B was not used because this was a bounded loader verifier on Dopey.

## 2026-08-18 — Matched eager-versus-lazy H100 loader A/B

The eager whole-step implementation and the prior asynchronous consumption
path were compared on two H100s using the same master seed (`2019407807`). The
two runs matched exactly at every optimizer step for global sample count,
DIEGESIS/MV-Kubric source counts, source view counts, source trajectory counts
and total global trajectories. This removed the workload mismatch in the
earlier smoke comparison.

Across warm optimizer steps 2–10, eager materialization reduced measured data
wait from 8.68 to 1.42 seconds in aggregate, but increased forward time from
40.88 to 64.60 seconds and backward time from 23.91 to 35.45 seconds. Total
warm-step time increased from 79.33 to 109.34 seconds. Lazy asynchronous
consumption was therefore 27.4% faster end-to-end. The first update was also
128.69 seconds with lazy consumption versus 241.94 seconds with eager
materialization.

The result rejects eager GPU materialization of all eight global samples as a
training optimization. It drains the ready queues before compute, creates a
large startup barrier, holds more decoded data resident and causes background
refill/decode work to overlap adversely with model compute. Production keeps
the deterministic eight-sample optimizer-step schedule but consumes prepared
batches lazily, allowing loader work to remain naturally overlapped. Eager
materialization remains available only as an explicit benchmark switch.

- Eager W&B: https://wandb.ai/jeetucl-ucl/mvtracker-continual-training/runs/785db7bf6150
- Lazy W&B: https://wandb.ai/jeetucl-ucl/mvtracker-continual-training/runs/f1aac438efd2
- Eager output: `jeet-mvtracker-runs-v2/continual-training/whole-step-loader-smoke10-20260818T091500Z/`
- Lazy output: `jeet-mvtracker-runs-v2/continual-training/lazy-loader-matched-smoke10-20260818/`

## 2026-08-18 — Planned physical loader on Dopey Titans

The mixed training input path was split into deterministic metadata planning
and payload materialization. Four complete optimizer updates are kept ahead in
CPU RAM. Within each eight-scene update, same-view scenes may form physical
batches of two regardless of their individual trajectory counts; trajectory
axes are padded to the larger scene and the padding mask excludes those slots
from the model and metrics. Pairing is admitted only by the measured H100
batch-two limits: 1/2/4 views at at most 1,024 tracks and 3 views at at most
1,280 tracks. Five- and six-view samples remain singletons.

The final bounded loader/decode profile ran concurrently on Dopey's physical
Titan X devices 1 and 2. It used four optimizer updates, 32 total scenes, four
CPU materialization workers per lane, CUDA RGB/depth streams, eight-image
decode chunks, and one pass. It did not run the model, optimizer, training, or
GPU 0. W&B was unavailable on Dopey, so the durable JSON and log are the run
record.

Artifacts:

```text
/media/data3/jthakwani/mvtracker-runs/mixed-physical-loader-final-20260818/
  plan.json
  profile.log
  report.json
```

| Metric | Titan 1 | Titan 2 |
|---|---:|---:|
| Scenes | 16 | 16 |
| Trajectories | 20,215 | 21,052 |
| Encoded payload | 1.455 GB | 1.350 GB |
| Wall time | 13.495 s | 13.967 s |
| CPU materialization | 0.936 s | 0.843 s |
| CUDA decode events | 12.163 s | 12.819 s |
| Exposed loader wait | 12.073 s | 12.742 s |
| Scenes/s | 1.186 | 1.146 |
| Trajectories/s | 1,498 | 1,507 |
| Peak sampled process RSS | 6.260 GiB | 6.744 GiB |
| Peak sampled GPU memory | 8.427 GiB | 8.441 GiB |

Metadata planning took 2.167, 3.481, 1.961 and 2.106 seconds for the four
updates; the four-step lookahead is intended to hide this behind training.
There was one deterministic invalid-plan retry. The scheduler used two safe
pairs: rank 0 executed 3/4/4/4 physical groups and rank 1 executed 4/3/4/4,
reducing 32 logical scene forwards to 30 physical forwards. The first pair had
256/256 tracks and no padding; the second had 701/912 tracks and 211 padded
slots. Allowing unequal
rank-local group counts was necessary: DDP only requires both ranks to meet at
the final synchronized backward, not to execute the same number of no-sync
forwards.

An earlier two-pass Titan check completed the cold pass but exhausted the
12 GiB Titan during repeated nvTIFF decoding in the warm pass. Bounded decode
chunks made allocation failures explicit and prevented null decoder outputs
from reaching DLPack. The successful one-pass profile peaked at 8.441 GiB.
This is a Titan loader validation, not an H100 end-to-end speed claim; the
physical batching limits remain those measured with full H100 training steps.

## 2026-08-18 — Physical batching two-H100 smoke

The production training path completed a bounded ten-update DDP smoke on two
H100s with seed 72. The run used the deterministic 4-DIEGESIS/4-MV-Kubric
logical optimizer step, four-step CPU planning lookahead, lazy RGB/depth
materialization, physical same-view batches of at most two scenes, gradient
accumulation and W&B logging. Evaluation was disabled. It completed without an
OOM, deadlock, sample-ratio drift or rank failure and released both GPUs.

The first update took 100.11 seconds because it included kernel compilation and
cold initialization. Across warm updates 2–10, mean optimizer-step time was
5.83 seconds and median was 5.64 seconds. Exposed data wait averaged 0.46
seconds. Meanwhile, CPU planning averaged 4.11 seconds and payload
materialization averaged 1.26 seconds, demonstrating that the four-step
lookahead hid most input work behind model computation rather than putting it
on the critical path. Warm throughput averaged 1,958 trajectories/s with a
median of 2,093 trajectories/s.

Five same-view pairs were formed across the 80 logical scene samples. This
reduced 80 logical forwards to 75 physical forwards, a 6.25% reduction, with
631 total padded trajectory slots. Exact global source counts remained four
DIEGESIS and four MV-Kubric scenes per optimizer update. At update 10, the
point-in-time hardware metrics were 79% utilization and 19.8/79.6 GiB on rank
0, and 100% utilization and 30.1/79.6 GiB on rank 1. Container RAM was 39.7
GiB.

This proves the physical-batching path is operational on the intended H100 DDP
setup and that planning is successfully overlapped. It is not a controlled
speedup measurement against the old loader because the earlier smoke used a
different workload seed and code state. A matched A/B would be required for an
exact speedup claim.

- Modal app: https://modal.com/apps/ucl-prism/main/ap-BdiMNdYYpEleiVtYHC64Zb
- W&B: https://wandb.ai/jeetucl-ucl/mvtracker-continual-training/runs/b4d7a46acca2
- Output: `jeet-mvtracker-runs-v2/continual-training/smoke10-physical-batching-7c6a46c-20260818T093720Z/`
- Billing tags: `owner=jeet`, `project=mvtracker`, `purpose=training`

## 20 August 2026: H200 universal-pair training restart

The preceding two-H100 run failed at optimizer step 44. GPU memory had reached
79.18/79.65 GiB; while the model retained the current batch's activations, the
one-group-ahead DALI decoder requested another contiguous 192 MiB and failed.
The stale rank was stopped after the failure.

The continual-training lane was moved to two H200s. Physical batching still
has a maximum of two scenes and the logical optimizer batch remains eight
scenes. Same-view scenes may now pair up to 2,048 tracks for views 1--4, 819
for view 5 and 512 for view 6; views 5 and 6 are no longer forced to remain
singletons. These limits use the existing H200 profiles.

The fresh run on source `532dbe83e8aab01bc56a614538e3b000f3e5784a`
completed distributed initial validation and optimizer step 1. Rank 0 evaluated
12 MV-Kubric scenes and rank 1 evaluated 15. Step 1 scheduled five physical
groups for eight logical scenes: one pair on rank 0 and two pairs on rank 1.
It took 40.70 seconds cold (16.98 data, 14.72 forward, 8.87 backward). Sampled
peak memory during that first forward was 37,925 MiB on GPU 0 and 105,127 MiB
on GPU 1. The run remained healthy past step 32; its median step time at that
point was 5.80 seconds.

- Run: `gt-replay-main-ddp2-h200-paired-532dbe8-20260820T220000Z`
- Modal app: `ap-r6vflpmIFACf2d8jT7RwB4`
- Function call: `fc-01M0GBJ5Q7ZD0PYMXT1M79NHR7`
- W&B: https://wandb.ai/jeetucl-ucl/mvtracker-continual-training/runs/8dd8bd8df50a
- Billing tags: `owner=jeet`, `project=mvtracker`, `purpose=training`

### Stopped-run audit and mmap correction

The H200 run was stopped by request after 526 completed updates. The durable
resume point is `model_000500.pth`; steps 501--526 exist only in telemetry.
Container RAM grew monotonically from 69.23 GiB at step 1 to 454.96 GiB at
step 500 and 470.28 GiB at step 520. The DALI WebDataset reader had retained
its default mmap behavior on the mounted network Volume. Source `2b4109c`
switches it to plain file I/O and changes the Modal training allocation to a
64 GiB request with a hard 128 GiB limit.

Trailing 50-update means changed as follows from steps 1--50 to 477--526:

| Source | Total loss | Trajectory loss | Visibility loss |
|---|---:|---:|---:|
| Combined | 0.25637 → 0.22589 | 0.05878 → 0.05162 | 0.19758 → 0.17427 |
| DIEGESIS | 0.34045 → 0.26618 | 0.05748 → 0.03704 | 0.28297 → 0.22914 |
| MV-Kubric | 0.17228 → 0.18560 | 0.06008 → 0.06620 | 0.11220 → 0.11939 |

DIEGESIS validation improved from step 0 to 500: AJ 66.84→71.76,
average points within threshold 88.84→93.30, ATE 5.30→3.93, and occlusion
accuracy 78.08→80.06. Most of the gain was already present at step 250.

For a strict MV-Kubric comparison, scenes 101--102 were recovered from the
step-0 full-validation per-scene rows and compared with the same subset at
steps 250 and 500:

| Step | AJ | Average points within threshold | ATE | Occlusion accuracy |
|---:|---:|---:|---:|---:|
| 0 | 71.78 | 82.02 | 7.86 | 91.62 |
| 250 | 69.33 | 80.19 | 8.84 | 91.13 |
| 500 | 68.46 | 79.58 | 9.03 | 90.59 |

The step-0 27-scene MV-Kubric aggregate is not used in that trend because the
later scheduled validations contain only scenes 101--102.

### Step-500 continuation after the mmap fix

The same run resumed from canonical `model_000500.pth` on source
`34ceaabf956c038df628b5f306e6dc0192dfbe83`. Model, optimizer, OneCycle
scheduler, completed step, master seed, W&B identity and source schedule state
were restored; telemetry-only steps 501--526 from the stopped process were not
loaded. MV-Kubric DALI streaming resumed at saved cursor 1005 (group 125,
offset 5) rather than replaying from the first shard. Initial step-500
validation was skipped because its metrics already existed.

Update 501 completed in 41.61 seconds cold: 20.76 seconds data, 15.58 forward
and 5.11 backward. Rank RSS was approximately 45.3 and 44.3 GiB; system
anonymous pages were 47.6 GiB and cached/mapped data approximately 24.2 GiB,
well below the new 128 GiB hard limit.

The legacy step-500 checkpoint stores rank-0 source cursors only. Reconstructing
the logs showed rank 1's DIEGESIS cursor was 1067 versus rank 0's 1053, so the
resume rewinds 14 rank-1 DIEGESIS attempts. Both ranks' MV-Kubric cursor was
1005, so the large replay dataset continues at the correct position.

- Resume app: `ap-e5wPL69oknXFAWO8jaRlVQ`
- Function call: `fc-01M0GGB3NMHSJJ962CC4RHZVKM`
- W&B: https://wandb.ai/jeetucl-ucl/mvtracker-continual-training/runs/8dd8bd8df50a

### Final 1,000-step result

The resumed run completed all 1,000 optimizer updates. Canonical
`latest_checkpoint.json` points to `model_final.pth` at step 1000; periodic
`model_001000.pth` is also present. Median optimizer-step time was 5.29
seconds; total active container runtime across both segments was approximately
96 minutes. The final logged LR was effectively zero at `-5.24e-8`, a tiny
linear-scheduler endpoint overshoot. After mmap was disabled, container RAM stabilized near 85.5 GiB
through steps 930--990 and ended at 82.5 GiB.

Trailing 50-update training means changed from steps 1--50 to 951--1000:

| Source | Total loss | Trajectory loss | Visibility loss |
|---|---:|---:|---:|
| Combined | 0.25637 → 0.21949 | 0.05878 → 0.05172 | 0.19758 → 0.16777 |
| DIEGESIS | 0.34045 → 0.26190 | 0.05748 → 0.03666 | 0.28297 → 0.22523 |
| MV-Kubric | 0.17228 → 0.17709 | 0.06008 → 0.06678 | 0.11220 → 0.11032 |

DIEGESIS held-out validation:

| Step | AJ | Average points within threshold | ATE | Occlusion accuracy |
|---:|---:|---:|---:|---:|
| 0 | 66.84 | 88.84 | 5.30 | 78.08 |
| 250 | 71.48 | 93.11 | 3.92 | 79.62 |
| 500 | 71.76 | 93.30 | 3.93 | 80.06 |
| 750 | 72.70 | 94.14 | 3.50 | 80.27 |
| 1000 | 72.66 | 94.04 | 3.47 | 80.33 |

Strictly matched MV-Kubric scenes 101--102:

| Step | AJ | Average points within threshold | ATE | Occlusion accuracy |
|---:|---:|---:|---:|---:|
| 0 | 71.78 | 82.02 | 7.86 | 91.62 |
| 250 | 69.33 | 80.19 | 8.84 | 91.13 |
| 500 | 68.46 | 79.58 | 9.03 | 90.59 |
| 750 | 69.14 | 79.74 | 8.99 | 91.35 |
| 1000 | 70.39 | 80.63 | 9.34 | 91.21 |

The 27-scene MV-Kubric aggregate also remained below baseline at step 1000:
AJ 73.79→72.00, average points 84.46→83.15, ATE 7.42→8.16, and
occlusion accuracy 91.66→91.14. There was genuine late AJ recovery on the
matched subset after step 500, but not a complete return to the initial model.
The equal-source matched AJ `(DIEGESIS + MV-Kubric) / 2` improved from 69.31
at step 0 to 71.53 at step 1000.

## 2026-08-19 — Direct Modal Volume v2 dataset experiment

The abandoned 2,000-scene dataset-image build was stopped and its six
dataset-specific image layers were deleted. Shared dependency images and the
older 100-scene image were preserved. Training, CPU profiling and T4 profiling
were rewired to mount `jeet-mvtracker-data-v2` read-only at
`/mnt/mvtracker-data`; the results Volume remains the only writable training
mount.

The two pinned MV-Kubric archives were expanded once into Volume v2. Extraction
of scenes 1001--2000 took 3,724.4 seconds and scenes 2001--3000 took 3,969.7
seconds. The published inventory contains 1,992 training scenes and the 27
held-out validation scenes 101--127. DIEGESIS contains 21 raw scenes with the
17/2/2 train/validation/test links and its existing JPEG cache. Both published
MV-Tracker checkpoints remain present.

The original serial MV-Kubric index pass was stopped after both archive
extractions had committed. The replacement indexer uses 16 scene workers,
reports progress every 25 scenes and constructs the source fingerprint from
the inventory already collected during indexing. It indexed all 2,019
train-plus-validation scenes in 1,171.1 seconds, avoiding the former second
filesystem-stat pass.

- Ingestion W&B: https://wandb.ai/jeetucl-ucl/mvtracker-continual-training/runs/dmwu2i0j
- Manifest: `jeet-mvtracker-data-v2/direct-volume-data-manifest.json`
- Index: `datasets/kubric-multiview/train/MVTracker_index`

Direct-Volume CPU loading was acceptable for DIEGESIS but very slow for native
MV-Kubric. DIEGESIS warm median was 0.766 seconds/sample, while MV-Kubric warm
median was 13.999 seconds/sample with 21.507-second p95 and 0.0736 samples/s.

- CPU profile W&B: https://wandb.ai/jeetucl-ucl/mvtracker-continual-training/runs/na8soubm

The first T4 matrix exposed a benchmark-only CUDA retention bug: each case had
eight unused prefetched requests, leaving producer threads and GPU tensors alive
for later cases. The profiler now requests exactly the samples each source will
consume, joins producers and releases the CUDA allocator between cases. The
rerun completed all cases on one T4. The representative alternating
DIEGESIS/MV-Kubric four-view schedule measured 5.60 samples/s, 0.00048-second
median exposed wait and 0.045-second p95 exposed wait after warm-up, peaking at
7.87 GiB VRAM.

- T4 profile W&B: https://wandb.ai/jeetucl-ucl/mvtracker-modal-profiling/runs/fuwpv8v4
- Artifact: `jeet-mvtracker-runs-v2/t4-loader-benchmark/t4-loader-9cad3860-20260818T235041Z.json`

The bounded two-H100, ten-update training smoke did not complete optimizer step
1 and was stopped. Dataset initialization eventually found 17 DIEGESIS and
1,992 MV-Kubric training scenes per rank, but first-update loader waits reached
655.96 and 1,316.14 seconds. Immediately before termination one rank remained
at 0% GPU utilization with 64.5 GiB allocated while the other reported 100%
with 34.9 GiB; both CPUs were busy. There was no OOM or code exception.

This rejects direct access to the expanded native MV-Kubric small-file tree on
Modal Volume v2 as the production training layout. Prefetch can hide steady
state in the bounded T4 loader test, but it cannot hide the enormous DDP
startup/first-update fill cost. The full 1,000-step run was not launched.

- Failed smoke W&B: https://wandb.ai/jeetucl-ucl/mvtracker-continual-training/runs/eda46c990e5a
- Modal app: `ap-UK0R5GUxCWYvn8Mud7IlkB`
- Run: `direct-volume-v2-smoke10-ddp2-h100-20260819T0110Z`

Correction: subsequent inspection of the saved crash batch found a real code
exception after loading. The physical-batch visibility assertion treated
padded trajectory slots as real tracks. Every real trajectory in the batch was
visible somewhere; only the deliberately padded slots were invisible. The
assertion is now padding-aware. This was separate from, and occurred after, the
large native MV-Kubric loader stall.

## 2026-08-19 — MV-Kubric WebDataset/DALI pilot

A 32-scene MV-Kubric pilot was converted into eight uncompressed, indexed TAR
shards with four scenes per shard. Each scene record contains its metadata and
the encoded RGB/depth payloads for all ten source cameras. Conversion completed
in roughly 3.5 minutes and produced about 12 GB of derived data under
`datasets/kubric-multiview-webdataset/v1/train` on the data Volume. The
conversion is parallel, reports shard/scene progress, and generates standard
NVIDIA `wds2idx` sidecars.

The production-side prototype uses DALI's indexed WebDataset reader, preserves
the existing view-count and track/augmentation distributions, and continues to
decode RGB/depth on CUDA. Rank-local physical batching, padding-aware
visibility checks, and ahead-of-time correlation-extension compilation were
also completed. No H100 training run was launched.

The one-T4 native-versus-DALI pilot completed 4 warm-up and 16 measured samples
per path at 1, 2, 4 and 6 selected views. Results were:

| Selected views | Native samples/s | DALI samples/s |
|---:|---:|---:|
| 1 | 7.869 | 0.137 |
| 2 | 1.131 | 0.137 |
| 4 | 0.559 | 0.141 |
| 6 | 0.513 | 0.139 |

The pilot rejects the current **record layout**, not DALI itself. The DALI
reader materializes all ten view components in a scene record before Python
selects the requested 1--6 views. Its read/unpack median therefore stayed at
6.83--7.16 seconds regardless of selected view count, while CUDA decode took
only 0.044--0.211 seconds. The loader moves a full scene even for a one-view
sample and is 3.7--57 times slower than the warmed native path in this test.
This layout must not be used for the full training run. The next storage design
must permit indexed reads of only the selected camera payloads while retaining
scene-level metadata and statistical sampling behavior.

- Successful T4 W&B: https://wandb.ai/jeetucl-ucl/mvtracker-modal-profiling/runs/kiwexyf3
- Report: `jeet-mvtracker-runs-v2/t4-mvkubric-webdataset/mvkubric-webdataset-t4-pilot-v3.json`
- Modal app: https://modal.com/apps/ucl-prism/main/ap-Wj4MIHq6n1kfBG3GwNgN7p

## 2026-08-19 — Full MV-Kubric scene/view WebDataset publication

The two pinned training archives were copied to one Modal CPU container's local
SSD one at a time, extracted locally with rapidgzip, and converted with eight
concurrent shard workers. Only the large completed TAR shards and their compact
inventories were written back to the data Volume.

Archive 1001--2000 copied 377.17 GiB in 276.4 seconds, extracted 450.29 GiB and
771,344 files in 1,402.2 seconds, and produced 994 scenes in 249 shards. Archive
2001--3000 copied 379.28 GiB in 317.6 seconds, extracted 451.84 GiB and 774,448
files in 1,410.1 seconds, and produced 998 scenes in 250 shards. All 499 training
shards were durable after 5,554.3 seconds of the primary run.

The initial finalizer exposed that the `widsindex` CLI imports PyTorch and would
also reread every TAR to compute MD5 and sample counts. Finalization now writes
the standard WIDS-v1 descriptor atomically from the already committed shard
inventories (`url`, `nsamples`, and `filesize`), avoiding PyTorch and an
unnecessary pass over the full dataset. The successful resume also converted
the fixed validation scenes 101--127 in 142.4 seconds.

Published dataset:

- Train: 1,992 scenes, 499 shards at `datasets/kubric-multiview-webdataset/train`
- Validation: 27 scenes, 7 shards at `datasets/kubric-multiview-webdataset/validation`
- Data Volume: `jeet-mvtracker-data-v2`
- Successful finalization W&B: https://wandb.ai/jeetucl-ucl/mvtracker-modal-profiling/runs/7sxzztsh
- Successful Modal app: https://modal.com/apps/ucl-prism/main/ap-BM6BQdVVBFhOkNEEZaB0xD

## 2026-08-19 — Full WebDataset two-H100 DDP smoke

The GT-depth DIEGESIS/MV-Kubric 50/50 recipe completed ten optimizer updates on
two H100s against the full 1,992-scene MV-Kubric WebDataset. The global batch
remained eight scenes per update (four per source), with four rank-local logical
scenes and physical pairing only for compatible view counts.

The first optimizer update took 60.13 seconds, including 18.73 seconds of cold
data loading and 28.02 seconds of first-pass model compilation. Updates 2--10
averaged 9.35 seconds (9.17-second median). The final update took 9.57 seconds:
2.82 seconds loading, 3.26 seconds forward and 2.25 seconds backward. Its global
batch contained 9,712 trajectories, corresponding to 0.835 samples/s and 1,014
trajectories/s. The sampled final hardware state was 47.0/79.6 GiB at 100% on
GPU 0 and 52.3/79.6 GiB at 72% on GPU 1. Container memory was 42.2 GiB.

Training saved both `model_000010.pth` and `model_final.pth`. The training loop
then attempted a terminal evaluation even though the smoke configuration has no
evaluation datasets; one rank exited and the other waited in NCCL. The stuck
post-training app was stopped after confirming both checkpoints were durable.
The terminal-evaluation guard is fixed for future smoke runs; the ten-update
training result itself is complete and was not rerun.

- W&B: https://wandb.ai/jeetucl-ucl/mvtracker-continual-training/runs/36bd7c2a9283
- Run: `continual-training/gt-depth-replay-smoke10-ddp2-h100-76b9abe8-20260819T210601Z`
- Modal app: https://modal.com/apps/ucl-prism/main/ap-uL3MQBiFHg29YqtylNy4uj

## 2026-08-20 — Single-H100 UpdateFormer optimization contract

A locked single-H100 contract now checks UpdateFormer outputs, real-track input
gradients, every parameter gradient, one AdamW update, optimizer state, and the
3-D/visibility losses. BF16 results may differ by at most one ULP; FP32 values
use fixed `rtol=1e-4`, `atol=1e-5`. Five B1/B2/B4 workloads include irregular
track counts and exclude padded tracks exactly as the production loss does. The
full reference tensors, baseline state, hashes, and manifest are immutable under
`performance-contracts/updateformer-v3` on `jeet-mvtracker-runs-v2`.

The memory profile was confirmed: parameters and optimizer state are small;
retained UpdateFormer activations dominate. Non-reentrant checkpointing of each
complete UpdateFormer call passed the contract. Across twelve chained calls it
reduced peak allocation from 28.19 to 3.71 GiB for B1/1,536 tracks and from
41.69 to 5.15 GiB for B4/512 tracks (7.6--8.1x), while increasing isolated step
time by 35--43%. Block-by-block checkpointing saved less memory per unit of
compute and was discarded. RNG preservation is disabled because UpdateFormer
contains no stochastic operation; this also makes checkpointing CUDA-graph
capturable.

Checkpointed capacity saturated at B8 for 512 tracks (1.45 to 6.51 scenes/s,
4.5x over B1) and B2 for 1,536 tracks (1.53 to 1.99 scenes/s). Whole-step CUDA
Graph replay demonstrated the remaining launch-overhead ceiling: B1/512 improved
1.80 to 5.58 scenes/s and B1/1,536 improved 1.84 to 2.51. PyTorch's supported
autograd-aware partial graphs also passed the contract and achieved 3.7x for
B1/512, 1.48x for B4/512, and 2.0x for B1/1,536. They are not enabled in
production because variable trajectory counts would require either many
one-use graph captures or numerical-changing track buckets.

Two candidates were rejected before timing/promotion. Regional
`torch.compile` changed millions of low-precision values and produced up to
`9.2e-5` difference after one Adam update. Track-count bucketing likewise
produced up to `8.98e-5` update drift. A reusable custom forward-only graph
also failed the one-ULP gate. None remains in the production model.

The accepted whole-call checkpoint completed a three-update live mixed-data
smoke on one H100 with rank-local WebDataset physical batching. Warm updates
took 6.05 and 7.76 seconds for four logical scenes; final sampled GPU memory was
27.18/79.65 GiB. The rank-local scheduler's stale two-rank gate was removed;
global scheduling still requires two ranks.

- Contract capture: https://wandb.ai/jeetucl-ucl/mvtracker-modal-profiling/runs/665sa9sz
- Checkpoint study: https://wandb.ai/jeetucl-ucl/mvtracker-modal-profiling/runs/lj0vzrl8
- Capacity study: https://wandb.ai/jeetucl-ucl/mvtracker-modal-profiling/runs/61tsnzx5
- CUDA Graph study: https://wandb.ai/jeetucl-ucl/mvtracker-modal-profiling/runs/rsbgoojb
- Partial-graph study: https://wandb.ai/jeetucl-ucl/mvtracker-modal-profiling/runs/ywz00xip
- Live single-H100 smoke: https://wandb.ai/jeetucl-ucl/mvtracker-continual-training/runs/137699893d38
- Live run directory: `single-gpu-performance/updateformer-single-h100-adebc37d`

## 2026-08-20 — Matched full-model checkpoint throughput result

Whole-UpdateFormer checkpointing was compared against the recorded eager H100
baseline using the same full MVTracker profile path: deterministic cached
MV-Kubric tensors, 24 frames, 384×512, 1,024 trajectories per scene, BF16,
clean-depth checkpoint, forward/loss/backward/clipping/Adam, two warm-ups and
three measured updates. Dataset loading and decoding were excluded from both.

| Views | Eager best | Eager scenes/s | Checkpoint best | Checkpoint scenes/s | Ratio |
|---:|---:|---:|---:|---:|---:|
| 1 | B3 | 2.8742 | B8 | 2.7322 | 0.9506× |
| 4 | B2 | 1.6556 | B4 | 1.6354 | 0.9878× |

Checkpointing therefore increased safe physical capacity by 2.0--2.7× but did
not improve full-model throughput: its best result was 4.9% slower for one view
and 1.2% slower for four views. The recomputation cost almost completely
amortizes in the four-view case but still slightly exceeds the batching gain.
The optimization should be treated as a memory/capacity lever, not a standalone
speed optimization. It remains useful if another optimization exploits the
headroom or if an otherwise required shape OOMs.

- Checkpointed H100 sweep: https://wandb.ai/jeetucl-ucl/mvtracker-modal-profiling/runs/96kk758r
- Results: `checkpoint-net-throughput-8fa3abc8/summary.json` on `jeet-mvtracker-runs-v2`

### H200 B1--B16 follow-up

The same 1-view and 4-view, 1,024-track full-model comparison was repeated on
one H200. Deterministic B8 tensors were duplicated along the batch axis to
provide B16 shapes; B1--B8 therefore use the same deterministic prefix as the
H100 test, and duplication affects neither per-scene computation nor throughput
arithmetic. The eager reference is the prior matched H200 profile.

| Views | Eager best | Eager scenes/s | Checkpoint best | Checkpoint scenes/s | Ratio |
|---:|---:|---:|---:|---:|---:|
| 1 | B5 | 3.3005 | B16 | 3.0482 | 0.9235× |
| 4 | B3 | 1.9328 | B7 | 1.8279 | 0.9457× |

For one view, checkpoint B16 was safe at 66.5% peak H200 memory but remained
7.6% slower than eager B5. For four views, checkpoint B7 was the largest safe
batch at 83.3% memory and remained 5.4% slower than eager B3; B8 exceeded the
90% safety threshold and B9+ OOMed. Extra H200 VRAM therefore does not turn
checkpointing into a throughput optimization. The GPU reaches its compute
plateau before the checkpointed capacity frontier.

- H200 sweep: https://wandb.ai/jeetucl-ucl/mvtracker-modal-profiling/runs/jtrt2nk6
- H200 cache preparation: https://wandb.ai/jeetucl-ucl/mvtracker-modal-profiling/runs/he7wltnk
- Results: `checkpoint-net-throughput-h200-3cbae668/summary.json` on `jeet-mvtracker-runs-v2`

## 2026-08-20 — Integrated UpdateFormer fused-backend rejection

The eager reference was replayed after the no-math cleanup and passed the
locked v3 golden contract. Virtual-track `expand` and disabling UpdateFormer
checkpointing by default are accepted; checkpointing remains an explicit OOM
lever.

A full real-update gate then compared eager against fused candidates on the
saved mixed-source B2 crash batch (one view, 599 trajectory slots). It checks
every refinement coordinate/logit, final predictions, losses, all parameter
gradients, the clipped Adam update, five repeated updates, timing, and memory.

The fixed-capacity 1,024-track candidate failed with 28.4 mm final trajectory
RMS, 2.59% visibility flips, gradient cosine 0.9853, and Adam-update cosine
0.5406. A component diagnostic found that fixed padding, not fused QKV, was the
main local source: padding caused up to `7.91e-5` one-update difference, versus
`2.38e-7` for QKV.

Removing padding and CUDA graphs produced an exact-shape dynamic Inductor
candidate. Warm full updates improved from 0.750 to 0.539 seconds (1.39x), but
the cold compile took about 565 seconds across the sliding-window shapes and
the candidate still drifted by 27.5 mm RMS with Adam-update cosine 0.4302. It
was rejected.

QKV-only fusion produced exactly identical first forward traces, predictions,
visibility and loss, and improved warm update time by roughly 2--5%. Its
backward accumulation still changed the first Adam update (cosine about 0.998).
After five identical updates, trajectory RMS divergence reached 33--40 mm and
cumulative update cosine approximately 0.95. Manual and recomputed-original
custom backward variants did not meet the gate. QKV remains experimental and
is not enabled by default.

This pass cost $1.7447 on Modal: $1.3960 H100, $0.1718 CPU and $0.1768 memory.
All jobs used `owner=jeet`, `project=mvtracker`, `purpose=profiling`; unrelated
workspace jobs were excluded.

- Exact replay: https://wandb.ai/jeetucl-ucl/mvtracker-modal-profiling/runs/5pwgqitz
- Fixed-padding candidate: https://wandb.ai/jeetucl-ucl/mvtracker-modal-profiling/runs/qrp6tvk9
- Component diagnostic: https://wandb.ai/jeetucl-ucl/mvtracker-modal-profiling/runs/myhpn5un
- Dynamic Inductor candidate: https://wandb.ai/jeetucl-ucl/mvtracker-modal-profiling/runs/5dcc96ai
- Five-update QKV gate: https://wandb.ai/jeetucl-ucl/mvtracker-modal-profiling/runs/j5qf7qf3
## 20 August 2026: single-H100 performance autoresearch

The performance harness was extended from isolated UpdateFormer tests to a
complete real forward/loss/backward/clip/Adam update. It records eager
run-to-run nondeterminism, candidate drift after one and five updates, memory,
steady-state time and amortized 1,000-update time. Candidates run in fresh GPU
subprocesses so compiler and CUDA-graph pools cannot contaminate one another.

The strongest repeated-batch result was a whole-update CUDA graph at 1.46x,
with a bit-identical first forward and first loss. It is not usable for live
training: changing trajectory schedules caused 66.3 seconds of graph capture
in the first optimizer step and 61.2 GiB of retained graph-private pools before
an OOM in step two. Fixed buckets, Transformer Engine MLP, channels-last CNN and
external FlashAttention-2 were all rejected on speed, behavior, or both.

A real operator profile found KNN (67.9 ms), FlashAttention backward (61.9 ms),
copies (54.0 ms), GEMMs (46.5 ms), reductions (31.7 ms), adds (23.4 ms) and
LayerNorm (25.8 ms) as the primary GPU costs. A tiled KNN plus exact serial tie
fallback restored bit-identical first-pass behavior but cost 124 ms because
invalid-depth point clouds produce many ties; it was rejected.

Key W&B runs:

- Isolated candidate sweep: `spr9xn8p`
- Whole-graph nondeterminism calibration: `9nzq3u66`
- Live exact-graph failure: `116c68a5bfed`
- Transformer Engine MLP: `5hfzxmg6`
- Serial operator profile: `48cp0l06`
- Exact tiled-KNN gate: `4o4ov1fi`
- Tiled-KNN operator profile: `rpojngwh`
- Channels-last CNN: `61s9iiw0`
- External FlashAttention-2: `o7oz7cj4`

The current production default remains eager UpdateFormer, contiguous CNN and
the original serial capture-safe KNN. No rejected candidate was promoted.

The final static upper-bound candidate combined default dynamic Inductor with
the whole-update graph. It reached 2.08x steady-state (0.807 s to 0.389 s) and
1.51x over a 1,000-update horizon after its 154-second setup. It was rejected
for production because it is static-shape-only and its first Adam update cosine
was 0.548. W&B: `jj3wr5c3` (combined) and `o1im1wxb` (Inductor only).

## 20 August 2026: CPU-native DALI WebDataset throughput

A CPU-only Modal function tested the exact storage reader used by Modal's
ResNet50 example: `fn.readers.webdataset` received eight randomly ordered
MV-Kubric TARs and their DALI indexes directly from the mounted Volume, with
`random_shuffle=False`. No GPU, WIDS, Python TAR reader, local staging, image
decode, model, or validation was involved.

The pipeline built in 4.51 seconds. It streamed 32 scenes / 12.31 GB in 40.75
seconds: 288.0 MiB/s and 0.785 scenes/s. Four-scene batches had a 5.31-second
median; the first took 5.73 seconds. All eight batches preserved the expected
one metadata plus ten paired RGB/depth view-record grouping.

- Source: `165d4c626f336618f83418d968a21a4a26e09dde`
- Modal app: `ap-k5yiMFEKFRsDpoorbe4qK3`
- W&B: https://wandb.ai/jeetucl-ucl/mvtracker-modal-profiling/runs/1v5bptbx

## 20 August 2026: native DALI two-H100 training smoke

The mixed DIEGESIS/MV-Kubric production path completed ten optimizer updates
on two H100s with validation and visualization disabled. MV-Kubric TARs and
indexes were read directly by `fn.readers.webdataset`; no WIDS, Python
`pread`, TAR copying, or local staging remained in the live path. Complete
four-scene groups are reused as `A,B,C,D,A,B,C,D`, with independent live
view/track/augmentation sampling from each request.

Each rank's DALI stream built over its shard partition in 61.9--65.7 seconds.
The first optimizer step took 81.09 seconds, including 54.07 seconds of exposed
first-use loading and decode setup. Steps 2--10 had a 7.26-second median and a
0.25-second median exposed data wait. By steps 5 and 9, the next 1.4--1.5 GB
scene groups were already prefetched and reported approximately 1 ms reader
wait. Step 10 took 4.43 seconds with 0.24 seconds of data wait. Both final
checkpoints were saved and both H100s were released.

- Source: `5e02fdc298418de28252a05ac40a50b64139d2c0`
- Run: `gt-replay-prod-smoke10-5e02fdc2-20260820T184325Z`
- Modal worker: `ap-tUkVEprbY1lYixfRLGOXJh`
- W&B: https://wandb.ai/jeetucl-ucl/mvtracker-continual-training/runs/7c737df3cd1f

## 20 August 2026: distributed batched validation and main training launch

Validation was moved from rank 0 to both DDP ranks. DIEGESIS scenes are split
one per rank; the 27-scene MV-Kubric validation set is partitioned by whole
DALI shards and evaluated in physical batches of two scenes. Rank 0 gathers
the small metric dictionaries and writes the combined TensorBoard, W&B and CSV
outputs once. The same finite validation loaders cycle cleanly for later
checkpoints.

The first integration launch reached DIEGESIS validation but exposed a missing
persistent DALI image decoder in the generic CUDA prefetch wrapper. It stopped
before optimizer step 1 and released both H100s. The wrapper now constructs one
DALI decoder per iterator and reuses it. A physical validation batch remains
two scenes; one such batch is submitted to the decoder at a time.

The fresh run on source `dd10c76bf21f4847c2dfd918bc7b51ec4f8be1a0`
completed initial distributed validation: rank 0 evaluated 12 MV-Kubric scenes
and rank 1 evaluated 15, while both ranks also evaluated their DIEGESIS scene.
Training then completed optimizer steps 1 and 2. Step 1 took 31.73 seconds and
step 2 took 8.50 seconds. The run remains live.

- Run: `gt-replay-main-ddp2-h100-batched-val-dd10c76-20260820T210000Z`
- Modal app: `ap-5Nq2R5RjEZatlxmos742LI`
- Function call: `fc-01M0GADM4K96P8PNRQRTAKDMEQ`
- W&B: https://wandb.ai/jeetucl-ucl/mvtracker-continual-training/runs/e70a1c037710
- Billing tags: `owner=jeet`, `project=mvtracker`, `purpose=training`

## 21 August 2026: final mixed-replay checkpoint external evaluation

The canonical 1,000-step `model_final.pth` from the mixed DIEGESIS/MV-Kubric
continuation was transferred to dopey and verified as SHA-256
`f70af6596ad29ebce8d60ac5c00d69bb06e1d43f573fb646cf3db6c58df6aab0`.
It was evaluated with the same checkout, cached inputs, four-view selections
and evaluator settings as the released mixed/noisy-depth checkpoint. The job
completed all 46 sequences on dopey's RTX 3090 with exit code zero in 3 minutes
11 seconds. W&B was unavailable on dopey, so the matched evaluation remained
file-logged like the original baseline.

| Benchmark | Metric | Published mixed-depth | Final step 1000 | Delta |
|---|---|---:|---:|---:|
| MV-Kubric | AJ | 73.59 | 72.58 | -1.01 |
| MV-Kubric | Delta-avg | 84.22 | 83.54 | -0.68 |
| MV-Kubric | MTE | 7.76 | 8.03 | +0.27 |
| MV-Kubric | Occlusion accuracy | 91.88 | 91.66 | -0.22 |
| Panoptic | AJ | 86.03 | 86.84 | +0.81 |
| Panoptic | Delta-avg | 94.71 | 95.79 | +1.08 |
| Panoptic | MTE | 3.13 | 2.94 | -0.19 |
| Panoptic | Occlusion accuracy | 92.28 | 92.13 | -0.15 |
| DexYCB | AJ | 72.99 | 72.46 | -0.53 |
| DexYCB | Delta-avg | 82.22 | 82.07 | -0.15 |
| DexYCB | MTE | 1.85 | 1.87 | +0.02 |
| DexYCB | Occlusion accuracy | 91.05 | 91.05 | 0.00 |

The continuation modestly regressed MV-Kubric, improved Panoptic trajectory
metrics, and was nearly neutral on DexYCB. It did not produce broad external
catastrophic forgetting, but it also did not improve all benchmark domains.

- UCL run: `mvtracker-final-gt-replay-eval-20260821T000300Z`
- Checkpoint: `/media/data3/jthakwani/mvtracker/checkpoints/diegesis-mvkubric-gt-replay-step1000/model_final.pth`
- Results: `/media/data3/jthakwani/mvtracker-evals/mvtracker-final-gt-replay-eval-20260821T000300Z/final-gt-replay-step1000-mixed-init/`
- Log: `/media/data3/jthakwani/mvtracker-evals/mvtracker-final-gt-replay-eval-20260821T000300Z.ucl.log`

## 21 August 2026: three-source Syn4D launch gate

Source `e376e2dc7689ca1f4614adec9d713bba86ae31ad` added Syn4D to the
continual-training scheduler at a global 25% DIEGESIS / 25% Syn4D / 50%
MV-Kubric ratio. The selected cache contains 20 `lab_bald` sequences; sequences
0--15 train and 16--19 validate. Syn4D uses its own 300 m depth ceiling. The
run starts from the released mixed/noisy-depth checkpoint and otherwise keeps
the prior BF16, global-batch-eight, peak-5e-5 recipe, extended to 2,000 steps.

A one-update two-H200 smoke completed forward, loss, backward, optimizer,
checkpointing and W&B logging with the exact global 2/2/4 source counts. The
cold step took 63.20 seconds, including 44.05 seconds exposed data wait. H200
memory was 42.7 and 41.2 GiB; container RAM was 39.4/128 GiB. W&B:
https://wandb.ai/jeetucl-ucl/mvtracker-continual-training/runs/a022701d2fb1

The first main attempt stopped before training because Syn4D lacked an
evaluator-name mapping. Source `48b71a8ac902ecff4eb57c5d782467fce798798c`
mapped it to the existing Kubric-style 3D tracking protocol. The fresh run
then completed all initial DIEGESIS, four-sequence Syn4D, and 27-scene
MV-Kubric validation and reached 16 optimizer updates. Warm steps took
5.7--16.2 seconds; exposed data wait was usually 0.15--2.17 seconds after the
cold step, and both H200s reached 98--100% utilization at the step-10 sample.

The run was deliberately cancelled under the requested early safety gate:
container RAM rose from 77.57 GiB at step 1 to 110.19 GiB at step 10 under a
128 GiB hard limit. No periodic checkpoint existed before the step-250 save.
The H200s were released and unrelated Modal containers were not interrupted.
The likely bounded contributors are the four-step encoded lookahead plus
Syn4D's mapped sequence cache, but the available two memory samples do not
prove that the process had stabilized. W&B:
https://wandb.ai/jeetucl-ucl/mvtracker-continual-training/runs/a96efc42247a

### 256-GiB relaunch

Source `d44c73db12b999b8f6257b89689b0675c229cd00` retained the 64-GiB RAM
request and raised only the hard limit to 256 GiB. The fresh two-H200 run
completed all three initial validation datasets and remained healthy beyond
100 optimizer updates. The first cold update took 147.7 seconds, but the warm
median remained around 6--7 seconds with roughly 0.6 seconds median exposed
data wait.

Container RAM followed a decelerating cache-fill curve: 105.5 GiB at step 10,
125.7 at step 20, 133.4 at step 30, 135.9 at step 50, 140.2 at step 90, and
142.1 at step 100. The process therefore still crept upward, but had 114 GiB
headroom and no longer resembled the earlier linear mmap leak. The run remains
active and monitored.

- Run: `gt-replay-syn4d-main-ddp2-h200-256g-d44c73d-20260821T231000Z`
- Function call: `fc-01M0K63D2YX68PJ3TVCDKPHJ06`
- W&B: https://wandb.ai/jeetucl-ucl/mvtracker-continual-training/runs/b7f6de8d6ff6
