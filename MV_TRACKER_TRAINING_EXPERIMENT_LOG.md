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
