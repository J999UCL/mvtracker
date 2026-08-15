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
