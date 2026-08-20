# MV-Tracker Performance Autoresearch

Status: active research record, 2026-08-20

This document records the performance investigation, the correctness contract,
the optimization results, and the proposed autoresearch loop for accelerating
MV-Tracker training. It is intentionally separate from the general training
experiment log: this file is about preserving learning behavior while changing
the implementation aggressively.

## 1. Objective

The objective is to maximize useful MV-Tracker training throughput without
quietly changing the tracker being trained.

Primary score:

- Full training-update wall time.
- Scenes per second.
- Real trajectories per second.

Secondary scores:

- Peak allocated and observed GPU memory.
- CPU and GPU utilization.
- Data-wait, forward, backward, optimizer, and synchronization time.
- Startup and graph/compile amortization cost.

The target is not merely a small code cleanup. Large rewrites, custom Triton or
CUDA kernels, altered execution schedules, and recomputation are in scope. The
correctness and training-behavior gates—not the size of the rewrite—decide
whether a candidate is accepted.

All optimization work is single-GPU first. DDP communication and distributed
loading are reintroduced only after a candidate is clearly better on one GPU.

## 2. What the model is actually processing

The tracker has 22,607,356 trainable parameters. The weights themselves are not
the memory problem.

A training sample contains:

- 1--6 camera views.
- 24 RGB-D frames per view.
- 384×512 crops.
- A variable number of 3-D trajectories, capped at 2,048 for ordinary views.
- Camera intrinsics/extrinsics, visibility, validity, and query information.

The model uses 12-frame sliding windows. A typical 24-frame sample produces up
to three overlapping windows. Every window performs four refinement iterations.
Each refinement calls UpdateFormer once, so one scene can execute UpdateFormer
12 times.

The production UpdateFormer configuration is:

- Input width: 581.
- Hidden width: 256.
- Output width: 131.
- Six temporal stages.
- Six spatial stages.
- Six attention heads.
- 64 virtual tracks.

One UpdateFormer call contains one temporal block and three spatial/cross blocks
per stage: 24 attention/cross-attention blocks. Across 12 calls, a scene can
execute 288 such blocks.

The earlier operator profile observed roughly:

- 3,018 `mm` operations.
- 1,491 `addmm` operations.
- 720 layer-normalization calls.
- 288 attention calls.
- 71 KNN queries.
- 48 correlation operations.
- More than 6,000 copies.

This is not “a few large matrix multiplications.” It is a deep, Python-directed
sequence of many relatively small kernels.

## 3. Where the memory goes

Measured fixed state was small:

- Model parameters: about 90 MB.
- Adam optimizer tensors after initialization: about 181 MB.
- Non-PyTorch CUDA/process memory: roughly 2--3 GB.

Saved activations dominated live memory. Representative real microbatches saved:

| Shape | Saved tensors | Live allocated |
|---|---:|---:|
| 2 views, 1,750 tracks | 38.83 GB | 40.19 GB |
| 3 views, 1,052 tracks | 30.58 GB | 31.75 GB |
| 4 views, 1,309 tracks | 38.50 GB | 40.39 GB |
| 6 views, 427 tracks | 30.52 GB | 32.33 GB |
| 3 views, 2,048 tracks | 47.50 GB | approximately 49 GB |

After backward, saved-tensor memory returned to zero and live PyTorch allocation
fell to roughly 1 GB. The CUDA allocator retained cached blocks, but allocator
caching was not the cause of OOM: the live activation peak was real.

The largest saved-tensor call sites were UpdateFormer attention, MLP, and
residual operations. Correlation and point-cloud construction were materially
smaller.

## 4. Locked correctness contract

The verifier lives in `mvtracker/profiling/updateformer_contract.py`. The active
contract is `mvtracker-updateformer-contract-v3`; its golden artifacts live at:

```text
jeet-mvtracker-runs-v2/performance-contracts/updateformer-v3/
```

It uses one H100 and checks:

- Real-track UpdateFormer outputs.
- Real-track input gradients.
- Every parameter gradient.
- One AdamW parameter update.
- Optimizer tensor state.
- The 3-D trajectory loss and its gradients.
- The visibility loss and its gradients.

The workloads cover B1, ragged B2, ragged B4, and irregular trajectory counts
such as 777 and 900. Padded trajectories are excluded from the contract just as
they are excluded from the production loss.

Current numerical bounds:

- Shapes and dtypes must match exactly.
- BF16 values may differ by at most one ULP.
- FP32 values use fixed `rtol=1e-4`, `atol=1e-5`.

The verifier source file, baseline state, and golden reference tensors are
hashed. A changed verifier source cannot validate against the existing golden.
The benchmark verifies first and times second.

This contract is deliberately strict. It is suitable for proving that
checkpointing has not changed the computation. It is not yet sufficient for
deciding whether a small local numerical change is safe across the complete
refinement process.

## 5. Why local numerical error may propagate

UpdateFormer is iterative. A small difference does not simply appear at the
final output and stop.

For each refinement:

1. UpdateFormer predicts coordinate and feature deltas.
2. Coordinate deltas update the 3-D trajectory estimate.
3. Feature deltas update the persistent learned track features.
4. The next iteration performs new KNN queries and correlation sampling at the
   changed coordinates.
5. Final coordinates and visibility from one window initialize the next window.

Coordinates are detached between refinement iterations for backward, but their
changed forward values still affect the next KNN/correlation computation.
Feature state is not detached and continues through the refinements.

KNN introduces a discontinuity. A tiny coordinate change near a neighbour
boundary can change an integer neighbour index. That can create a much larger
correlation difference than the original floating-point perturbation. Sliding
windows provide another opportunity for drift to accumulate.

Therefore, “maximum UpdateFormer output error is only `1e-7`” is not enough to
accept an optimization. The complete tracker must be compared after every
refinement and every window.

## 6. Optimization results

### 6.1 Whole-UpdateFormer activation checkpointing

Non-reentrant checkpointing of each complete UpdateFormer call passed the
strict contract. RNG preservation is disabled because UpdateFormer contains no
stochastic operation.

Across 12 chained calls:

| Shape | Eager peak | Checkpoint peak | Eager time | Checkpoint time |
|---|---:|---:|---:|---:|
| B1, 1,536 tracks | 28.19 GB | 3.71 GB | 0.351 s | 0.503 s |
| B4, 512 tracks | 41.69 GB | 5.15 GB | 0.444 s | 0.599 s |

This is a 7.6--8.1× reduction in accumulated UpdateFormer activation memory at
a 35--43% compute-time penalty.

Checkpointing increased physical capacity substantially but did not increase
full-model throughput.

#### H100 matched full-model result

| Views | Eager optimum | Checkpoint optimum | Throughput ratio |
|---:|---:|---:|---:|
| 1, 1,024 tracks | B3, 2.874 scenes/s | B8, 2.732 scenes/s | 0.9506× |
| 4, 1,024 tracks | B2, 1.656 scenes/s | B4, 1.635 scenes/s | 0.9878× |

#### H200 matched full-model result

| Views | Eager optimum | Checkpoint optimum | Throughput ratio |
|---:|---:|---:|---:|
| 1, 1,024 tracks | B5, 3.301 scenes/s | B16, 3.048 scenes/s | 0.9235× |
| 4, 1,024 tracks | B3, 1.933 scenes/s | B7, 1.828 scenes/s | 0.9457× |

For four H200 views, checkpoint B8 exceeded the 90% safety threshold and B9+
OOMed. Additional VRAM did not turn recomputation into a throughput win. The GPU
became compute-saturated before reaching the checkpointed capacity frontier.

Conclusion: checkpointing is an OOM/capacity tool, not a speed optimization. It
should be disabled for ordinary speed-focused training and enabled only for a
shape that otherwise cannot run or for another optimization that genuinely uses
the freed memory.

### 6.2 CUDA Graphs

Whole-step CUDA Graph replay demonstrated large launch-overhead headroom:

| Shape | Eager | CUDA Graph | Speedup |
|---|---:|---:|---:|
| B1, 512 tracks | 1.80 scenes/s | 5.58 scenes/s | 3.10× |
| B4, 512 tracks | 6.49 scenes/s | 7.66 scenes/s | 1.18× |
| B1, 1,536 tracks | 1.84 scenes/s | 2.51 scenes/s | 1.37× |
| B2, 1,536 tracks | 2.60 scenes/s | 2.73 scenes/s | 1.05× |

PyTorch's official autograd-aware partial-network graphs also passed the strict
contract:

| Shape | Eager | Partial graph | Speedup |
|---|---:|---:|---:|
| B1, 512 tracks | 1.25 scenes/s | 4.60 scenes/s | 3.69× |
| B4, 512 tracks | 4.75 scenes/s | 7.00 scenes/s | 1.48× |
| B1, 1,536 tracks | 1.12 scenes/s | 2.24 scenes/s | 2.01× |

This is the strongest measured speed lever. The production integration problem
is shape reuse: CUDA Graphs require static tensor shapes, while trajectory counts
are variable. Capturing a graph for a one-use exact shape costs more than it
saves.

### 6.3 Track-count bucketing

Padding trajectories into reusable buckets would make CUDA Graphs practical,
but the tested implementation failed the strict contract.

Observed drift:

- Maximum real-output absolute difference: about `1.2e-7`.
- Millions of BF16 elements moved by more than one ULP, often around zero.
- Maximum first Adam-update parameter difference: `8.98e-5`.

Because this local drift may propagate through refinement/KNN/window loops,
bucketing was removed rather than silently accepted.

### 6.4 Regional `torch.compile`

Compiling repeated attention blocks also failed the strict contract before it
was promoted.

Observed drift:

- Maximum output absolute difference: about `1.2e-7`.
- Millions of low-precision values moved by more than one ULP.
- Maximum first Adam-update difference: about `9.2e-5`.

The candidate was removed. Its speed was not used to justify moving the gate.

### 6.5 Reusable custom forward graph

A custom forward-only graph plus eager recomputation backward was prototyped to
reuse one graph across repeated calls of the same shape. It failed the one-ULP
gate (`2.98e-8` maximum absolute difference in the first failing output) and was
not integrated.

### 6.6 Existing accepted lower-level work

The indexed correlation path already uses the accepted hybrid implementation:
custom fused/Triton-CUDA work where it preserved the intended behavior, with the
selected forward/backward division recorded in the training research log.

Other previously completed semantics-preserving cleanup includes removing or
throttling GPU synchronization loops, reprojection audits, stationary baseline
work, and expensive dashboard diagnostics from ordinary microbatches.

## 7. What changes if small numerical drift is acceptable

If strict one-ULP equivalence is relaxed, the opportunity becomes materially
larger. Track bucketing can provide a small fixed set of reusable shapes, which
allows CUDA Graphs to amortize capture across training.

Based on isolated measurements, the plausible target becomes:

- Approximately 2--3.7× for UpdateFormer in launch-bound shapes.
- Approximately 1.4--2× for complete training steps.
- Potentially 1.5--2.5× after combining graphs, kernel fusion, and better
  physical grouping.

These are hypotheses, not promises. Full-model measurement remains the score.

Accepting drift must not mean accepting any result that happens to train. The
gate must move from local bit similarity to measured training stability.

## 8. Full-refinement drift verifier (next required harness)

The next verifier should operate on complete real MVTracker microbatches. It
should record and compare:

### Forward state

- Coordinates after every refinement.
- Feature state after every refinement.
- Visibility logits after every refinement/window.
- Final trajectories and visibility.
- KNN neighbour indices and the fraction that change.
- Correlation tensors or compact numerical summaries at every level.

### Training state

- Every loss component.
- Input and parameter gradients.
- Gradient cosine similarity and norm ratios.
- Adam parameter and optimizer-state divergence.
- Clipping behavior.

### Multi-update stability

- Run identical baseline/candidate batches for 10, then 100 updates.
- Compare loss curves, parameters, optimizer state, and predictions throughout.
- Evaluate both resulting checkpoints on the same small validation subset.
- Compare divergence with ordinary baseline nondeterminism and run-to-run seed
  variation before locking acceptance thresholds.

Thresholds should be calibrated from the model's natural variance. They should
not be invented after observing whether a desired candidate passes.

## 9. Autoresearch loop

The intended loop is:

1. Lock the baseline implementation, input batches, environment, checkpoint,
   verifier, and timer.
2. Apply one candidate optimization.
3. Run the correctness/training-stability gate first.
4. Reject immediately on a gate failure.
5. Run paired performance measurements only for passing candidates.
6. Keep a candidate only when the full-step score improves consistently.
7. Record the implementation, result, cost, W&B run, and rejection reason.
8. Start the next candidate from the best accepted implementation, not from an
   arbitrary stack of unverified changes.

### Timing protocol

- One GPU initially.
- Same GPU type and immutable software image.
- Same checkpoint and cached batches.
- Fixed warm-up and measured iteration counts.
- Synchronize outside timed regions.
- Paired or alternating baseline/candidate order where practical.
- Report median and individual durations, not only the fastest iteration.
- Include forward, backward, optimizer, and full-step wall time.
- Exclude data loading only in explicitly labelled model-step benchmarks.
- Confirm promising candidates on the live loader and full training path.

### Anti-reward-hacking rules

A candidate may not improve its score by:

- Reducing frames, views, trajectories, layers, windows, or refinements.
- Omitting loss components or gradients.
- Changing optimizer semantics.
- Caching outputs instead of computing them.
- Moving work outside the timer while leaving it necessary for every update.
- Ignoring padding/visibility semantics.
- Modifying the verifier, golden artifacts, or timer.

The harness—not the candidate—owns workload construction, synchronization, and
timing.

## 10. Candidate research queue

### Priority A: full-loop drift harness

Build the verifier described above. This determines whether measured compile or
bucketing drift is genuinely harmful after four refinements and three windows.

### Priority B: bucketing plus official partial CUDA Graphs

If the calibrated drift gate passes:

1. Choose trajectory buckets using the observed training distribution.
2. Measure padding overhead and graph-cache hit rate.
3. Pre-capture only common B1/B2 physical shapes.
4. Use PyTorch's official autograd-aware graph mechanism.
5. Keep graph capture/startup cost separate and report its amortization horizon.
6. Re-run the full matched H100/H200 benchmark.

### Priority C: UpdateFormer fusion

- Fuse self-attention Q/K/V projection where practical.
- Fuse LayerNorm/residual/MLP epilogues.
- Evaluate packed or variable-length attention for ragged padded tracks.
- Replace launch-heavy blocks with Triton/CUDA only when the full-loop drift
  harness passes.

### Priority D: physical-batch planning

Couple source view-count choices so compatible DIEGESIS/MV-Kubric scenes pair
more often while preserving each source's marginal view-count distribution.
This changes cross-scene correlation, not the per-source distribution, and must
be logged as a sampling-policy choice.

### Priority E: remaining synchronization and host overhead

- Reprofile the accepted single-GPU candidate.
- Remove remaining `.item()`, CPU copies, debug assertions, and per-step Python
  scans from the hot path.
- Measure optimizer/gradient-diagnostic overhead independently.

### Priority F: DDP

Only after the single-GPU path is accepted:

- Re-enable two-rank DDP.
- Measure communication inside synchronized backward using profiler/NCCL events.
- Confirm no-sync accumulation and rank balance.
- Compare scaling efficiency against the accepted single-GPU baseline.

## 11. Current recommendation

For immediate training with current code:

- Disable UpdateFormer checkpointing when the ordinary eager shape fits.
- Enable checkpointing only to prevent OOM or to support a required larger
  physical batch; do not claim it is faster.
- Keep CUDA Graphs and track bucketing experimental until the full-refinement
  drift harness is complete.
- Continue optimizing on one GPU before spending effort on DDP scaling.

## 12. Key artifacts

- Performance contract:
  `jeet-mvtracker-runs-v2/performance-contracts/updateformer-v3/`
- H100 checkpoint throughput:
  `jeet-mvtracker-runs-v2/checkpoint-net-throughput-8fa3abc8/summary.json`
- H200 checkpoint throughput:
  `jeet-mvtracker-runs-v2/checkpoint-net-throughput-h200-3cbae668/summary.json`
- General training research record: `MV_TRACKER_TRAINING_EXPERIMENT_LOG.md`
- Reusable Modal commands: `scripts.md`

Relevant W&B runs:

- Contract capture: `665sa9sz`
- Checkpoint study: `lj0vzrl8`
- CUDA Graph study: `rsbgoojb`
- Official partial-graph study: `ywz00xip`
- H100 full-model checkpoint sweep: `96kk758r`
- H200 full-model checkpoint sweep: `jtrt2nk6`
