# Optimization Status & Future Levers

Snapshot of where performance work on the `webgpu-avbd` lane stopped
(2026-06-11, through commit `dde2361`) and the ranked list of remaining
optimization candidates, so this work can be resumed cold.

## Where We Are

All numbers: RTX 4090 laptop GPU, 300-frame averages, and only
**same-thermal-window A/B comparisons** are trustworthy — this machine
throttles up to ~50% during long benchmark sessions, so absolute numbers
drift between sessions. The gates assert generous ceilings for that reason.

| Scene | Frame avg | Notes |
|---|---|---|
| Sphere Pour 20000 on Cylinders | ~9.1 ms | was ~24 ms at migration start, ~13.5 ms at start of 2026-06-11 |
| Sphere Pour 5000 on Cylinders | ~3.1 ms | |
| Soft Body 8x8x8 | ~4.2 ms | |
| Pyramid | ~1.3 ms | small-scene floor ≈ 0.85 ms (see below) |

20k-pour frame composition after the latest work (cool-thermal session):

| Section | ms | State |
|---|---|---|
| GPU iterate (submit 0.65 + wait 1.8-3.1 + apply 0.25) | ~4.1 | wait dominates; see Lever 1 |
| Uniform-grid broadphase | ~2.3 | parallel; replaced SAP 2026-06-11 |
| buildFrame | ~1.8 | sub-stats below |
| bodyInit (Solver::prepareStep warmstart) | ~0.7 | |
| velocity update + sync | ~0.75 | |

buildFrame sub-stats (exposed as `avbdGpuBuild{Bodies,Records,Slots,Adj,Color}AvgMs`
in the benchmark JSON — instrumentation is permanent, use it):

| Sub-section | ms | What it is |
|---|---|---|
| slots | 0.59 | serial CSR `put` loop for sphere pairs + miss inserts (probes are parallel now) |
| color | 0.61 | serial greedy coloring (partner reads are contiguous now) |
| bodies | 0.35 | parallel body record fill |
| adj | 0.29 | CSR count/prefix/fill for CPU-manifold records |

## What Was Done (2026-06-11 session, newest first)

- `dde2361` — parallel sphere-slot probes + `adjPartnersFlat` for coloring.
  buildFrame 2.84 → 1.84 ms at 20k. Bit-identical physics.
- `3b704fe` — fused contact+joint dual kernels into one `constraintDual`
  dispatch. **No measurable perf change** — see Finding 1 below.
- `9c60106` — large-body broadphase culling via tight world AABBs +
  velocity slack (a 1000x1000 ground slab's bounding sphere used to pair
  with every body every frame).
- `59741b5` — uniform-grid broadphase replaced SAP as default
  (broadphase 3.78 → 2.38 ms at 20k). SAP stays selectable
  (`--broadphase sap`); pair-parity vs all-pairs validated.
- `6833684` — merged the three end-of-step readbacks into one buffer,
  single MapAsync.

## Hard-Won Findings (do not re-learn these)

1. **Dispatch count does not matter.** Cutting ~10 dispatches/step via the
   dual-kernel fuse changed nothing measurable. Per-dispatch launch inside a
   D3D12 compute pass is nearly free. The ~0.85 ms small-scene floor is
   **submit + synchronous MapAsync wait**, not dispatch overhead. Do not
   spend more effort reducing dispatch counts.
2. **Measure before designing.** The "GPU-resident pairs / temporal
   coherence" plan was sized against ~5.7 ms of CPU pair work; profiling
   showed most of it was two serial loops that parallelized in an afternoon
   with zero architectural risk. The sub-stat instrumentation that revealed
   this is permanent — start every future phase by reading it.
3. **Pour-scene ground pairs are mostly real.** The AABB cull removed only
   in-flight sphere-ground pairs because settled pours scatter spheres in
   shallow layers where most are genuinely near the ground. Tall-pile or
   container scenes would benefit much more.
4. **Validation pattern that works:** compare battery (Pyramid/Rope/Soft
   Body, both lanes, same broadphase) catches behavior changes; pure
   restructurings should produce **bit-identical** compare numbers (slot
   assignment order, adjacency content, coloring all preserved). Pair-set
   changes (broadphase) need the all-pairs parity check instead.

## Future Levers (ranked)

1. **Async readback / frame pipelining** — the big one left. The step ends
   with a synchronous MapAsync wait (~1.8-3.1 ms at 20k, and most of the
   small-scene floor). Options: (a) double-buffer and apply the readback one
   frame late (cheap, adds a frame of physics latency to CPU-visible state —
   audit consumers of post-step positions first: warmstart, viewer snapshot,
   stats); (b) overlap the wait with next-frame CPU work (broadphase +
   buildFrame don't need GPU results until upload — restructure step order
   to submit, then do CPU prep, then map). Estimated win: 1.5-3 ms at 20k,
   floor → ~0.3-0.5 ms. Risk: moderate (step-order restructure).
2. **Remaining buildFrame serial bits** — the CSR `put` loop for sphere
   pairs (0.59 ms) and greedy coloring (0.61 ms). The put loop could go
   parallel with per-chunk counting + offsets (deterministic two-pass).
   Coloring could go incremental (reuse last frame's colors, recolor only
   bodies whose adjacency changed — pours are ~95%+ persistent) or
   speculative-parallel (Jones-Plassmann-ish). Estimated win: ~0.8 ms
   combined. Risk: low-moderate; coloring incrementality needs careful
   invalidation.
3. **Broadphase second pass** (~2.3 ms at 20k): the grid itself is fast;
   most of the remaining time is the per-hit classification (constrainedTo
   walks) and the serial apply pass. Candidates: cache constrainedTo results
   for persistent pairs (temporal again), parallelize the apply by
   pre-partitioning sink vs manifold pairs. Estimated win: ~0.5-1 ms.
4. **bodyInit / prepareStep warmstart (~0.7 ms)** and **syncFromLegacy
   parallelization** — straightforward parallel-for candidates, small wins.
5. **GPU-resident pairs / GPU broadphase** — the original Phase-2-finale
   idea, now de-prioritized: after the parallelization wins, all CPU pair
   work combined is ~4 ms and a GPU pair list would need a mid-step readback
   (sync point) for CPU-manifold routing + coloring, or full GPU coloring +
   GPU slot hashing. Only revisit if scenes grow past ~50k bodies or the
   CPU becomes the bottleneck again.
6. **GPU narrowphase coverage** (dynamic box/cylinder partners for the
   sphere kernel, capsules, box-box) — not a perf lever for current scenes
   (CPU manifolds are cheap at current counts); becomes one if scenes lean
   on non-sphere shapes at scale. Tracked as a feature need, not perf.
7. **Delete the OpenGL shell + webgpu_backend.cpp** (~15.8k lines) — zero
   runtime impact, build hygiene only.

## How to Resume

1. Build: `cmake --build build-webgpu-dawn-nmake3` from a VS x64 prompt
   (new source files need a reconfigure — globbed).
2. Read the current numbers: 20k benchmark with
   `--benchmark-scene "Sphere Pour 20000 on Cylinders"` and look at the
   `avbdGpu*AvgMs` fields. Same thermal window for any A/B.
3. Keep the gates green: `npm run checkpoint:gpu-playable` +
   `npm run benchmark:gpu-avbd-20k` from `viewer/`.
4. Kill orphaned `avbd_headless_server.exe` before relinking.
