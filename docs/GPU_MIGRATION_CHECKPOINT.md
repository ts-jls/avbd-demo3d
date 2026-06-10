# GPU Migration Checkpoint

This document records the current trusted state of the AVBD GPU migration.

## Current Production-Test Path

- **Primary viewer:** three.js browser viewer connected to the headless native server.
- **Default backend:** `webgpu-avbd` (what `npm run server:gpu` launches).
- **OpenGL app:** retained as a debug/reference shell, not the desired production visualization path.

## Backend Lanes

After consolidation there are exactly two solver lanes:

### CPU Reference (`cpu`)

The correctness reference. It runs the original AVBD solver and is the
baseline every GPU result is compared against. Also the automatic fallback
when WebGPU/Dawn is unavailable.

### WebGPU AVBD (`webgpu-avbd`)

The paper-faithful GPU solver lane (source/avbd_gpu_solver.cpp) and the
default for all scenes:

- CPU keeps broadphase, narrowphase, manifold warmstarting, and body
  warmstarting (`Solver::prepareStep`).
- The GPU runs the full iteration loop: graph-colored per-body 6x6 LDL
  primal solves (contacts, joints, and springs stamped exactly as the CPU
  reference) plus per-constraint dual lambda/penalty updates.
- One upload at step start, one readback at step end (positions + dual
  state, so the next frame's CPU warmstart matches the reference).

Because joints ride the same kernel as contacts, this lane handles soft-body
and bridge scenes — Soft Body 8x8x8 runs ~2.6x faster than CPU Reference and
settles to the same rest state. The 5,000-sphere pour runs at ~24 ms/frame on
an uncontended RTX 4090 laptop GPU (inside the 25 ms playable budget that
previously required the specialized contact-direct path). Backend comparisons
against `cpu` track to 1e-4..1e-2 over 180 frames on stacking/joint scenes
(residual drift comes from colored vs. sequential solve ordering, not from a
different algorithm).

Run a comparison with:

```powershell
avbd_headless_server.exe --compare-scene "Soft Body" --compare-frames 180 --compare-backend webgpu-avbd --no-stream
```

### Removed lanes

The historical experimental lanes (fast, counterless-fast, direct-fast,
contact-resident, contact-resident-async, resident-ground-fast,
joint-proposal, joint-replace, joint-direct, joint-contact-direct, the bare
`webgpu` runtime spine, the `webgpu-contact-direct` arcade path, and the
`auto` router) were all removed after `webgpu-avbd` superseded them. Their
kernel/runtime internals still exist inside `source/webgpu_backend.cpp` but
are unreachable from the headless server; deleting/splitting that file is the
remaining consolidation task.

## What Is Actually GPU Today

Implemented and benchmarked:

- WebGPU device/runtime initialization.
- Full AVBD primal/dual iteration loop on GPU (`webgpu-avbd`): graph-colored
  per-body 6x6 solves + per-constraint dual updates for contacts, joints, and
  springs, validated against the CPU reference.

Still incomplete:

- GPU broadphase/narrowphase as the default path (CPU still generates
  manifolds for `webgpu-avbd`).
- GPU-resident constraint state across frames (today: full upload per step).
- Async end-of-step readback (today: synchronous map, ~7 ms small-scene floor).

## Current Acceptance Checkpoint

Run from `C:\code\avbd-demo3d\viewer` after building `build-webgpu-dawn-nmake3`:

```powershell
npm run checkpoint:gpu-playable
```

This runs:

- JS/package sanity checks.
- `npm run benchmark:gpu-avbd-5k` — 5,000 sphere pour, 600 frames: physics
  avg < 35 ms, ground penetration < 0.05, settled speeds/heights, no
  fallbacks.
- `npm run benchmark:gpu-avbd-softbody` — Soft Body 8x8x8, 240 frames:
  physics avg < 45 ms, stable rest state, no ground penetration.
- `npm run benchmark:gpu-avbd-compare` — Pyramid/Rope/Soft Body stepped in
  lockstep against `cpu`: max position error < 0.05, RMS < 0.02.

## Next Recommended Engineering Step

Keep this checkpoint green and use it as the baseline. The next GPU migration
slices for `webgpu-avbd`:

1. Keep constraint/body buffers GPU-resident across frames (upload deltas,
   not the world) and make the end-of-step readback asynchronous — this
   removes the ~7 ms small-scene floor and most of the 5k-scene cost.
2. Move narrowphase to GPU pair-type by pair-type (sphere pairs first,
   box-box SAT/clipping last).
3. Delete/split `source/webgpu_backend.cpp` now that only device init and
   diagnostics are reachable from the server.
