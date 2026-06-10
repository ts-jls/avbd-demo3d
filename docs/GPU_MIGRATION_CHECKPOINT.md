# GPU Migration Checkpoint

This document records the current trusted state of the AVBD GPU migration. It is intentionally conservative: it separates the playable path from experimental probes so we do not mistake partial GPU ownership for the final paper-style GPU solver.

## Current Production-Test Path

- **Primary viewer:** three.js browser viewer connected to the headless native server.
- **Recommended browser backend:** `Auto Playable Safe`.
- **Headless auto backend policy:**
  - `Soft Body*` and `Bridge*` scenes use `CPU Reference`.
  - Other scenes use `WebGPU Physics Experimental Contact Direct`.
- **OpenGL app:** retained as a debug/reference shell, not the desired production visualization path.

## Trusted Modes

### CPU Reference

The correctness reference. It runs the original AVBD solver and should remain available for comparisons and joint-heavy scenes.

### WebGPU Physics Experimental Contact Direct

The current fast playable path for sphere/round-body-heavy scenes. It uses WebGPU work for direct contact handling and avoids the full CPU AVBD solve for supported contact-heavy scenes. It is not a complete general GPU AVBD solver.

### Auto Playable Safe

The recommended user-facing mode while the GPU solver is still maturing. It routes scenes to CPU or WebGPU Contact Direct based on known stability.

### WebGPU AVBD (`webgpu-avbd`)

The paper-faithful GPU solver lane (source/avbd_gpu_solver.cpp). Unlike the
Contact Direct experiments, this runs the *actual* AVBD algorithm on the GPU:

- CPU keeps broadphase, narrowphase, manifold warmstarting, and body
  warmstarting (`Solver::prepareStep`).
- The GPU runs the full iteration loop: graph-colored per-body 6x6 LDL
  primal solves (contacts, joints, and springs stamped exactly as the CPU
  reference) plus per-constraint dual lambda/penalty updates.
- One upload at step start, one readback at step end (positions + dual
  state, so the next frame's CPU warmstart matches the reference).

Because joints ride the same kernel as contacts, this lane handles soft-body
and bridge scenes — Soft Body 8x8x8 runs ~2.6x faster than CPU Reference and
settles to the same rest state. Backend comparisons against `cpu` track to
1e-4..1e-2 over 180 frames on stacking/joint scenes (residual drift comes
from colored vs. sequential solve ordering, not from a different algorithm).

Run a comparison with:

```powershell
avbd_headless_server.exe --compare-scene "Soft Body" --compare-frames 180 --compare-backend webgpu-avbd --no-stream
```

Current limitations: per-frame upload/readback and synchronous mapping make
small scenes overhead-bound (~7 ms floor), and the 5,000-sphere pour runs
~63 ms (faster than CPU's ~100 ms, but not yet at the Contact Direct arcade
path's 25 ms budget). The next migration slices are GPU-resident constraint
state across frames and async readback.

## Backend Lanes (post-consolidation)

The experimental lanes (fast, counterless-fast, direct-fast, contact-resident,
contact-resident-async, resident-ground-fast, joint-proposal, joint-replace,
joint-direct, joint-contact-direct, and the bare `webgpu` runtime spine) were
removed after `webgpu-avbd` superseded them. The supported lanes are:

- `cpu` — the correctness reference.
- `webgpu-avbd` — the paper-faithful GPU solver (general purpose, handles
  joints/soft bodies).
- `webgpu-contact-direct` — the specialized arcade path for the 5,000-sphere
  pour (fast position projection, not a general solver).
- `auto` — routes Soft Body/Bridge scenes to `cpu`, everything else to
  `webgpu-contact-direct`.

Remaining opt-in probes (ride the contact-direct lane):

- `--gpu-apply-prediction`: applies WebGPU body prediction output back into the legacy body state.
- `--gpu-apply-velocity`: applies WebGPU velocity output back into the legacy body state.
- `--gpu-runtime-validate`: turns sample readback validation back on for runtime kernels.

## What Is Actually GPU Today

Implemented and benchmarked:

- WebGPU device/runtime initialization.
- Full AVBD primal/dual iteration loop on GPU (`webgpu-avbd`): graph-colored
  per-body 6x6 solves + per-constraint dual updates for contacts, joints, and
  springs, validated against the CPU reference.
- Body prediction and velocity update kernels (contact-direct probes).
- Contact-direct path for the 5,000 sphere pour scene.

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
- `npm run benchmark:gpu-playable-5k`
- `npm run benchmark:gpu-softbody-playable`
- `npm run benchmark:gpu-apply-prediction-5k`
- `npm run benchmark:gpu-apply-velocity-5k`
- `npm run benchmark:gpu-avbd-softbody`
- `npm run benchmark:gpu-avbd-compare`

Passing this checkpoint means:

- The 5,000 sphere scene stays within the current playable performance/stability target.
- Soft Body 8x8x8 stays stable on the safe auto route.
- GPU prediction and velocity apply probes still work as opt-in migration steps.
- The `webgpu-avbd` solver stays stable on Soft Body 8x8x8 and tracks the CPU
  reference on Pyramid/Rope/Soft Body within position/rotation tolerances.
- No WebGPU fallbacks occur in the GPU tests.

## Next Recommended Engineering Step

Keep this checkpoint green and use it as the baseline. The next GPU migration
slice for `webgpu-avbd` is reducing per-frame CPU<->GPU traffic: keep
constraint/body buffers resident across frames (upload deltas, not the world)
and make the end-of-step readback asynchronous/double-buffered.
