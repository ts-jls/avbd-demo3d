import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "../..");

const args = new Map();
for (let i = 2; i < process.argv.length; i += 1) {
  const arg = process.argv[i];
  if (arg.startsWith("--")) {
    const key = arg.slice(2);
    const value = process.argv[i + 1]?.startsWith("--") ? "true" : process.argv[++i] ?? "true";
    args.set(key, value);
  }
}

const exe = args.get("exe") ?? path.join(repoRoot, "build-webgpu-dawn-nmake3", "avbd_headless_server.exe");
const frames = Number(args.get("frames") ?? 120);
const iterations = Number(args.get("iterations") ?? 0);
const referenceBackend = args.get("reference-backend") ?? "cpu";
const backends = (args.get("backends") ?? "webgpu-avbd")
  .split(",")
  .map((backend) => backend.trim())
  .filter(Boolean);
const scenes = (args.get("scenes") ?? "Pyramid,Sphere Pour on Cylinders")
  .split(",")
  .map((scene) => scene.trim())
  .filter(Boolean);
const positionRelTolerance = Number(args.get("position-rel-tolerance") ?? 0.02);
const rotationRelTolerance = Number(args.get("rotation-rel-tolerance") ?? 0.02);
const linearVelocityRelTolerance = Number(args.get("linear-velocity-rel-tolerance") ?? args.get("velocity-rel-tolerance") ?? 0.10);
const angularVelocityRelTolerance = Number(args.get("angular-velocity-rel-tolerance") ?? args.get("velocity-rel-tolerance") ?? 0.10);
const maxSpeedRelTolerance = Number(args.get("max-speed-rel-tolerance") ?? 0.20);
const requireNoFallbacks = args.get("require-no-fallbacks") === "true";

function runBenchmark(scene, backend) {
  const benchmarkArgs = [
    "--benchmark-scene",
    scene,
    "--benchmark-frames",
    String(frames),
    "--no-stream",
  ];
  if (backend && backend !== "cpu") {
    benchmarkArgs.push("--physics-backend", backend);
  }
  if (iterations > 0) {
    benchmarkArgs.push("--iterations", String(iterations));
  }

  const result = spawnSync(exe, benchmarkArgs, {
    cwd: path.dirname(exe),
    encoding: "utf8",
  });

  if (result.status !== 0) {
    throw new Error(`Benchmark failed for ${scene} / ${backend}: ${result.stderr || result.stdout}`);
  }

  const line = result.stdout.trim().split(/\r?\n/).find((entry) => entry.includes("\"headlessBenchmark\""));
  if (!line) {
    throw new Error(`No benchmark JSON emitted for ${scene} / ${backend}: ${result.stdout}`);
  }
  return JSON.parse(line);
}

function relativeDelta(a, b) {
  const denominator = Math.max(1, Math.abs(a));
  return Math.abs(b - a) / denominator;
}

function compareCount(scene, backend, key, reference, candidate) {
  if (reference[key] !== candidate[key]) {
    throw new Error(`${scene} / ${backend}: ${key} changed from ${reference[key]} to ${candidate[key]}`);
  }
}

function compareMetric(scene, backend, key, tolerance, reference, candidate) {
  const delta = relativeDelta(Number(reference[key] ?? 0), Number(candidate[key] ?? 0));
  if (delta > tolerance) {
    throw new Error(`${scene} / ${backend}: ${key} relative delta ${delta.toFixed(6)} exceeded ${tolerance}`);
  }
  return delta;
}

const results = [];
for (const scene of scenes) {
  const reference = runBenchmark(scene, referenceBackend);
  const comparisons = [];
  for (const backend of backends) {
    const candidate = runBenchmark(scene, backend);
    compareCount(scene, backend, "finalActiveBodies", reference, candidate);
    compareCount(scene, backend, "finalDynamicBodies", reference, candidate);
    compareCount(scene, backend, "finalBoxes", reference, candidate);
    compareCount(scene, backend, "finalSpheres", reference, candidate);
    compareCount(scene, backend, "finalCapsules", reference, candidate);
    compareCount(scene, backend, "finalCylinders", reference, candidate);
    if (requireNoFallbacks && Number(candidate.webgpuRuntimeFallbacks ?? 0) !== 0) {
      throw new Error(`${scene} / ${backend}: used ${candidate.webgpuRuntimeFallbacks} WebGPU fallback(s)`);
    }

    const deltas = {
      finalPositionChecksum: compareMetric(scene, backend, "finalPositionChecksum", positionRelTolerance, reference, candidate),
      finalPositionAbsChecksum: compareMetric(scene, backend, "finalPositionAbsChecksum", positionRelTolerance, reference, candidate),
      finalRotationChecksum: compareMetric(scene, backend, "finalRotationChecksum", rotationRelTolerance, reference, candidate),
      finalLinearVelocityChecksum: compareMetric(scene, backend, "finalLinearVelocityChecksum", linearVelocityRelTolerance, reference, candidate),
      finalAngularVelocityChecksum: compareMetric(scene, backend, "finalAngularVelocityChecksum", angularVelocityRelTolerance, reference, candidate),
      finalMaxSpeed: compareMetric(scene, backend, "finalMaxSpeed", maxSpeedRelTolerance, reference, candidate),
    };

    comparisons.push({
      backend,
      physicsAvgMs: candidate.physicsAvgMs,
      webgpuRuntimeFallbacks: candidate.webgpuRuntimeFallbacks ?? 0,
      deltas,
      signature: {
        finalPositionChecksum: candidate.finalPositionChecksum,
        finalPositionAbsChecksum: candidate.finalPositionAbsChecksum,
        finalRotationChecksum: candidate.finalRotationChecksum,
        finalLinearVelocityChecksum: candidate.finalLinearVelocityChecksum,
        finalAngularVelocityChecksum: candidate.finalAngularVelocityChecksum,
        finalMaxSpeed: candidate.finalMaxSpeed,
      },
    });
  }
  results.push({
    scene,
    reference: {
      backend: reference.physicsBackend,
      physicsAvgMs: reference.physicsAvgMs,
      signature: {
        finalActiveBodies: reference.finalActiveBodies,
        finalDynamicBodies: reference.finalDynamicBodies,
        finalPositionChecksum: reference.finalPositionChecksum,
        finalPositionAbsChecksum: reference.finalPositionAbsChecksum,
        finalRotationChecksum: reference.finalRotationChecksum,
        finalLinearVelocityChecksum: reference.finalLinearVelocityChecksum,
        finalAngularVelocityChecksum: reference.finalAngularVelocityChecksum,
        finalMaxSpeed: reference.finalMaxSpeed,
      },
    },
    comparisons,
  });
}

console.log(JSON.stringify({
  ok: true,
  exe,
  frames,
  iterations,
  referenceBackend,
  backends,
  scenes,
  tolerances: {
    positionRelTolerance,
    rotationRelTolerance,
    linearVelocityRelTolerance,
    angularVelocityRelTolerance,
    maxSpeedRelTolerance,
    requireNoFallbacks,
  },
  results,
}, null, 2));
