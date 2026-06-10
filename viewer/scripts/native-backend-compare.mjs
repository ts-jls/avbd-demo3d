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
const backend = args.get("backend") ?? "webgpu-avbd";
const scenes = (args.get("scenes") ?? "Pyramid,Sphere Pour on Cylinders")
  .split(",")
  .map((scene) => scene.trim())
  .filter(Boolean);
const assertMaxPositionError = Number(args.get("assert-max-position-error") ?? -1);
const assertRmsPositionError = Number(args.get("assert-rms-position-error") ?? -1);
const assertMaxRotationError = Number(args.get("assert-max-rotation-error") ?? -1);
const assertRmsRotationError = Number(args.get("assert-rms-rotation-error") ?? -1);
const assertMaxLinearVelocityError = Number(args.get("assert-max-linear-velocity-error") ?? -1);
const assertMaxAngularVelocityError = Number(args.get("assert-max-angular-velocity-error") ?? -1);
const assertNoFallbacks = args.get("assert-no-webgpu-fallbacks") === "true";
const assertNoWebgpuValidation = args.get("assert-no-webgpu-validation") === "true" || assertNoFallbacks;

function hasWebGpuValidationProblem(value) {
  if (typeof value !== "string") {
    return false;
  }
  return /webgpu validation error|invalid commandbuffer|invalid computepipeline|invalid shadermodule|device lost|validation failed/i.test(value);
}

function runNativeCompare(scene) {
  const nativeArgs = [
    "--compare-scene",
    scene,
    "--compare-frames",
    String(frames),
    "--compare-backend",
    backend,
  ];
  if (iterations > 0) {
    nativeArgs.push("--iterations", String(iterations));
  }

  const result = spawnSync(exe, nativeArgs, {
    cwd: path.dirname(exe),
    encoding: "utf8",
  });

  if (result.status !== 0) {
    throw new Error(`Native backend comparison failed for ${scene}: ${result.stderr || result.stdout}`);
  }

  const line = result.stdout.trim().split(/\r?\n/).find((entry) => entry.includes("\"backendComparison\""));
  if (!line) {
    throw new Error(`No comparison JSON emitted for ${scene}: ${result.stdout}`);
  }
  return JSON.parse(line);
}

function assertThreshold(scene, label, value, limit) {
  if (limit >= 0 && value > limit) {
    throw new Error(`${scene}: ${label} ${value} exceeded ${limit}`);
  }
}

const results = [];
for (const scene of scenes) {
  const comparison = runNativeCompare(scene);
  if (comparison.bodyCountMismatch !== 0) {
    throw new Error(`${scene}: body count mismatch ${comparison.referenceBodies} vs ${comparison.candidateBodies}`);
  }
  if (assertNoFallbacks && Number(comparison.candidateWebgpuRuntimeFallbacks ?? 0) !== 0) {
    throw new Error(`${scene}: used ${comparison.candidateWebgpuRuntimeFallbacks} WebGPU fallback(s)`);
  }
  if (assertNoWebgpuValidation) {
    const statusFields = [
      "candidateWebgpuStatus",
      "candidateWebgpuRuntimeStatus",
      "candidateWebgpuSapStatus",
      "candidateWebgpuJointTopologyStatus",
    ];
    for (const field of statusFields) {
      if (hasWebGpuValidationProblem(comparison[field])) {
        throw new Error(`${scene}: unhealthy ${field}: ${comparison[field]}`);
      }
    }
  }
  assertThreshold(scene, "maxPositionError", comparison.maxPositionError, assertMaxPositionError);
  assertThreshold(scene, "rmsPositionError", comparison.rmsPositionError, assertRmsPositionError);
  assertThreshold(scene, "maxRotationError", comparison.maxRotationError, assertMaxRotationError);
  assertThreshold(scene, "rmsRotationError", comparison.rmsRotationError, assertRmsRotationError);
  assertThreshold(scene, "maxLinearVelocityError", comparison.maxLinearVelocityError, assertMaxLinearVelocityError);
  assertThreshold(scene, "maxAngularVelocityError", comparison.maxAngularVelocityError, assertMaxAngularVelocityError);
  results.push(comparison);
}

console.log(JSON.stringify({
  ok: true,
  exe,
  frames,
  iterations,
  backend,
  scenes,
  assertions: {
    assertMaxPositionError,
    assertRmsPositionError,
    assertMaxRotationError,
    assertRmsRotationError,
    assertMaxLinearVelocityError,
    assertMaxAngularVelocityError,
    assertNoFallbacks,
    assertNoWebgpuValidation,
  },
  results,
}, null, 2));
