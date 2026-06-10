import { spawn, spawnSync } from "node:child_process";
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

const exe = args.get("exe") ?? path.join(repoRoot, "build-viewer-bridge-nmake", "avbd_headless_server.exe");
const frames = Number(args.get("frames") ?? 600);
const warmupFrames = Number(args.get("warmup-frames") ?? 0);
const resetAfterWarmup = args.get("reset-after-warmup") === "true";
const streamFrames = Number(args.get("stream-frames") ?? 180);
const port = Number(args.get("port") ?? 8765);
const iterations = Number(args.get("iterations") ?? 0);
const gravity = args.get("gravity");
const physicsBackend = args.get("physics-backend") ?? "cpu";
const snapshotMode = args.get("snapshot-mode") ?? "json";
const collisionOnly = args.get("collision-only") === "true";
const gpuSphereContacts = args.get("gpu-sphere-contacts") === "true";
const gpuGroundContactFeed = args.get("gpu-ground-contact-feed") === "true";
const gpuContactSolveDiagnostic = args.get("gpu-contact-solve-diagnostic") === "true";
const gpuContactSolveNoReadback = args.get("gpu-contact-solve-no-readback") === "true";
const gpuDirectCounterReadback = args.get("gpu-direct-counter-readback") === "true";
const gpuRuntimeValidate = args.get("gpu-runtime-validate") === "true";
const gpuApplyPrediction = args.get("gpu-apply-prediction") === "true";
const gpuApplyVelocity = args.get("gpu-apply-velocity") === "true";
const gpuResidentCounterlessContacts = args.get("gpu-resident-counterless-contacts") === "true";
const gpuDeferFinalPositionReadback = args.get("gpu-defer-final-position-readback") === "true";
const gpuAsyncFinalPositionReadback = args.get("gpu-async-final-position-readback") === "true";
const gpuApplyContactPositions = args.get("gpu-apply-contact-positions") === "true";
const gpuGroundContacts = args.get("gpu-ground-contacts") === "true";
const gpuJointTopologyDiagnostic = args.get("gpu-joint-topology-diagnostic") === "true";
const gpuApplyJointProposals = args.get("gpu-apply-joint-proposals") === "true";
const gpuRuntimeJointProposals = args.get("gpu-runtime-joint-proposals") === "true";
const gpuReplaceCpuJoints = args.get("gpu-replace-cpu-joints") === "true";
const gpuJointTopologyRepeats = Number(args.get("gpu-joint-topology-repeats") ?? 1);
const gpuJointProposalIterations = Number(args.get("gpu-joint-proposal-iterations") ?? 1);
const gpuContactIterations = Number(args.get("gpu-contact-iterations") ?? 4);
const gpuContactRelaxation = Number(args.get("gpu-contact-relaxation") ?? 0.10);
const gpuJointPerturb = Number(args.get("gpu-joint-perturb") ?? 0);
const effectiveFastWebGpu = physicsBackend === "webgpu-fast" ||
  physicsBackend === "webgpu-sphere-ground-fast" ||
  physicsBackend === "webgpu-counterless-fast" ||
  physicsBackend === "webgpu-direct-fast" ||
  physicsBackend === "webgpu-resident-ground-fast";
const effectiveRuntimeJointProposals = gpuRuntimeJointProposals || physicsBackend === "webgpu-joint-proposal" || physicsBackend === "webgpu-joint-replace" || physicsBackend === "webgpu-joint-direct" || physicsBackend === "webgpu-joint-contact-direct";
const effectiveReplaceCpuJoints = gpuReplaceCpuJoints || physicsBackend === "webgpu-joint-replace";
const effectiveDirectJointSolve = physicsBackend === "webgpu-joint-direct" || physicsBackend === "webgpu-joint-contact-direct";
const effectiveDirectContactSolve = physicsBackend === "webgpu-contact-direct" ||
  physicsBackend === "webgpu-contact-resident" ||
  physicsBackend === "webgpu-contact-resident-async" ||
  physicsBackend === "webgpu-joint-contact-direct";
const effectiveContactResidentAsync = physicsBackend === "webgpu-contact-resident-async";
const effectiveContactNoReadback = gpuContactSolveNoReadback || effectiveFastWebGpu || effectiveContactResidentAsync;
const effectiveResidentCounterlessContacts = gpuResidentCounterlessContacts || effectiveContactResidentAsync;
const effectiveAsyncFinalPositionReadback = gpuAsyncFinalPositionReadback || effectiveContactResidentAsync;
const assertPhysicsAvgUnder = Number(args.get("assert-physics-avg-under") ?? 0);
const assertNoWebgpuFallbacks = args.get("assert-no-webgpu-fallbacks") === "true";
const assertNoWebgpuValidation = args.get("assert-no-webgpu-validation") === "true" || assertNoWebgpuFallbacks;
const assertGroundCandidatesMin = Number(args.get("assert-ground-candidates-min") ?? 0);
const assertDirectSphereCylinderCandidatesMin = Number(args.get("assert-direct-sphere-cylinder-candidates-min") ?? 0);
const assertDirectSphereCapsuleCandidatesMin = Number(args.get("assert-direct-sphere-capsule-candidates-min") ?? 0);
const assertDirectSphereBoxCandidatesMin = Number(args.get("assert-direct-sphere-box-candidates-min") ?? 0);
const assertDirectBoxPairCandidatesMin = Number(args.get("assert-direct-box-pair-candidates-min") ?? 0);
const assertDirectSphereCylinderCandidatesMaxMin = Number(args.get("assert-direct-sphere-cylinder-candidates-max-min") ?? 0);
const assertDirectSphereCapsuleCandidatesMaxMin = Number(args.get("assert-direct-sphere-capsule-candidates-max-min") ?? 0);
const assertDirectSphereBoxCandidatesMaxMin = Number(args.get("assert-direct-sphere-box-candidates-max-min") ?? 0);
const assertDirectBoxPairCandidatesMaxMin = Number(args.get("assert-direct-box-pair-candidates-max-min") ?? 0);
const assertDirectRoundPairCandidatesMin = Number(args.get("assert-direct-round-pair-candidates-min") ?? 0);
const assertDirectRoundPairCandidatesMaxMin = Number(args.get("assert-direct-round-pair-candidates-max-min") ?? 0);
const assertDirectGpuContactRecordsMin = Number(args.get("assert-direct-gpu-contact-records-min") ?? 0);
const assertDirectGpuContactRecordsMaxMin = Number(args.get("assert-direct-gpu-contact-records-max-min") ?? 0);
const assertDirectGpuRoundPairCandidatesMin = Number(args.get("assert-direct-gpu-round-pair-candidates-min") ?? 0);
const assertDirectGpuRoundPairCandidatesMaxMin = Number(args.get("assert-direct-gpu-round-pair-candidates-max-min") ?? 0);
const assertDirectGpuBoxPairCandidatesMin = Number(args.get("assert-direct-gpu-box-pair-candidates-min") ?? 0);
const assertDirectGpuBoxPairCandidatesMaxMin = Number(args.get("assert-direct-gpu-box-pair-candidates-max-min") ?? 0);
const assertDirectGpuCounterReadbackBytesMin = Number(args.get("assert-direct-gpu-counter-readback-bytes-min") ?? 0);
const assertDirectGpuCounterReadbackBytesMax = Number(args.get("assert-direct-gpu-counter-readback-bytes-max") ?? -1);
const assertPredictionAppliedBodiesMin = Number(args.get("assert-prediction-applied-bodies-min") ?? 0);
const assertPredictionAppliedReadbackBytesMin = Number(args.get("assert-prediction-applied-readback-bytes-min") ?? 0);
const assertVelocityAppliedBodiesMin = Number(args.get("assert-velocity-applied-bodies-min") ?? 0);
const assertVelocityAppliedReadbackBytesMin = Number(args.get("assert-velocity-applied-readback-bytes-min") ?? 0);
const assertExternalContactsMin = Number(args.get("assert-external-contacts-min") ?? 0);
const assertExternalGroundContactsMin = Number(args.get("assert-external-ground-contacts-min") ?? 0);
const assertContactProposalActiveBodiesMin = Number(args.get("assert-contact-proposal-active-bodies-min") ?? 0);
const assertContactFinalPositionReady = args.get("assert-contact-final-position-ready") === "true";
const assertFinalPositionReadbackDeferred = args.get("assert-final-position-readback-deferred") === "true";
const assertFinalPositionAsyncReadbackConsumed = args.get("assert-final-position-async-readback-consumed") === "true";
const assertFinalPositionAsyncReadbackDroppedMax = Number(args.get("assert-final-position-async-readback-dropped-max") ?? -1);
const assertFinalPositionAsyncReadbackWaitMax = Number(args.get("assert-final-position-async-readback-wait-max") ?? -1);
const assertContactProposalResidualAfterMax = Number(args.get("assert-contact-proposal-residual-after-max") ?? -1);
const assertContactIterationResidualAfterMax = Number(args.get("assert-contact-iteration-residual-after-max") ?? -1);
const assertContactAdjacencyReadbackBytesMax = Number(args.get("assert-contact-adjacency-readback-bytes-max") ?? -1);
const assertContactGatherReadbackBytesMax = Number(args.get("assert-contact-gather-readback-bytes-max") ?? -1);
const assertContactProposalResidualReadbackBytesMax = Number(args.get("assert-contact-proposal-residual-readback-bytes-max") ?? -1);
const assertSapCandidatesMin = Number(args.get("assert-sap-candidates-min") ?? 0);
const assertSphereHitsMin = Number(args.get("assert-sphere-hits-min") ?? 0);
const assertPairChecksMin = Number(args.get("assert-pair-checks-min") ?? 0);
const assertAppliedPositionBodiesMin = Number(args.get("assert-applied-position-bodies-min") ?? 0);
const assertDirectSphereAppliedBodiesMin = Number(args.get("assert-direct-sphere-applied-bodies-min") ?? 0);
const assertDirectSphereAppliedBodiesMaxMin = Number(args.get("assert-direct-sphere-applied-bodies-max-min") ?? 0);
const assertDirectRoundAppliedBodiesMin = Number(args.get("assert-direct-round-applied-bodies-min") ?? 0);
const assertDirectRoundAppliedBodiesMaxMin = Number(args.get("assert-direct-round-applied-bodies-max-min") ?? 0);
const assertDirectGroundAppliedBodiesMin = Number(args.get("assert-direct-ground-applied-bodies-min") ?? 0);
const assertDirectGroundAppliedBodiesMaxMin = Number(args.get("assert-direct-ground-applied-bodies-max-min") ?? 0);
const assertManifoldCountMax = Number(args.get("assert-manifold-count-max") ?? -1);
const assertFinalMaxSpeedMin = Number(args.get("assert-final-max-speed-min") ?? -1);
const assertFinalMaxSpeedMax = Number(args.get("assert-final-max-speed-max") ?? -1);
const assertFinalDynamicMaxHeightMin = Number(args.get("assert-final-dynamic-max-height-min") ?? -1);
const assertFinalDynamicMaxHeightMax = Number(args.get("assert-final-dynamic-max-height-max") ?? -1);
const assertPrimalAvgMax = Number(args.get("assert-primal-avg-max") ?? -1);
const assertDualAvgMax = Number(args.get("assert-dual-avg-max") ?? -1);
const assertBelowGroundSpheresMax = Number(args.get("assert-below-ground-spheres-max") ?? -1);
const assertMaxGroundPenetration = Number(args.get("assert-max-ground-penetration") ?? -1);
const assertBelowGroundBodiesMax = Number(args.get("assert-below-ground-bodies-max") ?? -1);
const assertMaxGroundBodyPenetration = Number(args.get("assert-max-ground-body-penetration") ?? -1);
const assertMaxLinearError = Number(args.get("assert-max-linear-error") ?? -1);
const assertMaxAngularError = Number(args.get("assert-max-angular-error") ?? -1);
const assertSapCounterReadbackBytesMax = Number(args.get("assert-sap-counter-readback-bytes-max") ?? -1);
const assertSapPairReadbackBytesMax = Number(args.get("assert-sap-pair-readback-bytes-max") ?? -1);
const assertJointTopologyMismatchesMax = Number(args.get("assert-joint-topology-mismatches-max") ?? -1);
const assertJointTopologyJointsMin = Number(args.get("assert-joint-topology-joints-min") ?? 0);
const assertJointTopologyLastMsUnder = Number(args.get("assert-joint-topology-last-ms-under") ?? -1);
const assertJointColorConflictsMax = Number(args.get("assert-joint-color-conflicts-max") ?? -1);
const assertJointColorCountMax = Number(args.get("assert-joint-color-count-max") ?? -1);
const assertJointResidualMax = Number(args.get("assert-joint-residual-max") ?? -1);
const assertJointResidualMin = Number(args.get("assert-joint-residual-min") ?? -1);
const assertJointProposalMaxCorrection = Number(args.get("assert-joint-proposal-max-correction") ?? -1);
const assertJointProposalActiveBodiesMin = Number(args.get("assert-joint-proposal-active-bodies-min") ?? 0);
const assertJointProposalResidualAfterMax = Number(args.get("assert-joint-proposal-residual-after-max") ?? -1);
const assertJointProposalResidualRmsRatioMax = Number(args.get("assert-joint-proposal-residual-rms-ratio-max") ?? -1);
const assertJointProposalAppliedBodiesMin = Number(args.get("assert-joint-proposal-applied-bodies-min") ?? 0);
const assertJointProposalAppliedMaxDeltaMin = Number(args.get("assert-joint-proposal-applied-max-delta-min") ?? -1);
const assertJointFinalPositionAsyncReadbackConsumed = args.get("assert-joint-final-position-async-readback-consumed") === "true";
const assertJointFinalPositionAsyncReadbackDroppedMax = Number(args.get("assert-joint-final-position-async-readback-dropped-max") ?? -1);
const assertJointFinalPositionAsyncReadbackWaitMax = Number(args.get("assert-joint-final-position-async-readback-wait-max") ?? -1);
const assertPrimalJointSkippedMin = Number(args.get("assert-primal-joint-skipped-min") ?? 0);
const assertDualJointSkippedMin = Number(args.get("assert-dual-joint-skipped-min") ?? 0);
const assertPrimalIgnoreSkippedMin = Number(args.get("assert-primal-ignore-skipped-min") ?? 0);
const assertDualIgnoreSkippedMin = Number(args.get("assert-dual-ignore-skipped-min") ?? 0);
const assertJointInitializationSkippedMin = Number(args.get("assert-joint-initialization-skipped-min") ?? 0);
const assertIgnoreInitializationSkippedMin = Number(args.get("assert-ignore-initialization-skipped-min") ?? 0);
const assertDirectJointSolve = args.get("assert-direct-joint-solve") === "true";
const assertDirectContactSolve = args.get("assert-direct-contact-solve") === "true";
const assertStreamSnapshotsPerSecondMin = Number(args.get("assert-stream-snapshots-per-second-min") ?? 0);
const assertStreamAvgSnapshotBytesMax = Number(args.get("assert-stream-avg-snapshot-bytes-max") ?? -1);
const assertStreamAvgParseMsMax = Number(args.get("assert-stream-avg-parse-ms-max") ?? -1);
const streamTimeoutMs = Number(args.get("stream-timeout-ms") ?? 30000);
const scenes = (args.get("scenes") ?? "Pyramid,Sphere Pour on Cylinders,Sphere Pour 5000 on Cylinders,Soft Body 8x8x8")
  .split(",")
  .map((scene) => scene.trim())
  .filter(Boolean);

function hasWebGpuValidationProblem(value) {
  if (typeof value !== "string") {
    return false;
  }
  return /webgpu validation error|invalid commandbuffer|invalid computepipeline|invalid shadermodule|device lost|validation failed/i.test(value);
}

async function eventDataToArrayBuffer(data) {
  if (data instanceof ArrayBuffer) {
    return data;
  }
  if (ArrayBuffer.isView(data)) {
    return data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
  }
  if (typeof Blob !== "undefined" && data instanceof Blob) {
    return await data.arrayBuffer();
  }
  return Buffer.from(String(data), "utf8").buffer;
}

async function eventDataToText(data) {
  if (typeof data === "string") {
    return data;
  }
  const buffer = await eventDataToArrayBuffer(data);
  return Buffer.from(buffer).toString("utf8");
}

function decodeBinarySnapshotHeader(buffer) {
  const view = new DataView(buffer);
  let offset = 0;
  const magic = view.getUint32(offset, true);
  offset += 4;
  if (magic !== 0x53425641) {
    throw new Error("Invalid binary snapshot magic");
  }
  const version = view.getUint32(offset, true);
  offset += 4;
  if (version !== 1) {
    throw new Error(`Unsupported binary snapshot version ${version}`);
  }
  const frame = Number(view.getBigUint64(offset, true));
  offset += 8;
  const sceneBytes = view.getUint32(offset, true);
  offset += 4;
  const bodyCount = view.getUint32(offset, true);
  offset += 4 + sceneBytes;
  const bytesPerBody = 64;
  if (buffer.byteLength < offset + bodyCount * bytesPerBody) {
    throw new Error("Truncated binary snapshot body payload");
  }
  return { type: "snapshot", version, frame, bodies: bodyCount };
}

function isBinarySnapshotBuffer(buffer) {
  return buffer.byteLength >= 24 && new DataView(buffer).getUint32(0, true) === 0x53425641;
}

function runHeadlessBenchmark(scene) {
  const benchmarkArgs = [
    "--benchmark-scene",
    scene,
    "--benchmark-frames",
    String(frames),
    "--no-stream",
  ];
  if (warmupFrames > 0) {
    benchmarkArgs.push("--warmup-frames", String(warmupFrames));
  }
  if (resetAfterWarmup) {
    benchmarkArgs.push("--reset-after-warmup");
  }
  if (physicsBackend !== "cpu") {
    benchmarkArgs.push("--physics-backend", physicsBackend);
  }
  if (iterations > 0) {
    benchmarkArgs.push("--iterations", String(iterations));
  }
  if (gravity !== undefined) {
    benchmarkArgs.push("--gravity", String(gravity));
  }
  if (collisionOnly) {
    benchmarkArgs.push("--collision-only");
  }
  if (gpuSphereContacts) {
    benchmarkArgs.push("--gpu-sphere-contacts");
  }
  if (gpuGroundContactFeed) {
    benchmarkArgs.push("--gpu-ground-contact-feed");
  }
  if (gpuContactSolveDiagnostic) {
    benchmarkArgs.push("--gpu-contact-solve-diagnostic");
  }
  if (gpuContactSolveNoReadback) {
    benchmarkArgs.push("--gpu-contact-solve-no-readback");
  }
  if (gpuDirectCounterReadback) {
    benchmarkArgs.push("--gpu-direct-counter-readback");
  }
  if (gpuRuntimeValidate) {
    benchmarkArgs.push("--gpu-runtime-validate");
  }
  if (gpuApplyPrediction) {
    benchmarkArgs.push("--gpu-apply-prediction");
  }
  if (gpuApplyVelocity) {
    benchmarkArgs.push("--gpu-apply-velocity");
  }
  if (gpuResidentCounterlessContacts) {
    benchmarkArgs.push("--gpu-resident-counterless-contacts");
  }
  if (gpuDeferFinalPositionReadback) {
    benchmarkArgs.push("--gpu-defer-final-position-readback");
  }
  if (gpuAsyncFinalPositionReadback) {
    benchmarkArgs.push("--gpu-async-final-position-readback");
  }
  if (gpuApplyContactPositions) {
    benchmarkArgs.push("--gpu-apply-contact-positions");
  }
  if (gpuGroundContacts) {
    benchmarkArgs.push("--gpu-ground-contacts");
  }
  if (gpuRuntimeJointProposals) {
    benchmarkArgs.push("--gpu-runtime-joint-proposals");
  }
  if (gpuReplaceCpuJoints) {
    benchmarkArgs.push("--gpu-replace-cpu-joints");
  }
  if (effectiveRuntimeJointProposals && gpuJointProposalIterations > 1) {
    benchmarkArgs.push("--gpu-joint-proposal-iterations", String(gpuJointProposalIterations));
  }
  if (gpuContactIterations !== 4) {
    benchmarkArgs.push("--gpu-contact-iterations", String(gpuContactIterations));
  }
  if (gpuContactRelaxation !== 0.10) {
    benchmarkArgs.push("--gpu-contact-relaxation", String(gpuContactRelaxation));
  }
  if (gpuJointTopologyDiagnostic) {
    benchmarkArgs.push("--gpu-joint-topology-diagnostic");
    if (gpuJointTopologyRepeats > 1) {
      benchmarkArgs.push("--gpu-joint-topology-repeats", String(gpuJointTopologyRepeats));
    }
    if (gpuJointProposalIterations > 1) {
      benchmarkArgs.push("--gpu-joint-proposal-iterations", String(gpuJointProposalIterations));
    }
  }
  if (gpuApplyJointProposals) {
    benchmarkArgs.push("--gpu-apply-joint-proposals");
  }
  if (gpuJointPerturb !== 0) {
    benchmarkArgs.push("--gpu-joint-perturb", String(gpuJointPerturb));
  }
  const result = spawnSync(exe, benchmarkArgs, {
    cwd: path.dirname(exe),
    encoding: "utf8",
  });

  if (result.status !== 0) {
    throw new Error(`Headless benchmark failed for ${scene}: ${result.stderr || result.stdout}`);
  }

  const line = result.stdout.trim().split(/\r?\n/).find((entry) => entry.includes("\"headlessBenchmark\""));
  if (!line) {
    throw new Error(`No benchmark JSON emitted for ${scene}: ${result.stdout}`);
  }
  return JSON.parse(line);
}

function connectAndMeasure(scene) {
  return new Promise((resolve, reject) => {
    if (typeof WebSocket !== "function") {
      reject(new Error("This Node.js runtime does not provide a global WebSocket implementation."));
      return;
    }

    const serverArgs = [
      "--scene",
      scene,
      "--port",
      String(port),
      "--tick-rate",
      "120",
    ];
    if (physicsBackend !== "cpu") {
      serverArgs.push("--physics-backend", physicsBackend);
    }
    if (iterations > 0) {
      serverArgs.push("--iterations", String(iterations));
    }
    if (gravity !== undefined) {
      serverArgs.push("--gravity", String(gravity));
    }
    if (gpuSphereContacts) {
      serverArgs.push("--gpu-sphere-contacts");
    }
    if (gpuGroundContactFeed) {
      serverArgs.push("--gpu-ground-contact-feed");
    }
    if (gpuContactSolveDiagnostic) {
      serverArgs.push("--gpu-contact-solve-diagnostic");
    }
    if (gpuContactSolveNoReadback) {
      serverArgs.push("--gpu-contact-solve-no-readback");
    }
    if (gpuDirectCounterReadback) {
      serverArgs.push("--gpu-direct-counter-readback");
    }
    if (gpuRuntimeValidate) {
      serverArgs.push("--gpu-runtime-validate");
    }
    if (gpuApplyPrediction) {
      serverArgs.push("--gpu-apply-prediction");
    }
    if (gpuApplyVelocity) {
      serverArgs.push("--gpu-apply-velocity");
    }
    if (gpuResidentCounterlessContacts) {
      serverArgs.push("--gpu-resident-counterless-contacts");
    }
    if (gpuDeferFinalPositionReadback) {
      serverArgs.push("--gpu-defer-final-position-readback");
    }
    if (gpuAsyncFinalPositionReadback) {
      serverArgs.push("--gpu-async-final-position-readback");
    }
    if (gpuApplyContactPositions) {
      serverArgs.push("--gpu-apply-contact-positions");
    }
    if (gpuGroundContacts) {
      serverArgs.push("--gpu-ground-contacts");
    }
    if (gpuRuntimeJointProposals) {
      serverArgs.push("--gpu-runtime-joint-proposals");
    }
    if (gpuReplaceCpuJoints) {
      serverArgs.push("--gpu-replace-cpu-joints");
    }
    if (effectiveRuntimeJointProposals && gpuJointProposalIterations > 1) {
      serverArgs.push("--gpu-joint-proposal-iterations", String(gpuJointProposalIterations));
    }
    if (gpuContactIterations !== 4) {
      serverArgs.push("--gpu-contact-iterations", String(gpuContactIterations));
    }
    if (gpuContactRelaxation !== 0.10) {
      serverArgs.push("--gpu-contact-relaxation", String(gpuContactRelaxation));
    }
    if (gpuJointTopologyDiagnostic) {
      serverArgs.push("--gpu-joint-topology-diagnostic");
      if (gpuJointTopologyRepeats > 1) {
        serverArgs.push("--gpu-joint-topology-repeats", String(gpuJointTopologyRepeats));
      }
      if (gpuJointProposalIterations > 1) {
        serverArgs.push("--gpu-joint-proposal-iterations", String(gpuJointProposalIterations));
      }
    }
    if (gpuApplyJointProposals) {
      serverArgs.push("--gpu-apply-joint-proposals");
    }
    if (gpuJointPerturb !== 0) {
      serverArgs.push("--gpu-joint-perturb", String(gpuJointPerturb));
    }
    const server = spawn(exe, serverArgs, {
      cwd: path.dirname(exe),
      stdio: ["ignore", "pipe", "pipe"],
      windowsHide: true,
    });

    let stderr = "";
    server.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });

    let socket;
    let framesSeen = 0;
    let totalBytes = 0;
    let parseTotalMs = 0;
    let firstAt = 0;
    let lastAt = 0;
    let done = false;

    function stopServer() {
      if (!server.killed) {
        server.kill();
      }
    }

    const timeout = setTimeout(() => {
      if (done) {
        return;
      }
      done = true;
      socket?.close();
      stopServer();
      reject(new Error(`Timed out waiting for streaming benchmark for ${scene}. ${stderr}`));
    }, streamTimeoutMs);

    function finish() {
      if (done) {
        return;
      }
      done = true;
      clearTimeout(timeout);
      socket?.close();
      stopServer();
      const durationMs = Math.max(1, lastAt - firstAt);
      resolve({
        scene,
        frames: framesSeen,
        durationMs,
        snapshotsPerSecond: framesSeen * 1000 / durationMs,
        avgSnapshotBytes: framesSeen ? totalBytes / framesSeen : 0,
        avgParseMs: framesSeen ? parseTotalMs / framesSeen : 0,
        snapshotMode,
      });
    }

    function tryConnect() {
      if (done) {
        return;
      }
      socket = new WebSocket(`ws://127.0.0.1:${port}`);
      socket.addEventListener("open", () => {
        if (snapshotMode === "binary") {
          socket.send(JSON.stringify({ type: "command", command: "snapshotMode", value: "binary" }));
        }
      });
      socket.addEventListener("error", () => {
        if (!done && framesSeen === 0) {
          setTimeout(tryConnect, 250);
        }
      });
      socket.addEventListener("message", async (event) => {
        if (done) {
          return;
        }
        const parseBegin = performance.now();
        let bytes = 0;
        if (snapshotMode === "binary" && typeof event.data !== "string") {
          const buffer = await eventDataToArrayBuffer(event.data);
          if (isBinarySnapshotBuffer(buffer)) {
            decodeBinarySnapshotHeader(buffer);
            bytes = buffer.byteLength;
          } else {
            const text = Buffer.from(buffer).toString("utf8");
            if (!text.includes("\"type\":\"snapshot\"")) {
              return;
            }
            JSON.parse(text);
            bytes = buffer.byteLength;
          }
        } else {
          const text = await eventDataToText(event.data);
          if (!text.includes("\"type\":\"snapshot\"")) {
            return;
          }
          JSON.parse(text);
          bytes = typeof event.data === "string" ? text.length : (await eventDataToArrayBuffer(event.data)).byteLength;
        }
        const now = performance.now();
        if (framesSeen === 0) {
          firstAt = now;
        }
        lastAt = now;
        parseTotalMs += performance.now() - parseBegin;
        totalBytes += bytes;
        framesSeen += 1;
        if (framesSeen >= streamFrames) {
          finish();
        }
      });
    }

    setTimeout(tryConnect, 500);
  });
}

const output = [];
for (const scene of scenes) {
  const headless = runHeadlessBenchmark(scene);
  if (assertPhysicsAvgUnder > 0 && headless.physicsAvgMs >= assertPhysicsAvgUnder) {
    throw new Error(`${scene} physicsAvgMs ${headless.physicsAvgMs} exceeded ${assertPhysicsAvgUnder}`);
  }
  if (assertNoWebgpuFallbacks && headless.webgpuRuntimeFallbacks !== 0) {
    throw new Error(`${scene} used ${headless.webgpuRuntimeFallbacks} WebGPU runtime fallback(s)`);
  }
  if (assertNoWebgpuValidation) {
    const statusFields = [
      "webgpuStatus",
      "webgpuRuntimeStatus",
      "webgpuSapStatus",
      "webgpuJointTopologyStatus",
    ];
    for (const field of statusFields) {
      if (hasWebGpuValidationProblem(headless[field])) {
        throw new Error(`${scene} reported unhealthy ${field}: ${headless[field]}`);
      }
    }
  }
  if (assertGroundCandidatesMin > 0 && (headless.webgpuSphereGroundCandidates ?? 0) < assertGroundCandidatesMin) {
    throw new Error(`${scene} sphere-ground candidates ${headless.webgpuSphereGroundCandidates ?? 0} below ${assertGroundCandidatesMin}`);
  }
  if (assertDirectSphereCylinderCandidatesMin > 0 && (headless.webgpuDirectSphereCylinderCandidates ?? 0) < assertDirectSphereCylinderCandidatesMin) {
    throw new Error(`${scene} direct sphere-cylinder candidates ${headless.webgpuDirectSphereCylinderCandidates ?? 0} below ${assertDirectSphereCylinderCandidatesMin}`);
  }
  if (assertDirectSphereCylinderCandidatesMaxMin > 0 && (headless.webgpuDirectSphereCylinderCandidatesMax ?? 0) < assertDirectSphereCylinderCandidatesMaxMin) {
    throw new Error(`${scene} max direct sphere-cylinder candidates ${headless.webgpuDirectSphereCylinderCandidatesMax ?? 0} below ${assertDirectSphereCylinderCandidatesMaxMin}`);
  }
  if (assertDirectSphereCapsuleCandidatesMin > 0 && (headless.webgpuDirectSphereCapsuleCandidates ?? 0) < assertDirectSphereCapsuleCandidatesMin) {
    throw new Error(`${scene} direct sphere-capsule candidates ${headless.webgpuDirectSphereCapsuleCandidates ?? 0} below ${assertDirectSphereCapsuleCandidatesMin}`);
  }
  if (assertDirectSphereCapsuleCandidatesMaxMin > 0 && (headless.webgpuDirectSphereCapsuleCandidatesMax ?? 0) < assertDirectSphereCapsuleCandidatesMaxMin) {
    throw new Error(`${scene} max direct sphere-capsule candidates ${headless.webgpuDirectSphereCapsuleCandidatesMax ?? 0} below ${assertDirectSphereCapsuleCandidatesMaxMin}`);
  }
  if (assertDirectSphereBoxCandidatesMin > 0 && (headless.webgpuDirectSphereBoxCandidates ?? 0) < assertDirectSphereBoxCandidatesMin) {
    throw new Error(`${scene} direct sphere-box candidates ${headless.webgpuDirectSphereBoxCandidates ?? 0} below ${assertDirectSphereBoxCandidatesMin}`);
  }
  if (assertDirectSphereBoxCandidatesMaxMin > 0 && (headless.webgpuDirectSphereBoxCandidatesMax ?? 0) < assertDirectSphereBoxCandidatesMaxMin) {
    throw new Error(`${scene} max direct sphere-box candidates ${headless.webgpuDirectSphereBoxCandidatesMax ?? 0} below ${assertDirectSphereBoxCandidatesMaxMin}`);
  }
  if (assertDirectBoxPairCandidatesMin > 0 && (headless.webgpuDirectBoxPairCandidates ?? 0) < assertDirectBoxPairCandidatesMin) {
    throw new Error(`${scene} direct box-box candidates ${headless.webgpuDirectBoxPairCandidates ?? 0} below ${assertDirectBoxPairCandidatesMin}`);
  }
  if (assertDirectBoxPairCandidatesMaxMin > 0 && (headless.webgpuDirectBoxPairCandidatesMax ?? 0) < assertDirectBoxPairCandidatesMaxMin) {
    throw new Error(`${scene} max direct box-box candidates ${headless.webgpuDirectBoxPairCandidatesMax ?? 0} below ${assertDirectBoxPairCandidatesMaxMin}`);
  }
  if (assertDirectRoundPairCandidatesMin > 0 && (headless.webgpuDirectRoundPairCandidates ?? 0) < assertDirectRoundPairCandidatesMin) {
    throw new Error(`${scene} direct round-pair candidates ${headless.webgpuDirectRoundPairCandidates ?? 0} below ${assertDirectRoundPairCandidatesMin}`);
  }
  if (assertDirectRoundPairCandidatesMaxMin > 0 && (headless.webgpuDirectRoundPairCandidatesMax ?? 0) < assertDirectRoundPairCandidatesMaxMin) {
    throw new Error(`${scene} max direct round-pair candidates ${headless.webgpuDirectRoundPairCandidatesMax ?? 0} below ${assertDirectRoundPairCandidatesMaxMin}`);
  }
  if (assertDirectGpuContactRecordsMin > 0 && (headless.webgpuDirectGpuContactRecords ?? 0) < assertDirectGpuContactRecordsMin) {
    throw new Error(`${scene} GPU direct contact records ${headless.webgpuDirectGpuContactRecords ?? 0} below ${assertDirectGpuContactRecordsMin}`);
  }
  if (assertDirectGpuContactRecordsMaxMin > 0 && (headless.webgpuDirectGpuContactRecordsMax ?? 0) < assertDirectGpuContactRecordsMaxMin) {
    throw new Error(`${scene} max GPU direct contact records ${headless.webgpuDirectGpuContactRecordsMax ?? 0} below ${assertDirectGpuContactRecordsMaxMin}`);
  }
  if (assertDirectGpuRoundPairCandidatesMin > 0 && (headless.webgpuDirectGpuRoundPairCandidates ?? 0) < assertDirectGpuRoundPairCandidatesMin) {
    throw new Error(`${scene} GPU direct round-pair candidates ${headless.webgpuDirectGpuRoundPairCandidates ?? 0} below ${assertDirectGpuRoundPairCandidatesMin}`);
  }
  if (assertDirectGpuRoundPairCandidatesMaxMin > 0 && (headless.webgpuDirectGpuRoundPairCandidatesMax ?? 0) < assertDirectGpuRoundPairCandidatesMaxMin) {
    throw new Error(`${scene} max GPU direct round-pair candidates ${headless.webgpuDirectGpuRoundPairCandidatesMax ?? 0} below ${assertDirectGpuRoundPairCandidatesMaxMin}`);
  }
  if (assertDirectGpuBoxPairCandidatesMin > 0 && (headless.webgpuDirectGpuBoxPairCandidates ?? 0) < assertDirectGpuBoxPairCandidatesMin) {
    throw new Error(`${scene} GPU direct box-box candidates ${headless.webgpuDirectGpuBoxPairCandidates ?? 0} below ${assertDirectGpuBoxPairCandidatesMin}`);
  }
  if (assertDirectGpuBoxPairCandidatesMaxMin > 0 && (headless.webgpuDirectGpuBoxPairCandidatesMax ?? 0) < assertDirectGpuBoxPairCandidatesMaxMin) {
    throw new Error(`${scene} max GPU direct box-box candidates ${headless.webgpuDirectGpuBoxPairCandidatesMax ?? 0} below ${assertDirectGpuBoxPairCandidatesMaxMin}`);
  }
  if (assertDirectGpuCounterReadbackBytesMax >= 0 && (headless.webgpuDirectGpuCounterReadbackBytes ?? 0) > assertDirectGpuCounterReadbackBytesMax) {
    throw new Error(`${scene} direct GPU counter readback bytes ${headless.webgpuDirectGpuCounterReadbackBytes ?? 0} exceeded ${assertDirectGpuCounterReadbackBytesMax}`);
  }
  if (assertDirectGpuCounterReadbackBytesMin > 0 && (headless.webgpuDirectGpuCounterReadbackBytes ?? 0) < assertDirectGpuCounterReadbackBytesMin) {
    throw new Error(`${scene} direct GPU counter readback bytes ${headless.webgpuDirectGpuCounterReadbackBytes ?? 0} below ${assertDirectGpuCounterReadbackBytesMin}`);
  }
  if (assertPredictionAppliedBodiesMin > 0 && (headless.webgpuPredictionAppliedBodies ?? 0) < assertPredictionAppliedBodiesMin) {
    throw new Error(`${scene} GPU prediction applied bodies ${headless.webgpuPredictionAppliedBodies ?? 0} below ${assertPredictionAppliedBodiesMin}`);
  }
  if (assertPredictionAppliedReadbackBytesMin > 0 && (headless.webgpuPredictionAppliedReadbackBytes ?? 0) < assertPredictionAppliedReadbackBytesMin) {
    throw new Error(`${scene} GPU prediction applied readback bytes ${headless.webgpuPredictionAppliedReadbackBytes ?? 0} below ${assertPredictionAppliedReadbackBytesMin}`);
  }
  if (assertVelocityAppliedBodiesMin > 0 && (headless.webgpuVelocityAppliedBodies ?? 0) < assertVelocityAppliedBodiesMin) {
    throw new Error(`${scene} GPU velocity applied bodies ${headless.webgpuVelocityAppliedBodies ?? 0} below ${assertVelocityAppliedBodiesMin}`);
  }
  if (assertVelocityAppliedReadbackBytesMin > 0 && (headless.webgpuVelocityAppliedReadbackBytes ?? 0) < assertVelocityAppliedReadbackBytesMin) {
    throw new Error(`${scene} GPU velocity applied readback bytes ${headless.webgpuVelocityAppliedReadbackBytes ?? 0} below ${assertVelocityAppliedReadbackBytesMin}`);
  }
  if (assertExternalContactsMin > 0 && (headless.webgpuExternalContacts ?? 0) < assertExternalContactsMin) {
    throw new Error(`${scene} external contacts ${headless.webgpuExternalContacts ?? 0} below ${assertExternalContactsMin}`);
  }
  if (assertExternalGroundContactsMin > 0 && (headless.webgpuExternalGroundContacts ?? 0) < assertExternalGroundContactsMin) {
    throw new Error(`${scene} external ground contacts ${headless.webgpuExternalGroundContacts ?? 0} below ${assertExternalGroundContactsMin}`);
  }
  if (assertContactProposalActiveBodiesMin > 0 && (headless.webgpuSphereContactProposalActiveBodies ?? 0) < assertContactProposalActiveBodiesMin) {
    throw new Error(`${scene} contact proposal active bodies ${headless.webgpuSphereContactProposalActiveBodies ?? 0} below ${assertContactProposalActiveBodiesMin}`);
  }
  if (assertContactFinalPositionReady && (headless.webgpuSphereContactFinalPositionReady ?? 0) < 1) {
    throw new Error(`${scene} contact final position buffer was not ready`);
  }
  if (assertFinalPositionReadbackDeferred && (headless.webgpuSphereContactFinalPositionReadbackDeferred ?? 0) < 1) {
    throw new Error(`${scene} contact final position readback was not deferred`);
  }
  if (assertFinalPositionAsyncReadbackConsumed && (headless.webgpuSphereContactFinalPositionAsyncReadbackConsumed ?? 0) < 1) {
    throw new Error(`${scene} contact final position async readback was not consumed`);
  }
  if (assertFinalPositionAsyncReadbackDroppedMax >= 0 &&
      (headless.webgpuSphereContactFinalPositionAsyncReadbackDropped ?? 0) > assertFinalPositionAsyncReadbackDroppedMax) {
    throw new Error(`${scene} contact final position async readback drops ${headless.webgpuSphereContactFinalPositionAsyncReadbackDropped ?? 0} exceeded ${assertFinalPositionAsyncReadbackDroppedMax}`);
  }
  if (assertFinalPositionAsyncReadbackWaitMax >= 0 &&
      (headless.webgpuSphereContactFinalPositionAsyncReadbackWaitMs ?? 0) > assertFinalPositionAsyncReadbackWaitMax) {
    throw new Error(`${scene} contact final position async readback wait ${headless.webgpuSphereContactFinalPositionAsyncReadbackWaitMs ?? 0} ms exceeded ${assertFinalPositionAsyncReadbackWaitMax}`);
  }
  if (assertContactProposalResidualAfterMax >= 0 && (headless.webgpuSphereContactProposalResidualAfterMax ?? 0) > assertContactProposalResidualAfterMax) {
    throw new Error(`${scene} contact proposal residual after max ${headless.webgpuSphereContactProposalResidualAfterMax ?? 0} exceeded ${assertContactProposalResidualAfterMax}`);
  }
  if (assertContactIterationResidualAfterMax >= 0 && (headless.webgpuSphereContactIterationResidualAfterMax ?? 0) > assertContactIterationResidualAfterMax) {
    throw new Error(`${scene} contact iteration residual after max ${headless.webgpuSphereContactIterationResidualAfterMax ?? 0} exceeded ${assertContactIterationResidualAfterMax}`);
  }
  if (assertContactAdjacencyReadbackBytesMax >= 0 && (headless.webgpuSphereContactAdjacencyReadbackBytes ?? 0) > assertContactAdjacencyReadbackBytesMax) {
    throw new Error(`${scene} GPU contact adjacency readback bytes ${headless.webgpuSphereContactAdjacencyReadbackBytes ?? 0} exceeded ${assertContactAdjacencyReadbackBytesMax}`);
  }
  if (assertContactGatherReadbackBytesMax >= 0 && (headless.webgpuSphereContactGatherReadbackBytes ?? 0) > assertContactGatherReadbackBytesMax) {
    throw new Error(`${scene} GPU contact gather readback bytes ${headless.webgpuSphereContactGatherReadbackBytes ?? 0} exceeded ${assertContactGatherReadbackBytesMax}`);
  }
  if (assertContactProposalResidualReadbackBytesMax >= 0 && (headless.webgpuSphereContactProposalResidualReadbackBytes ?? 0) > assertContactProposalResidualReadbackBytesMax) {
    throw new Error(`${scene} GPU contact residual readback bytes ${headless.webgpuSphereContactProposalResidualReadbackBytes ?? 0} exceeded ${assertContactProposalResidualReadbackBytesMax}`);
  }
  if (assertSapCandidatesMin > 0 && (headless.webgpuSapCandidates ?? 0) < assertSapCandidatesMin) {
    throw new Error(`${scene} SAP candidates ${headless.webgpuSapCandidates ?? 0} below ${assertSapCandidatesMin}`);
  }
  const sphereHits = Number(headless.sphereHits ?? headless.webgpuSapSphereHits ?? 0);
  if (assertSphereHitsMin > 0 && sphereHits < assertSphereHitsMin) {
    throw new Error(`${scene} sphere hits ${sphereHits} below ${assertSphereHitsMin}`);
  }
  if (assertPairChecksMin > 0 && (headless.pairChecks ?? 0) < assertPairChecksMin) {
    throw new Error(`${scene} pair checks ${headless.pairChecks ?? 0} below ${assertPairChecksMin}`);
  }
  if (assertManifoldCountMax >= 0 && (headless.manifoldCount ?? 0) > assertManifoldCountMax) {
    throw new Error(`${scene} manifold count ${headless.manifoldCount ?? 0} exceeded ${assertManifoldCountMax}`);
  }
  if (assertFinalMaxSpeedMin >= 0 && (headless.finalMaxSpeed ?? 0) < assertFinalMaxSpeedMin) {
    throw new Error(`${scene} final max speed ${headless.finalMaxSpeed ?? 0} below ${assertFinalMaxSpeedMin}`);
  }
  if (assertFinalMaxSpeedMax >= 0 && (headless.finalMaxSpeed ?? 0) > assertFinalMaxSpeedMax) {
    throw new Error(`${scene} final max speed ${headless.finalMaxSpeed ?? 0} exceeded ${assertFinalMaxSpeedMax}`);
  }
  if (assertFinalDynamicMaxHeightMin >= 0 && (headless.finalDynamicMaxHeight ?? 0) < assertFinalDynamicMaxHeightMin) {
    throw new Error(`${scene} final dynamic max height ${headless.finalDynamicMaxHeight ?? 0} below ${assertFinalDynamicMaxHeightMin}`);
  }
  if (assertFinalDynamicMaxHeightMax >= 0 && (headless.finalDynamicMaxHeight ?? 0) > assertFinalDynamicMaxHeightMax) {
    throw new Error(`${scene} final dynamic max height ${headless.finalDynamicMaxHeight ?? 0} exceeded ${assertFinalDynamicMaxHeightMax}`);
  }
  if (assertPrimalAvgMax >= 0 && (headless.primalAvgMs ?? 0) > assertPrimalAvgMax) {
    throw new Error(`${scene} primal avg ${headless.primalAvgMs ?? 0} exceeded ${assertPrimalAvgMax}`);
  }
  if (assertDualAvgMax >= 0 && (headless.dualAvgMs ?? 0) > assertDualAvgMax) {
    throw new Error(`${scene} dual avg ${headless.dualAvgMs ?? 0} exceeded ${assertDualAvgMax}`);
  }
  if (assertDirectJointSolve && headless.webgpuDirectJointSolve !== true) {
    throw new Error(`${scene} did not report direct joint solve`);
  }
  if (assertDirectContactSolve && headless.webgpuDirectContactSolve !== true) {
    throw new Error(`${scene} did not report direct contact solve`);
  }
  if (assertAppliedPositionBodiesMin > 0 && (headless.webgpuSphereContactAppliedPositionBodies ?? 0) < assertAppliedPositionBodiesMin) {
    throw new Error(`${scene} applied position bodies ${headless.webgpuSphereContactAppliedPositionBodies ?? 0} below ${assertAppliedPositionBodiesMin}`);
  }
  if (assertDirectSphereAppliedBodiesMin > 0 && (headless.webgpuDirectSphereContactAppliedPositionBodies ?? 0) < assertDirectSphereAppliedBodiesMin) {
    throw new Error(`${scene} direct sphere applied position bodies ${headless.webgpuDirectSphereContactAppliedPositionBodies ?? 0} below ${assertDirectSphereAppliedBodiesMin}`);
  }
  if (assertDirectSphereAppliedBodiesMaxMin > 0 && (headless.webgpuDirectSphereContactAppliedPositionBodiesMax ?? 0) < assertDirectSphereAppliedBodiesMaxMin) {
    throw new Error(`${scene} max direct sphere applied position bodies ${headless.webgpuDirectSphereContactAppliedPositionBodiesMax ?? 0} below ${assertDirectSphereAppliedBodiesMaxMin}`);
  }
  const directRoundAppliedBodies = headless.webgpuDirectRoundContactAppliedPositionBodies ?? headless.webgpuDirectSphereContactAppliedPositionBodies ?? 0;
  const directRoundAppliedBodiesMax = headless.webgpuDirectRoundContactAppliedPositionBodiesMax ?? headless.webgpuDirectSphereContactAppliedPositionBodiesMax ?? 0;
  if (assertDirectRoundAppliedBodiesMin > 0 && directRoundAppliedBodies < assertDirectRoundAppliedBodiesMin) {
    throw new Error(`${scene} direct round-body applied position bodies ${directRoundAppliedBodies} below ${assertDirectRoundAppliedBodiesMin}`);
  }
  if (assertDirectRoundAppliedBodiesMaxMin > 0 && directRoundAppliedBodiesMax < assertDirectRoundAppliedBodiesMaxMin) {
    throw new Error(`${scene} max direct round-body applied position bodies ${directRoundAppliedBodiesMax} below ${assertDirectRoundAppliedBodiesMaxMin}`);
  }
  if (assertDirectGroundAppliedBodiesMin > 0 && (headless.webgpuDirectGroundAppliedPositionBodies ?? 0) < assertDirectGroundAppliedBodiesMin) {
    throw new Error(`${scene} direct ground applied position bodies ${headless.webgpuDirectGroundAppliedPositionBodies ?? 0} below ${assertDirectGroundAppliedBodiesMin}`);
  }
  if (assertDirectGroundAppliedBodiesMaxMin > 0 && (headless.webgpuDirectGroundAppliedPositionBodiesMax ?? 0) < assertDirectGroundAppliedBodiesMaxMin) {
    throw new Error(`${scene} max direct ground applied position bodies ${headless.webgpuDirectGroundAppliedPositionBodiesMax ?? 0} below ${assertDirectGroundAppliedBodiesMaxMin}`);
  }
  if (assertBelowGroundSpheresMax >= 0 && (headless.finalBelowGroundSpheres ?? 0) > assertBelowGroundSpheresMax) {
    throw new Error(`${scene} below-ground spheres ${headless.finalBelowGroundSpheres ?? 0} exceeded ${assertBelowGroundSpheresMax}`);
  }
  if (assertMaxGroundPenetration >= 0 && (headless.finalMaxSphereGroundPenetration ?? 0) > assertMaxGroundPenetration) {
    throw new Error(`${scene} max sphere-ground penetration ${headless.finalMaxSphereGroundPenetration ?? 0} exceeded ${assertMaxGroundPenetration}`);
  }
  if (assertBelowGroundBodiesMax >= 0 && (headless.finalBelowGroundBodies ?? 0) > assertBelowGroundBodiesMax) {
    throw new Error(`${scene} below-ground bodies ${headless.finalBelowGroundBodies ?? 0} exceeded ${assertBelowGroundBodiesMax}`);
  }
  if (assertMaxGroundBodyPenetration >= 0 && (headless.finalMaxGroundBodyPenetration ?? 0) > assertMaxGroundBodyPenetration) {
    throw new Error(`${scene} max ground-body penetration ${headless.finalMaxGroundBodyPenetration ?? 0} exceeded ${assertMaxGroundBodyPenetration}`);
  }
  if (assertMaxLinearError >= 0 && (headless.webgpuRuntimeMaxLinearError ?? 0) > assertMaxLinearError) {
    throw new Error(`${scene} WebGPU max linear error ${headless.webgpuRuntimeMaxLinearError ?? 0} exceeded ${assertMaxLinearError}`);
  }
  if (assertMaxAngularError >= 0 && (headless.webgpuRuntimeMaxAngularError ?? 0) > assertMaxAngularError) {
    throw new Error(`${scene} WebGPU max angular error ${headless.webgpuRuntimeMaxAngularError ?? 0} exceeded ${assertMaxAngularError}`);
  }
  if (assertSapCounterReadbackBytesMax >= 0 && (headless.webgpuSapCounterReadbackBytes ?? 0) > assertSapCounterReadbackBytesMax) {
    throw new Error(`${scene} SAP counter readback bytes ${headless.webgpuSapCounterReadbackBytes ?? 0} exceeded ${assertSapCounterReadbackBytesMax}`);
  }
  if (assertSapPairReadbackBytesMax >= 0 && (headless.webgpuSapPairReadbackBytes ?? 0) > assertSapPairReadbackBytesMax) {
    throw new Error(`${scene} SAP pair readback bytes ${headless.webgpuSapPairReadbackBytes ?? 0} exceeded ${assertSapPairReadbackBytesMax}`);
  }
  if (assertJointTopologyMismatchesMax >= 0 && (headless.webgpuJointTopologyMismatches ?? 0) > assertJointTopologyMismatchesMax) {
    throw new Error(`${scene} joint topology mismatches ${headless.webgpuJointTopologyMismatches ?? 0} exceeded ${assertJointTopologyMismatchesMax}`);
  }
  if (assertJointTopologyJointsMin > 0 && (headless.webgpuJointTopologyJoints ?? 0) < assertJointTopologyJointsMin) {
    throw new Error(`${scene} joint topology joints ${headless.webgpuJointTopologyJoints ?? 0} below ${assertJointTopologyJointsMin}`);
  }
  if (assertJointTopologyLastMsUnder >= 0 && (headless.webgpuJointTopologyMs ?? 0) > assertJointTopologyLastMsUnder) {
    throw new Error(`${scene} joint topology last ms ${headless.webgpuJointTopologyMs ?? 0} exceeded ${assertJointTopologyLastMsUnder}`);
  }
  if (assertJointColorConflictsMax >= 0 && (headless.webgpuJointColorConflicts ?? 0) > assertJointColorConflictsMax) {
    throw new Error(`${scene} joint color conflicts ${headless.webgpuJointColorConflicts ?? 0} exceeded ${assertJointColorConflictsMax}`);
  }
  if (assertJointColorCountMax >= 0 && (headless.webgpuJointColorCount ?? 0) > assertJointColorCountMax) {
    throw new Error(`${scene} joint color count ${headless.webgpuJointColorCount ?? 0} exceeded ${assertJointColorCountMax}`);
  }
  if (assertJointResidualMax >= 0 && (headless.webgpuJointResidualMax ?? 0) > assertJointResidualMax) {
    throw new Error(`${scene} joint residual max ${headless.webgpuJointResidualMax ?? 0} exceeded ${assertJointResidualMax}`);
  }
  if (assertJointResidualMin >= 0 && (headless.webgpuJointResidualMax ?? 0) < assertJointResidualMin) {
    throw new Error(`${scene} joint residual max ${headless.webgpuJointResidualMax ?? 0} below ${assertJointResidualMin}`);
  }
  if (assertJointProposalMaxCorrection >= 0 && (headless.webgpuJointProposalMaxCorrection ?? 0) > assertJointProposalMaxCorrection) {
    throw new Error(`${scene} joint proposal max correction ${headless.webgpuJointProposalMaxCorrection ?? 0} exceeded ${assertJointProposalMaxCorrection}`);
  }
  if (assertJointProposalActiveBodiesMin > 0 && (headless.webgpuJointProposalActiveBodies ?? 0) < assertJointProposalActiveBodiesMin) {
    throw new Error(`${scene} joint proposal active bodies ${headless.webgpuJointProposalActiveBodies ?? 0} below ${assertJointProposalActiveBodiesMin}`);
  }
  if (assertJointProposalResidualAfterMax >= 0 && (headless.webgpuJointProposalResidualAfterMax ?? 0) > assertJointProposalResidualAfterMax) {
    throw new Error(`${scene} joint proposal residual after max ${headless.webgpuJointProposalResidualAfterMax ?? 0} exceeded ${assertJointProposalResidualAfterMax}`);
  }
  if (assertJointProposalResidualRmsRatioMax >= 0) {
    const before = Number(headless.webgpuJointResidualRms ?? 0);
    const after = Number(headless.webgpuJointProposalResidualAfterRms ?? 0);
    if (before <= 1e-9) {
      throw new Error(`${scene} joint residual RMS ${before} too small for ratio assertion`);
    }
    const ratio = after / before;
    if (ratio > assertJointProposalResidualRmsRatioMax) {
      throw new Error(`${scene} joint proposal residual RMS ratio ${ratio} exceeded ${assertJointProposalResidualRmsRatioMax}`);
    }
  }
  if (assertJointProposalAppliedBodiesMin > 0 && (headless.webgpuJointProposalAppliedPositionBodies ?? 0) < assertJointProposalAppliedBodiesMin) {
    throw new Error(`${scene} joint proposal applied bodies ${headless.webgpuJointProposalAppliedPositionBodies ?? 0} below ${assertJointProposalAppliedBodiesMin}`);
  }
  if (assertJointProposalAppliedMaxDeltaMin >= 0 && (headless.webgpuJointProposalAppliedPositionMaxDelta ?? 0) < assertJointProposalAppliedMaxDeltaMin) {
    throw new Error(`${scene} joint proposal applied max delta ${headless.webgpuJointProposalAppliedPositionMaxDelta ?? 0} below ${assertJointProposalAppliedMaxDeltaMin}`);
  }
  if (assertJointFinalPositionAsyncReadbackConsumed && (headless.webgpuJointProposalFinalPositionAsyncReadbackConsumed ?? 0) < 1) {
    throw new Error(`${scene} joint final position async readback was not consumed`);
  }
  if (assertJointFinalPositionAsyncReadbackDroppedMax >= 0 &&
      (headless.webgpuJointProposalFinalPositionAsyncReadbackDropped ?? 0) > assertJointFinalPositionAsyncReadbackDroppedMax) {
    throw new Error(`${scene} joint final position async readback drops ${headless.webgpuJointProposalFinalPositionAsyncReadbackDropped ?? 0} exceeded ${assertJointFinalPositionAsyncReadbackDroppedMax}`);
  }
  if (assertJointFinalPositionAsyncReadbackWaitMax >= 0 &&
      (headless.webgpuJointProposalFinalPositionAsyncReadbackWaitMs ?? 0) > assertJointFinalPositionAsyncReadbackWaitMax) {
    throw new Error(`${scene} joint final position async readback wait ${headless.webgpuJointProposalFinalPositionAsyncReadbackWaitMs ?? 0} ms exceeded ${assertJointFinalPositionAsyncReadbackWaitMax}`);
  }
  if (assertPrimalJointSkippedMin > 0 && (headless.primalJointSkipped ?? 0) < assertPrimalJointSkippedMin) {
    throw new Error(`${scene} primal joint skipped ${headless.primalJointSkipped ?? 0} below ${assertPrimalJointSkippedMin}`);
  }
  if (assertDualJointSkippedMin > 0 && (headless.dualJointSkipped ?? 0) < assertDualJointSkippedMin) {
    throw new Error(`${scene} dual joint skipped ${headless.dualJointSkipped ?? 0} below ${assertDualJointSkippedMin}`);
  }
  if (assertPrimalIgnoreSkippedMin > 0 && (headless.primalIgnoreSkipped ?? 0) < assertPrimalIgnoreSkippedMin) {
    throw new Error(`${scene} primal ignore skipped ${headless.primalIgnoreSkipped ?? 0} below ${assertPrimalIgnoreSkippedMin}`);
  }
  if (assertDualIgnoreSkippedMin > 0 && (headless.dualIgnoreSkipped ?? 0) < assertDualIgnoreSkippedMin) {
    throw new Error(`${scene} dual ignore skipped ${headless.dualIgnoreSkipped ?? 0} below ${assertDualIgnoreSkippedMin}`);
  }
  if (assertJointInitializationSkippedMin > 0 && (headless.jointInitializationSkipped ?? 0) < assertJointInitializationSkippedMin) {
    throw new Error(`${scene} joint initialization skipped ${headless.jointInitializationSkipped ?? 0} below ${assertJointInitializationSkippedMin}`);
  }
  if (assertIgnoreInitializationSkippedMin > 0 && (headless.ignoreInitializationSkipped ?? 0) < assertIgnoreInitializationSkippedMin) {
    throw new Error(`${scene} ignore initialization skipped ${headless.ignoreInitializationSkipped ?? 0} below ${assertIgnoreInitializationSkippedMin}`);
  }
  let streaming = null;
  if (streamFrames > 0 && !collisionOnly) {
    try {
      streaming = await connectAndMeasure(scene);
      if (assertStreamSnapshotsPerSecondMin > 0 &&
          (streaming.snapshotsPerSecond ?? 0) < assertStreamSnapshotsPerSecondMin) {
        throw new Error(`${scene} streaming snapshots/sec ${streaming.snapshotsPerSecond ?? 0} below ${assertStreamSnapshotsPerSecondMin}`);
      }
      if (assertStreamAvgSnapshotBytesMax >= 0 &&
          (streaming.avgSnapshotBytes ?? 0) > assertStreamAvgSnapshotBytesMax) {
        throw new Error(`${scene} streaming avg snapshot bytes ${streaming.avgSnapshotBytes ?? 0} exceeded ${assertStreamAvgSnapshotBytesMax}`);
      }
      if (assertStreamAvgParseMsMax >= 0 &&
          (streaming.avgParseMs ?? 0) > assertStreamAvgParseMsMax) {
        throw new Error(`${scene} streaming avg parse ms ${streaming.avgParseMs ?? 0} exceeded ${assertStreamAvgParseMsMax}`);
      }
    } catch (error) {
      throw error;
    }
  } else {
    streaming = { scene, skipped: true };
  }
  output.push({ scene, headless, streaming });
}

console.log(JSON.stringify({ ok: true, exe, frames, warmupFrames, resetAfterWarmup, streamFrames, streamTimeoutMs, iterations, gravity: gravity ?? null, physicsBackend, snapshotMode, collisionOnly, gpuSphereContacts, gpuGroundContactFeed, gpuContactSolveDiagnostic, gpuContactSolveNoReadback: effectiveContactNoReadback, gpuDirectCounterReadback, gpuRuntimeValidate, gpuApplyPrediction, gpuApplyVelocity, gpuResidentCounterlessContacts: effectiveResidentCounterlessContacts, gpuDeferFinalPositionReadback, gpuAsyncFinalPositionReadback: effectiveAsyncFinalPositionReadback, gpuApplyContactPositions: gpuApplyContactPositions || effectiveFastWebGpu, gpuGroundContacts: gpuGroundContacts || effectiveFastWebGpu, gpuJointTopologyDiagnostic, gpuApplyJointProposals, gpuRuntimeJointProposals: effectiveRuntimeJointProposals, gpuReplaceCpuJoints: effectiveReplaceCpuJoints, gpuDirectJointSolve: effectiveDirectJointSolve, gpuDirectContactSolve: effectiveDirectContactSolve, gpuJointTopologyRepeats, gpuJointProposalIterations, gpuContactIterations, gpuContactRelaxation, gpuJointPerturb, assertPhysicsAvgUnder, assertNoWebgpuFallbacks, assertNoWebgpuValidation, assertGroundCandidatesMin, assertDirectSphereCylinderCandidatesMin, assertDirectSphereCapsuleCandidatesMin, assertDirectSphereBoxCandidatesMin, assertDirectSphereCylinderCandidatesMaxMin, assertDirectSphereCapsuleCandidatesMaxMin, assertDirectSphereBoxCandidatesMaxMin, assertDirectRoundPairCandidatesMin, assertDirectRoundPairCandidatesMaxMin, assertDirectGpuContactRecordsMin, assertDirectGpuContactRecordsMaxMin, assertDirectGpuRoundPairCandidatesMin, assertDirectGpuRoundPairCandidatesMaxMin, assertDirectGpuCounterReadbackBytesMin, assertDirectGpuCounterReadbackBytesMax, assertPredictionAppliedBodiesMin, assertPredictionAppliedReadbackBytesMin, assertVelocityAppliedBodiesMin, assertVelocityAppliedReadbackBytesMin, assertExternalContactsMin, assertExternalGroundContactsMin, assertContactProposalActiveBodiesMin, assertContactFinalPositionReady, assertFinalPositionReadbackDeferred, assertFinalPositionAsyncReadbackConsumed, assertFinalPositionAsyncReadbackDroppedMax, assertFinalPositionAsyncReadbackWaitMax, assertContactProposalResidualAfterMax, assertContactIterationResidualAfterMax, assertContactAdjacencyReadbackBytesMax, assertContactGatherReadbackBytesMax, assertContactProposalResidualReadbackBytesMax, assertSapCandidatesMin, assertSphereHitsMin, assertPairChecksMin, assertAppliedPositionBodiesMin, assertDirectSphereAppliedBodiesMin, assertDirectSphereAppliedBodiesMaxMin, assertDirectRoundAppliedBodiesMin, assertDirectRoundAppliedBodiesMaxMin, assertDirectGroundAppliedBodiesMin, assertDirectGroundAppliedBodiesMaxMin, assertManifoldCountMax, assertFinalMaxSpeedMin, assertFinalMaxSpeedMax, assertFinalDynamicMaxHeightMin, assertFinalDynamicMaxHeightMax, assertPrimalAvgMax, assertDualAvgMax, assertBelowGroundSpheresMax, assertMaxGroundPenetration, assertBelowGroundBodiesMax, assertMaxGroundBodyPenetration, assertMaxLinearError, assertMaxAngularError, assertSapCounterReadbackBytesMax, assertSapPairReadbackBytesMax, assertJointTopologyMismatchesMax, assertJointTopologyJointsMin, assertJointTopologyLastMsUnder, assertJointResidualMin, assertJointProposalResidualAfterMax, assertJointProposalResidualRmsRatioMax, assertJointProposalAppliedBodiesMin, assertJointProposalAppliedMaxDeltaMin, assertJointFinalPositionAsyncReadbackConsumed, assertJointFinalPositionAsyncReadbackDroppedMax, assertJointFinalPositionAsyncReadbackWaitMax, assertPrimalJointSkippedMin, assertDualJointSkippedMin, assertPrimalIgnoreSkippedMin, assertDualIgnoreSkippedMin, assertJointInitializationSkippedMin, assertIgnoreInitializationSkippedMin, assertDirectJointSolve, assertDirectContactSolve, assertStreamSnapshotsPerSecondMin, assertStreamAvgSnapshotBytesMax, assertStreamAvgParseMsMax, results: output }, null, 2));

