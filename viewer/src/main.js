import "./styles.css";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { makeSampleSnapshot, SHAPES } from "./samples.js";

const canvas = document.getElementById("renderer-canvas");
const sampleSelect = document.getElementById("sample-scene");
const hud = {
  mode: document.getElementById("viewer-mode"),
  status: document.getElementById("connection-status"),
  scene: document.getElementById("scene-name"),
  frame: document.getElementById("frame-number"),
  bodies: document.getElementById("body-count"),
  batches: document.getElementById("batch-count"),
  shapes: document.getElementById("shape-counts"),
  fps: document.getElementById("render-fps"),
  dropped: document.getElementById("dropped-frames"),
  snapshotSize: document.getElementById("snapshot-size"),
  bridgeUrl: document.getElementById("bridge-url"),
};

const pageUrl = new URL(window.location.href);
const bridgeUrl = pageUrl.searchParams.get("ws") ?? "ws://127.0.0.1:8765";
const initialSample = pageUrl.searchParams.get("sample");

const renderer = new THREE.WebGLRenderer({ antialias: true, canvas });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x17202c);
scene.fog = new THREE.Fog(0x17202c, 28, 90);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.05, 300);
camera.up.set(0, 0, 1);
camera.position.set(9, -14, 8);
camera.lookAt(0, 0, 2);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 2);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

const ambient = new THREE.HemisphereLight(0xcbd8ff, 0x3b2f28, 1.6);
scene.add(ambient);

const sun = new THREE.DirectionalLight(0xffffff, 2.8);
sun.position.set(9, -8, 16);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.left = -26;
sun.shadow.camera.right = 26;
sun.shadow.camera.top = 26;
sun.shadow.camera.bottom = -26;
sun.shadow.camera.near = 1;
sun.shadow.camera.far = 60;
scene.add(sun);

const grid = new THREE.GridHelper(48, 48, 0x6d7780, 0x3c4650);
grid.rotation.x = Math.PI / 2;
grid.position.z = 0.002;
scene.add(grid);

const materials = [
  new THREE.MeshStandardMaterial({ color: 0x9ba3a7, roughness: 0.72, metalness: 0.0 }),
  new THREE.MeshStandardMaterial({ color: 0x7d94b6, roughness: 0.62, metalness: 0.0 }),
  new THREE.MeshStandardMaterial({ color: 0x8d928f, roughness: 0.75, metalness: 0.0 }),
  new THREE.MeshStandardMaterial({ color: 0xb35b57, roughness: 0.58, metalness: 0.0 }),
  new THREE.MeshStandardMaterial({ color: 0x72a06e, roughness: 0.6, metalness: 0.0 }),
  new THREE.MeshStandardMaterial({ color: 0xc0a45d, roughness: 0.6, metalness: 0.0 }),
  new THREE.MeshStandardMaterial({ color: 0x6baeb0, roughness: 0.55, metalness: 0.0 }),
  new THREE.MeshStandardMaterial({ color: 0x9f77b8, roughness: 0.58, metalness: 0.0 }),
  new THREE.MeshStandardMaterial({ color: 0xb07f58, roughness: 0.62, metalness: 0.0 }),
];

const geometries = {
  [SHAPES.BOX]: new THREE.BoxGeometry(1, 1, 1),
  [SHAPES.SPHERE]: new THREE.SphereGeometry(0.5, 32, 16),
  [SHAPES.CYLINDER]: new THREE.CylinderGeometry(0.5, 0.5, 1, 32),
  [SHAPES.CAPSULE]: new THREE.CapsuleGeometry(0.5, 1, 10, 24),
};
geometries[SHAPES.CYLINDER].rotateX(Math.PI / 2);
geometries[SHAPES.CAPSULE].rotateX(Math.PI / 2);

const batches = new Map();
const tempObject = new THREE.Object3D();
const tempColor = new THREE.Color();
let currentSnapshot = makeSampleSnapshot("pyramid");
let frameCounter = 0;
let droppedFrames = 0;
let lastSnapshotBytes = 0;
let lastFpsTime = performance.now();
let framesThisSecond = 0;
let latestNativeSnapshot = null;
let socket = null;
let reconnectTimer = 0;
let reconnectAttempts = 0;
let nativeConnected = false;
let lastError = "";
let renderFps = 0;
let lastShapeCounts = {};
let lastFramedSignature = "";

const reconnectDelayMs = 1500;

function materialFor(body) {
  if (!body.dynamic) {
    return 0;
  }
  return Math.abs(body.material ?? 1) % materials.length;
}

function batchKey(shape, materialIndex) {
  return `${shape}:${materialIndex}`;
}

function clearBatches() {
  for (const mesh of batches.values()) {
    scene.remove(mesh);
  }
  batches.clear();
}

function buildBatch(shape, materialIndex, count) {
  const geometry = geometries[shape] ?? geometries[SHAPES.BOX];
  const material = materials[materialIndex] ?? materials[1];
  const mesh = new THREE.InstancedMesh(geometry, material, count);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  mesh.frustumCulled = false;
  mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  scene.add(mesh);
  return mesh;
}

function shapeScale(body) {
  const size = body.size ?? [1, 1, 1];
  if (body.shape === SHAPES.CAPSULE) {
    return [body.radius * 2, body.radius * 2, Math.max(0.001, size[2] * 0.5)];
  }
  return size;
}

function boundsForBodies(bodies) {
  const dynamicBodies = bodies.filter((body) => body.dynamic);
  const frameBodies = dynamicBodies.length > 0 ? dynamicBodies : bodies;
  if (frameBodies.length === 0) {
    return null;
  }

  const min = new THREE.Vector3(Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY);
  const max = new THREE.Vector3(Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY);
  for (const body of frameBodies) {
    const position = body.position ?? [0, 0, 0];
    const scale = shapeScale(body);
    const half = new THREE.Vector3(
      Math.max(0.05, scale[0] * 0.5),
      Math.max(0.05, scale[1] * 0.5),
      Math.max(0.05, scale[2] * 0.5),
    );
    const center = new THREE.Vector3(position[0], position[1], position[2]);
    min.min(center.clone().sub(half));
    max.max(center.clone().add(half));
  }
  return { min, max };
}

function frameSnapshot(snapshot, bodies) {
  const signature = `${snapshot?.scene ?? ""}:${bodies.length}`;
  if (signature === lastFramedSignature) {
    return;
  }
  const bounds = boundsForBodies(bodies);
  if (!bounds) {
    return;
  }

  const center = bounds.min.clone().add(bounds.max).multiplyScalar(0.5);
  const size = bounds.max.clone().sub(bounds.min);
  const radius = Math.max(2.0, size.length() * 0.5);
  const fovRadians = THREE.MathUtils.degToRad(camera.fov);
  const distance = Math.max(6.0, (radius / Math.tan(fovRadians * 0.5)) * 1.35);
  const viewDirection = new THREE.Vector3(0.75, -1.0, 0.55).normalize();

  controls.target.copy(center);
  camera.position.copy(center).addScaledVector(viewDirection, distance);
  camera.near = Math.max(0.02, distance / 500);
  camera.far = Math.max(300, distance * 8);
  camera.updateProjectionMatrix();
  controls.update();
  lastFramedSignature = signature;
}

function summarizeShapes(bodies) {
  const counts = {};
  for (const body of bodies) {
    if (geometries[body.shape]) {
      counts[body.shape] = (counts[body.shape] ?? 0) + 1;
    }
  }
  return counts;
}

function formatShapeCounts(counts) {
  const entries = Object.entries(counts);
  if (entries.length === 0) {
    return "none";
  }
  return entries.map(([shape, count]) => `${shape}:${count}`).join(" ");
}

function applySnapshot(snapshot, snapshotBytes = 0) {
  const bodies = Array.isArray(snapshot?.bodies) ? snapshot.bodies : [];
  const groups = new Map();
  for (const body of bodies) {
    if (!geometries[body.shape]) {
      continue;
    }
    const materialIndex = materialFor(body);
    const key = batchKey(body.shape, materialIndex);
    if (!groups.has(key)) {
      groups.set(key, { shape: body.shape, materialIndex, bodies: [] });
    }
    groups.get(key).bodies.push(body);
  }

  const neededKeys = new Set(groups.keys());
  for (const [key, mesh] of batches.entries()) {
    if (!neededKeys.has(key) || mesh.count < groups.get(key)?.bodies.length) {
      scene.remove(mesh);
      batches.delete(key);
    }
  }

  for (const [key, group] of groups.entries()) {
    let mesh = batches.get(key);
    if (!mesh) {
      mesh = buildBatch(group.shape, group.materialIndex, group.bodies.length);
      batches.set(key, mesh);
    }

    mesh.count = group.bodies.length;
    for (let i = 0; i < group.bodies.length; i += 1) {
      const body = group.bodies[i];
      const [sx, sy, sz] = shapeScale(body);
      tempObject.position.fromArray(body.position ?? [0, 0, 0]);
      tempObject.quaternion.fromArray(body.rotation ?? [0, 0, 0, 1]);
      tempObject.scale.set(sx, sy, sz);
      tempObject.updateMatrix();
      mesh.setMatrixAt(i, tempObject.matrix);
      if (mesh.instanceColor) {
        tempColor.copy(materials[group.materialIndex].color);
        mesh.setColorAt(i, tempColor);
      }
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) {
      mesh.instanceColor.needsUpdate = true;
    }
  }

  currentSnapshot = snapshot;
  lastSnapshotBytes = snapshotBytes;
  lastShapeCounts = summarizeShapes(bodies);
  frameSnapshot(snapshot, bodies);
  hud.scene.textContent = snapshot.scene ?? "Unknown";
  hud.frame.textContent = String(snapshot.frame ?? 0);
  hud.bodies.textContent = String(bodies.length);
  hud.batches.textContent = String(batches.size);
  hud.shapes.textContent = formatShapeCounts(lastShapeCounts);
  hud.snapshotSize.textContent = `${(lastSnapshotBytes / 1024).toFixed(1)} KB`;
}

function updateSampleScene() {
  clearBatches();
  currentSnapshot = makeSampleSnapshot(sampleSelect.value, frameCounter);
  applySnapshot(currentSnapshot, JSON.stringify(currentSnapshot).length);
}

sampleSelect.addEventListener("change", updateSampleScene);
if (initialSample && sampleSelect.querySelector(`option[value="${CSS.escape(initialSample)}"]`)) {
  sampleSelect.value = initialSample;
}
hud.bridgeUrl.textContent = bridgeUrl.replace("ws://", "");
updateSampleScene();

function setSampleMode(statusText = "Sample scene") {
  nativeConnected = false;
  latestNativeSnapshot = null;
  hud.mode.textContent = "Sample";
  hud.status.textContent = statusText;
  updateSampleScene();
}

function scheduleReconnect() {
  if (reconnectTimer) {
    return;
  }

  reconnectTimer = window.setTimeout(() => {
    reconnectTimer = 0;
    connectWebSocket();
  }, reconnectDelayMs);
}

function connectWebSocket() {
  if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
    return;
  }

  const url = new URL(window.location.href);
  const wsUrl = url.searchParams.get("ws") ?? "ws://127.0.0.1:8765";
  reconnectAttempts += 1;
  hud.status.textContent = reconnectAttempts === 1 ? "Connecting..." : `Retrying bridge (${reconnectAttempts})`;

  try {
    socket = new WebSocket(wsUrl);
  } catch (error) {
    console.warn("WebSocket creation failed", error);
    lastError = error.message;
    setSampleMode("Bridge unavailable");
    scheduleReconnect();
    return;
  }

  socket.binaryType = "arraybuffer";
  socket.addEventListener("open", () => {
    reconnectAttempts = 0;
    nativeConnected = true;
    hud.mode.textContent = "Native";
    hud.status.textContent = "Connected";
  });
  socket.addEventListener("close", () => {
    socket = null;
    setSampleMode(nativeConnected ? "Bridge disconnected" : "Bridge unavailable");
    scheduleReconnect();
  });
  socket.addEventListener("error", () => {
    lastError = "WebSocket error";
    hud.status.textContent = "Bridge error";
  });
  socket.addEventListener("message", (event) => {
    if (latestNativeSnapshot) {
      droppedFrames += 1;
    }
    latestNativeSnapshot = event.data;
  });
}

connectWebSocket();

function consumeNativeSnapshot() {
  if (!latestNativeSnapshot) {
    return;
  }

  const payload = latestNativeSnapshot;
  latestNativeSnapshot = null;
  try {
    let text;
    if (payload instanceof ArrayBuffer) {
      text = new TextDecoder().decode(payload);
    } else {
      text = String(payload);
    }
    const snapshot = JSON.parse(text);
    if (snapshot?.type === "snapshot") {
      applySnapshot(snapshot, text.length);
    }
  } catch (error) {
    console.warn("Snapshot decode failed", error);
    lastError = error.message;
    hud.status.textContent = "Decode error";
  }
}

function viewerState() {
  return {
    mode: hud.mode.textContent,
    status: hud.status.textContent,
    scene: currentSnapshot?.scene ?? "Unknown",
    frame: currentSnapshot?.frame ?? 0,
    bodyCount: Array.isArray(currentSnapshot?.bodies) ? currentSnapshot.bodies.length : 0,
    batchCount: batches.size,
    shapeCounts: { ...lastShapeCounts },
    renderFps,
    droppedFrames,
    snapshotBytes: lastSnapshotBytes,
    bridgeUrl,
    nativeConnected,
    reconnectAttempts,
    pendingNativeSnapshot: Boolean(latestNativeSnapshot),
    lastError,
    camera: {
      position: camera.position.toArray(),
      target: controls.target.toArray(),
    },
  };
}

window.__avbdViewer = {
  getState: viewerState,
  applySnapshot(snapshot) {
    applySnapshot(snapshot, JSON.stringify(snapshot).length);
    hud.mode.textContent = "Test";
    hud.status.textContent = "Injected snapshot";
  },
  setSampleScene(name) {
    if (!sampleSelect.querySelector(`option[value="${CSS.escape(name)}"]`)) {
      throw new Error(`Unknown sample scene: ${name}`);
    }
    sampleSelect.value = name;
    setSampleMode("Sample scene");
  },
  reconnect: connectWebSocket,
  disconnect() {
    if (socket) {
      socket.close();
    }
  },
};

window.dispatchEvent(new CustomEvent("avbd-viewer-ready", { detail: viewerState() }));

function animate(now) {
  requestAnimationFrame(animate);
  consumeNativeSnapshot();
  controls.update();
  renderer.render(scene, camera);

  framesThisSecond += 1;
  if (now - lastFpsTime >= 500) {
    renderFps = (framesThisSecond * 1000) / (now - lastFpsTime);
    hud.fps.textContent = renderFps.toFixed(1);
    hud.dropped.textContent = String(droppedFrames);
    framesThisSecond = 0;
    lastFpsTime = now;
  }

  if (hud.mode.textContent === "Sample") {
    frameCounter += 1;
    hud.frame.textContent = String(frameCounter);
  }
}

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

requestAnimationFrame(animate);
