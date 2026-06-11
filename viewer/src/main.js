import "./styles.css";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { makeSampleSnapshot, SHAPES } from "./samples.js";

const canvas = document.getElementById("renderer-canvas");
const sampleSelect = document.getElementById("sample-scene");
const nativeSceneSelect = document.getElementById("native-scene");
const nativeBackendSelect = document.getElementById("native-backend");
const nativeLoadButton = document.getElementById("native-load");
const nativePauseButton = document.getElementById("native-pause");
const nativeStepButton = document.getElementById("native-step");
const nativeResetButton = document.getElementById("native-reset");
const debugToggleButton = document.getElementById("debug-toggle");
const debugPanel = document.getElementById("debug-panel");
const debugMetrics = document.getElementById("debug-metrics");
const copyMetricsButton = document.getElementById("copy-metrics");
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
  parseMs: document.getElementById("parse-ms"),
  applyMs: document.getElementById("apply-ms"),
  renderMs: document.getElementById("render-ms"),
  wsKbps: document.getElementById("ws-kbps"),
  interaction: document.getElementById("interaction-status"),
  bridgeUrl: document.getElementById("bridge-url"),
};

const pageUrl = new URL(window.location.href);
const bridgeUrl = pageUrl.searchParams.get("ws") ?? "ws://127.0.0.1:8765";
const initialSample = pageUrl.searchParams.get("sample");
const snapshotMode = pageUrl.searchParams.get("snapshot") ?? "binary";
const bridgeTextDecoder = new TextDecoder();

const renderer = new THREE.WebGLRenderer({ antialias: true, canvas });
renderer.domElement.tabIndex = 0;
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
const raycaster = new THREE.Raycaster();
const pointerNdc = new THREE.Vector2();
const dragPlane = new THREE.Plane();
const dragPlaneNormal = new THREE.Vector3();
const dragPoint = new THREE.Vector3();
const dragLocalPoint = new THREE.Vector3();
const dragBodyPosition = new THREE.Vector3();
const dragBodyQuaternion = new THREE.Quaternion();
const worldUp = new THREE.Vector3(0, 0, 1);
const flyForward = new THREE.Vector3();
const flyRight = new THREE.Vector3();
const flyMove = new THREE.Vector3();
const flyLookTarget = new THREE.Vector3();
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
let nativePaused = false;
let lastError = "";
let renderFps = 0;
let renderMs = 0;
let parseMs = 0;
let applyMs = 0;
let wsKbps = 0;
let websocketBytesThisSecond = 0;
let websocketMessageCount = 0;
let lastShapeCounts = {};
let lastFramedSignature = "";
let bodyById = new Map();
let latestNativeMetricsText = "";
let latestSnapshotMode = "json";
let activeDrag = null;
let pendingDragTarget = null;
let dragCommandQueued = false;
let flyNavigationActive = false;
let flyPointerId = null;
let flyYaw = 0;
let flyPitch = 0;
let flyTargetYaw = 0;
let flyTargetPitch = 0;
let previousAnimationTime = performance.now();

const flyKeys = new Set();
const flyLookSensitivity = 0.0025;
const flyLookDamping = 28.0;
let flyMoveSpeed = 8.0;
const flyMinMoveSpeed = 0.5;
const flyMaxMoveSpeed = 120.0;
const flyFastMultiplier = 4.0;

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
  removeClothOverlay();
}

// Scenes whose sphere particles form a cloth grid, keyed by scene name. The
// particles are skinned as a deforming sheet instead of rendered as spheres;
// ids ascend in creation order, which the scene builds x-major.
const clothGrids = {
  Cloth: { width: 30, depth: 30, maxParticleDiameter: 0.3 },
};
let clothOverlay = null;

function removeClothOverlay() {
  if (clothOverlay) {
    scene.remove(clothOverlay.mesh);
    clothOverlay.geometry.dispose();
    clothOverlay.mesh.material.dispose();
    clothOverlay = null;
  }
}

// Returns the set of body ids consumed by the cloth sheet (so the caller can
// skip instancing them), or null when the snapshot has no cloth.
function updateClothOverlay(snapshot, bodies) {
  const grid = clothGrids[snapshot?.scene];
  if (!grid) {
    removeClothOverlay();
    return null;
  }
  const particles = bodies.filter(
    (body) => body.dynamic && body.shape === SHAPES.SPHERE && (body.size?.[0] ?? 1) < grid.maxParticleDiameter,
  );
  if (particles.length !== grid.width * grid.depth) {
    removeClothOverlay();
    return null;
  }
  particles.sort((a, b) => a.id - b.id);

  if (!clothOverlay || clothOverlay.count !== particles.length) {
    removeClothOverlay();
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      "position",
      new THREE.BufferAttribute(new Float32Array(particles.length * 3), 3).setUsage(THREE.DynamicDrawUsage),
    );
    const indices = [];
    for (let x = 0; x < grid.width - 1; x += 1) {
      for (let y = 0; y < grid.depth - 1; y += 1) {
        const a = x * grid.depth + y;
        const b = (x + 1) * grid.depth + y;
        const c = (x + 1) * grid.depth + y + 1;
        const d = x * grid.depth + y + 1;
        indices.push(a, b, c, a, c, d);
      }
    }
    geometry.setIndex(indices);
    const material = new THREE.MeshStandardMaterial({
      color: 0xe8e2d2,
      roughness: 0.9,
      metalness: 0.0,
      side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.frustumCulled = false;
    scene.add(mesh);
    clothOverlay = { mesh, geometry, count: particles.length };
  }

  const positions = clothOverlay.geometry.attributes.position;
  for (let i = 0; i < particles.length; i += 1) {
    const p = particles[i].position ?? [0, 0, 0];
    positions.setXYZ(i, p[0], p[1], p[2]);
  }
  positions.needsUpdate = true;
  clothOverlay.geometry.computeVertexNormals();
  return new Set(particles.map((body) => body.id));
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
  const applyBegin = performance.now();
  const bodies = Array.isArray(snapshot?.bodies) ? snapshot.bodies : [];
  const clothIds = updateClothOverlay(snapshot, bodies);
  bodyById = new Map();
  const groups = new Map();
  for (const body of bodies) {
    if (!geometries[body.shape]) {
      continue;
    }
    bodyById.set(body.id, body);
    if (clothIds?.has(body.id)) {
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
    mesh.userData.bodyIds = group.bodies.map((body) => body.id);
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
    mesh.boundingSphere = null;
    mesh.boundingBox = null;
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
  applyMs = performance.now() - applyBegin;
  hud.applyMs.textContent = `${applyMs.toFixed(2)} ms`;
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

function updateNativeControls() {
  nativeSceneSelect.disabled = !nativeConnected;
  nativeBackendSelect.disabled = !nativeConnected;
  nativeLoadButton.disabled = !nativeConnected;
  nativePauseButton.disabled = !nativeConnected;
  nativeStepButton.disabled = !nativeConnected;
  nativeResetButton.disabled = !nativeConnected;
  nativePauseButton.textContent = nativePaused ? "Play" : "Pause";
}

function buildMetricsReport() {
  const state = viewerState();
  const lines = [
    `Viewer mode: ${state.mode}`,
    `Viewer status: ${state.status}`,
    `Scene: ${state.scene}`,
    `Frame: ${state.frame}`,
    `Bodies: ${state.bodyCount}`,
    `Batches: ${state.batchCount}`,
    `Shapes: ${formatShapeCounts(state.shapeCounts)}`,
    `Render FPS: ${state.renderFps.toFixed(1)}`,
    `Dropped snapshots: ${state.droppedFrames}`,
    `Snapshot mode: ${state.snapshotMode}`,
    `Snapshot KB: ${(state.snapshotBytes / 1024).toFixed(1)}`,
    `WebSocket KB/s: ${state.webSocketKbps.toFixed(1)}`,
    `WebSocket messages: ${state.webSocketMessageCount}`,
    `JSON parse: ${state.parseMs.toFixed(2)} ms`,
    `Apply snapshot: ${state.applyMs.toFixed(2)} ms`,
    `Render: ${state.renderMs.toFixed(2)} ms`,
    `Native connected: ${state.nativeConnected ? "On" : "Off"}`,
    `Native paused: ${state.nativePaused ? "On" : "Off"}`,
    `Native backend selection: ${nativeBackendSelect.selectedOptions[0]?.textContent ?? "Server Default"}`,
    `Interaction: ${hud.interaction.textContent}`,
    `Bridge URL: ${state.bridgeUrl}`,
  ];
  if (latestNativeMetricsText) {
    lines.push("", "Native Metrics", latestNativeMetricsText.trim());
  }
  return `${lines.join("\n")}\n`;
}

function updateDebugMetrics() {
  debugMetrics.textContent = buildMetricsReport();
}

async function copyMetrics() {
  const text = buildMetricsReport();
  try {
    await navigator.clipboard.writeText(text);
    copyMetricsButton.textContent = "Copied";
    window.setTimeout(() => {
      copyMetricsButton.textContent = "Copy Metrics";
    }, 1200);
  } catch (error) {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.style.position = "fixed";
    textarea.style.left = "-1000px";
    document.body.appendChild(textarea);
    textarea.focus();
    textarea.select();
    document.execCommand("copy");
    document.body.removeChild(textarea);
  }
}

function setSampleMode(statusText = "Sample scene") {
  endBrowserDrag(false);
  nativeConnected = false;
  nativePaused = false;
  latestNativeSnapshot = null;
  hud.mode.textContent = "Sample";
  hud.status.textContent = statusText;
  updateNativeControls();
  updateSampleScene();
}

function sendCommand(command, value) {
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    return false;
  }
  socket.send(JSON.stringify({ type: "command", command, value }));
  return true;
}

function applyNativeBackendSelection() {
  const backend = nativeBackendSelect.value;
  if (!backend) {
    return true;
  }
  return sendCommand("physicsBackend", backend);
}

function shapeNameFromId(shapeId) {
  switch (shapeId) {
    case 0:
      return "box";
    case 1:
      return "sphere";
    case 2:
      return "capsule";
    case 3:
      return "cylinder";
    default:
      return "box";
  }
}

function isBinarySnapshot(payload) {
  if (!(payload instanceof ArrayBuffer) || payload.byteLength < 24) {
    return false;
  }
  return new DataView(payload).getUint32(0, true) === 0x53425641;
}

function decodeBinarySnapshot(payload) {
  const view = new DataView(payload);
  let offset = 0;
  const magic = view.getUint32(offset, true);
  offset += 4;
  if (magic !== 0x53425641) {
    throw new Error("Invalid AVBD binary snapshot magic");
  }
  const version = view.getUint32(offset, true);
  offset += 4;
  if (version !== 1) {
    throw new Error(`Unsupported AVBD binary snapshot version ${version}`);
  }
  const frame = Number(view.getBigUint64(offset, true));
  offset += 8;
  const sceneBytes = view.getUint32(offset, true);
  offset += 4;
  const bodyCount = view.getUint32(offset, true);
  offset += 4;
  const sceneName = bridgeTextDecoder.decode(new Uint8Array(payload, offset, sceneBytes));
  offset += sceneBytes;

  const bodies = [];
  for (let i = 0; i < bodyCount; i += 1) {
    const id = view.getUint32(offset, true);
    offset += 4;
    const shapeId = view.getUint32(offset, true);
    offset += 4;
    const material = view.getUint32(offset, true);
    offset += 4;
    const dynamic = view.getUint32(offset, true) !== 0;
    offset += 4;
    const position = [
      view.getFloat32(offset, true),
      view.getFloat32(offset + 4, true),
      view.getFloat32(offset + 8, true),
    ];
    offset += 12;
    const rotation = [
      view.getFloat32(offset, true),
      view.getFloat32(offset + 4, true),
      view.getFloat32(offset + 8, true),
      view.getFloat32(offset + 12, true),
    ];
    offset += 16;
    const size = [
      view.getFloat32(offset, true),
      view.getFloat32(offset + 4, true),
      view.getFloat32(offset + 8, true),
    ];
    offset += 12;
    const radius = view.getFloat32(offset, true);
    offset += 4;
    const halfLength = view.getFloat32(offset, true);
    offset += 4;

    bodies.push({
      id,
      shape: shapeNameFromId(shapeId),
      position,
      rotation,
      size,
      radius,
      halfLength,
      dynamic,
      material,
    });
  }

  return { type: "snapshot", version, frame, scene: sceneName, bodies };
}

function setPointerFromEvent(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointerNdc.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointerNdc.y = -(((event.clientY - rect.top) / rect.height) * 2 - 1);
  raycaster.setFromCamera(pointerNdc, camera);
}

function arrayFromVector(vector) {
  return [vector.x, vector.y, vector.z];
}

function pickBody(event) {
  if (!nativeConnected) {
    return null;
  }

  setPointerFromEvent(event);
  const hits = raycaster.intersectObjects([...batches.values()], false);
  for (const hit of hits) {
    const bodyId = hit.object.userData.bodyIds?.[hit.instanceId];
    const body = bodyById.get(bodyId);
    if (!body?.dynamic) {
      continue;
    }
    return { hit, body };
  }
  return null;
}

function bodyLocalPoint(body, worldPoint) {
  dragBodyPosition.fromArray(body.position ?? [0, 0, 0]);
  dragBodyQuaternion.fromArray(body.rotation ?? [0, 0, 0, 1]);
  return dragLocalPoint.copy(worldPoint).sub(dragBodyPosition).applyQuaternion(dragBodyQuaternion.invert());
}

function updateDragPlane(hitPoint) {
  camera.getWorldDirection(dragPlaneNormal);
  dragPlane.setFromNormalAndCoplanarPoint(dragPlaneNormal, hitPoint);
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function setFlyAnglesFromCamera() {
  camera.getWorldDirection(flyForward);
  flyYaw = Math.atan2(flyForward.y, flyForward.x);
  flyPitch = Math.asin(clamp(flyForward.z, -0.98, 0.98));
  flyTargetYaw = flyYaw;
  flyTargetPitch = flyPitch;
}

function applyFlyLook() {
  const cp = Math.cos(flyPitch);
  flyForward.set(cp * Math.cos(flyYaw), cp * Math.sin(flyYaw), Math.sin(flyPitch)).normalize();
  flyLookTarget.copy(camera.position).add(flyForward);
  camera.lookAt(flyLookTarget);
  controls.target.copy(flyLookTarget);
}

function updateFlyLook(deltaSeconds) {
  if (!flyNavigationActive) {
    return;
  }

  const t = 1.0 - Math.exp(-flyLookDamping * deltaSeconds);
  flyYaw += (flyTargetYaw - flyYaw) * t;
  flyPitch += (flyTargetPitch - flyPitch) * t;
  applyFlyLook();
}

function endFlyNavigation() {
  if (!flyNavigationActive) {
    return;
  }
  flyNavigationActive = false;
  flyPointerId = null;
  flyKeys.clear();
  controls.enabled = true;
  renderer.domElement.style.cursor = activeDrag ? "grabbing" : "";
  hud.interaction.textContent = activeDrag ? `Dragging #${activeDrag.bodyId}` : "Orbit";
}

function updateFlyHud() {
  hud.interaction.textContent = `Fly ${flyMoveSpeed.toFixed(1)} u/s`;
}

function beginFlyNavigation(event) {
  if (event.button !== 2 || activeDrag) {
    return;
  }

  setFlyAnglesFromCamera();
  flyNavigationActive = true;
  flyPointerId = event.pointerId;
  controls.enabled = false;
  renderer.domElement.focus();
  renderer.domElement.setPointerCapture?.(event.pointerId);
  renderer.domElement.style.cursor = "crosshair";
  updateFlyHud();
  event.preventDefault();
  event.stopImmediatePropagation();
}

function updateFlyNavigation(event) {
  if (!flyNavigationActive || event.pointerId !== flyPointerId) {
    return;
  }

  flyTargetYaw -= event.movementX * flyLookSensitivity;
  flyTargetPitch = clamp(flyTargetPitch - event.movementY * flyLookSensitivity, -1.45, 1.45);
  event.preventDefault();
}

function updateFlyMovement(deltaSeconds) {
  if (!flyNavigationActive) {
    return;
  }

  flyMove.set(0, 0, 0);
  camera.getWorldDirection(flyForward).normalize();
  flyRight.crossVectors(flyForward, worldUp);
  if (flyRight.lengthSq() < 0.0001) {
    flyRight.set(1, 0, 0);
  } else {
    flyRight.normalize();
  }

  if (flyKeys.has("w")) flyMove.add(flyForward);
  if (flyKeys.has("s")) flyMove.sub(flyForward);
  if (flyKeys.has("d")) flyMove.add(flyRight);
  if (flyKeys.has("a")) flyMove.sub(flyRight);
  if (flyKeys.has("e")) flyMove.add(worldUp);
  if (flyKeys.has("q")) flyMove.sub(worldUp);

  if (flyMove.lengthSq() === 0) {
    return;
  }

  const multiplier = flyKeys.has("shift") ? flyFastMultiplier : 1.0;
  flyMove.normalize().multiplyScalar(flyMoveSpeed * multiplier * deltaSeconds);
  camera.position.add(flyMove);
  controls.target.add(flyMove);
}

function endBrowserDrag(sendNative = true) {
  if (!activeDrag) {
    return;
  }
  if (sendNative) {
    sendCommand("endDrag", true);
  }
  activeDrag = null;
  pendingDragTarget = null;
  dragCommandQueued = false;
  controls.enabled = true;
  renderer.domElement.style.cursor = "";
  hud.interaction.textContent = "Orbit";
}

function queueDragUpdate(worldTarget) {
  pendingDragTarget = arrayFromVector(worldTarget);
  dragCommandQueued = true;
}

function flushDragUpdate() {
  if (!activeDrag || !dragCommandQueued || !pendingDragTarget) {
    return;
  }
  sendCommand("updateDrag", { worldTarget: pendingDragTarget });
  dragCommandQueued = false;
}

function updateBrowserDrag(event) {
  if (!activeDrag) {
    return;
  }
  setPointerFromEvent(event);
  if (raycaster.ray.intersectPlane(dragPlane, dragPoint)) {
    queueDragUpdate(dragPoint);
  }
  event.preventDefault();
}

function beginBrowserDrag(event) {
  if (event.button !== 0) {
    return;
  }

  const pick = pickBody(event);
  if (!pick) {
    return;
  }

  const worldHit = pick.hit.point.clone();
  const localHit = bodyLocalPoint(pick.body, worldHit).clone();
  updateDragPlane(worldHit);

  const started = sendCommand("beginDrag", {
    bodyId: pick.body.id,
    localHit: arrayFromVector(localHit),
    worldHit: arrayFromVector(worldHit),
  });
  if (!started) {
    return;
  }

  activeDrag = {
    pointerId: event.pointerId,
    bodyId: pick.body.id,
  };
  controls.enabled = false;
  renderer.domElement.setPointerCapture?.(event.pointerId);
  renderer.domElement.style.cursor = "grabbing";
  hud.interaction.textContent = `Dragging #${pick.body.id}`;
  queueDragUpdate(worldHit);
  event.preventDefault();
  event.stopImmediatePropagation();
}

nativeLoadButton.addEventListener("click", () => {
  endBrowserDrag();
  sendCommand("scene", nativeSceneSelect.value);
  applyNativeBackendSelection();
});

nativeBackendSelect.addEventListener("change", () => {
  applyNativeBackendSelection();
});

nativePauseButton.addEventListener("click", () => {
  nativePaused = !nativePaused;
  updateNativeControls();
  sendCommand("pause", nativePaused);
});

nativeStepButton.addEventListener("click", () => {
  sendCommand("step", true);
});

nativeResetButton.addEventListener("click", () => {
  endBrowserDrag();
  sendCommand("reset", true);
});
debugToggleButton.addEventListener("click", () => {
  debugPanel.hidden = !debugPanel.hidden;
  if (!debugPanel.hidden) {
    updateDebugMetrics();
  }
});
copyMetricsButton.addEventListener("click", copyMetrics);
updateNativeControls();

renderer.domElement.addEventListener("pointerdown", beginBrowserDrag, { capture: true });
renderer.domElement.addEventListener("pointerdown", beginFlyNavigation, { capture: true });
renderer.domElement.addEventListener("pointermove", updateBrowserDrag);
renderer.domElement.addEventListener("pointermove", updateFlyNavigation);
renderer.domElement.addEventListener("pointerup", (event) => {
  if (activeDrag?.pointerId === event.pointerId) {
    renderer.domElement.releasePointerCapture?.(event.pointerId);
    endBrowserDrag();
  }
  if (flyNavigationActive && flyPointerId === event.pointerId) {
    renderer.domElement.releasePointerCapture?.(event.pointerId);
    endFlyNavigation();
  }
});
renderer.domElement.addEventListener("pointercancel", () => {
  endBrowserDrag();
  endFlyNavigation();
});
renderer.domElement.addEventListener("contextmenu", (event) => event.preventDefault());
renderer.domElement.addEventListener("wheel", (event) => {
  if (!flyNavigationActive) {
    return;
  }
  const factor = Math.exp(-event.deltaY * 0.0015);
  flyMoveSpeed = clamp(flyMoveSpeed * factor, flyMinMoveSpeed, flyMaxMoveSpeed);
  updateFlyHud();
  event.preventDefault();
}, { passive: false });
window.addEventListener("blur", () => {
  endBrowserDrag();
  endFlyNavigation();
});
window.addEventListener("keydown", (event) => {
  if (!flyNavigationActive) {
    return;
  }
  const key = event.key.toLowerCase();
  if (["w", "a", "s", "d", "q", "e"].includes(key)) {
    flyKeys.add(key);
    event.preventDefault();
  } else if (event.key === "Shift") {
    flyKeys.add("shift");
    event.preventDefault();
  }
});
window.addEventListener("keyup", (event) => {
  const key = event.key.toLowerCase();
  if (["w", "a", "s", "d", "q", "e"].includes(key)) {
    flyKeys.delete(key);
  } else if (event.key === "Shift") {
    flyKeys.delete("shift");
  }
});

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
    if (snapshotMode === "binary") {
      sendCommand("snapshotMode", "binary");
      latestSnapshotMode = "binary";
    } else {
      sendCommand("snapshotMode", "json");
      latestSnapshotMode = "json";
    }
    updateNativeControls();
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
    try {
      if (event.data instanceof ArrayBuffer && isBinarySnapshot(event.data)) {
        websocketBytesThisSecond += event.data.byteLength;
        websocketMessageCount += 1;
        if (latestNativeSnapshot) {
          droppedFrames += 1;
        }
        latestNativeSnapshot = event.data;
        latestSnapshotMode = "binary";
        return;
      }

      const text = event.data instanceof ArrayBuffer ? bridgeTextDecoder.decode(event.data) : String(event.data);
      websocketBytesThisSecond += event.data instanceof ArrayBuffer ? event.data.byteLength : text.length;
      websocketMessageCount += 1;
      if (text.includes("\"type\":\"snapshot\"")) {
        if (latestNativeSnapshot) {
          droppedFrames += 1;
        }
        latestNativeSnapshot = text;
      } else {
        const message = JSON.parse(text);
        if (message?.type !== "status") {
          return;
        }
        latestNativeMetricsText = String(message.metrics ?? "");
        hud.status.textContent = nativeConnected ? "Connected" : "Status received";
      }
    } catch (error) {
      console.warn("Bridge message decode failed", error);
      lastError = error.message;
    }
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
    const parseBegin = performance.now();
    const snapshot = payload instanceof ArrayBuffer && isBinarySnapshot(payload)
      ? decodeBinarySnapshot(payload)
      : JSON.parse(String(payload));
    parseMs = performance.now() - parseBegin;
    hud.parseMs.textContent = `${parseMs.toFixed(2)} ms`;
    if (snapshot?.type === "snapshot") {
      applySnapshot(snapshot, payload instanceof ArrayBuffer ? payload.byteLength : String(payload).length);
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
    renderMs,
    parseMs,
    applyMs,
    webSocketKbps: wsKbps,
    webSocketMessageCount: websocketMessageCount,
    droppedFrames,
    snapshotBytes: lastSnapshotBytes,
    snapshotMode: latestSnapshotMode,
    bridgeUrl,
    nativeConnected,
    nativePaused,
    draggingBodyId: activeDrag?.bodyId ?? null,
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
  sendCommand,
};

window.dispatchEvent(new CustomEvent("avbd-viewer-ready", { detail: viewerState() }));

function animate(now) {
  requestAnimationFrame(animate);
  const deltaSeconds = Math.min(0.05, Math.max(0.0, (now - previousAnimationTime) / 1000));
  previousAnimationTime = now;
  consumeNativeSnapshot();
  flushDragUpdate();
  updateFlyLook(deltaSeconds);
  updateFlyMovement(deltaSeconds);
  if (!flyNavigationActive) {
    controls.update();
  }
  const renderBegin = performance.now();
  renderer.render(scene, camera);
  renderMs = performance.now() - renderBegin;
  hud.renderMs.textContent = `${renderMs.toFixed(2)} ms`;

  framesThisSecond += 1;
  if (now - lastFpsTime >= 500) {
    renderFps = (framesThisSecond * 1000) / (now - lastFpsTime);
    wsKbps = (websocketBytesThisSecond / 1024) * (1000 / (now - lastFpsTime));
    hud.fps.textContent = renderFps.toFixed(1);
    hud.dropped.textContent = String(droppedFrames);
    hud.wsKbps.textContent = `${wsKbps.toFixed(1)} KB/s`;
    if (!debugPanel.hidden) {
      updateDebugMetrics();
    }
    framesThisSecond = 0;
    websocketBytesThisSecond = 0;
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
