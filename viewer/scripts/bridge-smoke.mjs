const args = new Map();
for (let i = 2; i < process.argv.length; i += 1) {
  const arg = process.argv[i];
  if (arg.startsWith("--")) {
    const key = arg.slice(2);
    const value = process.argv[i + 1]?.startsWith("--") ? "true" : process.argv[++i] ?? "true";
    args.set(key, value);
  }
}

const url = args.get("url") ?? "ws://127.0.0.1:8765";
const timeoutMs = Number(args.get("timeout") ?? 5000);
const binaryMode = args.get("binary") === "true";
const command = args.get("command") ?? "";
const commandValue = args.get("value") ?? "";
const expectStatusContains = args.get("expect-status-contains") ?? "";

function fail(message) {
  console.error(JSON.stringify({ ok: false, error: message }, null, 2));
  process.exit(1);
}

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function validateSnapshot(snapshot) {
  assert(snapshot.type === "snapshot", "snapshot type must be 'snapshot'");
  assert(snapshot.version === 1, "snapshot version must be 1");
  assert(typeof snapshot.scene === "string" && snapshot.scene.length > 0, "snapshot scene is missing");
  assert(Number.isFinite(snapshot.frame), "snapshot frame is invalid");
  assert(Array.isArray(snapshot.bodies) && snapshot.bodies.length > 0, "snapshot has no bodies");

  const inspected = snapshot.bodies.slice(0, Math.min(8, snapshot.bodies.length));
  for (const [index, body] of inspected.entries()) {
    assert(Number.isInteger(body.id), `body ${index} id is invalid`);
    assert(typeof body.shape === "string" && body.shape.length > 0, `body ${index} shape is invalid`);
    assert(Array.isArray(body.position) && body.position.length === 3, `body ${index} position is invalid`);
    assert(Array.isArray(body.rotation) && body.rotation.length === 4, `body ${index} rotation is invalid`);
    assert(Array.isArray(body.size) && body.size.length === 3, `body ${index} size is invalid`);
    assert(typeof body.dynamic === "boolean", `body ${index} dynamic flag is invalid`);
    assert(Number.isFinite(body.material), `body ${index} material is invalid`);
  }
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

function decodeBinarySnapshot(buffer) {
  const view = new DataView(buffer);
  let offset = 0;
  const magic = view.getUint32(offset, true);
  offset += 4;
  assert(magic === 0x53425641, "binary snapshot magic is invalid");
  const version = view.getUint32(offset, true);
  offset += 4;
  assert(version === 1, "binary snapshot version is invalid");
  const frame = Number(view.getBigUint64(offset, true));
  offset += 8;
  const sceneBytes = view.getUint32(offset, true);
  offset += 4;
  const bodyCount = view.getUint32(offset, true);
  offset += 4;
  const scene = Buffer.from(buffer.slice(offset, offset + sceneBytes)).toString("utf8");
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
    const position = [view.getFloat32(offset, true), view.getFloat32(offset + 4, true), view.getFloat32(offset + 8, true)];
    offset += 12;
    const rotation = [view.getFloat32(offset, true), view.getFloat32(offset + 4, true), view.getFloat32(offset + 8, true), view.getFloat32(offset + 12, true)];
    offset += 16;
    const size = [view.getFloat32(offset, true), view.getFloat32(offset + 4, true), view.getFloat32(offset + 8, true)];
    offset += 12;
    const radius = view.getFloat32(offset, true);
    offset += 4;
    const halfLength = view.getFloat32(offset, true);
    offset += 4;
    bodies.push({ id, shape: shapeNameFromId(shapeId), position, rotation, size, radius, halfLength, dynamic, material });
  }
  return { type: "snapshot", version, scene, frame, bodies };
}

function isBinarySnapshotBuffer(buffer) {
  return buffer.byteLength >= 24 && new DataView(buffer).getUint32(0, true) === 0x53425641;
}

if (typeof WebSocket !== "function") {
  fail("This Node.js runtime does not provide a global WebSocket implementation.");
}

let settled = false;
let validatedSnapshot = null;
let matchedStatus = "";
const timer = setTimeout(() => {
  if (!settled) {
    settled = true;
    fail(`Timed out waiting ${timeoutMs} ms for ${url}`);
  }
}, timeoutMs);

let socket;
try {
  socket = new WebSocket(url);
} catch (error) {
  clearTimeout(timer);
  fail(`WebSocket creation failed: ${error.message}`);
}

socket.addEventListener("error", () => {
  if (!settled) {
    settled = true;
    clearTimeout(timer);
    fail(`Could not connect to ${url}`);
  }
});

socket.addEventListener("open", () => {
  if (binaryMode) {
    socket.send(JSON.stringify({ type: "command", command: "snapshotMode", value: "binary" }));
  }
  if (command) {
    socket.send(JSON.stringify({ type: "command", command, value: commandValue }));
  }
});

function maybePass() {
  if (!validatedSnapshot) {
    return;
  }
  if (expectStatusContains && !matchedStatus) {
    return;
  }

  settled = true;
  clearTimeout(timer);
  socket.close();

  const shapeCounts = {};
  for (const body of validatedSnapshot.bodies) {
    shapeCounts[body.shape] = (shapeCounts[body.shape] ?? 0) + 1;
  }
  console.log(JSON.stringify({
    ok: true,
    url,
    mode: binaryMode ? "binary" : "json",
    command: command || null,
    value: command ? commandValue : null,
    statusMatched: matchedStatus || null,
    scene: validatedSnapshot.scene,
    frame: validatedSnapshot.frame,
    bodies: validatedSnapshot.bodies.length,
    shapes: shapeCounts,
    firstBody: validatedSnapshot.bodies[0],
  }, null, 2));
  process.exit(0);
}

socket.addEventListener("message", async (event) => {
  if (settled) {
    return;
  }

  try {
    let snapshot;
    if (binaryMode && typeof event.data !== "string") {
      const buffer = await eventDataToArrayBuffer(event.data);
      if (!isBinarySnapshotBuffer(buffer)) {
        const text = Buffer.from(buffer).toString("utf8");
        if (!text.includes("\"type\":\"snapshot\"")) {
          return;
        }
        snapshot = JSON.parse(text);
      } else {
        snapshot = decodeBinarySnapshot(buffer);
      }
    } else {
      const text = await eventDataToText(event.data);
      if (!text.includes("\"type\":\"snapshot\"")) {
        if (expectStatusContains && text.includes("\"type\":\"status\"")) {
          const message = JSON.parse(text);
          const metrics = String(message.metrics ?? "");
          if (metrics.includes(expectStatusContains)) {
            matchedStatus = expectStatusContains;
            maybePass();
          }
        }
        return;
      }
      snapshot = JSON.parse(text);
    }
    validateSnapshot(snapshot);
    validatedSnapshot = snapshot;
    maybePass();
  } catch (error) {
    settled = true;
    clearTimeout(timer);
    socket.close();
    fail(error.message);
  }
});
