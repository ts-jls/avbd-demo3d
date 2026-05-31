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

if (typeof WebSocket !== "function") {
  fail("This Node.js runtime does not provide a global WebSocket implementation.");
}

let settled = false;
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

socket.addEventListener("message", (event) => {
  if (settled) {
    return;
  }

  try {
    const text = typeof event.data === "string" ? event.data : Buffer.from(event.data).toString("utf8");
    const snapshot = JSON.parse(text);
    validateSnapshot(snapshot);
    settled = true;
    clearTimeout(timer);
    socket.close();

    const shapeCounts = {};
    for (const body of snapshot.bodies) {
      shapeCounts[body.shape] = (shapeCounts[body.shape] ?? 0) + 1;
    }
    console.log(JSON.stringify({
      ok: true,
      url,
      scene: snapshot.scene,
      frame: snapshot.frame,
      bodies: snapshot.bodies.length,
      shapes: shapeCounts,
      firstBody: snapshot.bodies[0],
    }, null, 2));
    process.exit(0);
  } catch (error) {
    settled = true;
    clearTimeout(timer);
    socket.close();
    fail(error.message);
  }
});
