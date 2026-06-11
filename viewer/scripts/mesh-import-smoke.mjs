// Smoke test for the importMesh bridge command: sends a small torus mesh,
// expects a meshImport reply with particle ids + per-vertex bindings, and a
// later snapshot containing the new particle bodies.
const url = process.argv[2] ?? "ws://127.0.0.1:8765";

function makeTorus(majorR, minorR, majorSeg, minorSeg) {
  const verts = [];
  const tris = [];
  for (let i = 0; i < majorSeg; i++) {
    const a = (2 * Math.PI * i) / majorSeg;
    for (let j = 0; j < minorSeg; j++) {
      const b = (2 * Math.PI * j) / minorSeg;
      const r = majorR + minorR * Math.cos(b);
      verts.push(Math.cos(a) * r, Math.sin(a) * r, minorR * Math.sin(b));
    }
  }
  const at = (i, j) => (i % majorSeg) * minorSeg + (j % minorSeg);
  for (let i = 0; i < majorSeg; i++) {
    for (let j = 0; j < minorSeg; j++) {
      tris.push(at(i, j), at(i + 1, j), at(i + 1, j + 1));
      tris.push(at(i, j), at(i + 1, j + 1), at(i, j + 1));
    }
  }
  return { verts, tris };
}

const torus = makeTorus(0.8, 0.35, 20, 10);
const ws = new WebSocket(url);
let reply = null;
let particleIds = null;
let sawParticlesInSnapshot = false;

const timeout = setTimeout(() => {
  console.error(JSON.stringify({ ok: false, error: "timeout", gotReply: !!reply, sawParticlesInSnapshot }));
  process.exit(1);
}, 15000);

ws.addEventListener("open", () => {
  ws.send(
    JSON.stringify({
      type: "command",
      command: "importMesh",
      name: "smoke-torus",
      mode: "soft",
      spacing: 0.22,
      scale: 1.0,
      position: [0, 0, 4],
      vertices: torus.verts,
      triangles: torus.tris,
    }),
  );
});

ws.addEventListener("message", (event) => {
  if (typeof event.data !== "string") return;
  let msg;
  try {
    msg = JSON.parse(event.data);
  } catch {
    return;
  }
  if (msg.type === "meshImport") {
    reply = msg;
    if (!msg.ok) {
      clearTimeout(timeout);
      console.error(JSON.stringify({ ok: false, error: msg.error ?? "import failed" }));
      process.exit(1);
    }
    particleIds = new Set(msg.particleIds);
  }
  if (msg.type === "snapshot" && particleIds) {
    const present = msg.bodies.filter((b) => particleIds.has(b.id)).length;
    if (present === particleIds.size) {
      sawParticlesInSnapshot = true;
      clearTimeout(timeout);
      const bindings = reply.bindings;
      const sane =
        Array.isArray(bindings) &&
        bindings.length === torus.verts.length / 3 &&
        bindings.every((b) => Array.isArray(b) && b.length >= 1 && b.length <= 4);
      console.log(
        JSON.stringify({
          ok: sane,
          particles: particleIds.size,
          vertices: bindings.length,
          maxBindsPerVertex: Math.max(...bindings.map((b) => b.length)),
          weightSumSample: bindings[0].reduce((s, x) => s + x[1], 0),
        }),
      );
      process.exit(sane ? 0 : 1);
    }
  }
});

ws.addEventListener("error", () => {
  clearTimeout(timeout);
  console.error(JSON.stringify({ ok: false, error: "websocket error" }));
  process.exit(1);
});
