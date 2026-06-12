// Regression test for thin/open meshes (the "squiggle" case): imports a
// helix tube whose radius is far below the lattice spacing — the interior
// parity fill finds nothing, so particles must come from the surface
// voxelization and stay connected through diagonal links. Asserts the
// settled particle cloud stays coherent instead of scattering.
const url = process.argv[2] ?? "ws://127.0.0.1:8765";

const TURNS = 2.5, HR = 0.7, HEIGHT = 1.4, TR = 0.05, SEG = 120, RING = 6;
const verts = [];
const tris = [];
for (let i = 0; i <= SEG; i++) {
  const t = i / SEG;
  const a = 2 * Math.PI * TURNS * t;
  const cx = Math.cos(a) * HR, cy = Math.sin(a) * HR, cz = HEIGHT * (t - 0.5);
  const tx = -Math.sin(a), ty = Math.cos(a);
  for (let j = 0; j < RING; j++) {
    const b = (2 * Math.PI * j) / RING;
    const nx = Math.cos(b) * ty, ny = -Math.cos(b) * tx, nz = Math.sin(b);
    verts.push(cx + TR * nx, cy + TR * ny, cz + TR * nz);
  }
}
for (let i = 0; i < SEG; i++) {
  for (let j = 0; j < RING; j++) {
    const a = i * RING + j;
    const b = i * RING + ((j + 1) % RING);
    const c = (i + 1) * RING + j;
    const d = (i + 1) * RING + ((j + 1) % RING);
    tris.push(a, c, d, a, d, b);
  }
}

const ws = new WebSocket(url);
let ids = null;
let importedAt = null;

const timer = setTimeout(() => {
  console.error(JSON.stringify({ ok: false, error: "timeout" }));
  process.exit(1);
}, 25000);

ws.addEventListener("open", () => {
  ws.send(
    JSON.stringify({
      type: "command",
      command: "importMesh",
      name: "helix",
      mode: "soft",
      spacing: 2 / 7,
      scale: 1,
      position: [0, 0, 2.5],
      vertices: verts,
      triangles: tris,
    }),
  );
});

ws.addEventListener("message", (ev) => {
  if (typeof ev.data !== "string") return;
  let m;
  try {
    m = JSON.parse(ev.data);
  } catch {
    return;
  }
  if (m.type === "meshImport") {
    if (!m.ok) {
      clearTimeout(timer);
      console.error(JSON.stringify({ ok: false, error: m.error }));
      process.exit(1);
    }
    ids = new Set(m.particleIds);
    importedAt = Date.now();
    console.error(`imported: ${m.particleIds.length} particles`);
  }
  if (m.type === "snapshot" && ids && Date.now() - importedAt > 6000) {
    clearTimeout(timer);
    const ps = m.bodies.filter((b) => ids.has(b.id)).map((b) => b.position);
    const mn = [Infinity, Infinity, Infinity];
    const mx = [-Infinity, -Infinity, -Infinity];
    for (const p of ps) {
      for (let a = 0; a < 3; a++) {
        mn[a] = Math.min(mn[a], p[a]);
        mx[a] = Math.max(mx[a], p[a]);
      }
    }
    const spread = Math.max(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]);
    // Coherent settled squiggle: stays within a few units, rests on (not
    // through) the ground.
    const ok = ps.length === ids.size && spread < 4.0 && mn[2] > -0.5;
    console.log(JSON.stringify({ ok, particles: ps.length, spread: +spread.toFixed(2), minZ: +mn[2].toFixed(2) }));
    process.exit(ok ? 0 : 1);
  }
});

ws.addEventListener("error", () => {
  clearTimeout(timer);
  console.error(JSON.stringify({ ok: false, error: "websocket error" }));
  process.exit(1);
});
