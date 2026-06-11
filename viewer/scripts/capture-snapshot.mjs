// Capture one JSON snapshot from the headless server after a delay and write
// it to a file, for offline visual inspection of a scene's body positions.
import fs from "node:fs";

const args = new Map();
for (let i = 2; i < process.argv.length; i += 1) {
  const arg = process.argv[i];
  if (arg.startsWith("--")) {
    const value = process.argv[i + 1]?.startsWith("--") ? "true" : process.argv[++i] ?? "true";
    args.set(arg.slice(2), value);
  }
}

const url = args.get("url") ?? "ws://127.0.0.1:8765";
const delayMs = Number(args.get("delay") ?? 4000);
const outPath = args.get("out") ?? "snapshot.json";

const ws = new WebSocket(url);
let latest = null;

ws.addEventListener("message", async (event) => {
  const text = typeof event.data === "string" ? event.data : null;
  if (!text) return;
  try {
    const msg = JSON.parse(text);
    if (msg.type === "snapshot") latest = msg;
  } catch {}
});

ws.addEventListener("open", () => {
  setTimeout(() => {
    if (!latest) {
      console.error(JSON.stringify({ ok: false, error: "no snapshot received" }));
      process.exit(1);
    }
    fs.writeFileSync(outPath, JSON.stringify(latest));
    console.log(JSON.stringify({ ok: true, frame: latest.frame, scene: latest.scene, bodies: latest.bodies.length }));
    process.exit(0);
  }, delayMs);
});

ws.addEventListener("error", () => {
  console.error(JSON.stringify({ ok: false, error: "websocket error" }));
  process.exit(1);
});
