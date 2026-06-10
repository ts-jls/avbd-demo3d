import { spawn, spawnSync } from "node:child_process";
import net from "node:net";
import path from "node:path";

function parseArgs(argv) {
  const args = new Map();
  for (let i = 0; i < argv.length; i++) {
    const token = argv[i];
    if (!token.startsWith("--")) continue;
    const key = token.slice(2);
    const next = argv[i + 1];
    if (!next || next.startsWith("--")) {
      args.set(key, "true");
    } else {
      args.set(key, next);
      i++;
    }
  }
  return args;
}

async function isPortAvailable(port) {
  return await new Promise((resolve) => {
    const server = net.createServer();
    server.once("error", () => resolve(false));
    server.once("listening", () => server.close(() => resolve(true)));
    server.listen(port, "127.0.0.1");
  });
}

async function findAvailablePort(startPort) {
  for (let port = startPort; port < startPort + 50; port++) {
    if (await isPortAvailable(port)) return port;
  }
  throw new Error(`No available localhost port found from ${startPort} to ${startPort + 49}`);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

const args = parseArgs(process.argv.slice(2));
const viewerRoot = path.resolve(import.meta.dirname, "..");
const port = await findAvailablePort(Number(args.get("port-start") ?? 18800));
const timeoutMs = Number(args.get("timeout") ?? 12000);
const startupDelayMs = Number(args.get("startup-delay") ?? 5000);
const scene = args.get("scene") ?? "Pyramid";
const backend = args.get("physics-backend") ?? "webgpu-avbd";

const server = spawn(process.execPath, [
  path.join("scripts", "start-gpu-server.mjs"),
  "--port", String(port),
  "--strict-port",
  "--stdio", "ignore",
  "--scene", scene,
  "--physics-backend", backend,
], {
  cwd: viewerRoot,
  stdio: "ignore",
});

let serverExited = false;
server.on("exit", () => {
  serverExited = true;
});

try {
  await sleep(startupDelayMs);
  if (serverExited) {
    throw new Error("GPU server launcher exited before bridge smoke could connect");
  }

  const result = spawnSync(process.execPath, [
    path.join("scripts", "bridge-smoke.mjs"),
    "--url", `ws://127.0.0.1:${port}`,
    "--binary", "true",
    "--timeout", String(timeoutMs),
  ], {
    cwd: viewerRoot,
    encoding: "utf8",
  });

  if (result.status !== 0) {
    throw new Error(result.stderr || result.stdout || `bridge smoke exited ${result.status}`);
  }

  const parsed = JSON.parse(result.stdout.slice(result.stdout.indexOf("{")));
  console.log(JSON.stringify({
    ok: true,
    port,
    scene: parsed.scene,
    frame: parsed.frame,
    bodies: parsed.bodies,
    shapes: parsed.shapes,
    mode: parsed.mode,
    backend,
  }, null, 2));
} finally {
  if (!server.killed) {
    server.kill("SIGINT");
  }
}
