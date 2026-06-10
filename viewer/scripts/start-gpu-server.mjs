import fs from "node:fs";
import net from "node:net";
import path from "node:path";
import { spawn } from "node:child_process";

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

function boolArg(args, key) {
  const value = args.get(key);
  return value === "true" || value === "1";
}

async function isPortAvailable(port) {
  return await new Promise((resolve) => {
    const server = net.createServer();
    server.once("error", () => resolve(false));
    server.once("listening", () => {
      server.close(() => resolve(true));
    });
    server.listen(port, "127.0.0.1");
  });
}

async function findAvailablePort(startPort, scanCount) {
  for (let port = startPort; port < startPort + scanCount; port++) {
    if (await isPortAvailable(port)) return port;
  }
  return 0;
}

const args = parseArgs(process.argv.slice(2));
const repoRoot = path.resolve(import.meta.dirname, "..", "..");
const defaultExe = path.join(repoRoot, "build-webgpu-dawn-nmake3", "avbd_headless_server.exe");
const exe = args.get("exe") ?? defaultExe;
const requestedPort = Number(args.get("port") ?? 8765);
const scanCount = Number(args.get("scan") ?? 32);
const strictPort = boolArg(args, "strict-port");
const checkOnly = boolArg(args, "check-only");
const scene = args.get("scene") ?? "Pyramid";
const backend = args.get("physics-backend") ?? "webgpu-avbd";
const snapshotMode = args.get("snapshot-mode") ?? "binary";
const tickRate = args.get("tick-rate") ?? "60";
const metricsInterval = args.get("metrics-interval");
const stdioMode = args.get("stdio") === "ignore" ? "ignore" : "inherit";

if (!Number.isInteger(requestedPort) || requestedPort <= 0 || requestedPort > 65535) {
  console.error(`Invalid --port ${args.get("port")}`);
  process.exit(1);
}

if (!fs.existsSync(exe)) {
  console.error(`Headless server executable not found: ${exe}`);
  console.error("Build the Dawn target first, then rerun this launcher.");
  process.exit(1);
}

const port = strictPort
  ? ((await isPortAvailable(requestedPort)) ? requestedPort : 0)
  : await findAvailablePort(requestedPort, scanCount);

if (!port) {
  const range = strictPort ? `${requestedPort}` : `${requestedPort}-${requestedPort + scanCount - 1}`;
  console.error(`No available localhost port found in ${range}.`);
  process.exit(1);
}

const wsUrl = `ws://127.0.0.1:${port}`;
const viewerUrl = port === 8765 ? "http://127.0.0.1:5173/" : `http://127.0.0.1:5173/?ws=${encodeURIComponent(wsUrl)}`;

console.log("AVBD GPU headless server launcher");
console.log(`Executable: ${exe}`);
console.log(`Scene: ${scene}`);
console.log(`Physics backend: ${backend}`);
console.log(`Snapshot mode: ${snapshotMode}`);
console.log(`Bridge: ${wsUrl}`);
console.log(`Viewer URL: ${viewerUrl}`);

if (checkOnly) {
  console.log("Check-only mode passed; server was not started.");
  process.exit(0);
}

const childArgs = [
  "--scene", scene,
  "--port", String(port),
  "--physics-backend", backend,
  "--snapshot-mode", snapshotMode,
  "--tick-rate", tickRate,
];
if (metricsInterval) {
  childArgs.push("--metrics-interval", metricsInterval);
}

const child = spawn(exe, childArgs, {
  cwd: path.dirname(exe),
  stdio: stdioMode,
});

let stopping = false;
function stop() {
  if (stopping) return;
  stopping = true;
  if (!child.killed) child.kill("SIGINT");
}

process.on("SIGINT", stop);
process.on("SIGTERM", stop);

child.on("exit", (code, signal) => {
  if (signal) {
    process.exit(0);
  }
  process.exit(code ?? 0);
});
