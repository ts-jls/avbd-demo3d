import { createServer } from "node:http";
import { readFile, rm, stat, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { extname, join, normalize, resolve } from "node:path";
import { spawn } from "node:child_process";
import { tmpdir } from "node:os";
import { inflateSync } from "node:zlib";

const args = new Map();
for (let i = 2; i < process.argv.length; i += 1) {
  const arg = process.argv[i];
  if (arg.startsWith("--")) {
    const key = arg.slice(2);
    const value = process.argv[i + 1]?.startsWith("--") ? "true" : process.argv[++i] ?? "true";
    args.set(key, value);
  }
}

const repoViewerRoot = resolve(new URL("..", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1"));
const distRoot = resolve(repoViewerRoot, "dist");
const screenshotPath = args.get("screenshot") ?? "C:\\tmp\\avbd-three-viewer-smoke.png";
const timeoutMs = Number(args.get("timeout") ?? 12000);
const sample = args.get("sample") ?? "mixed";
const bridgeUrl = args.get("ws") ?? "ws://127.0.0.1:8765";
const expectNative = args.get("expect-native") === "true";
const expectedShapes = (args.get("expect-shapes") ?? "")
  .split(",")
  .map((shape) => shape.trim())
  .filter(Boolean);

function fail(message) {
  console.error(JSON.stringify({ ok: false, error: message }, null, 2));
  process.exit(1);
}

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function contentType(path) {
  switch (extname(path)) {
    case ".html": return "text/html; charset=utf-8";
    case ".js": return "text/javascript; charset=utf-8";
    case ".css": return "text/css; charset=utf-8";
    case ".png": return "image/png";
    case ".svg": return "image/svg+xml";
    default: return "application/octet-stream";
  }
}

async function fileExists(path) {
  try {
    const info = await stat(path);
    return info.isFile();
  } catch {
    return false;
  }
}

function findBrowser() {
  const explicit = args.get("browser");
  const candidates = [
    explicit,
    "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
    "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
    "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
  ].filter(Boolean);
  return candidates.find((path) => existsSync(path));
}

function wait(ms) {
  return new Promise((resolveWait) => setTimeout(resolveWait, ms));
}

function paethPredictor(left, up, upLeft) {
  const p = left + up - upLeft;
  const pa = Math.abs(p - left);
  const pb = Math.abs(p - up);
  const pc = Math.abs(p - upLeft);
  if (pa <= pb && pa <= pc) return left;
  if (pb <= pc) return up;
  return upLeft;
}

function parsePng(buffer) {
  const signature = buffer.subarray(0, 8).toString("hex");
  assert(signature === "89504e470d0a1a0a", "screenshot is not a PNG");

  let offset = 8;
  let width = 0;
  let height = 0;
  let bitDepth = 0;
  let colorType = 0;
  const idat = [];
  while (offset < buffer.length) {
    const length = buffer.readUInt32BE(offset);
    const type = buffer.subarray(offset + 4, offset + 8).toString("ascii");
    const data = buffer.subarray(offset + 8, offset + 8 + length);
    offset += 12 + length;

    if (type === "IHDR") {
      width = data.readUInt32BE(0);
      height = data.readUInt32BE(4);
      bitDepth = data[8];
      colorType = data[9];
      const interlace = data[12];
      assert(bitDepth === 8, `unsupported PNG bit depth ${bitDepth}`);
      assert(interlace === 0, "interlaced PNG screenshots are unsupported");
      assert(colorType === 2 || colorType === 6, `unsupported PNG color type ${colorType}`);
    } else if (type === "IDAT") {
      idat.push(data);
    } else if (type === "IEND") {
      break;
    }
  }

  const channels = colorType === 6 ? 4 : 3;
  const stride = width * channels;
  const inflated = inflateSync(Buffer.concat(idat));
  const pixels = Buffer.alloc(width * height * channels);
  let src = 0;
  for (let y = 0; y < height; y += 1) {
    const filter = inflated[src++];
    const row = pixels.subarray(y * stride, (y + 1) * stride);
    const previous = y > 0 ? pixels.subarray((y - 1) * stride, y * stride) : null;
    for (let x = 0; x < stride; x += 1) {
      const raw = inflated[src++];
      const left = x >= channels ? row[x - channels] : 0;
      const up = previous ? previous[x] : 0;
      const upLeft = previous && x >= channels ? previous[x - channels] : 0;
      if (filter === 0) row[x] = raw;
      else if (filter === 1) row[x] = (raw + left) & 255;
      else if (filter === 2) row[x] = (raw + up) & 255;
      else if (filter === 3) row[x] = (raw + Math.floor((left + up) / 2)) & 255;
      else if (filter === 4) row[x] = (raw + paethPredictor(left, up, upLeft)) & 255;
      else throw new Error(`unsupported PNG row filter ${filter}`);
    }
  }
  return { width, height, channels, pixels };
}

function analyzePng(buffer) {
  const image = parsePng(buffer);
  const colors = new Set();
  let minLuma = 255;
  let maxLuma = 0;
  let darkPixels = 0;
  let brightPixels = 0;
  let sampledPixels = 0;
  const pixelCount = image.width * image.height;
  const step = Math.max(1, Math.floor(pixelCount / 20000));
  for (let i = 0; i < pixelCount; i += step) {
    const base = i * image.channels;
    const r = image.pixels[base];
    const g = image.pixels[base + 1];
    const b = image.pixels[base + 2];
    const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    minLuma = Math.min(minLuma, luma);
    maxLuma = Math.max(maxLuma, luma);
    if (luma < 20) darkPixels += 1;
    if (luma > 45) brightPixels += 1;
    colors.add(`${r >> 4},${g >> 4},${b >> 4}`);
    sampledPixels += 1;
  }
  return {
    width: image.width,
    height: image.height,
    sampledPixels,
    lumaRange: Number((maxLuma - minLuma).toFixed(2)),
    colorBuckets: colors.size,
    brightRatio: Number((brightPixels / sampledPixels).toFixed(4)),
    darkRatio: Number((darkPixels / sampledPixels).toFixed(4)),
  };
}

function createStaticServer(root) {
  return createServer(async (request, response) => {
    try {
      const url = new URL(request.url ?? "/", "http://127.0.0.1");
      const requested = url.pathname === "/" ? "/index.html" : url.pathname;
      const filePath = normalize(join(root, decodeURIComponent(requested)));
      if (!filePath.startsWith(root)) {
        response.writeHead(403);
        response.end("Forbidden");
        return;
      }
      const finalPath = await fileExists(filePath) ? filePath : join(root, "index.html");
      const bytes = await readFile(finalPath);
      response.writeHead(200, { "Content-Type": contentType(finalPath) });
      response.end(bytes);
    } catch (error) {
      response.writeHead(500);
      response.end(String(error?.message ?? error));
    }
  });
}

async function listen(server) {
  await new Promise((resolveListen) => server.listen(0, "127.0.0.1", resolveListen));
  return server.address().port;
}

async function reserveFreePort() {
  const server = createServer();
  await new Promise((resolveListen, rejectListen) => {
    server.once("error", rejectListen);
    server.listen(0, "127.0.0.1", resolveListen);
  });
  const port = server.address().port;
  await new Promise((resolveClose) => server.close(resolveClose));
  return port;
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${url} returned ${response.status}`);
  }
  return response.json();
}

async function waitForPage(port, startedAt) {
  let lastError = "";
  while (Date.now() - startedAt < timeoutMs) {
    try {
      const pages = await fetchJson(`http://127.0.0.1:${port}/json/list`);
      const page = pages.find((item) => item.type === "page" && item.webSocketDebuggerUrl);
      if (page) {
        return page;
      }
    } catch (error) {
      lastError = error.message;
    }
    await wait(100);
  }
  throw new Error(`Timed out waiting for browser debug page. ${lastError}`);
}

class CdpClient {
  constructor(url) {
    this.url = url;
    this.nextId = 1;
    this.pending = new Map();
  }

  async connect() {
    this.socket = new WebSocket(this.url);
    await new Promise((resolveOpen, rejectOpen) => {
      this.socket.addEventListener("open", resolveOpen, { once: true });
      this.socket.addEventListener("error", () => rejectOpen(new Error("CDP WebSocket failed")), { once: true });
    });
    this.socket.addEventListener("message", (event) => {
      const message = JSON.parse(String(event.data));
      if (message.id && this.pending.has(message.id)) {
        const { resolveSend, rejectSend } = this.pending.get(message.id);
        this.pending.delete(message.id);
        if (message.error) {
          rejectSend(new Error(message.error.message));
        } else {
          resolveSend(message.result);
        }
      }
    });
  }

  send(method, params = {}) {
    const id = this.nextId++;
    this.socket.send(JSON.stringify({ id, method, params }));
    return new Promise((resolveSend, rejectSend) => {
      this.pending.set(id, { resolveSend, rejectSend });
    });
  }

  close() {
    this.socket?.close();
  }
}

async function waitForExpression(cdp, expression, startedAt) {
  while (Date.now() - startedAt < timeoutMs) {
    const result = await cdp.send("Runtime.evaluate", {
      expression,
      returnByValue: true,
      awaitPromise: true,
    });
    if (result.result?.value) {
      return result.result.value;
    }
    await wait(100);
  }
  throw new Error(`Timed out waiting for expression: ${expression}`);
}

async function waitForState(cdp, predicate, description, startedAt) {
  let lastState;
  while (Date.now() - startedAt < timeoutMs) {
    lastState = await evaluate(cdp, "window.__avbdViewer?.getState?.()");
    if (lastState && predicate(lastState)) {
      return lastState;
    }
    await wait(100);
  }
  throw new Error(`Timed out waiting for ${description}. Last state: ${JSON.stringify(lastState)}`);
}

async function evaluate(cdp, expression) {
  const result = await cdp.send("Runtime.evaluate", {
    expression,
    returnByValue: true,
    awaitPromise: true,
  });
  if (result.exceptionDetails) {
    throw new Error(result.exceptionDetails.text ?? "Runtime evaluation failed");
  }
  return result.result?.value;
}

async function main() {
  assert(await fileExists(join(distRoot, "index.html")), "viewer/dist/index.html is missing; run npm run build first");
  assert(typeof WebSocket === "function", "Node.js global WebSocket is unavailable");

  const browserPath = findBrowser();
  assert(browserPath, "Could not find Microsoft Edge or Google Chrome. Pass --browser <path>.");

  const server = createStaticServer(distRoot);
  const appPort = await listen(server);
  const debugPort = await reserveFreePort();
  const userDataDir = join(tmpdir(), `avbd-viewer-browser-smoke-${process.pid}`);
  const query = new URLSearchParams({ sample });
  if (bridgeUrl) {
    query.set("ws", bridgeUrl);
  }
  const url = `http://127.0.0.1:${appPort}/?${query.toString()}`;

  const browser = spawn(browserPath, [
    "--headless=new",
    "--disable-background-networking",
    "--disable-default-apps",
    "--disable-extensions",
    "--disable-sync",
    "--hide-scrollbars",
    `--remote-debugging-port=${debugPort}`,
    `--user-data-dir=${userDataDir}`,
    "--window-size=1280,720",
    url,
  ], { stdio: "ignore" });

  const startedAt = Date.now();
  let cdp;
  try {
    const page = await waitForPage(debugPort, startedAt);
    cdp = new CdpClient(page.webSocketDebuggerUrl);
    await cdp.connect();
    await cdp.send("Page.enable");
    await cdp.send("Runtime.enable");
    await waitForExpression(cdp, "Boolean(window.__avbdViewer)", startedAt);
    const state = expectNative
      ? await waitForState(
          cdp,
          (candidate) => candidate.mode === "Native"
            && candidate.nativeConnected
            && candidate.frame > 0
            && candidate.scene !== sample
            && candidate.bodyCount > 0,
          "native bridge snapshot",
          startedAt,
        )
      : await waitForState(cdp, (candidate) => candidate.mode === "Sample" && candidate.bodyCount > 0, "sample scene render", startedAt);
    assert(state.mode === (expectNative ? "Native" : "Sample"), `expected ${expectNative ? "Native" : "Sample"} mode, got ${state.mode}`);
    assert(state.bodyCount > 0, "viewer rendered no bodies");
    assert(state.batchCount > 0, "viewer rendered no batches");
    assert(Object.keys(state.shapeCounts ?? {}).length > 0, "viewer reported no shape counts");
    for (const shape of expectedShapes) {
      assert((state.shapeCounts?.[shape] ?? 0) > 0, `expected rendered shape '${shape}', got ${JSON.stringify(state.shapeCounts)}`);
    }

    const canvasReady = await evaluate(cdp, "(() => { const c = document.getElementById('renderer-canvas'); return Boolean(c && c.width > 0 && c.height > 0); })()");
    assert(canvasReady, "renderer canvas is not sized");

    const clip = await evaluate(cdp, "(() => { const c = document.getElementById('renderer-canvas'); const r = c.getBoundingClientRect(); return { x: Math.max(0, r.left + r.width * 0.18), y: Math.max(0, r.top + r.height * 0.12), width: Math.max(1, r.width * 0.56), height: Math.max(1, r.height * 0.66), scale: 1 }; })()");
    const screenshot = await cdp.send("Page.captureScreenshot", { format: "png", fromSurface: true, clip });
    const screenshotBytes = Buffer.from(screenshot.data, "base64");
    await writeFile(screenshotPath, screenshotBytes);
    const screenshotStats = analyzePng(screenshotBytes);
    assert(screenshotStats.brightRatio > 0.02, `render screenshot appears too dark: ${JSON.stringify(screenshotStats)}`);
    assert(screenshotStats.colorBuckets >= 3, `render screenshot has too little color variation: ${JSON.stringify(screenshotStats)}`);
    console.log(JSON.stringify({
      ok: true,
      url,
      browser: browserPath,
      screenshot: screenshotPath,
      screenshotStats,
      state,
    }, null, 2));
  } finally {
    cdp?.close();
    browser.kill();
    server.close();
    await wait(300);
    await rm(userDataDir, { recursive: true, force: true }).catch(() => {});
  }
}

main().catch((error) => fail(error.message));
