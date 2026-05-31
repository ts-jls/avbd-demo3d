import { readFile } from "node:fs/promises";

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

const [html, main, packageJson] = await Promise.all([
  readFile(new URL("../index.html", import.meta.url), "utf8"),
  readFile(new URL("../src/main.js", import.meta.url), "utf8"),
  readFile(new URL("../package.json", import.meta.url), "utf8"),
]);

const requiredTestIds = [
  "renderer-canvas",
  "viewer-mode",
  "connection-status",
  "scene-name",
  "frame-number",
  "body-count",
  "batch-count",
  "shape-counts",
  "render-fps",
  "dropped-frames",
  "snapshot-size",
  "bridge-url",
  "sample-scene",
];

for (const testId of requiredTestIds) {
  assert(html.includes(`data-testid="${testId}"`), `Missing data-testid="${testId}"`);
}

const requiredMainHooks = [
  "window.__avbdViewer",
  "getState: viewerState",
  "applySnapshot(snapshot)",
  "setSampleScene(name)",
  "reconnect: connectWebSocket",
  "avbd-viewer-ready",
  "shapeCounts",
  "batchCount",
  "nativeConnected",
  "reconnectAttempts",
];

for (const hook of requiredMainHooks) {
  assert(main.includes(hook), `Missing viewer hook: ${hook}`);
}

const pkg = JSON.parse(packageJson);
for (const script of ["build", "smoke:samples", "smoke:bridge", "smoke:contracts", "smoke:browser", "smoke:browser:native"]) {
  assert(pkg.scripts?.[script], `Missing npm script: ${script}`);
}

console.log(JSON.stringify({
  ok: true,
  testIds: requiredTestIds.length,
  hooks: requiredMainHooks.length,
  scripts: ["build", "smoke:samples", "smoke:bridge", "smoke:contracts", "smoke:browser", "smoke:browser:native"],
}, null, 2));
