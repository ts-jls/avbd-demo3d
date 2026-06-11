import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const viewerRoot = path.resolve(__dirname, "..");
const repoRoot = path.resolve(viewerRoot, "..");
const npmBin = process.platform === "win32" ? "npm.cmd" : "npm";

const args = new Set(process.argv.slice(2));
const skipBuild = args.has("--skip-build");
const includeBrowserSmoke = args.has("--browser-smoke");

function run(label, command, commandArgs, cwd = viewerRoot) {
  console.log(`\n== ${label}`);
  console.log(`${command} ${commandArgs.join(" ")}`);
  const result = spawnSync(command, commandArgs, {
    cwd,
    stdio: "inherit",
    shell: process.platform === "win32" && command.endsWith(".cmd"),
  });
  if (result.error) {
    throw result.error;
  }
  if (result.status !== 0) {
    throw new Error(`${label} failed with exit code ${result.status}`);
  }
}

const webgpuExe = path.join(repoRoot, "build-webgpu-dawn-nmake3", "avbd_headless_server.exe");
if (!fs.existsSync(webgpuExe)) {
  throw new Error(`Missing WebGPU headless executable: ${webgpuExe}`);
}

run("headless benchmark syntax", "node", ["--check", path.join("scripts", "headless-benchmark.mjs")]);
JSON.parse(fs.readFileSync(path.join(viewerRoot, "package.json"), "utf8"));
console.log("package json ok");

if (!skipBuild) {
  run("three.js viewer build", npmBin, ["run", "build"]);
}

run("AVBD GPU solver 5k sphere pour", npmBin, ["run", "benchmark:gpu-avbd-5k"]);
run("AVBD GPU solver 20k sphere pour", npmBin, ["run", "benchmark:gpu-avbd-20k"]);
run("AVBD GPU solver soft body", npmBin, ["run", "benchmark:gpu-avbd-softbody"]);
run("AVBD GPU solver vs CPU reference", npmBin, ["run", "benchmark:gpu-avbd-compare"]);

if (includeBrowserSmoke) {
  run("sample viewer smoke", npmBin, ["run", "smoke:samples"]);
}

console.log("\nGPU playable checkpoint passed.");
