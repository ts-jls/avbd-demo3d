import { makeSampleSnapshot, SHAPES } from "../src/samples.js";

const sampleNames = ["pyramid", "mixed", "dense"];
const supportedShapes = new Set(Object.values(SHAPES));

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function validateBody(body, index, sceneName) {
  assert(Number.isInteger(body.id), `${sceneName} body ${index} has invalid id`);
  assert(supportedShapes.has(body.shape), `${sceneName} body ${index} has unsupported shape ${body.shape}`);
  assert(Array.isArray(body.position) && body.position.length === 3, `${sceneName} body ${index} has invalid position`);
  assert(Array.isArray(body.rotation) && body.rotation.length === 4, `${sceneName} body ${index} has invalid rotation`);
  assert(Array.isArray(body.size) && body.size.length === 3, `${sceneName} body ${index} has invalid size`);
  assert(typeof body.dynamic === "boolean", `${sceneName} body ${index} has invalid dynamic flag`);
  assert(Number.isFinite(body.material), `${sceneName} body ${index} has invalid material`);
}

const summary = [];
for (const name of sampleNames) {
  const snapshot = makeSampleSnapshot(name, 1);
  assert(snapshot.type === "snapshot", `${name} snapshot has invalid type`);
  assert(snapshot.version === 1, `${name} snapshot has invalid version`);
  assert(Array.isArray(snapshot.bodies) && snapshot.bodies.length > 0, `${name} snapshot has no bodies`);
  snapshot.bodies.forEach((body, index) => validateBody(body, index, name));

  const counts = {};
  for (const body of snapshot.bodies) {
    counts[body.shape] = (counts[body.shape] ?? 0) + 1;
  }
  summary.push({ scene: name, bodies: snapshot.bodies.length, shapes: counts });
}

console.log(JSON.stringify({ ok: true, samples: summary }, null, 2));
