export const SHAPES = Object.freeze({
  BOX: "box",
  SPHERE: "sphere",
  CAPSULE: "capsule",
  CYLINDER: "cylinder",
});

const staticBody = {
  dynamic: false,
  material: 0,
  rotation: [0, 0, 0, 1],
};

const dynamicBody = {
  dynamic: true,
  material: 1,
  rotation: [0, 0, 0, 1],
};

function box(id, position, size, overrides = {}) {
  return {
    ...dynamicBody,
    id,
    shape: SHAPES.BOX,
    position,
    size,
    radius: Math.max(size[0], size[1], size[2]) * 0.5,
    halfLength: 0,
    ...overrides,
  };
}

function sphere(id, position, radius, overrides = {}) {
  const diameter = radius * 2;
  return {
    ...dynamicBody,
    id,
    shape: SHAPES.SPHERE,
    position,
    size: [diameter, diameter, diameter],
    radius,
    halfLength: 0,
    ...overrides,
  };
}

function capsule(id, position, radius, halfLength, overrides = {}) {
  return {
    ...dynamicBody,
    id,
    shape: SHAPES.CAPSULE,
    position,
    size: [radius * 2, radius * 2, halfLength * 2 + radius * 2],
    radius,
    halfLength,
    ...overrides,
  };
}

function cylinder(id, position, radius, halfLength, overrides = {}) {
  return {
    ...dynamicBody,
    id,
    shape: SHAPES.CYLINDER,
    position,
    size: [radius * 2, radius * 2, halfLength * 2],
    radius,
    halfLength,
    ...overrides,
  };
}

function pyramidBodies() {
  const bodies = [
    box(0, [0, 0, -0.12], [34, 34, 0.24], { ...staticBody, material: 2 }),
  ];
  const rows = 12;
  let id = 1;
  for (let z = 0; z < rows; z += 1) {
    const count = rows - z;
    for (let x = 0; x < count; x += 1) {
      bodies.push(box(id, [(x - count * 0.5) * 1.08, 0, 0.28 + z * 0.55], [1, 0.55, 0.5]));
      id += 1;
    }
  }
  return bodies;
}

function mixedBodies() {
  return [
    box(0, [0, 0, -0.12], [24, 18, 0.24], { ...staticBody, material: 2 }),
    box(1, [-4, -2, 0.6], [1.4, 1.2, 1.2], { material: 1 }),
    sphere(2, [-1.5, -2, 0.75], 0.65, { material: 3 }),
    cylinder(3, [1.2, -2, 0.8], 0.55, 0.75, { material: 4, rotation: [0.382683, 0, 0, 0.92388] }),
    capsule(4, [4, -2, 0.9], 0.42, 0.85, { material: 5, rotation: [0, 0.382683, 0, 0.92388] }),
    sphere(5, [-3, 1.5, 0.55], 0.45, { material: 6 }),
    box(6, [0, 1.5, 0.55], [0.8, 1.8, 0.8], { material: 7 }),
    cylinder(7, [3, 1.5, 0.55], 0.42, 0.55, { material: 8 }),
  ];
}

function denseBodies() {
  const bodies = [
    box(0, [0, 0, -0.12], [42, 42, 0.24], { ...staticBody, material: 2 }),
  ];
  let id = 1;
  for (let z = 0; z < 8; z += 1) {
    for (let y = 0; y < 8; y += 1) {
      for (let x = 0; x < 8; x += 1) {
        bodies.push(box(id, [(x - 3.5) * 0.72, (y - 3.5) * 0.72, 0.28 + z * 0.72], [0.58, 0.58, 0.58], { material: 1 + ((x + y + z) % 4) }));
        id += 1;
      }
    }
  }
  return bodies;
}

export function makeSampleSnapshot(name, frame = 0) {
  const builders = {
    pyramid: pyramidBodies,
    mixed: mixedBodies,
    dense: denseBodies,
  };
  const key = builders[name] ? name : "pyramid";
  return {
    type: "snapshot",
    version: 1,
    frame,
    scene: key,
    bodies: builders[key](),
  };
}
