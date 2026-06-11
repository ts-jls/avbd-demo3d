// Verifies the bridge reassembles fragmented client frames: speaks raw
// WebSocket over TCP, sends a large importMesh command split into many
// masked fragments (like a browser does), and expects a meshImport reply.
import net from "node:net";
import crypto from "node:crypto";

const PORT = Number(process.argv[2] ?? 8765);
const FRAGMENT = 32 * 1024;

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

function frame(opcode, fin, payload) {
  const mask = crypto.randomBytes(4);
  const masked = Buffer.from(payload);
  for (let i = 0; i < masked.length; i++) masked[i] ^= mask[i & 3];
  let header;
  if (masked.length < 126) {
    header = Buffer.from([(fin ? 0x80 : 0) | opcode, 0x80 | masked.length]);
  } else if (masked.length <= 0xffff) {
    header = Buffer.alloc(4);
    header[0] = (fin ? 0x80 : 0) | opcode;
    header[1] = 0x80 | 126;
    header.writeUInt16BE(masked.length, 2);
  } else {
    header = Buffer.alloc(10);
    header[0] = (fin ? 0x80 : 0) | opcode;
    header[1] = 0x80 | 127;
    header.writeBigUInt64BE(BigInt(masked.length), 2);
  }
  return Buffer.concat([header, mask, masked]);
}

// Dense torus so the JSON comfortably exceeds the old 64KB inbound cap.
const torus = makeTorus(0.9, 0.4, 64, 32);
const message = Buffer.from(
  JSON.stringify({
    type: "command",
    command: "importMesh",
    name: "fragtest",
    mode: "soft",
    spacing: 0.25,
    scale: 1,
    position: [0, 0, 4],
    vertices: torus.verts,
    triangles: torus.tris,
  }),
  "utf8",
);
console.error(`message bytes: ${message.length}, fragments: ${Math.ceil(message.length / FRAGMENT)}`);

const sock = net.connect(PORT, "127.0.0.1");
const key = crypto.randomBytes(16).toString("base64");
let upgraded = false;
let buffer = Buffer.alloc(0);

const timer = setTimeout(() => {
  console.error(JSON.stringify({ ok: false, error: "timeout" }));
  process.exit(1);
}, 20000);

sock.on("connect", () => {
  sock.write(
    `GET / HTTP/1.1\r\nHost: 127.0.0.1:${PORT}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: ${key}\r\nSec-WebSocket-Version: 13\r\n\r\n`,
  );
});

sock.on("data", (chunk) => {
  buffer = Buffer.concat([buffer, chunk]);
  if (!upgraded) {
    const end = buffer.indexOf("\r\n\r\n");
    if (end < 0) return;
    upgraded = true;
    buffer = buffer.subarray(end + 4);
    // Send the import as a fragmented message.
    for (let off = 0; off < message.length; off += FRAGMENT) {
      const part = message.subarray(off, Math.min(off + FRAGMENT, message.length));
      const first = off === 0;
      const last = off + FRAGMENT >= message.length;
      sock.write(frame(first ? 0x1 : 0x0, last, part));
    }
  }
  // Scan unmasked server frames for the meshImport reply.
  const text = buffer.toString("latin1");
  const at = text.indexOf('"type":"meshImport"');
  if (at >= 0) {
    clearTimeout(timer);
    const ok = text.indexOf('"ok":true', at) >= 0 && text.indexOf('"particleIds"', at) >= 0;
    console.log(JSON.stringify({ ok, sawReply: true }));
    process.exit(ok ? 0 : 1);
  }
  if (buffer.length > 32 * 1024 * 1024) buffer = buffer.subarray(buffer.length - 1024 * 1024);
});

sock.on("error", (e) => {
  clearTimeout(timer);
  console.error(JSON.stringify({ ok: false, error: e.message }));
  process.exit(1);
});
sock.on("close", () => {
  clearTimeout(timer);
  console.error(JSON.stringify({ ok: false, error: "socket closed before reply" }));
  process.exit(1);
});
