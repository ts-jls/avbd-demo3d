/*
 * WebGPU AVBD backend.
 *
 * This backend keeps broadphase/narrowphase/warmstarting on the CPU
 * (Solver::prepareStep) and runs the AVBD iteration loop on the GPU:
 *
 *   iterations x ( per-color primal body solve  +  contact dual  +  joint dual )
 *
 * The primal kernel is a direct port of the CPU reference: it stamps every
 * attached force's Hessian blocks into a 6x6 system and solves it with LDL^T.
 * Bodies are greedily graph-colored so no two bodies sharing a constraint
 * solve concurrently (Gauss-Seidel across colors, Jacobi within a color).
 * Dual kernels update lambda/penalty per constraint, exactly as the CPU
 * Manifold::updateDual / Joint::updateDual do.
 *
 * One upload at the start of the step, one readback at the end (positions
 * plus dual state so next frame's CPU warmstart matches the reference).
 */

#include "avbd_gpu_solver.h"

#include "webgpu_device.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace
{

struct CpuFallbackAvbdBackend : PhysicsBackend
{
    const char *name() const override { return "WebGPU AVBD (CPU fallback)"; }
    void step(Solver &solver) override { solver.stepCpuReference(); }
};

AvbdGpuSolverStats g_avbdGpuStats = {};

} // namespace

const AvbdGpuSolverStats &avbdGpuSolverStats()
{
    return g_avbdGpuStats;
}

#if AVBD_ENABLE_WEBGPU && AVBD_HAS_DAWN

namespace
{

using Clock = std::chrono::high_resolution_clock;

inline float elapsedMsAvbd(Clock::time_point begin, Clock::time_point end)
{
    return std::chrono::duration<float, std::milli>(end - begin).count();
}

// Mirrors manifold.cpp's file-local isCylinderCapContact.
inline bool isCylinderCapContactKey(int featureKey)
{
    return (featureKey & 0xFFFFFF00) == 0x05000000 || (featureKey & 0xFFFFFF00) == 0x06000000;
}

// Sentinel encoding for "infinite" stiffness/fracture in f32 GPU buffers.
// PENALTY_MAX is 1e10, so min(x, 1e30) behaves exactly like min(x, INFINITY).
const float GPU_INF_SENTINEL = 1.0e30f;

const uint32_t ADJ_KIND_CONTACT = 0u;
const uint32_t ADJ_KIND_JOINT = 1u;
const uint32_t ADJ_KIND_SPRING = 2u;

inline uint32_t adjEntryEncode(uint32_t kind, bool isBodyA, uint32_t index)
{
    return (kind << 30) | ((isBodyA ? 1u : 0u) << 29) | (index & 0x1FFFFFFFu);
}

// GPU-side structs. All members are vec4-sized so the C++ and WGSL layouts
// match without padding surprises.
struct GpuAvbdBody
{
    float posLin[4];     // xyz position, w mass
    float posAng[4];     // quat xyzw
    float initialLin[4]; // xyz, w unused
    float initialAng[4];
    float inertialLin[4];
    float inertialAng[4];
    float moment[4];   // xyz inertia diagonal, w shape type
    float shape[4];    // x radius, y halfLength
    float velocity[4]; // xyz linear velocity at q- (for the swept-sphere test)
    float halfSize[4]; // xyz box half extents
};

struct GpuAvbdContact
{
    uint32_t header[4]; // bodyA, bodyB, flags (bit0 cylinder cap), pad
    float n[4];         // basis row 0, w = friction
    float t1[4];        // basis row 1
    float t2[4];        // basis row 2
    float rA[4];
    float rB[4];
    float c0[4];
    float lambda[4]; // w = stick flag
    float penalty[4];
};

struct GpuAvbdJoint
{
    uint32_t header[4]; // bodyA, bodyB, flags (bit0 infLin, bit1 infAng, bit2 broken), pad
    float rA[4];        // w = stiffnessLin
    float rB[4];        // w = stiffnessAng
    float c0Lin[4];     // w = torqueArm
    float c0Ang[4];     // w = fracture
    float lambdaLin[4];
    float lambdaAng[4];
    float penaltyLin[4];
    float penaltyAng[4];
};

struct GpuAvbdSpring
{
    uint32_t header[4]; // bodyA, bodyB, pad, pad
    float rA[4];        // w = rest length
    float rB[4];        // w = stiffness
};

struct GpuAvbdParams
{
    float dt;
    float alpha;
    float betaLin;
    float betaAng;
    uint32_t bodyCount;
    uint32_t contactCount; // CPU manifold contacts + GPU sphere-pair contacts
    uint32_t jointCount;
    uint32_t springCount;
    float gamma;
    float collisionMargin;
    float padF0;
    float padF1;
    uint32_t spherePairCount;
    uint32_t spherePairBase; // u32 offset of pair records inside adjEntries
    uint32_t cpuContactBase; // first contact slot owned by the GPU narrowphase
    uint32_t padU0;
};

const uint32_t COLOR_SLOT_STRIDE = 256; // dynamic uniform offset alignment
const uint32_t MAX_COLOR_SLOTS = 256;

const char *AVBD_SOLVER_WGSL = R"(
struct Body {
    posLin : vec4<f32>,
    posAng : vec4<f32>,
    initialLin : vec4<f32>,
    initialAng : vec4<f32>,
    inertialLin : vec4<f32>,
    inertialAng : vec4<f32>,
    moment : vec4<f32>,
    shape : vec4<f32>,
    velocity : vec4<f32>,
    halfSize : vec4<f32>,
};

struct ContactC {
    header : vec4<u32>,
    n : vec4<f32>,
    t1 : vec4<f32>,
    t2 : vec4<f32>,
    rA : vec4<f32>,
    rB : vec4<f32>,
    c0 : vec4<f32>,
    lambda : vec4<f32>,
    penalty : vec4<f32>,
};

struct JointC {
    header : vec4<u32>,
    rA : vec4<f32>,
    rB : vec4<f32>,
    c0Lin : vec4<f32>,
    c0Ang : vec4<f32>,
    lambdaLin : vec4<f32>,
    lambdaAng : vec4<f32>,
    penaltyLin : vec4<f32>,
    penaltyAng : vec4<f32>,
};

struct SpringC {
    header : vec4<u32>,
    rA : vec4<f32>,
    rB : vec4<f32>,
};

struct Params {
    dtAlphaBeta : vec4<f32>, // dt, alpha, betaLin, betaAng
    counts : vec4<u32>,      // bodyCount, contactCount, jointCount, springCount
    misc : vec4<f32>,        // gamma, collisionMargin, unused, unused
    sphere : vec4<u32>,      // spherePairCount, spherePairBase, cpuContactBase, unused
};

struct ColorParams {
    offsetCount : vec4<u32>, // x offset into solvedBodies, y count
};

@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var<storage, read_write> bodies : array<Body>;
@group(0) @binding(2) var<storage, read_write> contacts : array<ContactC>;
@group(0) @binding(3) var<storage, read_write> joints : array<JointC>;
@group(0) @binding(4) var<storage, read> springs : array<SpringC>;
@group(0) @binding(5) var<storage, read> adjOffsets : array<u32>;
@group(0) @binding(6) var<storage, read> adjEntries : array<u32>;
@group(0) @binding(7) var<storage, read> solvedBodies : array<u32>;
@group(0) @binding(8) var<storage, read_write> sphereState : array<vec4<f32>>;
@group(1) @binding(0) var<uniform> colorParams : ColorParams;

const PENALTY_MIN : f32 = 1.0;
const PENALTY_MAX : f32 = 10000000000.0;
const STICK_THRESH : f32 = 0.00001;

// ---- 3x3 matrices stored as rows, matching maths.h conventions ----

struct M3 {
    r0 : vec3<f32>,
    r1 : vec3<f32>,
    r2 : vec3<f32>,
};

fn m3zero() -> M3 {
    return M3(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
}

fn m3diag(d : vec3<f32>) -> M3 {
    return M3(vec3<f32>(d.x, 0.0, 0.0), vec3<f32>(0.0, d.y, 0.0), vec3<f32>(0.0, 0.0, d.z));
}

fn m3v(m : M3, v : vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(m.r0, v), dot(m.r1, v), dot(m.r2, v));
}

fn m3t(m : M3) -> M3 {
    return M3(
        vec3<f32>(m.r0.x, m.r1.x, m.r2.x),
        vec3<f32>(m.r0.y, m.r1.y, m.r2.y),
        vec3<f32>(m.r0.z, m.r1.z, m.r2.z));
}

fn m3m(a : M3, b : M3) -> M3 {
    let bt = m3t(b);
    return M3(
        vec3<f32>(dot(a.r0, bt.r0), dot(a.r0, bt.r1), dot(a.r0, bt.r2)),
        vec3<f32>(dot(a.r1, bt.r0), dot(a.r1, bt.r1), dot(a.r1, bt.r2)),
        vec3<f32>(dot(a.r2, bt.r0), dot(a.r2, bt.r1), dot(a.r2, bt.r2)));
}

fn m3add(a : M3, b : M3) -> M3 {
    return M3(a.r0 + b.r0, a.r1 + b.r1, a.r2 + b.r2);
}

fn m3scale(m : M3, s : f32) -> M3 {
    return M3(m.r0 * s, m.r1 * s, m.r2 * s);
}

fn m3skew(r : vec3<f32>) -> M3 {
    return M3(
        vec3<f32>(0.0, -r.z, r.y),
        vec3<f32>(r.z, 0.0, -r.x),
        vec3<f32>(-r.y, r.x, 0.0));
}

fn m3outer(a : vec3<f32>, b : vec3<f32>) -> M3 {
    return M3(b * a.x, b * a.y, b * a.z);
}

fn m3col(m : M3, i : i32) -> vec3<f32> {
    if (i == 0) { return vec3<f32>(m.r0.x, m.r1.x, m.r2.x); }
    if (i == 1) { return vec3<f32>(m.r0.y, m.r1.y, m.r2.y); }
    return vec3<f32>(m.r0.z, m.r1.z, m.r2.z);
}

fn m3diagonalize(m : M3) -> M3 {
    return m3diag(vec3<f32>(length(m3col(m, 0)), length(m3col(m, 1)), length(m3col(m, 2))));
}

// ---- quaternion helpers matching maths.h ----

fn qmul(a : vec4<f32>, b : vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z);
}

fn qconj(q : vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.xyz, q.w);
}

fn qrot(q : vec4<f32>, v : vec3<f32>) -> vec3<f32> {
    let u = q.xyz;
    let t = cross(u, v) * 2.0;
    return v + t * q.w + cross(u, t);
}

// maths.h operator-(quat, quat): 2 * vec(a * inverse(b)); quats stay unit.
fn qsub(a : vec4<f32>, b : vec4<f32>) -> vec3<f32> {
    return qmul(a, qconj(b)).xyz * 2.0;
}

// maths.h operator+(quat, float3): normalize(a + quat(b,0)*a*0.5)
fn qaddv(q : vec4<f32>, w : vec3<f32>) -> vec4<f32> {
    return normalize(q + qmul(vec4<f32>(w, 0.0), q) * 0.5);
}

// ---- contact offset (port of manifold.cpp contactOffsetWorld) ----
// shape type: 0 box, 1 sphere, 2 capsule, 3 cylinder

fn contactOffsetWorld(b : Body, localOffset : vec3<f32>, n : vec3<f32>, isA : bool, capContact : bool) -> vec3<f32> {
    let st = b.moment.w;
    let radius = b.shape.x;
    let halfLen = b.shape.y;
    if (st == 1.0) {
        return n * select(radius, -radius, isA);
    }
    if (st == 2.0) {
        let axis = qrot(b.posAng, vec3<f32>(0.0, 0.0, 1.0));
        let nLocal = qrot(qconj(b.posAng), n);
        var centerZ = select(localOffset.z - nLocal.z * radius, localOffset.z + nLocal.z * radius, isA);
        centerZ = clamp(centerZ, -halfLen, halfLen);
        return axis * centerZ + n * select(radius, -radius, isA);
    }
    if (st == 3.0) {
        let supportW = n * select(1.0, -1.0, isA);
        let supportL = qrot(qconj(b.posAng), supportW);
        let storedRadial = sqrt(localOffset.x * localOffset.x + localOffset.y * localOffset.y);
        if (capContact) {
            var local = vec3<f32>(0.0);
            if (storedRadial > 1.0e-6) {
                local.x = localOffset.x / storedRadial * radius;
                local.y = localOffset.y / storedRadial * radius;
            }
            var capSign : f32;
            if (abs(localOffset.z) > 1.0e-6) {
                capSign = select(-1.0, 1.0, localOffset.z >= 0.0);
            } else {
                capSign = select(-1.0, 1.0, supportL.z >= 0.0);
            }
            local.z = halfLen * capSign;
            return qrot(b.posAng, local);
        }
        let radialLen = sqrt(supportL.x * supportL.x + supportL.y * supportL.y);
        var z : f32;
        if (abs(supportL.z) > 0.15) {
            z = select(-halfLen, halfLen, supportL.z >= 0.0);
        } else {
            z = clamp(localOffset.z, -halfLen, halfLen);
        }
        var local = vec3<f32>(0.0, 0.0, z);
        if (radialLen > 1.0e-6) {
            local.x = supportL.x / radialLen * radius;
            local.y = supportL.y / radialLen * radius;
        } else if (storedRadial > 1.0e-6) {
            local.x = localOffset.x / storedRadial * radius;
            local.y = localOffset.y / storedRadial * radius;
        }
        return qrot(b.posAng, local);
    }
    return qrot(b.posAng, localOffset);
}

// ---- shared contact evaluation (port of Manifold::updatePrimal/updateDual) ----

struct ContactEval {
    C : vec3<f32>,
    F : vec3<f32>,
    rAW : vec3<f32>,
    rBW : vec3<f32>,
    preFs : f32,   // friction magnitude before cone clamp
    bounds : f32,  // friction cone bound
};

fn evalContact(c : ContactC, alpha : f32) -> ContactEval {
    let bA = bodies[c.header.x];
    let bB = bodies[c.header.y];
    let capFlag = (c.header.z & 1u) != 0u;
    let rAW = contactOffsetWorld(bA, c.rA.xyz, c.n.xyz, true, capFlag);
    let rBW = contactOffsetWorld(bB, c.rB.xyz, c.n.xyz, false, capFlag);

    let dqALin = bA.posLin.xyz - bA.initialLin.xyz;
    let dqAAng = qsub(bA.posAng, bA.initialAng);
    let dqBLin = bB.posLin.xyz - bB.initialLin.xyz;
    let dqBAng = qsub(bB.posAng, bB.initialAng);

    let jALin = M3(c.n.xyz, c.t1.xyz, c.t2.xyz);
    let jBLin = m3scale(jALin, -1.0);
    let jAAng = M3(cross(rAW, jALin.r0), cross(rAW, jALin.r1), cross(rAW, jALin.r2));
    let jBAng = M3(cross(rBW, jBLin.r0), cross(rBW, jBLin.r1), cross(rBW, jBLin.r2));

    let C = c.c0.xyz * (1.0 - alpha) + m3v(jALin, dqALin) + m3v(jBLin, dqBLin) + m3v(jAAng, dqAAng) + m3v(jBAng, dqBAng);

    var F = c.penalty.xyz * C + c.lambda.xyz;
    F.x = min(F.x, 0.0);
    let bounds = abs(F.x) * c.n.w;
    let preFs = length(F.yz);
    if (preFs > bounds && preFs > 0.0) {
        F = vec3<f32>(F.x, F.yz * (bounds / preFs));
    }
    return ContactEval(C, F, rAW, rBW, preFs, bounds);
}

)" R"(
// ---- 6x6 LDL^T solve (port of maths.h solve) ----

struct SolveResult {
    lin : vec3<f32>,
    ang : vec3<f32>,
};

fn ldlSolve(aLin : M3, aAng : M3, aCross : M3, bLin : vec3<f32>, bAng : vec3<f32>) -> SolveResult {
    let A11 = aLin.r0.x;
    let A21 = aLin.r1.x;  let A22 = aLin.r1.y;
    let A31 = aLin.r2.x;  let A32 = aLin.r2.y;  let A33 = aLin.r2.z;
    let A41 = aCross.r0.x; let A42 = aCross.r0.y; let A43 = aCross.r0.z; let A44 = aAng.r0.x;
    let A51 = aCross.r1.x; let A52 = aCross.r1.y; let A53 = aCross.r1.z; let A54 = aAng.r1.x; let A55 = aAng.r1.y;
    let A61 = aCross.r2.x; let A62 = aCross.r2.y; let A63 = aCross.r2.z; let A64 = aAng.r2.x; let A65 = aAng.r2.y; let A66 = aAng.r2.z;

    let L21 = A21 / A11;
    let L31 = A31 / A11;
    let L41 = A41 / A11;
    let L51 = A51 / A11;
    let L61 = A61 / A11;

    let D1 = A11;
    let D2 = A22 - L21 * L21 * D1;

    let L32 = (A32 - L21 * L31 * D1) / D2;
    let L42 = (A42 - L21 * L41 * D1) / D2;
    let L52 = (A52 - L21 * L51 * D1) / D2;
    let L62 = (A62 - L21 * L61 * D1) / D2;

    let D3 = A33 - (L31 * L31 * D1 + L32 * L32 * D2);

    let L43 = (A43 - L31 * L41 * D1 - L32 * L42 * D2) / D3;
    let L53 = (A53 - L31 * L51 * D1 - L32 * L52 * D2) / D3;
    let L63 = (A63 - L31 * L61 * D1 - L32 * L62 * D2) / D3;

    let D4 = A44 - (L41 * L41 * D1 + L42 * L42 * D2 + L43 * L43 * D3);

    let L54 = (A54 - L41 * L51 * D1 - L42 * L52 * D2 - L43 * L53 * D3) / D4;
    let L64 = (A64 - L41 * L61 * D1 - L42 * L62 * D2 - L43 * L63 * D3) / D4;

    let D5 = A55 - (L51 * L51 * D1 + L52 * L52 * D2 + L53 * L53 * D3 + L54 * L54 * D4);

    let L65 = (A65 - L51 * L61 * D1 - L52 * L62 * D2 - L53 * L63 * D3 - L54 * L64 * D4) / D5;

    let D6 = A66 - (L61 * L61 * D1 + L62 * L62 * D2 + L63 * L63 * D3 + L64 * L64 * D4 + L65 * L65 * D5);

    let y1 = bLin.x;
    let y2 = bLin.y - L21 * y1;
    let y3 = bLin.z - L31 * y1 - L32 * y2;
    let y4 = bAng.x - L41 * y1 - L42 * y2 - L43 * y3;
    let y5 = bAng.y - L51 * y1 - L52 * y2 - L53 * y3 - L54 * y4;
    let y6 = bAng.z - L61 * y1 - L62 * y2 - L63 * y3 - L64 * y4 - L65 * y5;

    let z1 = y1 / D1;
    let z2 = y2 / D2;
    let z3 = y3 / D3;
    let z4 = y4 / D4;
    let z5 = y5 / D5;
    let z6 = y6 / D6;

    var ang : vec3<f32>;
    var lin : vec3<f32>;
    ang.z = z6;
    ang.y = z5 - L65 * ang.z;
    ang.x = z4 - L54 * ang.y - L64 * ang.z;
    lin.z = z3 - L43 * ang.x - L53 * ang.y - L63 * ang.z;
    lin.y = z2 - L32 * lin.z - L42 * ang.x - L52 * ang.y - L62 * ang.z;
    lin.x = z1 - L21 * lin.y - L31 * lin.z - L41 * ang.x - L51 * ang.y - L61 * ang.z;
    return SolveResult(lin, ang);
}

// geometricStiffnessBallSocket from joint.cpp: diag(-v[k]) with column k += v
fn gsBallSocket(k : i32, v : vec3<f32>) -> M3 {
    var m = m3diag(vec3<f32>(-v[k]));
    if (k == 0) {
        m.r0.x += v.x; m.r1.x += v.y; m.r2.x += v.z;
    } else if (k == 1) {
        m.r0.y += v.x; m.r1.y += v.y; m.r2.y += v.z;
    } else {
        m.r0.z += v.x; m.r1.z += v.y; m.r2.z += v.z;
    }
    return m;
}

// ---- primal kernel: one solved body per invocation ----

@compute @workgroup_size(64)
fn primal(@builtin(global_invocation_id) gid : vec3<u32>) {
    let slot = gid.x;
    if (slot >= colorParams.offsetCount.y) {
        return;
    }
    let bi = solvedBodies[colorParams.offsetCount.x + slot];
    let body = bodies[bi];
    let mass = body.posLin.w;
    let dt = params.dtAlphaBeta.x;
    let alpha = params.dtAlphaBeta.y;
    let invDt2 = 1.0 / (dt * dt);

    var lhsLin = m3diag(vec3<f32>(mass * invDt2));
    var lhsAng = m3diag(body.moment.xyz * invDt2);
    var lhsCross = m3zero();
    var rhsLin = (body.posLin.xyz - body.inertialLin.xyz) * (mass * invDt2);
    var rhsAng = qsub(body.posAng, body.inertialAng) * (body.moment.xyz * invDt2);

    let aStart = adjOffsets[bi];
    let aEnd = adjOffsets[bi + 1u];
    for (var k = aStart; k < aEnd; k = k + 1u) {
        let e = adjEntries[k];
        let kind = e >> 30u;
        let isA = ((e >> 29u) & 1u) == 1u;
        let idx = e & 0x1FFFFFFFu;

        if (kind == 0u) {
            // Contact (Manifold::updatePrimal for a single contact point)
            let c = contacts[idx];
            let ev = evalContact(c, alpha);
            let sgn = select(-1.0, 1.0, isA);
            let jLin = M3(c.n.xyz * sgn, c.t1.xyz * sgn, c.t2.xyz * sgn);
            let rW = select(ev.rBW, ev.rAW, isA);
            let jAng = M3(cross(rW, jLin.r0), cross(rW, jLin.r1), cross(rW, jLin.r2));

            let Kd = m3diag(c.penalty.xyz);
            let jLinT = m3t(jLin);
            let jAngT = m3t(jAng);
            let jAngTk = m3m(jAngT, Kd);

            lhsLin = m3add(lhsLin, m3m(m3m(jLinT, Kd), jLin));
            lhsAng = m3add(lhsAng, m3m(jAngTk, jAng));
            lhsCross = m3add(lhsCross, m3m(jAngTk, jLin));
            rhsLin = rhsLin + m3v(jLinT, ev.F);
            rhsAng = rhsAng + m3v(jAngT, ev.F);
        } else if (kind == 1u) {
            // Joint (Joint::updatePrimal)
            let j = joints[idx];
            let bA = bodies[j.header.x];
            let bB = bodies[j.header.y];
            let infLin = (j.header.z & 1u) != 0u;
            let infAng = (j.header.z & 2u) != 0u;
            let sgn = select(-1.0, 1.0, isA);

            if (dot(j.penaltyLin.xyz, j.penaltyLin.xyz) > 0.0) {
                let rAW = qrot(bA.posAng, j.rA.xyz);
                let rBW = qrot(bB.posAng, j.rB.xyz);
                let Kp = j.penaltyLin.xyz;
                var C = (bA.posLin.xyz + rAW) - (bB.posLin.xyz + rBW);
                if (infLin) {
                    C = C - j.c0Lin.xyz * alpha;
                }
                let F = Kp * C + j.lambdaLin.xyz;

                let jAng = m3skew(select(rBW, -rAW, isA));
                let jAngT = m3t(jAng);
                let jAngTk = m3m(jAngT, m3diag(Kp));

                lhsLin = m3add(lhsLin, m3diag(Kp));
                lhsAng = m3add(lhsAng, m3m(jAngTk, jAng));
                lhsCross = m3add(lhsCross, m3scale(jAngTk, sgn));

                let rG = select(-rBW, rAW, isA);
                let H = m3add(m3add(m3scale(gsBallSocket(0, rG), F.x), m3scale(gsBallSocket(1, rG), F.y)), m3scale(gsBallSocket(2, rG), F.z));
                lhsAng = m3add(lhsAng, m3diagonalize(H));

                rhsLin = rhsLin + F * sgn;
                rhsAng = rhsAng + m3v(jAngT, F);
            }

            if (dot(j.penaltyAng.xyz, j.penaltyAng.xyz) > 0.0) {
                let Ka = j.penaltyAng.xyz;
                let ta = j.c0Lin.w;
                var C = qsub(bA.posAng, bB.posAng) * ta;
                if (infAng) {
                    C = C - j.c0Ang.xyz * alpha;
                }
                let F = Ka * C + j.lambdaAng.xyz;
                lhsAng = m3add(lhsAng, m3diag(Ka * (ta * ta)));
                rhsAng = rhsAng + F * (sgn * ta);
            }
        } else if (kind == 2u) {
            // Spring (Spring::updatePrimal)
            let sp = springs[idx];
            let bA = bodies[sp.header.x];
            let bB = bodies[sp.header.y];
            let pA = bA.posLin.xyz + qrot(bA.posAng, sp.rA.xyz);
            let pB = bB.posLin.xyz + qrot(bB.posAng, sp.rB.xyz);
            let d = pA - pB;
            let dLen = length(d);
            if (dLen > 1.0e-6) {
                let n = d / dLen;
                let stiffness = sp.rB.w;
                let f = stiffness * (dLen - sp.rA.w);
                var jLin : vec3<f32>;
                var jAng : vec3<f32>;
                if (isA) {
                    let rWorld = qrot(bA.posAng, sp.rA.xyz);
                    jLin = n;
                    jAng = cross(rWorld, n);
                } else {
                    let rWorld = qrot(bB.posAng, sp.rB.xyz);
                    jLin = -n;
                    jAng = -cross(rWorld, n);
                }
                lhsLin = m3add(lhsLin, m3scale(m3outer(jLin, jLin), stiffness));
                lhsAng = m3add(lhsAng, m3scale(m3outer(jAng, jAng), stiffness));
                lhsCross = m3add(lhsCross, m3scale(m3outer(jAng, jLin), stiffness));
                rhsLin = rhsLin + jLin * f;
                rhsAng = rhsAng + jAng * f;
            }
        }
    }

    let dx = ldlSolve(lhsLin, lhsAng, lhsCross, -rhsLin, -rhsAng);
    bodies[bi].posLin = vec4<f32>(body.posLin.xyz + dx.lin, mass);
    bodies[bi].posAng = qaddv(body.posAng, dx.ang);
}

)" R"(
// ---- contact dual kernel (Manifold::updateDual per contact) ----

@compute @workgroup_size(64)
fn contactDual(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    if (i >= params.counts.y) {
        return;
    }
    let alpha = params.dtAlphaBeta.y;
    let betaLin = params.dtAlphaBeta.z;
    let c = contacts[i];
    let ev = evalContact(c, alpha);

    var pen = c.penalty.xyz;
    var stick = c.lambda.w;
    if (ev.F.x < 0.0) {
        pen.x = min(pen.x + betaLin * abs(ev.C.x), PENALTY_MAX);
    }
    if (ev.preFs <= ev.bounds) {
        pen.y = min(pen.y + betaLin * abs(ev.C.y), PENALTY_MAX);
        pen.z = min(pen.z + betaLin * abs(ev.C.z), PENALTY_MAX);
        stick = select(0.0, 1.0, length(ev.C.yz) < STICK_THRESH);
    }
    contacts[i].lambda = vec4<f32>(ev.F, stick);
    contacts[i].penalty = vec4<f32>(pen, c.penalty.w);
}

// ---- joint dual kernel (Joint::updateDual) ----

@compute @workgroup_size(64)
fn jointDual(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    if (i >= params.counts.z) {
        return;
    }
    let alpha = params.dtAlphaBeta.y;
    let betaLin = params.dtAlphaBeta.z;
    let betaAng = params.dtAlphaBeta.w;
    var j = joints[i];
    let bA = bodies[j.header.x];
    let bB = bodies[j.header.y];
    let infLin = (j.header.z & 1u) != 0u;
    let infAng = (j.header.z & 2u) != 0u;

    if (dot(j.penaltyLin.xyz, j.penaltyLin.xyz) > 0.0) {
        var C = (bA.posLin.xyz + qrot(bA.posAng, j.rA.xyz)) - (bB.posLin.xyz + qrot(bB.posAng, j.rB.xyz));
        if (infLin) {
            C = C - j.c0Lin.xyz * alpha;
            j.lambdaLin = vec4<f32>(j.penaltyLin.xyz * C + j.lambdaLin.xyz, j.lambdaLin.w);
        }
        let cap = min(j.rA.w, PENALTY_MAX);
        j.penaltyLin = vec4<f32>(min(j.penaltyLin.xyz + abs(C) * betaLin, vec3<f32>(cap)), j.penaltyLin.w);
    }

    if (dot(j.penaltyAng.xyz, j.penaltyAng.xyz) > 0.0) {
        let ta = j.c0Lin.w;
        var C = qsub(bA.posAng, bB.posAng) * ta;
        if (infAng) {
            C = C - j.c0Ang.xyz * alpha;
            j.lambdaAng = vec4<f32>(j.penaltyAng.xyz * C + j.lambdaAng.xyz, j.lambdaAng.w);
        }
        let cap = min(j.rB.w, PENALTY_MAX);
        j.penaltyAng = vec4<f32>(min(j.penaltyAng.xyz + abs(C) * betaAng, vec3<f32>(cap)), j.penaltyAng.w);
    }

    // Fracture test
    let fracture = j.c0Ang.w;
    if (dot(j.lambdaAng.xyz, j.lambdaAng.xyz) > fracture * fracture) {
        j.penaltyLin = vec4<f32>(0.0);
        j.penaltyAng = vec4<f32>(0.0);
        j.lambdaLin = vec4<f32>(0.0);
        j.lambdaAng = vec4<f32>(0.0);
        j.header.z = j.header.z | 4u;
    }

    joints[i] = j;
}

)" R"(
// ---- GPU sphere narrowphase: sphere-sphere, dynamic sphere vs static box,
// dynamic sphere vs static cylinder. Exact ports of addRoundContact /
// collideRoundBox (sphere path) / collideCylinderRound at q- positions, plus
// Manifold::initialize's warmstart for round shapes (sphere contacts are
// frictionless, so only normal lambda/penalty persist). ----

struct NarrowphaseHit {
    hit : bool,
    nAB : vec3<f32>,  // normal from A to B (basis row 0 = -nAB)
    xA : vec3<f32>,
    xB : vec3<f32>,
};

fn closestPointOnSegmentW(a : vec3<f32>, b : vec3<f32>, p : vec3<f32>) -> vec3<f32> {
    let ab = b - a;
    let denom = dot(ab, ab);
    if (denom < 1.0e-6) {
        return a;
    }
    let t = clamp(dot(p - a, ab) / denom, 0.0, 1.0);
    return a + ab * t;
}

// Port of collideRoundBox for a sphere vs a static box at q-. Returns the
// box->sphere normal, the surface point on the box, and whether contact
// exists. Branch order matches the CPU exactly: thin-static-top, inside-box
// (swept or static-thinnest-axis pick), separated (swept or none).
fn sphereStaticBoxContact(sphere : Body, box : Body, dt : f32,
                          normalOut : ptr<function, vec3<f32>>,
                          xBoxOut : ptr<function, vec3<f32>>) -> bool {
    let center = sphere.initialLin.xyz;
    let radius = sphere.shape.x;
    let boxCenter = box.initialLin.xyz;
    let half = box.halfSize.xyz;
    let ax0 = qrot(box.initialAng, vec3<f32>(1.0, 0.0, 0.0));
    let ax1 = qrot(box.initialAng, vec3<f32>(0.0, 1.0, 0.0));
    let ax2 = qrot(box.initialAng, vec3<f32>(0.0, 0.0, 1.0));
    var axes = array<vec3<f32>, 3>(ax0, ax1, ax2);
    var halfArr = array<f32, 3>(half.x, half.y, half.z);

    // Thin static box top (collideThinStaticBoxTopRound, sphere case).
    var thinAxis = 0;
    if (halfArr[1] < halfArr[thinAxis]) { thinAxis = 1; }
    if (halfArr[2] < halfArr[thinAxis]) { thinAxis = 2; }
    let axisA = (thinAxis + 1) % 3;
    let axisB = (thinAxis + 2) % 3;
    if (halfArr[thinAxis] * 4.0 <= min(halfArr[axisA], halfArr[axisB])) {
        let sign = select(-1.0, 1.0, axes[thinAxis].z >= 0.0);
        let normal = axes[thinAxis] * sign;
        let d = center - boxCenter;
        var local = vec3<f32>(dot(d, ax0), dot(d, ax1), dot(d, ax2));
        var localArr = array<f32, 3>(local.x, local.y, local.z);
        let signedDistance = localArr[thinAxis] * sign - halfArr[thinAxis];
        var inRange = signedDistance <= radius;
        for (var axis = 0; axis < 3; axis++) {
            if (axis == thinAxis) { continue; }
            if (localArr[axis] < -halfArr[axis] - radius || localArr[axis] > halfArr[axis] + radius) {
                inRange = false;
            }
        }
        if (inRange) {
            localArr[thinAxis] = halfArr[thinAxis] * sign;
            for (var axis = 0; axis < 3; axis++) {
                if (axis != thinAxis) {
                    localArr[axis] = clamp(localArr[axis], -halfArr[axis], halfArr[axis]);
                }
            }
            *normalOut = normal;
            *xBoxOut = boxCenter + ax0 * localArr[0] + ax1 * localArr[1] + ax2 * localArr[2];
            return true;
        }
        // Thin static box, sphere out of range: the CPU falls through to the
        // general path below.
    }

    let d = center - boxCenter;
    let local = vec3<f32>(dot(d, ax0), dot(d, ax1), dot(d, ax2));
    let closest = boxCenter
        + ax0 * clamp(local.x, -half.x, half.x)
        + ax1 * clamp(local.y, -half.y, half.y)
        + ax2 * clamp(local.z, -half.z, half.z);
    var boxToRound = center - closest;
    var distSq = dot(boxToRound, boxToRound);
    var surface = closest;

    let inside = abs(local.x) <= half.x && abs(local.y) <= half.y && abs(local.z) <= half.z;

    // Swept sphere vs expanded box (sweptSphereBoxContact); box is static so
    // relative velocity is the sphere's.
    var sweptHit = false;
    var sweptNormal = vec3<f32>(0.0);
    var sweptPoint = vec3<f32>(0.0);
    let vel = sphere.velocity.xyz;
    if (dt > 0.0 && dot(vel, vel) >= 1.0e-6) {
        let p1 = center - boxCenter;
        let p0 = p1 - vel * dt;
        let l0 = vec3<f32>(dot(p0, ax0), dot(p0, ax1), dot(p0, ax2));
        let l1 = vec3<f32>(dot(p1, ax0), dot(p1, ax1), dot(p1, ax2));
        let deltaL = l1 - l0;
        let expanded = half + vec3<f32>(radius);
        var l0Arr = array<f32, 3>(l0.x, l0.y, l0.z);
        var dArr = array<f32, 3>(deltaL.x, deltaL.y, deltaL.z);
        var expArr = array<f32, 3>(expanded.x, expanded.y, expanded.z);
        var tEnter = 0.0;
        var tExit = 1.0;
        var hitAxis = -1;
        var hitSign = 1.0;
        var rejected = false;
        for (var axis = 0; axis < 3; axis++) {
            if (abs(dArr[axis]) < 1.0e-6) {
                if (l0Arr[axis] < -expArr[axis] || l0Arr[axis] > expArr[axis]) {
                    rejected = true;
                }
                continue;
            }
            let invD = 1.0 / dArr[axis];
            var t0 = (-expArr[axis] - l0Arr[axis]) * invD;
            var t1v = (expArr[axis] - l0Arr[axis]) * invD;
            var sgn = -1.0;
            if (t0 > t1v) {
                let tmp = t0;
                t0 = t1v;
                t1v = tmp;
                sgn = 1.0;
            }
            if (t0 > tEnter) {
                tEnter = t0;
                hitAxis = axis;
                hitSign = sgn;
            }
            tExit = min(tExit, t1v);
            if (tEnter > tExit) {
                rejected = true;
            }
        }
        if (!rejected && hitAxis >= 0 && tEnter >= 0.0 && tEnter <= 1.0) {
            let hitLocal = l0 + deltaL * tEnter;
            var surf = array<f32, 3>(hitLocal.x, hitLocal.y, hitLocal.z);
            surf[hitAxis] = halfArr[hitAxis] * hitSign;
            for (var axis = 0; axis < 3; axis++) {
                if (axis != hitAxis) {
                    surf[axis] = clamp(surf[axis], -halfArr[axis], halfArr[axis]);
                }
            }
            sweptNormal = axes[hitAxis] * hitSign;
            sweptPoint = boxCenter + ax0 * surf[0] + ax1 * surf[1] + ax2 * surf[2];
            sweptHit = true;
        }
    }

    if (inside) {
        if (sweptHit) {
            boxToRound = sweptNormal;
            surface = sweptPoint;
            distSq = 1.0;
        } else {
            // Static box: exit along the thinnest axis, signed by world up.
            var axis = 0;
            if (halfArr[1] < halfArr[axis]) { axis = 1; }
            if (halfArr[2] < halfArr[axis]) { axis = 2; }
            let sign = select(-1.0, 1.0, axes[axis].z >= 0.0);
            var surf = array<f32, 3>(local.x, local.y, local.z);
            surf[axis] = halfArr[axis] * sign;
            surface = boxCenter + ax0 * surf[0] + ax1 * surf[1] + ax2 * surf[2];
            boxToRound = axes[axis] * sign;
            distSq = 1.0;
        }
    } else if (distSq > radius * radius) {
        if (!sweptHit) {
            return false;
        }
        boxToRound = sweptNormal;
        surface = sweptPoint;
        distSq = 1.0;
    }

    var normal = vec3<f32>(0.0, 0.0, 1.0);
    if (distSq > 1.0e-6) {
        normal = boxToRound / sqrt(distSq);
    }
    *normalOut = normal;
    *xBoxOut = surface;
    return true;
}

@compute @workgroup_size(64)
fn sphereNarrowphase(@builtin(global_invocation_id) gid : vec3<u32>) {
    let p = gid.x;
    if (p >= params.sphere.x) {
        return;
    }
    let base = params.sphere.y + p * 4u;
    let ia = adjEntries[base];
    let ib = adjEntries[base + 1u];
    let stateSlot = adjEntries[base + 2u];
    let fresh = adjEntries[base + 3u];
    let slot = params.sphere.z + p;

    let bA = bodies[ia];
    let bB = bodies[ib];
    let stA = bA.moment.w;
    let stB = bB.moment.w;
    var rec : ContactC; // zero-initialized; a non-overlapping pair contributes nothing
    rec.header = vec4<u32>(ia, ib, 0u, 0u);

    // Contacts are generated at q- (pre-warmstart positions), like the CPU.
    var result : NarrowphaseHit;
    result.hit = false;

    if (stA == 1.0 && stB == 1.0) {
        // Sphere-sphere (addRoundContact).
        let delta = bB.initialLin.xyz - bA.initialLin.xyz;
        let radiusSum = bA.shape.x + bB.shape.x;
        let distSq = dot(delta, delta);
        if (distSq <= radiusSum * radiusSum) {
            var nAB = vec3<f32>(1.0, 0.0, 0.0);
            if (distSq > 1.0e-6) {
                nAB = delta / sqrt(distSq);
            }
            result.hit = true;
            result.nAB = nAB;
            result.xA = bA.initialLin.xyz + nAB * bA.shape.x;
            result.xB = bB.initialLin.xyz - nAB * bB.shape.x;
        }
    } else if ((stA == 1.0 && stB == 0.0) || (stA == 0.0 && stB == 1.0)) {
        // Dynamic sphere vs static box (collideRoundBox, sphere path).
        let roundIsA = stA == 1.0;
        var sphere = bA;
        var box = bB;
        if (!roundIsA) {
            sphere = bB;
            box = bA;
        }
        var normalBoxToRound = vec3<f32>(0.0);
        var xBox = vec3<f32>(0.0);
        if (sphereStaticBoxContact(sphere, box, params.dtAlphaBeta.x, &normalBoxToRound, &xBox)) {
            let xRound = sphere.initialLin.xyz - normalBoxToRound * sphere.shape.x;
            result.hit = true;
            if (roundIsA) {
                result.nAB = -normalBoxToRound;
                result.xA = xRound;
                result.xB = xBox;
            } else {
                result.nAB = normalBoxToRound;
                result.xA = xBox;
                result.xB = xRound;
            }
        }
    } else {
        // Dynamic sphere vs static cylinder (collideCylinderRound + addRoundContact).
        let cylinderIsA = stA == 3.0;
        var sphere = bB;
        var cyl = bA;
        if (!cylinderIsA) {
            sphere = bA;
            cyl = bB;
        }
        let axis = qrot(cyl.initialAng, vec3<f32>(0.0, 0.0, 1.0));
        let segA = cyl.initialLin.xyz - axis * cyl.shape.y;
        let segB = cyl.initialLin.xyz + axis * cyl.shape.y;
        let centerCyl = closestPointOnSegmentW(segA, segB, sphere.initialLin.xyz);
        let centerSph = sphere.initialLin.xyz;
        var centerA = centerCyl;
        var radA = cyl.shape.x;
        var centerB = centerSph;
        var radB = sphere.shape.x;
        if (!cylinderIsA) {
            centerA = centerSph;
            radA = sphere.shape.x;
            centerB = centerCyl;
            radB = cyl.shape.x;
        }
        let delta = centerB - centerA;
        let radiusSum = radA + radB;
        let distSq = dot(delta, delta);
        if (distSq <= radiusSum * radiusSum) {
            var nAB = vec3<f32>(1.0, 0.0, 0.0);
            if (distSq > 1.0e-6) {
                nAB = delta / sqrt(distSq);
            }
            result.hit = true;
            result.nAB = nAB;
            result.xA = centerA + nAB * radA;
            result.xB = centerB - nAB * radB;
        }
    }

    if (result.hit) {
        let n = -result.nAB; // basis row 0 points from B to A
        var t1 = select(vec3<f32>(0.0, -n.z, n.y), vec3<f32>(-n.y, n.x, 0.0), abs(n.x) > abs(n.z));
        t1 = normalize(t1);
        let t2 = cross(n, t1);
        let diff = result.xA - result.xB;
        let c0 = vec3<f32>(dot(n, diff) + params.misc.y, dot(t1, diff), dot(t2, diff));

        var lambdaN = 0.0;
        var penaltyN = PENALTY_MIN;
        if (fresh == 0u) {
            let st = sphereState[stateSlot];
            lambdaN = st.x * params.dtAlphaBeta.y * params.misc.x;
            penaltyN = clamp(st.y * params.misc.x, PENALTY_MIN, PENALTY_MAX);
        }

        // Local offsets (addContact): needed by contactOffsetWorld for the
        // box/cylinder side; the sphere side ignores them.
        let rAL = qrot(qconj(bA.initialAng), result.xA - bA.initialLin.xyz);
        let rBL = qrot(qconj(bB.initialAng), result.xB - bB.initialLin.xyz);

        rec.n = vec4<f32>(n, 0.0); // sphere contacts are frictionless
        rec.t1 = vec4<f32>(t1, 0.0);
        rec.t2 = vec4<f32>(t2, 0.0);
        rec.rA = vec4<f32>(rAL, 0.0);
        rec.rB = vec4<f32>(rBL, 0.0);
        rec.c0 = vec4<f32>(c0, 0.0);
        rec.lambda = vec4<f32>(lambdaN, 0.0, 0.0, 0.0);
        rec.penalty = vec4<f32>(penaltyN, PENALTY_MIN, PENALTY_MIN, 0.0);
    }
    contacts[slot] = rec;
}

@compute @workgroup_size(64)
fn spherePersist(@builtin(global_invocation_id) gid : vec3<u32>) {
    let p = gid.x;
    if (p >= params.sphere.x) {
        return;
    }
    let base = params.sphere.y + p * 4u;
    let stateSlot = adjEntries[base + 2u];
    let slot = params.sphere.z + p;
    let c = contacts[slot];
    sphereState[stateSlot] = vec4<f32>(c.lambda.x, c.penalty.x, 0.0, 0.0);
}
)";

struct WebGpuAvbdBackend : PhysicsBackend
{
    WebGpuDevice *ctx;

    wgpu::BindGroupLayout group0Layout;
    wgpu::BindGroupLayout group1Layout;
    wgpu::ComputePipeline primalPipeline;
    wgpu::ComputePipeline contactDualPipeline;
    wgpu::ComputePipeline jointDualPipeline;
    wgpu::ComputePipeline sphereNarrowphasePipeline;
    wgpu::ComputePipeline spherePersistPipeline;
    bool pipelinesFailed = false;

    wgpu::Buffer paramsBuffer;
    wgpu::Buffer colorParamsBuffer;
    wgpu::Buffer bodiesBuffer;
    wgpu::Buffer contactsBuffer;
    wgpu::Buffer jointsBuffer;
    wgpu::Buffer springsBuffer;
    wgpu::Buffer adjOffsetsBuffer;
    wgpu::Buffer adjEntriesBuffer;
    wgpu::Buffer solvedBodiesBuffer;
    wgpu::Buffer sphereStateBuffer;
    wgpu::Buffer bodiesReadback;
    wgpu::Buffer contactsReadback;
    wgpu::Buffer jointsReadback;

    uint64_t bodiesCapacity = 0;
    uint64_t contactsCapacity = 0;
    uint64_t jointsCapacity = 0;
    uint64_t springsCapacity = 0;
    uint64_t adjOffsetsCapacity = 0;
    uint64_t adjEntriesCapacity = 0;
    uint64_t solvedBodiesCapacity = 0;
    uint64_t sphereStateCapacity = 0;
    uint32_t colorSlotsCapacity = 0;

    wgpu::BindGroup bindGroup0;
    wgpu::BindGroup bindGroup1;
    bool bindGroupsDirty = true;

    // Per-frame staging (reused to avoid realloc churn)
    std::vector<Rigid *> bodyPtrs;
    std::unordered_map<Rigid *, uint32_t> bodyIndex;
    std::vector<GpuAvbdBody> gpuBodies;
    std::vector<GpuAvbdContact> gpuContacts;
    std::vector<GpuAvbdJoint> gpuJoints;
    std::vector<GpuAvbdSpring> gpuSprings;
    std::vector<std::pair<Manifold *, int>> contactRefs;
    std::vector<Joint *> jointRefs;
    std::vector<std::vector<uint32_t>> adjLists;
    std::vector<std::vector<uint32_t>> neighborLists;
    std::vector<uint32_t> adjOffsetsFlat;
    std::vector<uint32_t> adjEntriesFlat;
    std::vector<int> bodyColors;
    std::vector<uint32_t> solvedBodiesFlat;
    std::vector<std::pair<uint32_t, uint32_t>> colorRanges; // offset, count
    std::vector<uint8_t> colorSlotStaging;

    float frameWaitMs = 0.0f; // accumulated synchronous map-wait time this step
    bool warnedFallback = false;
    bool gpuBroken = false; // permanent CPU fallback after device errors
    unsigned int errorCountAtStepStart = 0;

    // GPU-resident sphere contact state: the CPU assigns each persistent
    // sphere pair a stable state slot (so the GPU never hashes), and the GPU
    // narrowphase warmstarts normal lambda/penalty from it across frames.
    std::vector<std::pair<Rigid *, Rigid *>> spherePairScratch;
    std::unordered_map<uint64_t, uint32_t> sphereSlotMap;
    std::vector<uint32_t> sphereSlotLastSeen;
    std::vector<uint32_t> sphereSlotFreeList;
    uint32_t frameIndex = 0;
    uint32_t spherePairBase = 0;
    uint32_t cpuContactBase = 0;

    uint32_t acquireSphereSlot(uint64_t key, bool &fresh)
    {
        auto it = sphereSlotMap.find(key);
        if (it != sphereSlotMap.end())
        {
            uint32_t slot = it->second;
            fresh = sphereSlotLastSeen[slot] + 1 != frameIndex;
            sphereSlotLastSeen[slot] = frameIndex;
            return slot;
        }
        fresh = true;
        uint32_t slot;
        if (!sphereSlotFreeList.empty())
        {
            slot = sphereSlotFreeList.back();
            sphereSlotFreeList.pop_back();
        }
        else
        {
            slot = (uint32_t)sphereSlotLastSeen.size();
            sphereSlotLastSeen.push_back(0);
        }
        sphereSlotLastSeen[slot] = frameIndex;
        sphereSlotMap[key] = slot;
        return slot;
    }

    void retireSphereSlots()
    {
        if ((frameIndex & 255u) != 0u)
            return;
        for (auto it = sphereSlotMap.begin(); it != sphereSlotMap.end();)
        {
            if (sphereSlotLastSeen[it->second] + 64 < frameIndex)
            {
                sphereSlotFreeList.push_back(it->second);
                it = sphereSlotMap.erase(it);
            }
            else
            {
                ++it;
            }
        }
    }

    explicit WebGpuAvbdBackend(WebGpuDevice *device) : ctx(device) {}

    const char *name() const override { return "WebGPU AVBD"; }

    void step(Solver &solver) override;

    bool ensurePipelines();
    bool ensureBuffers(uint64_t bodyCount, uint64_t contactCount, uint64_t jointCount,
                       uint64_t springCount, uint64_t adjOffsetCount, uint64_t adjEntryCount,
                       uint64_t solvedCount, uint32_t colorCount);
    void buildFrame(Solver &solver);
    bool runGpuIterations(Solver &solver);
    bool readbackAndApply(Solver &solver);
    void applyGpuPairRollingFriction(Solver &solver);

    wgpu::Buffer makeBuffer(uint64_t bytes, wgpu::BufferUsage usage)
    {
        wgpu::BufferDescriptor desc = {};
        desc.size = bytes;
        desc.usage = usage;
        return ctx->device.CreateBuffer(&desc);
    }

    bool growStorage(wgpu::Buffer &buf, uint64_t &capacityBytes, uint64_t neededBytes, bool copySrc)
    {
        if (buf != nullptr && capacityBytes >= neededBytes)
            return true;
        uint64_t newCap = capacityBytes > 0 ? capacityBytes : 4096;
        while (newCap < neededBytes)
            newCap *= 2;
        wgpu::BufferUsage usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        if (copySrc)
            usage = usage | wgpu::BufferUsage::CopySrc;
        buf = makeBuffer(newCap, usage);
        capacityBytes = newCap;
        bindGroupsDirty = true;
        return buf != nullptr;
    }

    bool mapReadback(wgpu::Buffer &buf, uint64_t bytes, const void **outPtr)
    {
        Clock::time_point waitBegin = Clock::now();
        bool mapDone = false;
        wgpu::MapAsyncStatus mapStatus = wgpu::MapAsyncStatus::Error;
        wgpu::Future mapFuture = buf.MapAsync(
            wgpu::MapMode::Read, 0, bytes, wgpu::CallbackMode::WaitAnyOnly,
            [&](wgpu::MapAsyncStatus status, wgpu::StringView)
            {
                mapDone = true;
                mapStatus = status;
            });
        if (ctx->instance.WaitAny(mapFuture, UINT64_MAX) != wgpu::WaitStatus::Success ||
            !mapDone || mapStatus != wgpu::MapAsyncStatus::Success)
        {
            frameWaitMs += elapsedMsAvbd(waitBegin, Clock::now());
            return false;
        }
        frameWaitMs += elapsedMsAvbd(waitBegin, Clock::now());
        *outPtr = buf.GetConstMappedRange(0, bytes);
        return *outPtr != nullptr;
    }
};

bool WebGpuAvbdBackend::ensurePipelines()
{
    if (pipelinesFailed)
        return false;
    if (primalPipeline != nullptr)
        return true;

    wgpu::ShaderSourceWGSL wgsl = {};
    wgsl.code = AVBD_SOLVER_WGSL;
    wgpu::ShaderModuleDescriptor shaderDesc = {};
    shaderDesc.nextInChain = &wgsl;
    wgpu::ShaderModule shader = ctx->device.CreateShaderModule(&shaderDesc);
    if (shader == nullptr)
    {
        pipelinesFailed = true;
        return false;
    }

    wgpu::BindGroupLayoutEntry g0[9] = {};
    g0[0].binding = 0;
    g0[0].visibility = wgpu::ShaderStage::Compute;
    g0[0].buffer.type = wgpu::BufferBindingType::Uniform;
    for (int i = 1; i <= 3; ++i)
    {
        g0[i].binding = (uint32_t)i;
        g0[i].visibility = wgpu::ShaderStage::Compute;
        g0[i].buffer.type = wgpu::BufferBindingType::Storage;
    }
    for (int i = 4; i <= 7; ++i)
    {
        g0[i].binding = (uint32_t)i;
        g0[i].visibility = wgpu::ShaderStage::Compute;
        g0[i].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
    }
    g0[8].binding = 8;
    g0[8].visibility = wgpu::ShaderStage::Compute;
    g0[8].buffer.type = wgpu::BufferBindingType::Storage;
    wgpu::BindGroupLayoutDescriptor g0Desc = {};
    g0Desc.entryCount = 9;
    g0Desc.entries = g0;
    group0Layout = ctx->device.CreateBindGroupLayout(&g0Desc);

    wgpu::BindGroupLayoutEntry g1[1] = {};
    g1[0].binding = 0;
    g1[0].visibility = wgpu::ShaderStage::Compute;
    g1[0].buffer.type = wgpu::BufferBindingType::Uniform;
    g1[0].buffer.hasDynamicOffset = true;
    wgpu::BindGroupLayoutDescriptor g1Desc = {};
    g1Desc.entryCount = 1;
    g1Desc.entries = g1;
    group1Layout = ctx->device.CreateBindGroupLayout(&g1Desc);

    if (group0Layout == nullptr || group1Layout == nullptr)
    {
        pipelinesFailed = true;
        return false;
    }

    wgpu::BindGroupLayout layouts[2] = {group0Layout, group1Layout};
    wgpu::PipelineLayoutDescriptor layoutDesc = {};
    layoutDesc.bindGroupLayoutCount = 2;
    layoutDesc.bindGroupLayouts = layouts;
    wgpu::PipelineLayout pipelineLayout = ctx->device.CreatePipelineLayout(&layoutDesc);
    if (pipelineLayout == nullptr)
    {
        pipelinesFailed = true;
        return false;
    }

    const char *entries[5] = {"primal", "contactDual", "jointDual", "sphereNarrowphase", "spherePersist"};
    wgpu::ComputePipeline *pipelines[5] = {&primalPipeline, &contactDualPipeline, &jointDualPipeline,
                                           &sphereNarrowphasePipeline, &spherePersistPipeline};
    for (int i = 0; i < 5; ++i)
    {
        wgpu::ComputePipelineDescriptor desc = {};
        desc.layout = pipelineLayout;
        desc.compute.module = shader;
        desc.compute.entryPoint = entries[i];
        *pipelines[i] = ctx->device.CreateComputePipeline(&desc);
        if (*pipelines[i] == nullptr)
        {
            pipelinesFailed = true;
            return false;
        }
    }
    return true;
}

bool WebGpuAvbdBackend::ensureBuffers(uint64_t bodyCount, uint64_t contactCount, uint64_t jointCount,
                                      uint64_t springCount, uint64_t adjOffsetCount, uint64_t adjEntryCount,
                                      uint64_t solvedCount, uint32_t colorCount)
{
    bool ok = true;
    if (paramsBuffer == nullptr)
    {
        paramsBuffer = makeBuffer(sizeof(GpuAvbdParams), wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
        bindGroupsDirty = true;
        ok = ok && paramsBuffer != nullptr;
    }
    if (colorSlotsCapacity < colorCount || colorParamsBuffer == nullptr)
    {
        uint32_t slots = colorSlotsCapacity > 0 ? colorSlotsCapacity : 32;
        while (slots < colorCount)
            slots *= 2;
        if (slots > MAX_COLOR_SLOTS)
            slots = MAX_COLOR_SLOTS;
        colorParamsBuffer = makeBuffer((uint64_t)slots * COLOR_SLOT_STRIDE, wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
        colorSlotsCapacity = slots;
        bindGroupsDirty = true;
        ok = ok && colorParamsBuffer != nullptr;
    }

    ok = ok && growStorage(bodiesBuffer, bodiesCapacity, bodyCount * sizeof(GpuAvbdBody), true);
    ok = ok && growStorage(contactsBuffer, contactsCapacity, (contactCount > 0 ? contactCount : 1) * sizeof(GpuAvbdContact), true);
    ok = ok && growStorage(jointsBuffer, jointsCapacity, (jointCount > 0 ? jointCount : 1) * sizeof(GpuAvbdJoint), true);
    ok = ok && growStorage(springsBuffer, springsCapacity, (springCount > 0 ? springCount : 1) * sizeof(GpuAvbdSpring), false);
    ok = ok && growStorage(adjOffsetsBuffer, adjOffsetsCapacity, adjOffsetCount * sizeof(uint32_t), false);
    ok = ok && growStorage(adjEntriesBuffer, adjEntriesCapacity, (adjEntryCount > 0 ? adjEntryCount : 1) * sizeof(uint32_t), false);
    ok = ok && growStorage(solvedBodiesBuffer, solvedBodiesCapacity, (solvedCount > 0 ? solvedCount : 1) * sizeof(uint32_t), false);
    uint64_t stateSlots = sphereSlotLastSeen.size() > 0 ? sphereSlotLastSeen.size() : 1;
    ok = ok && growStorage(sphereStateBuffer, sphereStateCapacity, stateSlots * 4 * sizeof(float), false);
    if (!ok)
        return false;

    // Readback buffers track the storage capacities so copies always fit.
    auto ensureReadback = [&](wgpu::Buffer &buf, uint64_t bytes) -> bool
    {
        static_assert(sizeof(uint64_t) == 8, "");
        if (buf != nullptr && buf.GetSize() >= bytes)
            return true;
        buf = makeBuffer(bytes, wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst);
        return buf != nullptr;
    };
    ok = ok && ensureReadback(bodiesReadback, bodiesCapacity);
    ok = ok && ensureReadback(contactsReadback, contactsCapacity);
    ok = ok && ensureReadback(jointsReadback, jointsCapacity);
    if (!ok)
        return false;

    if (bindGroupsDirty)
    {
        wgpu::BindGroupEntry e0[9] = {};
        e0[0].binding = 0;
        e0[0].buffer = paramsBuffer;
        e0[0].size = sizeof(GpuAvbdParams);
        e0[1].binding = 1;
        e0[1].buffer = bodiesBuffer;
        e0[1].size = bodiesCapacity;
        e0[2].binding = 2;
        e0[2].buffer = contactsBuffer;
        e0[2].size = contactsCapacity;
        e0[3].binding = 3;
        e0[3].buffer = jointsBuffer;
        e0[3].size = jointsCapacity;
        e0[4].binding = 4;
        e0[4].buffer = springsBuffer;
        e0[4].size = springsCapacity;
        e0[5].binding = 5;
        e0[5].buffer = adjOffsetsBuffer;
        e0[5].size = adjOffsetsCapacity;
        e0[6].binding = 6;
        e0[6].buffer = adjEntriesBuffer;
        e0[6].size = adjEntriesCapacity;
        e0[7].binding = 7;
        e0[7].buffer = solvedBodiesBuffer;
        e0[7].size = solvedBodiesCapacity;
        e0[8].binding = 8;
        e0[8].buffer = sphereStateBuffer;
        e0[8].size = sphereStateCapacity;
        wgpu::BindGroupDescriptor g0Desc = {};
        g0Desc.layout = group0Layout;
        g0Desc.entryCount = 9;
        g0Desc.entries = e0;
        bindGroup0 = ctx->device.CreateBindGroup(&g0Desc);

        wgpu::BindGroupEntry e1[1] = {};
        e1[0].binding = 0;
        e1[0].buffer = colorParamsBuffer;
        e1[0].offset = 0;
        e1[0].size = 16; // one ColorParams struct per dynamic-offset slot
        wgpu::BindGroupDescriptor g1Desc = {};
        g1Desc.layout = group1Layout;
        g1Desc.entryCount = 1;
        g1Desc.entries = e1;
        bindGroup1 = ctx->device.CreateBindGroup(&g1Desc);

        if (bindGroup0 == nullptr || bindGroup1 == nullptr)
            return false;
        bindGroupsDirty = false;
    }
    return true;
}

void WebGpuAvbdBackend::buildFrame(Solver &solver)
{
    bodyPtrs.clear();
    bodyIndex.clear();
    for (Rigid *b = solver.bodies; b != 0; b = b->next)
    {
        bodyIndex[b] = (uint32_t)bodyPtrs.size();
        bodyPtrs.push_back(b);
    }
    uint32_t N = (uint32_t)bodyPtrs.size();

    gpuBodies.resize(N + 1);
    for (uint32_t i = 0; i < N; ++i)
    {
        Rigid *b = bodyPtrs[i];
        GpuAvbdBody &g = gpuBodies[i];
        g.posLin[0] = b->positionLin.x;
        g.posLin[1] = b->positionLin.y;
        g.posLin[2] = b->positionLin.z;
        g.posLin[3] = b->mass;
        g.posAng[0] = b->positionAng.x;
        g.posAng[1] = b->positionAng.y;
        g.posAng[2] = b->positionAng.z;
        g.posAng[3] = b->positionAng.w;
        g.initialLin[0] = b->initialLin.x;
        g.initialLin[1] = b->initialLin.y;
        g.initialLin[2] = b->initialLin.z;
        g.initialLin[3] = 0.0f;
        g.initialAng[0] = b->initialAng.x;
        g.initialAng[1] = b->initialAng.y;
        g.initialAng[2] = b->initialAng.z;
        g.initialAng[3] = b->initialAng.w;
        g.inertialLin[0] = b->inertialLin.x;
        g.inertialLin[1] = b->inertialLin.y;
        g.inertialLin[2] = b->inertialLin.z;
        g.inertialLin[3] = 0.0f;
        g.inertialAng[0] = b->inertialAng.x;
        g.inertialAng[1] = b->inertialAng.y;
        g.inertialAng[2] = b->inertialAng.z;
        g.inertialAng[3] = b->inertialAng.w;
        g.moment[0] = b->moment.x;
        g.moment[1] = b->moment.y;
        g.moment[2] = b->moment.z;
        g.moment[3] = (float)b->shape.type;
        g.shape[0] = b->shape.radius;
        g.shape[1] = b->shape.halfLength;
        g.shape[2] = 0.0f;
        g.shape[3] = 0.0f;
        g.velocity[0] = b->velocityLin.x;
        g.velocity[1] = b->velocityLin.y;
        g.velocity[2] = b->velocityLin.z;
        g.velocity[3] = 0.0f;
        g.halfSize[0] = b->size.x * 0.5f;
        g.halfSize[1] = b->size.y * 0.5f;
        g.halfSize[2] = b->size.z * 0.5f;
        g.halfSize[3] = 0.0f;
    }
    // Sentinel identity body for world-anchored joints (bodyA == null).
    GpuAvbdBody &world = gpuBodies[N];
    std::memset(&world, 0, sizeof(world));
    world.posAng[3] = 1.0f;
    world.initialAng[3] = 1.0f;
    world.inertialAng[3] = 1.0f;

    gpuContacts.clear();
    gpuJoints.clear();
    gpuSprings.clear();
    contactRefs.clear();
    jointRefs.clear();

    adjLists.assign(N, {});
    neighborLists.assign(N, {});

    auto addAdj = [&](uint32_t bi, uint32_t kind, bool isA, uint32_t idx)
    {
        if (bi < N && bodyPtrs[bi]->mass > 0.0f)
            adjLists[bi].push_back(adjEntryEncode(kind, isA, idx));
    };
    auto addNeighbors = [&](uint32_t a, uint32_t b)
    {
        if (a < N && b < N && bodyPtrs[a]->mass > 0.0f && bodyPtrs[b]->mass > 0.0f)
        {
            neighborLists[a].push_back(b);
            neighborLists[b].push_back(a);
        }
    };

    for (Force *force = solver.forces; force != 0; force = force->next)
    {
        if (Manifold *m = force->type == SIM_CONSTRAINT_MANIFOLD ? (Manifold *)force : 0)
        {
            uint32_t ia = bodyIndex[m->bodyA];
            uint32_t ib = bodyIndex[m->bodyB];
            for (int i = 0; i < m->numContacts; ++i)
            {
                GpuAvbdContact rec = {};
                rec.header[0] = ia;
                rec.header[1] = ib;
                rec.header[2] = isCylinderCapContactKey(m->contacts[i].feature.key) ? 1u : 0u;
                rec.n[0] = m->basis[0].x;
                rec.n[1] = m->basis[0].y;
                rec.n[2] = m->basis[0].z;
                rec.n[3] = m->friction;
                rec.t1[0] = m->basis[1].x;
                rec.t1[1] = m->basis[1].y;
                rec.t1[2] = m->basis[1].z;
                rec.t2[0] = m->basis[2].x;
                rec.t2[1] = m->basis[2].y;
                rec.t2[2] = m->basis[2].z;
                rec.rA[0] = m->contacts[i].rA.x;
                rec.rA[1] = m->contacts[i].rA.y;
                rec.rA[2] = m->contacts[i].rA.z;
                rec.rB[0] = m->contacts[i].rB.x;
                rec.rB[1] = m->contacts[i].rB.y;
                rec.rB[2] = m->contacts[i].rB.z;
                rec.c0[0] = m->contacts[i].C0.x;
                rec.c0[1] = m->contacts[i].C0.y;
                rec.c0[2] = m->contacts[i].C0.z;
                rec.lambda[0] = m->contacts[i].lambda.x;
                rec.lambda[1] = m->contacts[i].lambda.y;
                rec.lambda[2] = m->contacts[i].lambda.z;
                rec.lambda[3] = m->contacts[i].stick ? 1.0f : 0.0f;
                rec.penalty[0] = m->contacts[i].penalty.x;
                rec.penalty[1] = m->contacts[i].penalty.y;
                rec.penalty[2] = m->contacts[i].penalty.z;

                uint32_t idx = (uint32_t)gpuContacts.size();
                gpuContacts.push_back(rec);
                contactRefs.push_back({m, i});
                addAdj(ia, ADJ_KIND_CONTACT, true, idx);
                addAdj(ib, ADJ_KIND_CONTACT, false, idx);
            }
            addNeighbors(ia, ib);
        }
        else if (Joint *j = force->type == SIM_CONSTRAINT_JOINT ? (Joint *)force : 0)
        {
            uint32_t ia = j->bodyA ? bodyIndex[j->bodyA] : N;
            uint32_t ib = bodyIndex[j->bodyB];
            GpuAvbdJoint rec = {};
            rec.header[0] = ia;
            rec.header[1] = ib;
            rec.header[2] = (isinf(j->stiffnessLin) ? 1u : 0u) | (isinf(j->stiffnessAng) ? 2u : 0u);
            rec.rA[0] = j->rA.x;
            rec.rA[1] = j->rA.y;
            rec.rA[2] = j->rA.z;
            rec.rA[3] = isinf(j->stiffnessLin) ? GPU_INF_SENTINEL : j->stiffnessLin;
            rec.rB[0] = j->rB.x;
            rec.rB[1] = j->rB.y;
            rec.rB[2] = j->rB.z;
            rec.rB[3] = isinf(j->stiffnessAng) ? GPU_INF_SENTINEL : j->stiffnessAng;
            rec.c0Lin[0] = j->C0Lin.x;
            rec.c0Lin[1] = j->C0Lin.y;
            rec.c0Lin[2] = j->C0Lin.z;
            rec.c0Lin[3] = j->torqueArm;
            rec.c0Ang[0] = j->C0Ang.x;
            rec.c0Ang[1] = j->C0Ang.y;
            rec.c0Ang[2] = j->C0Ang.z;
            rec.c0Ang[3] = isinf(j->fracture) ? GPU_INF_SENTINEL : j->fracture;
            rec.lambdaLin[0] = j->lambdaLin.x;
            rec.lambdaLin[1] = j->lambdaLin.y;
            rec.lambdaLin[2] = j->lambdaLin.z;
            rec.lambdaAng[0] = j->lambdaAng.x;
            rec.lambdaAng[1] = j->lambdaAng.y;
            rec.lambdaAng[2] = j->lambdaAng.z;
            rec.penaltyLin[0] = j->penaltyLin.x;
            rec.penaltyLin[1] = j->penaltyLin.y;
            rec.penaltyLin[2] = j->penaltyLin.z;
            rec.penaltyAng[0] = j->penaltyAng.x;
            rec.penaltyAng[1] = j->penaltyAng.y;
            rec.penaltyAng[2] = j->penaltyAng.z;

            uint32_t idx = (uint32_t)gpuJoints.size();
            gpuJoints.push_back(rec);
            jointRefs.push_back(j);
            if (j->bodyA)
                addAdj(ia, ADJ_KIND_JOINT, true, idx);
            addAdj(ib, ADJ_KIND_JOINT, false, idx);
            if (j->bodyA)
                addNeighbors(ia, ib);
        }
        else if (Spring *s = force->type == SIM_CONSTRAINT_SPRING ? (Spring *)force : 0)
        {
            uint32_t ia = bodyIndex[s->bodyA];
            uint32_t ib = bodyIndex[s->bodyB];
            GpuAvbdSpring rec = {};
            rec.header[0] = ia;
            rec.header[1] = ib;
            rec.rA[0] = s->rA.x;
            rec.rA[1] = s->rA.y;
            rec.rA[2] = s->rA.z;
            rec.rA[3] = s->rest;
            rec.rB[0] = s->rB.x;
            rec.rB[1] = s->rB.y;
            rec.rB[2] = s->rB.z;
            rec.rB[3] = s->stiffness;

            uint32_t idx = (uint32_t)gpuSprings.size();
            gpuSprings.push_back(rec);
            addAdj(ia, ADJ_KIND_SPRING, true, idx);
            addAdj(ib, ADJ_KIND_SPRING, false, idx);
            addNeighbors(ia, ib);
        }
        // IgnoreCollision has no solver effect; skip.
    }

    // GPU-resident sphere pairs: assign each a persistent state slot and a
    // contact slot after the CPU manifold contacts, and add them to the
    // adjacency/coloring graph exactly like CPU contacts.
    cpuContactBase = (uint32_t)gpuContacts.size();
    retireSphereSlots();
    for (size_t k = 0; k < spherePairScratch.size(); ++k)
    {
        Rigid *a = spherePairScratch[k].first;
        Rigid *b = spherePairScratch[k].second;
        uint32_t ia = bodyIndex[a];
        uint32_t ib = bodyIndex[b];
        uint32_t contactSlot = cpuContactBase + (uint32_t)k;
        addAdj(ia, ADJ_KIND_CONTACT, true, contactSlot);
        addAdj(ib, ADJ_KIND_CONTACT, false, contactSlot);
        addNeighbors(ia, ib);
    }

    // Flatten adjacency
    adjOffsetsFlat.resize(N + 1);
    adjEntriesFlat.clear();
    for (uint32_t i = 0; i < N; ++i)
    {
        adjOffsetsFlat[i] = (uint32_t)adjEntriesFlat.size();
        adjEntriesFlat.insert(adjEntriesFlat.end(), adjLists[i].begin(), adjLists[i].end());
    }
    adjOffsetsFlat[N] = (uint32_t)adjEntriesFlat.size();

    // Pair records ride at the end of the adjEntries buffer: idxA, idxB,
    // stateSlot, freshFlag per pair.
    spherePairBase = (uint32_t)adjEntriesFlat.size();
    for (size_t k = 0; k < spherePairScratch.size(); ++k)
    {
        Rigid *a = spherePairScratch[k].first;
        Rigid *b = spherePairScratch[k].second;
        uint32_t lo = a->denseId < b->denseId ? a->denseId : b->denseId;
        uint32_t hi = a->denseId < b->denseId ? b->denseId : a->denseId;
        uint64_t key = ((uint64_t)lo << 32) | hi;
        bool fresh = true;
        uint32_t stateSlot = acquireSphereSlot(key, fresh);
        adjEntriesFlat.push_back(bodyIndex[a]);
        adjEntriesFlat.push_back(bodyIndex[b]);
        adjEntriesFlat.push_back(stateSlot);
        adjEntriesFlat.push_back(fresh ? 1u : 0u);
    }

    // Greedy graph coloring over solved bodies. Solved == dynamic with any
    // attached force (matches the CPU primal loop's skip conditions; a body
    // whose only force is IgnoreCollision converges to its inertial target,
    // exactly like the CPU reference).
    bodyColors.assign(N, -1);
    int maxColor = -1;
    std::vector<char> used;
    for (uint32_t i = 0; i < N; ++i)
    {
        Rigid *b = bodyPtrs[i];
        if (b->mass <= 0.0f || (b->forces == 0 && b->gpuPairCount == 0))
            continue;
        used.assign((size_t)maxColor + 2, 0);
        for (uint32_t nb : neighborLists[i])
        {
            int c = bodyColors[nb];
            if (c >= 0 && c < (int)used.size())
                used[c] = 1;
        }
        int color = 0;
        while (color < (int)used.size() && used[color])
            ++color;
        bodyColors[i] = color;
        if (color > maxColor)
            maxColor = color;
    }

    uint32_t colorCount = (uint32_t)(maxColor + 1);
    colorRanges.assign(colorCount, {0, 0});
    for (uint32_t i = 0; i < N; ++i)
        if (bodyColors[i] >= 0)
            colorRanges[bodyColors[i]].second++;
    uint32_t offset = 0;
    for (uint32_t c = 0; c < colorCount; ++c)
    {
        colorRanges[c].first = offset;
        offset += colorRanges[c].second;
        colorRanges[c].second = 0;
    }
    solvedBodiesFlat.resize(offset);
    for (uint32_t i = 0; i < N; ++i)
    {
        int c = bodyColors[i];
        if (c >= 0)
        {
            solvedBodiesFlat[colorRanges[c].first + colorRanges[c].second] = i;
            colorRanges[c].second++;
        }
    }
}

bool WebGpuAvbdBackend::runGpuIterations(Solver &solver)
{
    uint32_t bodyCount = (uint32_t)gpuBodies.size();
    uint32_t spherePairCount = (uint32_t)spherePairScratch.size();
    uint32_t contactCount = (uint32_t)gpuContacts.size() + spherePairCount;
    uint32_t jointCount = (uint32_t)gpuJoints.size();
    uint32_t springCount = (uint32_t)gpuSprings.size();
    uint32_t colorCount = (uint32_t)colorRanges.size();

    if (!ensureBuffers(bodyCount, contactCount, jointCount, springCount,
                       adjOffsetsFlat.size(), adjEntriesFlat.size(),
                       solvedBodiesFlat.size(), colorCount))
        return false;

    wgpu::Queue queue = ctx->queue;

    GpuAvbdParams params = {};
    params.dt = solver.dt;
    params.alpha = solver.alpha;
    params.betaLin = solver.betaLin;
    params.betaAng = solver.betaAng;
    params.bodyCount = bodyCount;
    params.contactCount = contactCount;
    params.jointCount = jointCount;
    params.springCount = springCount;
    params.gamma = solver.gamma;
    params.collisionMargin = COLLISION_MARGIN;
    params.spherePairCount = spherePairCount;
    params.spherePairBase = spherePairBase;
    params.cpuContactBase = cpuContactBase;
    queue.WriteBuffer(paramsBuffer, 0, &params, sizeof(params));

    colorSlotStaging.assign((size_t)colorCount * COLOR_SLOT_STRIDE, 0);
    for (uint32_t c = 0; c < colorCount; ++c)
    {
        uint32_t *slot = (uint32_t *)(colorSlotStaging.data() + (size_t)c * COLOR_SLOT_STRIDE);
        slot[0] = colorRanges[c].first;
        slot[1] = colorRanges[c].second;
    }
    queue.WriteBuffer(colorParamsBuffer, 0, colorSlotStaging.data(), colorSlotStaging.size());

    queue.WriteBuffer(bodiesBuffer, 0, gpuBodies.data(), gpuBodies.size() * sizeof(GpuAvbdBody));
    if (!gpuContacts.empty())
        queue.WriteBuffer(contactsBuffer, 0, gpuContacts.data(), gpuContacts.size() * sizeof(GpuAvbdContact));
    if (jointCount > 0)
        queue.WriteBuffer(jointsBuffer, 0, gpuJoints.data(), gpuJoints.size() * sizeof(GpuAvbdJoint));
    if (springCount > 0)
        queue.WriteBuffer(springsBuffer, 0, gpuSprings.data(), gpuSprings.size() * sizeof(GpuAvbdSpring));
    queue.WriteBuffer(adjOffsetsBuffer, 0, adjOffsetsFlat.data(), adjOffsetsFlat.size() * sizeof(uint32_t));
    if (!adjEntriesFlat.empty())
        queue.WriteBuffer(adjEntriesBuffer, 0, adjEntriesFlat.data(), adjEntriesFlat.size() * sizeof(uint32_t));
    if (!solvedBodiesFlat.empty())
        queue.WriteBuffer(solvedBodiesBuffer, 0, solvedBodiesFlat.data(), solvedBodiesFlat.size() * sizeof(uint32_t));

    wgpu::CommandEncoder encoder = ctx->device.CreateCommandEncoder();
    wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetBindGroup(0, bindGroup0);
    uint32_t zeroOffset = 0;
    pass.SetBindGroup(1, bindGroup1, 1, &zeroOffset);

    if (spherePairCount > 0)
    {
        pass.SetPipeline(sphereNarrowphasePipeline);
        pass.DispatchWorkgroups((spherePairCount + 63u) / 64u);
    }

    for (int it = 0; it < solver.iterations; ++it)
    {
        pass.SetPipeline(primalPipeline);
        for (uint32_t c = 0; c < (uint32_t)colorRanges.size(); ++c)
        {
            if (colorRanges[c].second == 0)
                continue;
            uint32_t dynOffset = c * COLOR_SLOT_STRIDE;
            pass.SetBindGroup(1, bindGroup1, 1, &dynOffset);
            pass.DispatchWorkgroups((colorRanges[c].second + 63u) / 64u);
        }
        if (contactCount > 0)
        {
            pass.SetPipeline(contactDualPipeline);
            pass.DispatchWorkgroups((contactCount + 63u) / 64u);
        }
        if (jointCount > 0)
        {
            pass.SetPipeline(jointDualPipeline);
            pass.DispatchWorkgroups((jointCount + 63u) / 64u);
        }
    }
    if (spherePairCount > 0)
    {
        pass.SetPipeline(spherePersistPipeline);
        pass.DispatchWorkgroups((spherePairCount + 63u) / 64u);
    }
    pass.End();

    encoder.CopyBufferToBuffer(bodiesBuffer, 0, bodiesReadback, 0, gpuBodies.size() * sizeof(GpuAvbdBody));
    if (contactCount > 0)
        encoder.CopyBufferToBuffer(contactsBuffer, 0, contactsReadback, 0, gpuContacts.size() * sizeof(GpuAvbdContact));
    if (jointCount > 0)
        encoder.CopyBufferToBuffer(jointsBuffer, 0, jointsReadback, 0, gpuJoints.size() * sizeof(GpuAvbdJoint));

    wgpu::CommandBuffer commands = encoder.Finish();
    queue.Submit(1, &commands);
    return true;
}

bool WebGpuAvbdBackend::readbackAndApply(Solver &solver)
{
    (void)solver;
    uint32_t N = (uint32_t)bodyPtrs.size();

    const void *ptr = nullptr;
    if (!mapReadback(bodiesReadback, gpuBodies.size() * sizeof(GpuAvbdBody), &ptr))
        return false;
    // A validation error anywhere this frame means the GPU work did not run
    // and the readback holds stale/zero data; do not apply it.
    if (ctx->errorCount != errorCountAtStepStart)
    {
        bodiesReadback.Unmap();
        return false;
    }
    const GpuAvbdBody *outBodies = (const GpuAvbdBody *)ptr;
    for (uint32_t i = 0; i < N; ++i)
    {
        Rigid *b = bodyPtrs[i];
        if (b->mass <= 0.0f || bodyColors[i] < 0)
            continue;
        b->positionLin = float3{outBodies[i].posLin[0], outBodies[i].posLin[1], outBodies[i].posLin[2]};
        b->positionAng = quat{outBodies[i].posAng[0], outBodies[i].posAng[1], outBodies[i].posAng[2], outBodies[i].posAng[3]};
    }
    bodiesReadback.Unmap();

    if (!gpuContacts.empty())
    {
        if (!mapReadback(contactsReadback, gpuContacts.size() * sizeof(GpuAvbdContact), &ptr))
            return false;
        const GpuAvbdContact *outContacts = (const GpuAvbdContact *)ptr;
        for (size_t k = 0; k < contactRefs.size(); ++k)
        {
            Manifold *m = contactRefs[k].first;
            int ci = contactRefs[k].second;
            const GpuAvbdContact &rec = outContacts[k];
            m->contacts[ci].lambda = float3{rec.lambda[0], rec.lambda[1], rec.lambda[2]};
            m->contacts[ci].penalty = float3{rec.penalty[0], rec.penalty[1], rec.penalty[2]};
            m->contacts[ci].stick = rec.lambda[3] > 0.5f;
        }
        contactsReadback.Unmap();
    }

    if (!gpuJoints.empty())
    {
        if (!mapReadback(jointsReadback, gpuJoints.size() * sizeof(GpuAvbdJoint), &ptr))
            return false;
        const GpuAvbdJoint *outJoints = (const GpuAvbdJoint *)ptr;
        for (size_t k = 0; k < jointRefs.size(); ++k)
        {
            Joint *j = jointRefs[k];
            const GpuAvbdJoint &rec = outJoints[k];
            j->lambdaLin = float3{rec.lambdaLin[0], rec.lambdaLin[1], rec.lambdaLin[2]};
            j->lambdaAng = float3{rec.lambdaAng[0], rec.lambdaAng[1], rec.lambdaAng[2]};
            j->penaltyLin = float3{rec.penaltyLin[0], rec.penaltyLin[1], rec.penaltyLin[2]};
            j->penaltyAng = float3{rec.penaltyAng[0], rec.penaltyAng[1], rec.penaltyAng[2]};
            if (rec.header[2] & 4u)
                j->broken = true;
        }
        jointsReadback.Unmap();
    }
    return true;
}

// Port of solver.cpp's applySphereRollingFriction for sphere-vs-static pairs
// whose manifolds now live on the GPU. Geometry is evaluated at q- (the same
// positions the narrowphase used); runs after finishStep so velocities exist,
// mirroring the CPU ordering.
void WebGpuAvbdBackend::applyGpuPairRollingFriction(Solver &solver)
{
    for (const std::pair<Rigid *, Rigid *> &pair : spherePairScratch)
    {
        Rigid *sphere = 0;
        Rigid *other = 0;
        if (pair.first->shape.type == RIGID_SHAPE_SPHERE && pair.second->shape.type != RIGID_SHAPE_SPHERE)
        {
            sphere = pair.first;
            other = pair.second;
        }
        else if (pair.second->shape.type == RIGID_SHAPE_SPHERE && pair.first->shape.type != RIGID_SHAPE_SPHERE)
        {
            sphere = pair.second;
            other = pair.first;
        }
        if (!sphere || sphere->mass <= 0.0f || other->mass > 0.0f)
            continue;

        float materialFriction = sqrtf(sphere->friction * other->friction);
        if (materialFriction <= 0.0f)
            continue;

        // Contact existence + box->sphere normal at q-.
        float3 normal;
        bool hit = false;
        if (other->shape.type == RIGID_SHAPE_CYLINDER)
        {
            float3 axis = rotate(other->initialAng, float3{0.0f, 0.0f, 1.0f});
            float3 segA = other->initialLin - axis * other->shape.halfLength;
            float3 segB = other->initialLin + axis * other->shape.halfLength;
            float3 ab = segB - segA;
            float denom = lengthSq(ab);
            float3 closest = segA;
            if (denom >= 1.0e-6f)
                closest = segA + ab * clamp(dot(sphere->initialLin - segA, ab) / denom, 0.0f, 1.0f);
            float3 d = sphere->initialLin - closest;
            float rSum = sphere->shape.radius + other->shape.radius;
            if (lengthSq(d) <= rSum * rSum)
            {
                hit = true;
                float len = sqrtf(lengthSq(d));
                normal = len > 1.0e-3f ? d / len : float3{0.0f, 0.0f, 1.0f};
            }
        }
        else
        {
            float3 half = other->size * 0.5f;
            quat invRot = conjugate(other->initialAng);
            float3 local = rotate(invRot, sphere->initialLin - other->initialLin);
            float3 clamped = {clamp(local.x, -half.x, half.x), clamp(local.y, -half.y, half.y), clamp(local.z, -half.z, half.z)};
            float3 d = local - clamped;
            bool inside = d.x == 0.0f && d.y == 0.0f && d.z == 0.0f;
            if (inside || lengthSq(d) <= sphere->shape.radius * sphere->shape.radius)
            {
                hit = true;
                if (inside)
                {
                    normal = float3{0.0f, 0.0f, 1.0f};
                }
                else
                {
                    float3 worldD = rotate(other->initialAng, d);
                    float len = sqrtf(lengthSq(worldD));
                    normal = len > 1.0e-6f ? worldD / len : float3{0.0f, 0.0f, 1.0f};
                }
            }
        }
        if (!hit)
            continue;

        float blend = clamp(materialFriction * 0.08f, 0.0f, 0.18f);
        float3 rWorld = -normal * sphere->shape.radius;
        float3 contactVelocity = sphere->velocityLin + cross(sphere->velocityAng, rWorld);
        float3 tangentVelocity = contactVelocity - normal * dot(contactVelocity, normal);
        sphere->velocityLin -= tangentVelocity * blend;
        float3 desiredOmega = cross(normal, sphere->velocityLin) / sphere->shape.radius;
        sphere->velocityAng += (desiredOmega - sphere->velocityAng) * blend;
        solver.world.updateBodyFromRigid(sphere);
    }
}

void WebGpuAvbdBackend::step(Solver &solver)
{
    if (ctx == nullptr || !ctx->deviceReady || gpuBroken || !ensurePipelines())
    {
        solver.stepCpuReference();
        return;
    }
    errorCountAtStepStart = ctx->errorCount;

    // Route broadphase-overlapping sphere-sphere pairs to the GPU
    // narrowphase instead of CPU manifolds.
    frameIndex++;
    spherePairScratch.clear();
    solver.spherePairSink = &spherePairScratch;

    // finishStep's end-of-step sync leaves SimWorld current, and nothing in
    // prepare/solve reads it, so skip the redundant start-of-step sync.
    solver.prepareStep(true);
    solver.spherePairSink = nullptr;
    Clock::time_point buildBegin = Clock::now();
    buildFrame(solver);
    g_avbdGpuStats.buildFrameMs = elapsedMsAvbd(buildBegin, Clock::now());

    bool gpuOk = false;
    g_avbdGpuStats.submitMs = 0.0f;
    g_avbdGpuStats.waitMs = 0.0f;
    g_avbdGpuStats.applyMs = 0.0f;
    if (!solvedBodiesFlat.empty() && (uint32_t)colorRanges.size() <= MAX_COLOR_SLOTS)
    {
        Clock::time_point begin = Clock::now();
        frameWaitMs = 0.0f;
        bool submitted = runGpuIterations(solver);
        g_avbdGpuStats.submitMs = elapsedMsAvbd(begin, Clock::now());
        Clock::time_point readbackBegin = Clock::now();
        gpuOk = submitted && readbackAndApply(solver);
        float readbackMs = elapsedMsAvbd(readbackBegin, Clock::now());
        g_avbdGpuStats.waitMs = frameWaitMs;
        g_avbdGpuStats.applyMs = readbackMs - frameWaitMs;
        if (gpuOk)
            solver.stats.primalSolveMs = elapsedMsAvbd(begin, Clock::now());
    }

    if (!gpuOk && !solvedBodiesFlat.empty())
    {
        if (ctx->errorCount != errorCountAtStepStart)
        {
            // Device errors are not transient: route every following step
            // through the full CPU reference path (with CPU sphere manifolds)
            // instead of limping on without the GPU-resident contacts.
            gpuBroken = true;
            std::fprintf(stderr, "WebGPU AVBD: device errors detected (%s); falling back to CPU permanently\n",
                         ctx->statusText());
        }
        else if (!warnedFallback)
        {
            std::fprintf(stderr, "WebGPU AVBD: GPU iteration unavailable, using CPU iterate fallback\n");
            warnedFallback = true;
        }
        g_avbdGpuStats.cpuIterateFallbacks++;
        solver.iteratePrimalDualCpu();
    }

    g_avbdGpuStats.active = true;
    g_avbdGpuStats.gpuIterateMs = gpuOk ? solver.stats.primalSolveMs : 0.0f;
    g_avbdGpuStats.bodies = (int)solvedBodiesFlat.size();
    g_avbdGpuStats.contacts = (int)(gpuContacts.size() + spherePairScratch.size());
    g_avbdGpuStats.spherePairs = (int)spherePairScratch.size();
    g_avbdGpuStats.joints = (int)gpuJoints.size();
    g_avbdGpuStats.springs = (int)gpuSprings.size();
    g_avbdGpuStats.colors = (int)colorRanges.size();

    solver.finishStep();

    if (gpuOk)
        applyGpuPairRollingFriction(solver);
}

} // namespace

std::unique_ptr<PhysicsBackend> makeWebGpuAvbdPhysicsBackend(WebGpuDevice *device)
{
    if (device == nullptr)
        return std::unique_ptr<PhysicsBackend>(new CpuFallbackAvbdBackend());
    return std::unique_ptr<PhysicsBackend>(new WebGpuAvbdBackend(device));
}

#else // !(AVBD_ENABLE_WEBGPU && AVBD_HAS_DAWN)

std::unique_ptr<PhysicsBackend> makeWebGpuAvbdPhysicsBackend(WebGpuDevice *device)
{
    (void)device;
    return std::unique_ptr<PhysicsBackend>(new CpuFallbackAvbdBackend());
}

#endif
