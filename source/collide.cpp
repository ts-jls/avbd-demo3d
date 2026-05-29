/*
* Copyright (c) 2026 Chris Giles
*
* Permission to use, copy, modify, distribute and sell this software
* and its documentation for any purpose is hereby granted without fee,
* provided that the above copyright notice appear in all copies.
* Chris Giles makes no representations about the suitability
* of this software for any purpose.
* It is provided "as is" without express or implied warranty.
*/

#include "solver.h"
#include <cfloat>
#include <cmath>

namespace
{
constexpr int MAX_CONTACTS = 8;
constexpr int MAX_POLY_VERTS = 16;
constexpr float SAT_AXIS_EPSILON = 1.0e-6f;
constexpr float PLANE_EPSILON = 1.0e-5f;
constexpr float CONTACT_MERGE_DIST_SQ = 1.0e-6f;

enum AxisType
{
    AXIS_FACE_A,
    AXIS_FACE_B,
    AXIS_EDGE
};

struct OBB
{
    float3 center;
    quat rotation;
    float3 half;
    float3 axis[3];
};

struct SatAxis
{
    AxisType type;
    int indexA;
    int indexB;
    float separation;
    float3 normalAB;
    bool valid;
};

struct FaceFrame
{
    int axisIndex;
    float3 normal;
    float3 center;
    float3 u;
    float3 v;
    float extentU;
    float extentV;
};

inline OBB makeOBB(const Rigid* body)
{
    OBB box{};
    box.center = body->positionLin;
    box.rotation = body->positionAng;
    box.half = body->size * 0.5f;
    box.axis[0] = rotate(body->positionAng, float3{ 1.0f, 0.0f, 0.0f });
    box.axis[1] = rotate(body->positionAng, float3{ 0.0f, 1.0f, 0.0f });
    box.axis[2] = rotate(body->positionAng, float3{ 0.0f, 0.0f, 1.0f });
    return box;
}

inline float absDot(float3 a, float3 b)
{
    return fabsf(dot(a, b));
}

inline float3 supportPoint(const OBB& box, const float3& dir)
{
    float sx = dot(dir, box.axis[0]) >= 0.0f ? 1.0f : -1.0f;
    float sy = dot(dir, box.axis[1]) >= 0.0f ? 1.0f : -1.0f;
    float sz = dot(dir, box.axis[2]) >= 0.0f ? 1.0f : -1.0f;

    return box.center
        + box.axis[0] * (box.half.x * sx)
        + box.axis[1] * (box.half.y * sy)
        + box.axis[2] * (box.half.z * sz);
}

inline void getFaceAxes(const OBB& box, int axisIndex, float3& u, float3& v, float& extentU, float& extentV)
{
    if (axisIndex == 0)
    {
        u = box.axis[1];
        v = box.axis[2];
        extentU = box.half.y;
        extentV = box.half.z;
    }
    else if (axisIndex == 1)
    {
        u = box.axis[0];
        v = box.axis[2];
        extentU = box.half.x;
        extentV = box.half.z;
    }
    else
    {
        u = box.axis[0];
        v = box.axis[1];
        extentU = box.half.x;
        extentV = box.half.y;
    }
}

inline void buildFaceFrame(const OBB& box, int axisIndex, const float3& outwardNormal, FaceFrame& frame)
{
    float sign = dot(outwardNormal, box.axis[axisIndex]) >= 0.0f ? 1.0f : -1.0f;
    frame.axisIndex = axisIndex;
    frame.normal = box.axis[axisIndex] * sign;
    frame.center = box.center + frame.normal * box.half[axisIndex];
    getFaceAxes(box, axisIndex, frame.u, frame.v, frame.extentU, frame.extentV);
}

inline int chooseIncidentFaceAxis(const OBB& box, const float3& referenceNormal)
{
    int axis = 0;
    float best = -FLT_MAX;

    for (int i = 0; i < 3; ++i)
    {
        float d = absDot(box.axis[i], referenceNormal);
        if (d > best)
        {
            best = d;
            axis = i;
        }
    }

    return axis;
}

inline void buildIncidentFace(const OBB& box, int axisIndex, const float3& referenceNormal, float3 outVerts[4])
{
    float sign = dot(box.axis[axisIndex], referenceNormal) > 0.0f ? -1.0f : 1.0f;
    float3 faceNormal = box.axis[axisIndex] * sign;
    float3 faceCenter = box.center + faceNormal * box.half[axisIndex];

    float3 u;
    float3 v;
    float extentU;
    float extentV;
    getFaceAxes(box, axisIndex, u, v, extentU, extentV);

    outVerts[0] = faceCenter + u * extentU + v * extentV;
    outVerts[1] = faceCenter - u * extentU + v * extentV;
    outVerts[2] = faceCenter - u * extentU - v * extentV;
    outVerts[3] = faceCenter + u * extentU - v * extentV;
}

inline int clipPolygonAgainstPlane(const float3* inVerts, int inCount, const float3& planeNormal, float planeOffset, float3* outVerts)
{
    if (inCount <= 0)
        return 0;

    int outCount = 0;
    float3 a = inVerts[inCount - 1];
    float da = dot(planeNormal, a) - planeOffset;

    for (int i = 0; i < inCount; ++i)
    {
        float3 b = inVerts[i];
        float db = dot(planeNormal, b) - planeOffset;

        bool aInside = da <= PLANE_EPSILON;
        bool bInside = db <= PLANE_EPSILON;

        if (aInside != bInside)
        {
            float t = 0.0f;
            float denom = da - db;
            if (fabsf(denom) > SAT_AXIS_EPSILON)
                t = clamp(da / denom, 0.0f, 1.0f);

            if (outCount < MAX_POLY_VERTS)
                outVerts[outCount++] = a + (b - a) * t;
        }

        if (bInside && outCount < MAX_POLY_VERTS)
            outVerts[outCount++] = b;

        a = b;
        da = db;
    }

    return outCount;
}

inline bool addContact(Rigid* bodyA, Rigid* bodyB, Manifold::Contact* contacts, int& contactCount, float3* contactMidpoints, float3 xA, float3 xB, int featureKey)
{
    float3 midpoint = (xA + xB) * 0.5f;

    for (int i = 0; i < contactCount; ++i)
    {
        float3 d = midpoint - contactMidpoints[i];
        if (lengthSq(d) < CONTACT_MERGE_DIST_SQ)
            return false;
    }

    if (contactCount >= MAX_CONTACTS)
        return false;

    Manifold::FeaturePair feature{};
    feature.key = featureKey;

    Manifold::Contact& c = contacts[contactCount];
    c.feature = feature;
    c.rA = rotate(conjugate(bodyA->positionAng), xA - bodyA->positionLin);
    c.rB = rotate(conjugate(bodyB->positionAng), xB - bodyB->positionLin);
    contactMidpoints[contactCount] = midpoint;
    ++contactCount;

    return true;
}

inline bool testAxis(const OBB& boxA, const OBB& boxB, const float3& delta, const float3& axis, AxisType type, int indexA, int indexB, SatAxis& best)
{
    float lenSq = lengthSq(axis);
    if (lenSq < SAT_AXIS_EPSILON)
        return true;

    float invLen = 1.0f / sqrtf(lenSq);
    float3 n = axis * invLen;
    if (dot(n, delta) < 0.0f)
        n = -n;

    float distance = fabsf(dot(delta, n));

    float rA =
        boxA.half.x * absDot(n, boxA.axis[0]) +
        boxA.half.y * absDot(n, boxA.axis[1]) +
        boxA.half.z * absDot(n, boxA.axis[2]);

    float rB =
        boxB.half.x * absDot(n, boxB.axis[0]) +
        boxB.half.y * absDot(n, boxB.axis[1]) +
        boxB.half.z * absDot(n, boxB.axis[2]);

    float separation = distance - (rA + rB);
    if (separation > 0.0f)
        return false;

    if (!best.valid || separation > best.separation)
    {
        best.valid = true;
        best.type = type;
        best.indexA = indexA;
        best.indexB = indexB;
        best.separation = separation;
        best.normalAB = n;
    }

    return true;
}

inline void supportEdge(const OBB& box, int axisIndex, const float3& dir, float3& edgeA, float3& edgeB)
{
    int axis1 = (axisIndex + 1) % 3;
    int axis2 = (axisIndex + 2) % 3;

    float sign1 = dot(dir, box.axis[axis1]) >= 0.0f ? 1.0f : -1.0f;
    float sign2 = dot(dir, box.axis[axis2]) >= 0.0f ? 1.0f : -1.0f;

    float3 edgeCenter = box.center
        + box.axis[axis1] * (box.half[axis1] * sign1)
        + box.axis[axis2] * (box.half[axis2] * sign2);

    edgeA = edgeCenter - box.axis[axisIndex] * box.half[axisIndex];
    edgeB = edgeCenter + box.axis[axisIndex] * box.half[axisIndex];
}

inline void closestPointsOnSegments(const float3& p0, const float3& p1, const float3& q0, const float3& q1, float3& c0, float3& c1)
{
    float3 d1 = p1 - p0;
    float3 d2 = q1 - q0;
    float3 r = p0 - q0;
    float a = dot(d1, d1);
    float e = dot(d2, d2);
    float f = dot(d2, r);

    float s = 0.0f;
    float t = 0.0f;

    if (a <= SAT_AXIS_EPSILON && e <= SAT_AXIS_EPSILON)
    {
        c0 = p0;
        c1 = q0;
        return;
    }

    if (a <= SAT_AXIS_EPSILON)
    {
        t = clamp(f / e, 0.0f, 1.0f);
    }
    else
    {
        float c = dot(d1, r);
        if (e <= SAT_AXIS_EPSILON)
        {
            s = clamp(-c / a, 0.0f, 1.0f);
        }
        else
        {
            float b = dot(d1, d2);
            float denom = a * e - b * b;

            if (fabsf(denom) > SAT_AXIS_EPSILON)
                s = clamp((b * f - c * e) / denom, 0.0f, 1.0f);

            t = (b * s + f) / e;

            if (t < 0.0f)
            {
                t = 0.0f;
                s = clamp(-c / a, 0.0f, 1.0f);
            }
            else if (t > 1.0f)
            {
                t = 1.0f;
                s = clamp((b - c) / a, 0.0f, 1.0f);
            }
        }
    }

    c0 = p0 + d1 * s;
    c1 = q0 + d2 * t;
}

inline int buildFaceManifold(Rigid* bodyA, Rigid* bodyB, const OBB& boxA, const OBB& boxB, bool referenceIsA, int referenceAxis, const float3& normalAB, Manifold::Contact* contacts)
{
    const OBB& referenceBox = referenceIsA ? boxA : boxB;
    const OBB& incidentBox = referenceIsA ? boxB : boxA;
    float3 referenceOutward = referenceIsA ? normalAB : -normalAB;

    FaceFrame referenceFace{};
    buildFaceFrame(referenceBox, referenceAxis, referenceOutward, referenceFace);

    int incidentAxis = chooseIncidentFaceAxis(incidentBox, referenceFace.normal);

    float3 clip0[MAX_POLY_VERTS];
    float3 clip1[MAX_POLY_VERTS];
    buildIncidentFace(incidentBox, incidentAxis, referenceFace.normal, clip0);
    int count = 4;

    float3 n0 = referenceFace.u;
    float o0 = dot(n0, referenceFace.center) + referenceFace.extentU;
    count = clipPolygonAgainstPlane(clip0, count, n0, o0, clip1);
    if (!count)
        return 0;

    float3 n1 = -referenceFace.u;
    float o1 = dot(n1, referenceFace.center) + referenceFace.extentU;
    count = clipPolygonAgainstPlane(clip1, count, n1, o1, clip0);
    if (!count)
        return 0;

    float3 n2 = referenceFace.v;
    float o2 = dot(n2, referenceFace.center) + referenceFace.extentV;
    count = clipPolygonAgainstPlane(clip0, count, n2, o2, clip1);
    if (!count)
        return 0;

    float3 n3 = -referenceFace.v;
    float o3 = dot(n3, referenceFace.center) + referenceFace.extentV;
    count = clipPolygonAgainstPlane(clip1, count, n3, o3, clip0);
    if (!count)
        return 0;

    int contactCount = 0;
    float3 contactMidpoints[MAX_CONTACTS];
    int featurePrefix = (referenceIsA ? AXIS_FACE_A : AXIS_FACE_B) << 24;
    featurePrefix |= (referenceAxis & 0xFF) << 16;
    featurePrefix |= (incidentAxis & 0xFF) << 8;

    for (int i = 0; i < count && contactCount < MAX_CONTACTS; ++i)
    {
        float3 pIncident = clip0[i];
        float distance = dot(pIncident - referenceFace.center, referenceFace.normal);
        if (distance > PLANE_EPSILON)
            continue;

        float3 pReference = pIncident - referenceFace.normal * distance;
        float3 xA = referenceIsA ? pReference : pIncident;
        float3 xB = referenceIsA ? pIncident : pReference;

        addContact(bodyA, bodyB, contacts, contactCount, contactMidpoints, xA, xB, featurePrefix | (i & 0xFF));
    }

    if (!contactCount)
    {
        float3 xA = supportPoint(boxA, normalAB);
        float3 xB = supportPoint(boxB, -normalAB);
        addContact(bodyA, bodyB, contacts, contactCount, contactMidpoints, xA, xB, featurePrefix);
    }

    return contactCount;
}

inline int buildEdgeContact(Rigid* bodyA, Rigid* bodyB, const OBB& boxA, const OBB& boxB, int axisA, int axisB, const float3& normalAB, Manifold::Contact* contacts)
{
    float3 a0;
    float3 a1;
    float3 b0;
    float3 b1;
    supportEdge(boxA, axisA, normalAB, a0, a1);
    supportEdge(boxB, axisB, -normalAB, b0, b1);

    float3 xA;
    float3 xB;
    closestPointsOnSegments(a0, a1, b0, b1, xA, xB);

    int contactCount = 0;
    float3 contactMidpoints[MAX_CONTACTS];
    int featureKey = (AXIS_EDGE << 24) | ((axisA & 0xFF) << 8) | (axisB & 0xFF);
    addContact(bodyA, bodyB, contacts, contactCount, contactMidpoints, xA, xB, featureKey);

    if (!contactCount)
    {
        xA = supportPoint(boxA, normalAB);
        xB = supportPoint(boxB, -normalAB);
        addContact(bodyA, bodyB, contacts, contactCount, contactMidpoints, xA, xB, featureKey);
    }

    return contactCount;
}

inline float3 closestPointOnOBB(const OBB& box, const float3& point)
{
    float3 d = point - box.center;
    return box.center
        + box.axis[0] * clamp(dot(d, box.axis[0]), -box.half.x, box.half.x)
        + box.axis[1] * clamp(dot(d, box.axis[1]), -box.half.y, box.half.y)
        + box.axis[2] * clamp(dot(d, box.axis[2]), -box.half.z, box.half.z);
}

inline bool pointInsideOBB(const OBB& box, const float3& point, float3& local)
{
    float3 d = point - box.center;
    local = {dot(d, box.axis[0]), dot(d, box.axis[1]), dot(d, box.axis[2])};
    return fabsf(local.x) <= box.half.x
        && fabsf(local.y) <= box.half.y
        && fabsf(local.z) <= box.half.z;
}

inline float3 localToOBBWorld(const OBB& box, const float3& local)
{
    return box.center + box.axis[0] * local.x + box.axis[1] * local.y + box.axis[2] * local.z;
}

inline bool isRoundShape(const Rigid* body)
{
    return body->shape.type == RIGID_SHAPE_SPHERE || body->shape.type == RIGID_SHAPE_CAPSULE;
}

inline bool isCylinderShape(const Rigid* body)
{
    return body->shape.type == RIGID_SHAPE_CYLINDER;
}

inline void capsuleSegment(Rigid* body, float3& a, float3& b)
{
    if (body->shape.type == RIGID_SHAPE_CAPSULE)
    {
        float3 axis = rotate(body->positionAng, float3{0.0f, 0.0f, 1.0f});
        a = body->positionLin - axis * body->shape.halfLength;
        b = body->positionLin + axis * body->shape.halfLength;
    }
    else
    {
        a = body->positionLin;
        b = body->positionLin;
    }
}

inline void cylinderSegment(Rigid* body, float3& a, float3& b)
{
    float3 axis = rotate(body->positionAng, float3{0.0f, 0.0f, 1.0f});
    a = body->positionLin - axis * body->shape.halfLength;
    b = body->positionLin + axis * body->shape.halfLength;
}

inline float3 closestPointOnSegment(float3 a, float3 b, float3 p)
{
    float3 ab = b - a;
    float denom = lengthSq(ab);
    if (denom < SAT_AXIS_EPSILON)
        return a;
    float t = clamp(dot(p - a, ab) / denom, 0.0f, 1.0f);
    return a + ab * t;
}

inline int addRoundContact(Rigid* bodyA, Rigid* bodyB, float3 centerA, float radiusA, float3 centerB, float radiusB, int featureKey, Manifold::Contact* contacts, float3x3& basisOut)
{
    float3 delta = centerB - centerA;
    float radiusSum = radiusA + radiusB;
    float distSq = lengthSq(delta);
    if (distSq > radiusSum * radiusSum)
        return 0;

    float3 normalAB = distSq > SAT_AXIS_EPSILON ? delta / sqrtf(distSq) : float3{1.0f, 0.0f, 0.0f};
    float3 xA = centerA + normalAB * radiusA;
    float3 xB = centerB - normalAB * radiusB;

    basisOut = orthonormal(-normalAB);
    int contactCount = 0;
    float3 contactMidpoints[MAX_CONTACTS];
    addContact(bodyA, bodyB, contacts, contactCount, contactMidpoints, xA, xB, featureKey);
    return contactCount;
}

inline bool sweptSphereBoxContact(const OBB& box, Rigid* sphere, Rigid* boxBody, float3& normalBoxToSphere, float3& xBox)
{
    float3 relativeVelocity = sphere->velocityLin - boxBody->velocityLin;
    float dt = sphere->solver ? sphere->solver->dt : 0.0f;
    if (dt <= 0.0f || lengthSq(relativeVelocity) < SAT_AXIS_EPSILON)
        return false;

    float3 current = sphere->positionLin;
    float3 previous = current - relativeVelocity * dt;
    float3 p0 = previous - box.center;
    float3 p1 = current - box.center;
    float3 local0 = {dot(p0, box.axis[0]), dot(p0, box.axis[1]), dot(p0, box.axis[2])};
    float3 local1 = {dot(p1, box.axis[0]), dot(p1, box.axis[1]), dot(p1, box.axis[2])};
    float3 delta = local1 - local0;
    float3 expandedHalf = box.half + float3{sphere->shape.radius, sphere->shape.radius, sphere->shape.radius};

    float tEnter = 0.0f;
    float tExit = 1.0f;
    int hitAxis = -1;
    float hitSign = 1.0f;

    for (int axis = 0; axis < 3; ++axis)
    {
        if (fabsf(delta[axis]) < SAT_AXIS_EPSILON)
        {
            if (local0[axis] < -expandedHalf[axis] || local0[axis] > expandedHalf[axis])
                return false;
            continue;
        }

        float invD = 1.0f / delta[axis];
        float t0 = (-expandedHalf[axis] - local0[axis]) * invD;
        float t1 = (expandedHalf[axis] - local0[axis]) * invD;
        float sign = -1.0f;
        if (t0 > t1)
        {
            float tmp = t0;
            t0 = t1;
            t1 = tmp;
            sign = 1.0f;
        }

        if (t0 > tEnter)
        {
            tEnter = t0;
            hitAxis = axis;
            hitSign = sign;
        }
        tExit = min(tExit, t1);
        if (tEnter > tExit)
            return false;
    }

    if (hitAxis < 0 || tEnter < 0.0f || tEnter > 1.0f)
        return false;

    float3 hitLocal = local0 + delta * tEnter;
    float3 surfaceLocal = hitLocal;
    surfaceLocal[hitAxis] = box.half[hitAxis] * hitSign;
    for (int axis = 0; axis < 3; ++axis)
    {
        if (axis != hitAxis)
            surfaceLocal[axis] = clamp(surfaceLocal[axis], -box.half[axis], box.half[axis]);
    }

    normalBoxToSphere = box.axis[hitAxis] * hitSign;
    xBox = localToOBBWorld(box, surfaceLocal);
    return true;
}

inline bool thinStaticBoxTopContact(const OBB& box, Rigid* sphere, Rigid* boxBody, float3& normalBoxToSphere, float3& xBox)
{
    if (boxBody->mass > 0.0f)
        return false;

    int thinAxis = 0;
    if (box.half.y < box.half[thinAxis])
        thinAxis = 1;
    if (box.half.z < box.half[thinAxis])
        thinAxis = 2;

    int axisA = (thinAxis + 1) % 3;
    int axisB = (thinAxis + 2) % 3;
    if (box.half[thinAxis] * 4.0f > min(box.half[axisA], box.half[axisB]))
        return false;

    float3 d = sphere->positionLin - box.center;
    float3 local = {dot(d, box.axis[0]), dot(d, box.axis[1]), dot(d, box.axis[2])};
    float sign = dot(box.axis[thinAxis], float3{0.0f, 0.0f, 1.0f}) >= 0.0f ? 1.0f : -1.0f;
    float signedDistance = local[thinAxis] * sign - box.half[thinAxis];
    if (signedDistance > sphere->shape.radius)
        return false;

    for (int axis = 0; axis < 3; ++axis)
    {
        if (axis == thinAxis)
            continue;
        if (local[axis] < -box.half[axis] - sphere->shape.radius || local[axis] > box.half[axis] + sphere->shape.radius)
            return false;
    }

    local[thinAxis] = box.half[thinAxis] * sign;
    for (int axis = 0; axis < 3; ++axis)
    {
        if (axis != thinAxis)
            local[axis] = clamp(local[axis], -box.half[axis], box.half[axis]);
    }

    normalBoxToSphere = box.axis[thinAxis] * sign;
    xBox = localToOBBWorld(box, local);
    return true;
}

inline bool thinStaticBoxTopContactRound(const OBB& box, Rigid* roundBody, Rigid* boxBody, float3& roundCenter, float3& normalBoxToRound, float3& xBox)
{
    if (boxBody->mass > 0.0f)
        return false;

    int thinAxis = 0;
    if (box.half.y < box.half[thinAxis])
        thinAxis = 1;
    if (box.half.z < box.half[thinAxis])
        thinAxis = 2;

    int axisA = (thinAxis + 1) % 3;
    int axisB = (thinAxis + 2) % 3;
    if (box.half[thinAxis] * 4.0f > min(box.half[axisA], box.half[axisB]))
        return false;

    float sign = dot(box.axis[thinAxis], float3{0.0f, 0.0f, 1.0f}) >= 0.0f ? 1.0f : -1.0f;
    float3 normal = box.axis[thinAxis] * sign;

    float3 segA;
    float3 segB;
    capsuleSegment(roundBody, segA, segB);
    float signedA = dot(segA, normal);
    float signedB = dot(segB, normal);
    if (roundBody->shape.type == RIGID_SHAPE_CAPSULE && fabsf(signedA - signedB) < 1.0e-4f)
        roundCenter = closestPointOnSegment(segA, segB, roundBody->positionLin);
    else
        roundCenter = signedA <= signedB ? segA : segB;

    float3 d = roundCenter - box.center;
    float3 local = {dot(d, box.axis[0]), dot(d, box.axis[1]), dot(d, box.axis[2])};
    float signedDistance = local[thinAxis] * sign - box.half[thinAxis];
    if (signedDistance > roundBody->shape.radius)
        return false;

    for (int axis = 0; axis < 3; ++axis)
    {
        if (axis == thinAxis)
            continue;
        if (local[axis] < -box.half[axis] - roundBody->shape.radius || local[axis] > box.half[axis] + roundBody->shape.radius)
            return false;
    }

    local[thinAxis] = box.half[thinAxis] * sign;
    for (int axis = 0; axis < 3; ++axis)
    {
        if (axis != thinAxis)
            local[axis] = clamp(local[axis], -box.half[axis], box.half[axis]);
    }

    normalBoxToRound = normal;
    xBox = localToOBBWorld(box, local);
    return true;
}

inline bool addThinStaticBoxTopRoundContact(const OBB& box, Rigid* roundBody, Rigid* boxBody, bool roundIsA, int thinAxis, float sign, const float3& normal, float3 roundCenter, int featureKey, Manifold::Contact* contacts, int& contactCount, float3* contactMidpoints)
{
    float3 d = roundCenter - box.center;
    float3 local = {dot(d, box.axis[0]), dot(d, box.axis[1]), dot(d, box.axis[2])};
    float signedDistance = local[thinAxis] * sign - box.half[thinAxis];
    if (signedDistance > roundBody->shape.radius)
        return false;

    for (int axis = 0; axis < 3; ++axis)
    {
        if (axis == thinAxis)
            continue;
        if (local[axis] < -box.half[axis] - roundBody->shape.radius || local[axis] > box.half[axis] + roundBody->shape.radius)
            return false;
    }

    local[thinAxis] = box.half[thinAxis] * sign;
    for (int axis = 0; axis < 3; ++axis)
    {
        if (axis != thinAxis)
            local[axis] = clamp(local[axis], -box.half[axis], box.half[axis]);
    }

    float3 xRound = roundCenter - normal * roundBody->shape.radius;
    float3 xBox = localToOBBWorld(box, local);
    float3 xA = roundIsA ? xRound : xBox;
    float3 xB = roundIsA ? xBox : xRound;
    return addContact(roundIsA ? roundBody : boxBody, roundIsA ? boxBody : roundBody, contacts, contactCount, contactMidpoints, xA, xB, featureKey);
}

inline int collideThinStaticBoxTopRound(const OBB& box, Rigid* roundBody, Rigid* boxBody, bool roundIsA, Manifold::Contact* contacts, float3x3& basisOut)
{
    if (boxBody->mass > 0.0f)
        return -1;

    int thinAxis = 0;
    if (box.half.y < box.half[thinAxis])
        thinAxis = 1;
    if (box.half.z < box.half[thinAxis])
        thinAxis = 2;

    int axisA = (thinAxis + 1) % 3;
    int axisB = (thinAxis + 2) % 3;
    if (box.half[thinAxis] * 4.0f > min(box.half[axisA], box.half[axisB]))
        return -1;

    float sign = dot(box.axis[thinAxis], float3{0.0f, 0.0f, 1.0f}) >= 0.0f ? 1.0f : -1.0f;
    float3 normal = box.axis[thinAxis] * sign;
    float3 normalAB = roundIsA ? -normal : normal;
    basisOut = orthonormal(-normalAB);

    int contactCount = 0;
    float3 contactMidpoints[MAX_CONTACTS];
    if (roundBody->shape.type == RIGID_SHAPE_CAPSULE)
    {
        float3 segA;
        float3 segB;
        capsuleSegment(roundBody, segA, segB);
        addThinStaticBoxTopRoundContact(box, roundBody, boxBody, roundIsA, thinAxis, sign, normal, segA, 0x04000000, contacts, contactCount, contactMidpoints);
        addThinStaticBoxTopRoundContact(box, roundBody, boxBody, roundIsA, thinAxis, sign, normal, segB, 0x04000001, contacts, contactCount, contactMidpoints);
    }
    else
    {
        addThinStaticBoxTopRoundContact(box, roundBody, boxBody, roundIsA, thinAxis, sign, normal, roundBody->positionLin, 0x02000000, contacts, contactCount, contactMidpoints);
    }

    return contactCount > 0 ? contactCount : -1;
}

inline int collideRoundRound(Rigid* bodyA, Rigid* bodyB, Manifold::Contact* contacts, float3x3& basisOut)
{
    float3 a0, a1, b0, b1;
    capsuleSegment(bodyA, a0, a1);
    capsuleSegment(bodyB, b0, b1);
    float3 centerA;
    float3 centerB;
    closestPointsOnSegments(a0, a1, b0, b1, centerA, centerB);
    return addRoundContact(bodyA, bodyB, centerA, bodyA->shape.radius, centerB, bodyB->shape.radius, 0x03000000, contacts, basisOut);
}

inline int collideRoundBox(Rigid* roundBody, Rigid* boxBody, bool roundIsA, Manifold::Contact* contacts, float3x3& basisOut)
{
    OBB box = makeOBB(boxBody);
    int topContactCount = collideThinStaticBoxTopRound(box, roundBody, boxBody, roundIsA, contacts, basisOut);
    if (topContactCount >= 0)
        return topContactCount;

    float3 roundCenter = roundBody->positionLin;
    if (roundBody->shape.type == RIGID_SHAPE_CAPSULE)
    {
        float3 segA, segB;
        capsuleSegment(roundBody, segA, segB);
        float3 closestA = closestPointOnOBB(box, segA);
        float3 closestB = closestPointOnOBB(box, segB);
        roundCenter = lengthSq(segA - closestA) <= lengthSq(segB - closestB) ? segA : segB;
    }

    float roundRadius = roundBody->shape.radius;
    float3 closest = closestPointOnOBB(box, roundCenter);
    float3 boxToRound = roundCenter - closest;
    float distSq = lengthSq(boxToRound);

    float3 local;
    float3 topNormal;
    float3 topPoint;
    float3 topRoundCenter;
    if (thinStaticBoxTopContactRound(box, roundBody, boxBody, topRoundCenter, topNormal, topPoint))
    {
        roundCenter = topRoundCenter;
        boxToRound = topNormal;
        closest = topPoint;
        distSq = 1.0f;
    }
    else if (pointInsideOBB(box, roundCenter, local))
    {
        float3 sweptNormal;
        float3 sweptBoxPoint;
        if (roundBody->shape.type == RIGID_SHAPE_SPHERE && sweptSphereBoxContact(box, roundBody, boxBody, sweptNormal, sweptBoxPoint))
        {
            boxToRound = sweptNormal;
            closest = sweptBoxPoint;
            distSq = 1.0f;
        }
        else
        {
            int axis = 0;
            float sign = local.x >= 0.0f ? 1.0f : -1.0f;
            float bestDistance = box.half.x - fabsf(local.x);
            float3 relativeVelocity = roundBody->velocityLin - boxBody->velocityLin;

            if (boxBody->mass <= 0.0f)
            {
                axis = 0;
                if (box.half.y < box.half[axis])
                    axis = 1;
                if (box.half.z < box.half[axis])
                    axis = 2;
                sign = dot(box.axis[axis], float3{0.0f, 0.0f, 1.0f}) >= 0.0f ? 1.0f : -1.0f;
            }
            else if (lengthSq(relativeVelocity) > SAT_AXIS_EPSILON)
            {
                float3 againstMotion = -relativeVelocity;
                float bestDot = -FLT_MAX;
                for (int candidateAxis = 0; candidateAxis < 3; ++candidateAxis)
                {
                    for (int candidateSignIndex = 0; candidateSignIndex < 2; ++candidateSignIndex)
                    {
                        float candidateSign = candidateSignIndex == 0 ? -1.0f : 1.0f;
                        float d = dot(box.axis[candidateAxis] * candidateSign, againstMotion);
                        if (d > bestDot)
                        {
                            bestDot = d;
                            axis = candidateAxis;
                            sign = candidateSign;
                        }
                    }
                }
            }
            else
            {

                float distanceY = box.half.y - fabsf(local.y);
                if (distanceY < bestDistance)
                {
                    axis = 1;
                    sign = local.y >= 0.0f ? 1.0f : -1.0f;
                    bestDistance = distanceY;
                }

                float distanceZ = box.half.z - fabsf(local.z);
                if (distanceZ < bestDistance)
                {
                    axis = 2;
                    sign = local.z >= 0.0f ? 1.0f : -1.0f;
                }
            }

            float3 surfaceLocal = local;
            surfaceLocal[axis] = box.half[axis] * sign;
            closest = localToOBBWorld(box, surfaceLocal);
            boxToRound = box.axis[axis] * sign;
            distSq = 1.0f;
        }
    }
    else if (distSq > roundRadius * roundRadius)
    {
        float3 sweptNormal;
        float3 sweptBoxPoint;
        if (roundBody->shape.type != RIGID_SHAPE_SPHERE || !sweptSphereBoxContact(box, roundBody, boxBody, sweptNormal, sweptBoxPoint))
            return 0;

        boxToRound = sweptNormal;
        closest = sweptBoxPoint;
        distSq = 1.0f;
    }

    float3 normalBoxToRound = distSq > SAT_AXIS_EPSILON ? boxToRound / sqrtf(distSq) : float3{0.0f, 0.0f, 1.0f};
    float3 normalAB = roundIsA ? -normalBoxToRound : normalBoxToRound;
    float3 xRound = roundCenter - normalBoxToRound * roundRadius;
    float3 xBox = closest;
    float3 xA = roundIsA ? xRound : xBox;
    float3 xB = roundIsA ? xBox : xRound;

    basisOut = orthonormal(-normalAB);
    int contactCount = 0;
    float3 contactMidpoints[MAX_CONTACTS];
    int featureKey = roundBody->shape.type == RIGID_SHAPE_CAPSULE ? 0x04000000 : 0x02000000;
    addContact(roundIsA ? roundBody : boxBody, roundIsA ? boxBody : roundBody, contacts, contactCount, contactMidpoints, xA, xB, featureKey);
    return contactCount;
}

inline int collideBoxes(Rigid* bodyA, Rigid* bodyB, Manifold::Contact* contacts, float3x3& basisOut)
{
    OBB boxA = makeOBB(bodyA);
    OBB boxB = makeOBB(bodyB);
    float3 delta = boxB.center - boxA.center;

    SatAxis bestFace{};
    bestFace.separation = -FLT_MAX;
    bestFace.valid = false;

    SatAxis bestEdge{};
    bestEdge.separation = -FLT_MAX;
    bestEdge.valid = false;

    for (int i = 0; i < 3; ++i)
    {
        if (!testAxis(boxA, boxB, delta, boxA.axis[i], AXIS_FACE_A, i, -1, bestFace))
            return 0;
    }

    for (int i = 0; i < 3; ++i)
    {
        if (!testAxis(boxA, boxB, delta, boxB.axis[i], AXIS_FACE_B, -1, i, bestFace))
            return 0;
    }

    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            float3 axis = cross(boxA.axis[i], boxB.axis[j]);
            if (!testAxis(boxA, boxB, delta, axis, AXIS_EDGE, i, j, bestEdge))
                return 0;
        }
    }

    if (!bestFace.valid)
        return 0;

    SatAxis best = bestFace;
    if (bestEdge.valid)
    {
        const float edgeRelTol = 0.95f;
        const float edgeAbsTol = 0.01f;
        if (edgeRelTol * bestEdge.separation > bestFace.separation + edgeAbsTol)
            best = bestEdge;
    }

    basisOut = orthonormal(-best.normalAB);

    if (best.type == AXIS_EDGE)
        return buildEdgeContact(bodyA, bodyB, boxA, boxB, best.indexA, best.indexB, best.normalAB, contacts);

    if (best.type == AXIS_FACE_A)
        return buildFaceManifold(bodyA, bodyB, boxA, boxB, true, best.indexA, best.normalAB, contacts);

    return buildFaceManifold(bodyA, bodyB, boxA, boxB, false, best.indexB, best.normalAB, contacts);
}

inline bool thinStaticBoxTopFrame(const OBB& box, Rigid* boxBody, int& thinAxis, float& topSign, float3& normal)
{
    if (boxBody->mass > 0.0f)
        return false;

    thinAxis = 0;
    if (box.half.y < box.half[thinAxis])
        thinAxis = 1;
    if (box.half.z < box.half[thinAxis])
        thinAxis = 2;

    int axisA = (thinAxis + 1) % 3;
    int axisB = (thinAxis + 2) % 3;
    if (box.half[thinAxis] * 4.0f > min(box.half[axisA], box.half[axisB]))
        return false;

    topSign = dot(box.axis[thinAxis], float3{0.0f, 0.0f, 1.0f}) >= 0.0f ? 1.0f : -1.0f;
    normal = box.axis[thinAxis] * topSign;
    return true;
}

inline bool addThinStaticBoxTopCylinderContact(const OBB& box, Rigid* cylinder, Rigid* boxBody, bool cylinderIsA, int thinAxis, float topSign, float3 xCylinder, int featureKey, Manifold::Contact* contacts, int& contactCount, float3* contactMidpoints)
{
    float3 local = {
        dot(xCylinder - box.center, box.axis[0]),
        dot(xCylinder - box.center, box.axis[1]),
        dot(xCylinder - box.center, box.axis[2])};
    float signedDistance = local[thinAxis] * topSign - box.half[thinAxis];
    if (signedDistance > COLLISION_MARGIN)
        return false;

    for (int axis = 0; axis < 3; ++axis)
    {
        if (axis == thinAxis)
            continue;
        if (local[axis] < -box.half[axis] - cylinder->shape.radius || local[axis] > box.half[axis] + cylinder->shape.radius)
            return false;
    }

    local[thinAxis] = (box.half[thinAxis] + COLLISION_MARGIN) * topSign;
    for (int axis = 0; axis < 3; ++axis)
    {
        if (axis != thinAxis)
            local[axis] = clamp(local[axis], -box.half[axis], box.half[axis]);
    }

    float3 xBox = localToOBBWorld(box, local);
    float3 xA = cylinderIsA ? xCylinder : xBox;
    float3 xB = cylinderIsA ? xBox : xCylinder;
    return addContact(cylinderIsA ? cylinder : boxBody, cylinderIsA ? boxBody : cylinder, contacts, contactCount, contactMidpoints, xA, xB, featureKey);
}

inline int collideThinStaticBoxTopCylinder(const OBB& box, Rigid* cylinder, Rigid* boxBody, bool cylinderIsA, Manifold::Contact* contacts, float3x3& basisOut)
{
    int thinAxis = 0;
    float topSign = 1.0f;
    float3 normal;
    if (!thinStaticBoxTopFrame(box, boxBody, thinAxis, topSign, normal))
        return -1;

    float3 normalAB = cylinderIsA ? -normal : normal;
    basisOut = orthonormal(-normalAB);

    float3 axis = rotate(cylinder->positionAng, float3{0.0f, 0.0f, 1.0f});
    float axisDot = dot(axis, normal);
    int contactCount = 0;
    float3 contactMidpoints[MAX_CONTACTS];

    if (fabsf(axisDot) > 0.65f)
    {
        float axisSign = axisDot >= 0.0f ? 1.0f : -1.0f;
        float3 capCenter = cylinder->positionLin - axis * (cylinder->shape.halfLength * axisSign);
        float3 u = fabsf(axis.x) > fabsf(axis.z) ? normalize(float3{-axis.y, axis.x, 0.0f}) : normalize(float3{0.0f, -axis.z, axis.y});
        float3 v = normalize(cross(axis, u));
        float3 offsets[4] = {
            u * cylinder->shape.radius,
            -u * cylinder->shape.radius,
            v * cylinder->shape.radius,
            -v * cylinder->shape.radius};
        for (int i = 0; i < 4; ++i)
            addThinStaticBoxTopCylinderContact(box, cylinder, boxBody, cylinderIsA, thinAxis, topSign, capCenter + offsets[i], 0x05000000 | i, contacts, contactCount, contactMidpoints);
    }
    else
    {
        float3 radial = normal - axis * axisDot;
        if (lengthSq(radial) < SAT_AXIS_EPSILON)
            radial = normal;
        radial = normalize(radial);

        float3 a = cylinder->positionLin - axis * cylinder->shape.halfLength;
        float3 b = cylinder->positionLin + axis * cylinder->shape.halfLength;
        addThinStaticBoxTopCylinderContact(box, cylinder, boxBody, cylinderIsA, thinAxis, topSign, a - radial * cylinder->shape.radius, 0x05000100, contacts, contactCount, contactMidpoints);
        addThinStaticBoxTopCylinderContact(box, cylinder, boxBody, cylinderIsA, thinAxis, topSign, b - radial * cylinder->shape.radius, 0x05000101, contacts, contactCount, contactMidpoints);
    }

    return contactCount > 0 ? contactCount : -1;
}

inline int collideCylinderBox(Rigid* cylinder, Rigid* boxBody, bool cylinderIsA, Manifold::Contact* contacts, float3x3& basisOut)
{
    OBB box = makeOBB(boxBody);
    int topContactCount = collideThinStaticBoxTopCylinder(box, cylinder, boxBody, cylinderIsA, contacts, basisOut);
    if (topContactCount >= 0)
        return topContactCount;

    return collideBoxes(cylinderIsA ? cylinder : boxBody, cylinderIsA ? boxBody : cylinder, contacts, basisOut);
}

inline int collideCylinderRound(Rigid* cylinder, Rigid* roundBody, bool cylinderIsA, Manifold::Contact* contacts, float3x3& basisOut)
{
    float3 cylA, cylB;
    cylinderSegment(cylinder, cylA, cylB);
    float3 roundA, roundB;
    capsuleSegment(roundBody, roundA, roundB);
    float3 centerCylinder;
    float3 centerRound;
    closestPointsOnSegments(cylA, cylB, roundA, roundB, centerCylinder, centerRound);

    if (cylinderIsA)
        return addRoundContact(cylinder, roundBody, centerCylinder, cylinder->shape.radius, centerRound, roundBody->shape.radius, 0x07000000, contacts, basisOut);
    return addRoundContact(roundBody, cylinder, centerRound, roundBody->shape.radius, centerCylinder, cylinder->shape.radius, 0x07000000, contacts, basisOut);
}

inline int addCylinderCapContacts(Rigid* bodyA, Rigid* bodyB, float3 normalAB, float3 capA, float3 capB, float radius, Manifold::Contact* contacts, float3x3& basisOut)
{
    basisOut = orthonormal(-normalAB);
    int contactCount = 0;
    float3 contactMidpoints[MAX_CONTACTS];
    float3 u = fabsf(normalAB.x) > fabsf(normalAB.z) ? normalize(float3{-normalAB.y, normalAB.x, 0.0f}) : normalize(float3{0.0f, -normalAB.z, normalAB.y});
    float3 v = normalize(cross(normalAB, u));
    float3 offsets[4] = {u * radius, -u * radius, v * radius, -v * radius};
    for (int i = 0; i < 4; ++i)
        addContact(bodyA, bodyB, contacts, contactCount, contactMidpoints, capA + offsets[i], capB + offsets[i], 0x06000000 | i);
    return contactCount;
}

inline int collideCylinders(Rigid* bodyA, Rigid* bodyB, Manifold::Contact* contacts, float3x3& basisOut)
{
    float3 axisA = rotate(bodyA->positionAng, float3{0.0f, 0.0f, 1.0f});
    float3 axisB = rotate(bodyB->positionAng, float3{0.0f, 0.0f, 1.0f});
    float parallel = dot(axisA, axisB);
    float3 delta = bodyB->positionLin - bodyA->positionLin;

    if (fabsf(parallel) > 0.95f)
    {
        if (parallel < 0.0f)
            axisB = -axisB;

        float axial = dot(delta, axisA);
        float3 radial = delta - axisA * axial;
        float radialLenSq = lengthSq(radial);
        float radiusSum = bodyA->shape.radius + bodyB->shape.radius;
        if (radialLenSq > radiusSum * radiusSum)
            return 0;

        float axialOverlap = bodyA->shape.halfLength + bodyB->shape.halfLength - fabsf(axial);
        if (axialOverlap < -COLLISION_MARGIN)
            return 0;

        if (fabsf(axial) > (bodyA->shape.halfLength + bodyB->shape.halfLength) * 0.5f)
        {
            float side = axial >= 0.0f ? 1.0f : -1.0f;
            float3 normalAB = axisA * side;
            float3 capA = bodyA->positionLin + normalAB * bodyA->shape.halfLength;
            float3 capB = bodyB->positionLin - normalAB * bodyB->shape.halfLength;
            return addCylinderCapContacts(bodyA, bodyB, normalAB, capA, capB, min(bodyA->shape.radius, bodyB->shape.radius) * 0.85f, contacts, basisOut);
        }

        float3 normalAB = radialLenSq > SAT_AXIS_EPSILON ? radial / sqrtf(radialLenSq) : orthonormal(axisA)[1];
        basisOut = orthonormal(-normalAB);
        int contactCount = 0;
        float3 contactMidpoints[MAX_CONTACTS];
        float lo = max(-bodyA->shape.halfLength, axial - bodyB->shape.halfLength);
        float hi = min(bodyA->shape.halfLength, axial + bodyB->shape.halfLength);
        float values[2] = {lo, hi};
        for (int i = 0; i < 2; ++i)
        {
            float tA = values[i];
            float tB = tA - axial;
            float3 xA = bodyA->positionLin + axisA * tA + normalAB * bodyA->shape.radius;
            float3 xB = bodyB->positionLin + axisA * tB - normalAB * bodyB->shape.radius;
            addContact(bodyA, bodyB, contacts, contactCount, contactMidpoints, xA, xB, 0x06000100 | i);
        }
        return contactCount;
    }

    float3 a0, a1, b0, b1;
    cylinderSegment(bodyA, a0, a1);
    cylinderSegment(bodyB, b0, b1);
    float3 centerA;
    float3 centerB;
    closestPointsOnSegments(a0, a1, b0, b1, centerA, centerB);
    return addRoundContact(bodyA, bodyB, centerA, bodyA->shape.radius, centerB, bodyB->shape.radius, 0x06000200, contacts, basisOut);
}

} // namespace

int Manifold::collide(Rigid* bodyA, Rigid* bodyB, Contact* contacts, float3x3& basisOut)
{
    if (isCylinderShape(bodyA) && isCylinderShape(bodyB))
        return collideCylinders(bodyA, bodyB, contacts, basisOut);

    if (isCylinderShape(bodyA) && isRoundShape(bodyB))
        return collideCylinderRound(bodyA, bodyB, true, contacts, basisOut);

    if (isRoundShape(bodyA) && isCylinderShape(bodyB))
        return collideCylinderRound(bodyB, bodyA, false, contacts, basisOut);

    if (isCylinderShape(bodyA) && bodyB->shape.type == RIGID_SHAPE_BOX)
        return collideCylinderBox(bodyA, bodyB, true, contacts, basisOut);

    if (bodyA->shape.type == RIGID_SHAPE_BOX && isCylinderShape(bodyB))
        return collideCylinderBox(bodyB, bodyA, false, contacts, basisOut);

    if (isRoundShape(bodyA) && isRoundShape(bodyB))
        return collideRoundRound(bodyA, bodyB, contacts, basisOut);

    if (isRoundShape(bodyA) && bodyB->shape.type == RIGID_SHAPE_BOX)
        return collideRoundBox(bodyA, bodyB, true, contacts, basisOut);

    if (bodyA->shape.type == RIGID_SHAPE_BOX && isRoundShape(bodyB))
        return collideRoundBox(bodyB, bodyA, false, contacts, basisOut);

    return collideBoxes(bodyA, bodyB, contacts, basisOut);
}
