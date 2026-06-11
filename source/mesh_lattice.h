/*
 * Mesh-to-particle-lattice importer: turns a closed triangle mesh into a
 * solver body made of sphere particles joined into a lattice.
 *
 * The mesh interior is voxelized with a row-parity fill (per (y,z) row, the
 * x-crossings of the surface are sorted and cells between odd/even pairs are
 * inside). Each inside cell becomes a sphere particle (these ride the GPU
 * narrowphase), and face-adjacent particles are linked with joints whose
 * anchors meet at the shared face midpoint — the same construction as the
 * hand-built Soft Body scenes.
 *
 * Stiffness: what decides whether the lattice settles is the
 * stiffness-to-MASS ratio (omega = sqrt(K/m); omega*dt must stay in the
 * low single digits for the iteration count to damp it). By default K is
 * derived from the particle mass to hit that regime. Passing
 * stiffnessLin = INFINITY instead produces a quasi-rigid lattice ("hard"
 * mesh) via the solver's infinite-stiffness joint path.
 */

#pragma once

#include "solver.h"

#include <algorithm>
#include <cmath>
#include <vector>

struct TriMesh
{
    std::vector<float3> verts;
    std::vector<uint32_t> tris; // 3 indices per triangle
};

struct MeshLatticeParams
{
    float spacing = 0.25f;       // lattice cell size; particle radius = spacing / 2
    float density = 12.0f;       // particle density (heavy particles keep omega*dt low)
    float friction = 0.6f;
    float stiffnessLin = -1.0f;  // < 0: auto from the omega*dt rule; INFINITY: rigid lattice
    float stiffnessAng = -1.0f;  // < 0: auto (0.2 * linear)
};

static void makeTorusMesh(TriMesh &out, float majorRadius, float minorRadius, int majorSegments, int minorSegments)
{
    out.verts.clear();
    out.tris.clear();
    const float TAU = 6.28318530718f;
    for (int i = 0; i < majorSegments; ++i)
    {
        float a = TAU * i / majorSegments;
        float ca = cosf(a), sa = sinf(a);
        for (int j = 0; j < minorSegments; ++j)
        {
            float b = TAU * j / minorSegments;
            float r = majorRadius + minorRadius * cosf(b);
            out.verts.push_back({ca * r, sa * r, minorRadius * sinf(b)});
        }
    }
    auto vertAt = [&](int i, int j) -> uint32_t
    {
        return (uint32_t)((i % majorSegments) * minorSegments + (j % minorSegments));
    };
    for (int i = 0; i < majorSegments; ++i)
    {
        for (int j = 0; j < minorSegments; ++j)
        {
            out.tris.push_back(vertAt(i, j));
            out.tris.push_back(vertAt(i + 1, j));
            out.tris.push_back(vertAt(i + 1, j + 1));
            out.tris.push_back(vertAt(i, j));
            out.tris.push_back(vertAt(i + 1, j + 1));
            out.tris.push_back(vertAt(i, j + 1));
        }
    }
}

// Builds the particle lattice for one mesh instance. Returns the number of
// particles created (0 means the mesh produced no interior cells — too small
// for the spacing, or not closed).
static int buildMeshLattice(Solver *solver, const TriMesh &mesh, float3 position, float scale, const MeshLatticeParams &params)
{
    if (mesh.verts.empty() || mesh.tris.size() < 3)
        return 0;

    std::vector<float3> verts(mesh.verts.size());
    float3 mn = {FLT_MAX, FLT_MAX, FLT_MAX};
    float3 mx = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    for (size_t i = 0; i < mesh.verts.size(); ++i)
    {
        float3 v = mesh.verts[i] * scale + position;
        verts[i] = v;
        mn = {min(mn.x, v.x), min(mn.y, v.y), min(mn.z, v.z)};
        mx = {max(mx.x, v.x), max(mx.y, v.y), max(mx.z, v.z)};
    }

    const float spacing = max(params.spacing, 1e-3f);
    const int nx = (int)((mx.x - mn.x) / spacing) + 1;
    const int ny = (int)((mx.y - mn.y) / spacing) + 1;
    const int nz = (int)((mx.z - mn.z) / spacing) + 1;
    if ((uint64_t)nx * ny * nz > (1ull << 22))
        return 0; // mesh far too large for this spacing

    // Row-parity interior fill: for each (y,z) row the surface crossings
    // along x are sorted and cells between consecutive pairs are inside.
    // The small irrational offset keeps row lines off mesh edges/vertices,
    // where parity counting is ambiguous.
    const float rowEps = spacing * 0.0137f;
    std::vector<char> inside((size_t)nx * ny * nz, 0);
    std::vector<float> crossings;
    int insideCount = 0;
    for (int iz = 0; iz < nz; ++iz)
    {
        float cz = mn.z + (iz + 0.5f) * spacing + rowEps;
        for (int iy = 0; iy < ny; ++iy)
        {
            float cy = mn.y + (iy + 0.5f) * spacing + rowEps;
            crossings.clear();
            for (size_t t = 0; t + 2 < mesh.tris.size(); t += 3)
            {
                const float3 &p0 = verts[mesh.tris[t]];
                const float3 &p1 = verts[mesh.tris[t + 1]];
                const float3 &p2 = verts[mesh.tris[t + 2]];
                float d = (p1.y - p0.y) * (p2.z - p0.z) - (p2.y - p0.y) * (p1.z - p0.z);
                if (fabsf(d) < 1e-12f)
                    continue;
                float u = ((cy - p0.y) * (p2.z - p0.z) - (p2.y - p0.y) * (cz - p0.z)) / d;
                float v = ((p1.y - p0.y) * (cz - p0.z) - (cy - p0.y) * (p1.z - p0.z)) / d;
                if (u < 0.0f || v < 0.0f || u + v > 1.0f)
                    continue;
                crossings.push_back(p0.x + u * (p1.x - p0.x) + v * (p2.x - p0.x));
            }
            if (crossings.size() < 2)
                continue;
            std::sort(crossings.begin(), crossings.end());
            for (size_t k = 0; k + 1 < crossings.size(); k += 2)
            {
                int x0 = (int)ceilf((crossings[k] - mn.x) / spacing - 0.5f);
                int x1 = (int)floorf((crossings[k + 1] - mn.x) / spacing - 0.5f);
                for (int ix = max(x0, 0); ix <= min(x1, nx - 1); ++ix)
                {
                    inside[(size_t)ix + (size_t)nx * (iy + (size_t)ny * iz)] = 1;
                    insideCount++;
                }
            }
        }
    }
    if (insideCount == 0)
        return 0;

    // Particles per inside cell.
    const float radius = spacing * 0.5f;
    const bool rigid = isinf(params.stiffnessLin);
    float Klin = params.stiffnessLin;
    float Kang = params.stiffnessAng;
    if (!rigid && Klin < 0.0f)
    {
        // omega*dt rule: K/m around 17000 1/s^2 puts omega*dt ~ 2.2 at 60 Hz,
        // the same regime as the stable hand-built soft scenes.
        float mass = (4.0f / 3.0f) * 3.14159265f * radius * radius * radius * params.density;
        Klin = 17000.0f * mass;
    }
    if (!rigid && Kang < 0.0f)
        Kang = 0.2f * Klin;

    std::vector<Rigid *> particles((size_t)nx * ny * nz, (Rigid *)0);
    int created = 0;
    for (int iz = 0; iz < nz; ++iz)
        for (int iy = 0; iy < ny; ++iy)
            for (int ix = 0; ix < nx; ++ix)
            {
                size_t cell = (size_t)ix + (size_t)nx * (iy + (size_t)ny * iz);
                if (!inside[cell])
                    continue;
                float3 c = {mn.x + (ix + 0.5f) * spacing,
                            mn.y + (iy + 0.5f) * spacing,
                            mn.z + (iz + 0.5f) * spacing};
                particles[cell] = Rigid::makeSphere(solver, radius, params.density, params.friction, c);
                created++;
            }

    // Structural joints between face-adjacent particles, anchors meeting at
    // the shared face midpoint.
    const float half = spacing * 0.5f;
    auto link = [&](Rigid *a, Rigid *b, float3 rA)
    {
        if (a && b)
            new Joint(solver, a, b, rA, {-rA.x, -rA.y, -rA.z},
                      rigid ? INFINITY : Klin, rigid ? INFINITY : Kang);
    };
    for (int iz = 0; iz < nz; ++iz)
        for (int iy = 0; iy < ny; ++iy)
            for (int ix = 0; ix < nx; ++ix)
            {
                size_t cell = (size_t)ix + (size_t)nx * (iy + (size_t)ny * iz);
                if (!particles[cell])
                    continue;
                if (ix + 1 < nx)
                    link(particles[cell], particles[cell + 1], {half, 0, 0});
                if (iy + 1 < ny)
                    link(particles[cell], particles[cell + (size_t)nx], {0, half, 0});
                if (iz + 1 < nz)
                    link(particles[cell], particles[cell + (size_t)nx * ny], {0, 0, half});
            }

    return created;
}
