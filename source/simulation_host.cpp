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

#include "simulation_host.h"

#include "scenes.h"

#include <chrono>
#include <cctype>
#include <iomanip>
#include <sstream>

namespace
{
float elapsedMs(std::chrono::high_resolution_clock::time_point begin,
                std::chrono::high_resolution_clock::time_point end)
{
    return std::chrono::duration<float, std::milli>(end - begin).count();
}
}

SimulationHost::SimulationHost()
    : currentScene(4),
      paused(false),
      lastStepMs(0.0f),
      snapshotFrame(0),
      draggedBody(INVALID_BODY_ID),
      dragJoint(0)
{
}

SimulationHost::~SimulationHost()
{
    releaseDrag();
}

Solver &SimulationHost::solver()
{
    return solverInstance;
}

const Solver &SimulationHost::solver() const
{
    return solverInstance;
}

const SimWorld &SimulationHost::world() const
{
    return solverInstance.world;
}

bool SimulationHost::loadScene(int index)
{
    if (index < 0 || index >= ::sceneCount)
        return false;

    releaseDrag();
    currentScene = index;
    scenes[currentScene](&solverInstance);
    solverInstance.world.syncFromLegacy(solverInstance);
    snapshotFrame = 0;
    return true;
}

bool SimulationHost::loadSceneByName(const char *name)
{
    std::string desired = normalizeSceneName(name);
    if (desired.empty())
        return false;

    for (int i = 0; i < ::sceneCount; ++i)
    {
        if (normalizeSceneName(sceneNames[i]) == desired)
            return loadScene(i);
    }
    return false;
}

void SimulationHost::resetScene()
{
    releaseDrag();
    loadScene(currentScene);
}

void SimulationHost::setPaused(bool value)
{
    paused = value;
}

bool SimulationHost::isPaused() const
{
    return paused;
}

bool &SimulationHost::pausedRef()
{
    return paused;
}

void SimulationHost::stepFrame()
{
    auto begin = std::chrono::high_resolution_clock::now();
    solverInstance.step();
    auto end = std::chrono::high_resolution_clock::now();
    lastStepMs = elapsedMs(begin, end);
}

void SimulationHost::singleStep()
{
    stepFrame();
}

bool SimulationHost::applyCommand(const SimulationCommand &command)
{
    std::string name = normalizeSceneName(command.command.c_str());
    if (name == "begindrag")
    {
        if (!command.hasBodyId || !command.hasLocalHit || !command.hasWorldHit)
            return false;
        return beginDrag(command.bodyId, command.localHit, command.worldHit);
    }
    if (name == "updatedrag")
    {
        if (!command.hasWorldTarget)
            return false;
        return updateDrag(command.worldTarget);
    }
    if (name == "enddrag")
    {
        releaseDrag();
        return true;
    }
    if (name == "pause")
    {
        setPaused(command.hasBool ? command.boolValue : true);
        return true;
    }
    if (name == "play" || name == "resume")
    {
        setPaused(false);
        return true;
    }
    if (name == "reset")
    {
        resetScene();
        return true;
    }
    if (name == "step" || name == "singlestep")
    {
        singleStep();
        return true;
    }
    if (name == "loadscene" || name == "scene")
    {
        if (!command.stringValue.empty() && loadSceneByName(command.stringValue.c_str()))
            return true;
        if (command.hasNumber)
            return loadScene((int)command.numberValue);
        return false;
    }
    if (name == "setdt" || name == "dt")
    {
        if (!command.hasNumber)
            return false;
        solverInstance.dt = (float)command.numberValue;
        return true;
    }
    if (name == "setgravity" || name == "gravity")
    {
        if (!command.hasNumber)
            return false;
        solverInstance.gravity = (float)command.numberValue;
        return true;
    }
    if (name == "setiterations" || name == "iterations")
    {
        if (!command.hasNumber)
            return false;
        int iterations = (int)command.numberValue;
        solverInstance.iterations = iterations < 1 ? 1 : iterations;
        return true;
    }
    if (name == "setbroadphase" || name == "broadphase")
    {
        if (!command.stringValue.empty())
        {
            std::string mode = normalizeSceneName(command.stringValue.c_str());
            if (mode == "allpairs")
                solverInstance.broadphaseMode = BROADPHASE_ALL_PAIRS;
            else if (mode == "spatialhashgrid" || mode == "spatialhash")
                solverInstance.broadphaseMode = BROADPHASE_SPATIAL_HASH;
            else if (mode == "sweepandprune" || mode == "sap")
                solverInstance.broadphaseMode = BROADPHASE_SWEEP_AND_PRUNE;
            else
                return false;
            return true;
        }
        if (command.hasNumber)
        {
            int mode = (int)command.numberValue;
            if (mode < 0 || mode >= BROADPHASE_COUNT)
                return false;
            solverInstance.broadphaseMode = (BroadphaseMode)mode;
            return true;
        }
        return false;
    }
    if (name == "skipignore" || name == "skipignoresolverwork")
    {
        if (!command.hasBool)
            return false;
        solverInstance.skipIgnoreCollisionSolverWork = command.boolValue;
        return true;
    }
    return false;
}

Rigid *SimulationHost::bodyForDrag(BodyId id) const
{
    const SimWorld &simWorld = solverInstance.world;
    if (id == INVALID_BODY_ID || id >= simWorld.bodies.size())
        return 0;

    const SimBodyData &bodyData = simWorld.bodies[id];
    Rigid *body = bodyData.source;
    if (!bodyData.active || !body || body->denseId != id || body->mass <= 0.0f)
        return 0;

    return body;
}

bool SimulationHost::beginDrag(BodyId id, const float3 &localHit, const float3 &worldHit)
{
    Rigid *body = bodyForDrag(id);
    if (!body)
        return false;

    releaseDrag();
    const float dragStiffness = 5000.0f;
    dragJoint = new Joint(&solverInstance, 0, body, worldHit, localHit, dragStiffness, 0.0f);
    draggedBody = id;
    solverInstance.world.syncFromLegacy(solverInstance);
    return true;
}

bool SimulationHost::updateDrag(const float3 &worldTarget)
{
    if (!dragJoint)
        return false;

    dragJoint->rA = worldTarget;
    return true;
}

void SimulationHost::releaseDrag()
{
    if (dragJoint)
    {
        delete dragJoint;
        dragJoint = 0;
        solverInstance.world.syncFromLegacy(solverInstance);
    }
    draggedBody = INVALID_BODY_ID;
}

int SimulationHost::sceneIndex() const
{
    return currentScene;
}

int &SimulationHost::sceneIndexRef()
{
    return currentScene;
}

const char *SimulationHost::currentSceneName() const
{
    return sceneNames[currentScene];
}

int SimulationHost::sceneCount() const
{
    return ::sceneCount;
}

const char *SimulationHost::sceneName(int index) const
{
    return (index >= 0 && index < ::sceneCount) ? sceneNames[index] : "";
}

float SimulationHost::lastPhysicsMs() const
{
    return lastStepMs;
}

float &SimulationHost::lastPhysicsMsRef()
{
    return lastStepMs;
}

uint64_t SimulationHost::nextSnapshotFrame()
{
    return snapshotFrame++;
}

std::string SimulationHost::metricsText() const
{
    const SolverStats &s = solverInstance.stats;
    std::ostringstream out;
    out << std::fixed << std::setprecision(2);
    out << "Scene: " << currentSceneName() << "\n";
    out << "Physics backend: " << solverInstance.physicsBackend->name() << "\n";
    out << "Paused: " << (paused ? "On" : "Off") << "\n";
    out << "Dragging body: " << (dragJoint ? (int)draggedBody : -1) << "\n";
    out << "Bodies: " << s.bodyCount << "\n";
    out << "Dense bodies: " << (int)solverInstance.world.bodies.size() << "\n";
    out << "Dense constraints: " << (int)solverInstance.world.constraints.size() << "\n";
    out << "Broadphase: " << s.broadphaseMs << " ms\n";
    out << "Pair checks: " << s.pairChecks << "\n";
    out << "Sphere hits: " << s.sphereHits << "\n";
    out << "Created manifolds: " << s.manifoldsCreated << "\n";
    out << "Active bodies: " << s.activeBodyCount << "\n";
    out << "Forces: " << s.forceCount << "\n";
    out << "Joints: " << s.jointCount << "\n";
    out << "Springs: " << s.springCount << "\n";
    out << "Manifolds: " << s.manifoldCount << "\n";
    out << "Ignores: " << s.ignoreCollisionCount << "\n";
    out << "Primal visits: " << s.primalForceVisits << "\n";
    out << "Dual visits: " << s.dualForceVisits << "\n";
    out << "Physics total: " << lastStepMs << " ms\n";
    out << "SimWorld sync: " << s.simWorldSyncMs << " ms\n";
    out << "Force init: " << s.forceInitMs << " ms\n";
    out << "Body init: " << s.bodyInitMs << " ms\n";
    out << "Primal solve: " << s.primalSolveMs << " ms\n";
    out << "Dual update: " << s.dualUpdateMs << " ms\n";
    out << "Velocity: " << s.velocityUpdateMs << " ms\n";
    return out.str();
}

std::string SimulationHost::normalizeSceneName(const char *text)
{
    std::string out;
    if (!text)
        return out;
    for (const char *c = text; *c; ++c)
    {
        unsigned char ch = (unsigned char)*c;
        if (std::isalnum(ch))
            out.push_back((char)std::tolower(ch));
    }
    return out;
}
