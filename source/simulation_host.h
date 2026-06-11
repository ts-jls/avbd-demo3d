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

#pragma once

#include "simulation_command.h"
#include "solver.h"

#include <stdint.h>
#include <string>
#include <vector>

struct SimulationHost
{
    SimulationHost();
    ~SimulationHost();

    Solver &solver();
    const Solver &solver() const;
    const SimWorld &world() const;

    bool loadScene(int sceneIndex);
    bool loadSceneByName(const char *name);
    void resetScene();
    void setPaused(bool value);
    bool isPaused() const;
    bool &pausedRef();
    void stepFrame();
    void singleStep();
    bool applyCommand(const SimulationCommand &command);
    void releaseDrag();

    int sceneIndex() const;
    int &sceneIndexRef();
    const char *currentSceneName() const;
    int sceneCount() const;
    const char *sceneName(int index) const;

    float lastPhysicsMs() const;
    float &lastPhysicsMsRef();
    uint64_t nextSnapshotFrame();
    std::string metricsText() const;

    // Messages queued for the bridge to broadcast (e.g. mesh import results).
    std::vector<std::string> takeOutboundMessages();

    static std::string normalizeSceneName(const char *text);

private:
    Rigid *bodyForDrag(BodyId id) const;
    bool beginDrag(BodyId id, const float3 &localHit, const float3 &worldHit);
    bool updateDrag(const float3 &worldTarget);
    bool importMesh(const SimulationCommand &command);

    std::vector<std::string> outboundMessages;

    Solver solverInstance;
    int currentScene;
    bool paused;
    float lastStepMs;
    uint64_t snapshotFrame;
    BodyId draggedBody;
    Joint *dragJoint;
};
