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
#include "viewer_bridge.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace
{
std::atomic<bool> running(true);

#ifdef _WIN32
BOOL WINAPI consoleHandler(DWORD event)
{
    if (event == CTRL_C_EVENT || event == CTRL_CLOSE_EVENT || event == CTRL_BREAK_EVENT)
    {
        running = false;
        return TRUE;
    }
    return FALSE;
}
#endif

const char *argValue(int argc, char **argv, const char *name)
{
    size_t nameLength = strlen(name);
    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], name) == 0 && i + 1 < argc)
            return argv[i + 1];
        if (strncmp(argv[i], name, nameLength) == 0 && argv[i][nameLength] == '=')
            return argv[i] + nameLength + 1;
    }
    return 0;
}

bool hasFlag(int argc, char **argv, const char *name)
{
    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], name) == 0)
            return true;
    }
    return false;
}
}

int main(int argc, char **argv)
{
#ifdef _WIN32
    SetConsoleCtrlHandler(consoleHandler, TRUE);
#endif

    const char *scene = argValue(argc, argv, "--scene");
    uint16_t port = (uint16_t)atoi(argValue(argc, argv, "--port") ? argValue(argc, argv, "--port") : "8765");
    double tickRate = atof(argValue(argc, argv, "--tick-rate") ? argValue(argc, argv, "--tick-rate") : "60");
    if (tickRate <= 0.0)
        tickRate = 60.0;

    SimulationHost host;
    if (scene && !host.loadSceneByName(scene))
    {
        std::printf("Unknown scene '%s', loading default scene '%s'\n", scene, host.currentSceneName());
        host.loadScene(host.sceneIndex());
    }
    else
    {
        host.loadScene(host.sceneIndex());
    }
    host.setPaused(hasFlag(argc, argv, "--paused"));

    ViewerBridge bridge;
    bridge.start(port);

    std::printf("AVBD headless server\n");
    std::printf("Scene: %s\n", host.currentSceneName());
    std::printf("Bridge: %s\n", bridge.statusText());
    std::printf("Press Ctrl+C to stop.\n");

    using Clock = std::chrono::high_resolution_clock;
    const auto tick = std::chrono::duration<double>(1.0 / tickRate);
    auto nextTick = Clock::now();
    int statusFrame = 0;

    while (running)
    {
        SimulationCommand command;
        while (bridge.pollCommand(command))
            host.applyCommand(command);

        if (!host.isPaused())
            host.stepFrame();

        bridge.broadcastSnapshot(host.world(), host.currentSceneName(), host.nextSnapshotFrame());
        if ((statusFrame++ % 30) == 0)
            bridge.broadcastStatus(host.metricsText().c_str());

        nextTick += std::chrono::duration_cast<Clock::duration>(tick);
        std::this_thread::sleep_until(nextTick);
        if (Clock::now() > nextTick + std::chrono::milliseconds(250))
            nextTick = Clock::now();
    }

    bridge.stop();
    std::printf("Shutdown complete.\n");
    return 0;
}
