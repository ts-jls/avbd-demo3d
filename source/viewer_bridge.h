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

#if AVBD_ENABLE_VIEWER_BRIDGE && defined(_WIN32)
#include <atomic>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>
#endif

struct ViewerBridgeStats
{
    uint64_t snapshotCount;
    uint64_t statusCount;
    uint64_t lastSnapshotBytes;
    uint64_t totalSnapshotBytes;
    int binarySnapshotMode;
    float lastSerializeMs;
    float avgSerializeMs;
    float lastSendMs;
    float avgSendMs;
    int lastSentClients;
    int sendFailures;
    int clients;
    int queuedCommands;
};

#if AVBD_ENABLE_VIEWER_BRIDGE && defined(_WIN32)
struct ViewerBridgeClient
{
    uintptr_t socket;
    bool binarySnapshots;
};
#endif

struct ViewerBridge
{
    ViewerBridge();
    ~ViewerBridge();

    bool start(uint16_t port);
    void stop();
    void broadcastSnapshot(const SimWorld &world, const char *sceneName, uint64_t frame);
    void broadcastStatus(const char *metricsText);
    bool pollCommand(SimulationCommand &command);
    const char *statusText() const;
    int clientCountValue() const;
    ViewerBridgeStats statsSnapshot() const;

#if AVBD_ENABLE_VIEWER_BRIDGE && defined(_WIN32)
    void serverLoop(uint16_t port);
    void setStatusText(const char *message);
    void updateSnapshotStats(uint64_t bytes, float serializeMs, float sendMs, int sentClients, int failures);
    void updateStatusStats();

    std::atomic<bool> running;
    std::atomic<int> clientCount;
    uintptr_t listenSocket;
    bool winsockStarted;
    std::thread serverThread;
    mutable std::mutex statusMutex;
    mutable std::mutex clientsMutex;
    mutable std::mutex commandsMutex;
    mutable std::mutex metricsMutex;
    std::vector<ViewerBridgeClient> clients;
    std::deque<SimulationCommand> commands;
    ViewerBridgeStats metrics;
    char status[256];
#endif
};
