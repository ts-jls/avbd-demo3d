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

#include "maths.h"

#include <stdint.h>
#include <string>

struct SimulationCommand
{
    std::string command;
    std::string stringValue;
    bool boolValue = false;
    bool hasBool = false;
    double numberValue = 0.0;
    bool hasNumber = false;
    uint32_t bodyId = 0;
    bool hasBodyId = false;
    float3 localHit = {0, 0, 0};
    bool hasLocalHit = false;
    float3 worldHit = {0, 0, 0};
    bool hasWorldHit = false;
    float3 worldTarget = {0, 0, 0};
    bool hasWorldTarget = false;
};
