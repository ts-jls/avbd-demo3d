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

struct SimWorld;

struct RenderBackend
{
    virtual ~RenderBackend() {}
    virtual const char *name() const = 0;
};

struct OpenGlReferenceRenderBackend : RenderBackend
{
    const char *name() const override { return "OpenGL Reference"; }
};

struct WebGpuInstancedPreviewRenderBackend : RenderBackend
{
    const char *name() const override { return "WebGPU Instanced Preview"; }
};
