// Copyright (c) 2026, SBEL GPU Development Team
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// raymath's float3 is an unrelated return-buffer type; rename it locally to avoid colliding with CUDA's float3.
// Include this wrapper everywhere in the viewer rather than exposing the workaround to public headers.
#define float3 RayMathFloat3
#include <raymath.h>
#undef float3
