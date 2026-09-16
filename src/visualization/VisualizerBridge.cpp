// Copyright (c) 2026, SBEL GPU Development Team
// SPDX-License-Identifier: BSD-3-Clause
#include "DEM/API.h"
#include "DEM/utils/DEMVisualizer.h"

namespace deme {

// This small bridge belongs to the solver archive (and the Python extension), never the graphics shared library.
// Keep the solver reference non-owning, as in the original API; the Python binding retains its keep_alive policy.
DEMVisualizer::DEMVisualizer(DEMSolver& solver)
    : DEMVisualizer([&solver]() { return solver.GetVisualizationScene(); },
                    [&solver](DEMVisualizationFrame& frame, bool velocities) {
                        solver.GetVisualizationFrame(frame, velocities);
                    }) {}

}  // namespace deme
