// Copyright (c) 2026, SBEL GPU Development Team
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <chrono>

#include "DEM/utils/DEMVisualizer.h"

namespace deme {

// Service input on wall time, independently of simulation output intervals. Call before each small dynamics advance;
// a long blocking solver call still prevents the window thread from handling input during that call.
class VisualizerDemoLoop {
  public:
    bool Update(DEMVisualizer& viewer) {
        const auto now = Clock::now();
        if (now < next_frame || !viewer.Run())
            return false;
        // DoDynamics has already joined dT. Capturing its state needs no reset of the contact-detection worker.
        viewer.Render();
        // A paused single-step request permits exactly the caller's next dynamics advance.
        while (viewer.Run() && !viewer.ShouldStep())
            viewer.Render();
        // Leave a physics work interval after drawing. Render may spend a full frame in its FPS limiter;
        // scheduling from the start would then redraw after every single timestep and starve the solver.
        next_frame = Clock::now() + std::chrono::milliseconds(16);
        return true;
    }

  private:
    using Clock = std::chrono::steady_clock;
    Clock::time_point next_frame{};
};

}  // namespace deme
