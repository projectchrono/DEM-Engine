// Copyright (c) 2021, SBEL GPU Development Team
// Copyright (c) 2021, University of Wisconsin - Madison
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DEME_VISUALIZER_H
#define DEME_VISUALIZER_H

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include <cuda_runtime.h>

#include "../VariableTypes.h"
#include "VisualizationData.h"

namespace deme {

class DEMSolver;

/// RGBA color used by the visualizer without exposing its rendering backend in DEME's public API.
struct DEMVisualizerColor {
    constexpr DEMVisualizerColor(std::uint8_t red = 255,
                                 std::uint8_t green = 255,
                                 std::uint8_t blue = 255,
                                 std::uint8_t alpha = 255)
        : r(red), g(green), b(blue), a(alpha) {}

    std::uint8_t r;
    std::uint8_t g;
    std::uint8_t b;
    std::uint8_t a;
};

/// Owner-based scalar coloring; Height uses the owner's global Z coordinate, Speed uses linear velocity magnitude.
enum class DEMVisualizerColorMode { FAMILY, HEIGHT, SPEED };

/// Step-wise interactive viewer for the current state of a DEMSolver.
///
/// The visualizer never advances the simulation. Each Render call synchronously captures and draws exactly one solver
/// state, allowing applications to choose their own simulation-to-render cadence.
class DEMVisualizer {
  public:
    explicit DEMVisualizer(DEMSolver& solver);
    ~DEMVisualizer();

    DEMVisualizer(const DEMVisualizer&) = delete;
    DEMVisualizer& operator=(const DEMVisualizer&) = delete;
    DEMVisualizer(DEMVisualizer&&) = delete;
    DEMVisualizer& operator=(DEMVisualizer&&) = delete;

    /// Create the native window and graphics resources. Calling this more than once has no effect.
    void Initialize();
    /// Return true while the initialized visualization window remains open.
    bool Run() const;
    /// Capture the current solver state and draw one frame. This does not advance the solver.
    void Render();
    /// Release the window and graphics resources. Calling this more than once has no effect.
    void Close();

    void SetWindowSize(int width, int height);
    void SetWindowTitle(const std::string& title);
    void SetTargetFPS(int fps);
    void SetCameraPosition(float3 position);
    void SetCameraTarget(float3 target);
    void SetBackgroundColor(DEMVisualizerColor color);
    void SetFamilyColor(family_t family, DEMVisualizerColor color);

    /// Enable or disable component-sphere rendering. Enabled by default.
    void SetRenderSpheres(bool render = true);
    /// Enable or disable triangle rendering. Enabled by default.
    void SetRenderTriangles(bool render = true);
    bool IsRenderingSpheres() const;
    bool IsRenderingTriangles() const;

    /// Simulation controls are requests: the application must call ShouldStep before advancing its solver.
    /// ShouldStep consumes one pending step when paused. Render itself never blocks on pause or advances dynamics.
    bool ShouldStep();
    void SetPaused(bool paused);
    bool IsPaused() const;
    void RequestStep();
    void SetFamilyVisible(family_t family, bool visible);
    void SetColorMode(DEMVisualizerColorMode mode);
    /// Fit visible geometry (or the selected owner) using the last captured frame.
    void FrameAll();
    void FrameSelected();
    /// Pick at logical window coordinates using the last rendered frame. Background clears the selection.
    void PickAt(int x, int y);
    bodyID_t GetSelectedOwner() const;
    /// Global sphere or triangle index, or SIZE_MAX when nothing is selected.
    size_t GetSelectedGeometryID() const;
    bool IsSelectedSphere() const;
    /// Save the next rendered frame, including the inspection UI, to a PNG path.
    void RequestScreenshot(const std::string& path);

  private:
    // The solver archive supplies the public constructor. Callbacks keep a second copy of the static solver/JIT
    // globals out of the shared graphics library, where ELF symbol interposition would otherwise double-destroy them.
    DEMVisualizer(std::function<DEMVisualizationScene()> scene,
                  std::function<void(DEMVisualizationFrame&, bool)> frame);
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace deme

#endif
