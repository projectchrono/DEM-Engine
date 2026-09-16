// Copyright (c) 2021, SBEL GPU Development Team
// Copyright (c) 2021, University of Wisconsin - Madison
// SPDX-License-Identifier: BSD-3-Clause

#include "DEM/utils/DEMVisualizer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "core/utils/Logger.hpp"
// raylib defines convenience macros such as PI, so keep it after DEME headers to avoid altering their declarations.
#include "NativeRenderer.h"
#include "ViewerInteraction.h"
#include <imgui_impl_opengl3.h>
#include <rlgl.h>

namespace deme {
namespace {

Color toRayColor(DEMVisualizerColor color) {
    return {color.r, color.g, color.b, color.a};
}

DEMVisualizerColor defaultFamilyColor(family_t family) {
    // The multiplicative hash keeps adjacent family IDs visually distinct without maintaining a fixed-size palette.
    const std::uint32_t hash = static_cast<std::uint32_t>(family) * 2654435761u;
    return {static_cast<std::uint8_t>(80u + (hash & 0x7fu)), static_cast<std::uint8_t>(80u + ((hash >> 8u) & 0x7fu)),
            static_cast<std::uint8_t>(80u + ((hash >> 16u) & 0x7fu)), 255};
}

bool finite(Vector3 p) {
    return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
}

}  // namespace

// Viewer-owned scene/frame storage insulates OpenGL from solver internals and keeps resource lifetime on the UI thread.
struct DEMVisualizer::Impl {
    Impl(std::function<DEMVisualizationScene()> scene_callback,
         std::function<void(DEMVisualizationFrame&, bool)> frame_callback)
        : capture_scene(std::move(scene_callback)), capture_frame(std::move(frame_callback)) {
        visible.fill(true);
    }
    std::function<DEMVisualizationScene()> capture_scene;
    std::function<void(DEMVisualizationFrame&, bool)> capture_frame;
    int width = 1440, height = 900, target_fps = 60;
    std::string title = "DEM-Engine | Simulation inspector";
    bool initialized = false, render_spheres = true, render_triangles = true;
    bool paused = false, step_requested = false, wireframe = false, custom_camera = false;
    bool scene_loaded = false, auto_range = true, grid = true;
    int resolution = 12, color_mode = 0;
    float scalar_min = 0, scalar_max = 1;
    std::string screenshot;
    DEMVisualizerColor background{24, 29, 39, 255};
    Camera3D camera{{4, 4, 3}, {0, 0, 0}, {0, 0, 1}, 45, CAMERA_PERSPECTIVE};
    std::map<family_t, DEMVisualizerColor> family_colors;
    std::array<bool, 256> visible;
    std::array<size_t, 256> family_counts{};
    bodyID_t selected = NULL_BODYID;
    size_t selected_geometry = std::numeric_limits<size_t>::max();
    bool selected_sphere = false;
    ImGuiContext* ui_context = nullptr;
    bool ui_renderer_ready = false;
    visualization::NativeRenderer renderer;
    DEMVisualizationScene scene;
    DEMVisualizationFrame frame;
    std::vector<std::array<float, 4>> colors;
    std::vector<unsigned char> owner_categories;

    int sidebarWidth() const { return std::min(340, GetScreenWidth() / 2); }
    int viewportWidth() const { return std::max(1, GetScreenWidth() - sidebarWidth()); }
    bool ownerVisible(bodyID_t owner) const {
        return owner < frame.families.size() && owner < owner_categories.size() && visible[frame.families[owner]] &&
               ((render_spheres && (owner_categories[owner] & 1)) ||
                (render_triangles && (owner_categories[owner] & 2)));
    }
    void clearSelection() {
        selected = NULL_BODYID;
        selected_geometry = std::numeric_limits<size_t>::max();
    }

    // Compute bounds only on demand, including component offsets and mesh extents rather than just owner centers.
    void fit(bool selection_only) {
        Vector3 lo{INFINITY, INFINITY, INFINITY}, hi{-INFINITY, -INFINITY, -INFINITY};
        bool found = false;
        auto add = [&](bodyID_t owner, float3 point, float radius) {
            if (!ownerVisible(owner) || (selection_only && owner != selected))
                return;
            Vector3 p = visualization::WorldPoint(point, frame.positions[owner], frame.orientations[owner]);
            if (!finite(p))
                return;
            lo = Vector3Min(lo, Vector3Subtract(p, {radius, radius, radius}));
            hi = Vector3Max(hi, Vector3Add(p, {radius, radius, radius}));
            found = true;
        };
        if (render_spheres)
            for (const auto& s : scene.spheres)
                add(s.owner, s.offset, s.radius);
        if (render_triangles)
            for (const auto& t : scene.triangles) {
                add(t.owner, t.a, 0);
                add(t.owner, t.b, 0);
                add(t.owner, t.c, 0);
            }
        if (!found)
            return;
        Vector3 direction = Vector3Normalize(Vector3Subtract(camera.position, camera.target));
        if (Vector3Length(direction) < 0.5f)
            direction = Vector3Normalize({1, 1, 0.7f});
        camera.target = Vector3Scale(Vector3Add(lo, hi), 0.5f);
        float radius = std::max(Vector3Distance(lo, hi) * 0.5f, 0.001f);
        float half_angle = camera.fovy * DEG2RAD * 0.5f;
        float aspect = static_cast<float>(viewportWidth()) / std::max(GetScreenHeight(), 1);
        half_angle = std::min(half_angle, std::atan(std::tan(half_angle) * aspect));
        camera.position = Vector3Add(camera.target, Vector3Scale(direction, radius / std::sin(half_angle) * 1.15f));
    }

    // Compute colors per owner. Only speed mode/selection requests velocity data from the solver.
    void updateColors() {
        colors.resize(frame.positions.size());
        family_counts.fill(0);
        float low = INFINITY, high = -INFINITY;
        auto scalar = [&](size_t i) {
            return color_mode == 2 && i < frame.velocities.size()
                       ? Vector3Length(visualization::ToVector(frame.velocities[i]))
                       : frame.positions[i].z;
        };
        for (size_t i = 0; i < colors.size(); ++i) {
            if (i < owner_categories.size() && owner_categories[i])
                ++family_counts[frame.families[i]];
            float value = scalar(i);
            if (ownerVisible(static_cast<bodyID_t>(i)) && std::isfinite(value)) {
                low = std::min(low, value);
                high = std::max(high, value);
            }
        }
        if (auto_range && std::isfinite(low)) {
            scalar_min = low;
            scalar_max = high;
        }
        for (size_t i = 0; i < colors.size(); ++i) {
            family_t family = frame.families[i];
            auto it = family_colors.find(family);
            auto color = it == family_colors.end() ? defaultFamilyColor(family) : it->second;
            if (color_mode) {
                float range = scalar_max - scalar_min;
                float t = range > 0 ? (scalar(i) - scalar_min) / range : 0.5f;
                colors[i] = visualization::ScalarColor(std::isfinite(t) ? t : 0.f);
            } else {
                colors[i] = {color.r / 255.f, color.g / 255.f, color.b / 255.f, color.a / 255.f};
            }
            colors[i][3] = ownerVisible(static_cast<bodyID_t>(i)) && finite(visualization::ToVector(frame.positions[i]))
                               ? colors[i][3]
                               : 0.f;
        }
    }

    // All inspection state stays in the viewer; buttons only queue application-visible stepping requests.
    void sidebar() {
        ImGui::SetNextWindowPos({static_cast<float>(viewportWidth()), 0});
        ImGui::SetNextWindowSize({static_cast<float>(sidebarWidth()), static_cast<float>(GetScreenHeight())});
        ImGui::Begin("Inspector", nullptr,
                     ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings);
        ImGui::TextColored({0.35f, 0.8f, 0.95f, 1}, "DEM-ENGINE");
        ImGui::TextDisabled("Simulation inspector  /  Z up");
        ImGui::Separator();
        ImGui::Text("Time  %.6g s", frame.simulation_time);
        ImGui::Text("%d FPS  |  %.1f ms", GetFPS(), GetFrameTime() * 1000.f);
        if (ImGui::Button(paused ? "Resume [Space]" : "Pause [Space]")) {
            paused = !paused;
            step_requested = false;
        }
        ImGui::SameLine();
        if (ImGui::Button("Step [.]")) {
            paused = true;
            step_requested = true;
        }
        if (ImGui::Button("Frame all [F]"))
            fit(false);
        ImGui::SameLine();
        if (ImGui::Button("Frame selected"))
            fit(true);
        ImGui::Spacing();
        if (ImGui::CollapsingHeader("Scene", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Text("%zu owners", frame.positions.size());
            ImGui::Checkbox("Spheres", &render_spheres);
            ImGui::SameLine();
            ImGui::TextDisabled("%zu", scene.spheres.size());
            ImGui::Checkbox("Meshes", &render_triangles);
            ImGui::SameLine();
            ImGui::TextDisabled("%zu facets", scene.triangles.size());
            ImGui::Checkbox("Wireframe", &wireframe);
            ImGui::SameLine();
            ImGui::Checkbox("XY grid", &grid);
            int quality = resolution == 6 ? 0 : (resolution == 12 ? 1 : 2);
            ImGui::TextUnformatted("Sphere detail");
            ImGui::SetNextItemWidth(-1);
            if (ImGui::Combo("##sphere_detail", &quality, "Low\0Medium\0High\0")) {
                resolution = quality == 0 ? 6 : (quality == 1 ? 12 : 24);
                renderer.SetScene(scene, resolution);
            }
        }
        if (ImGui::CollapsingHeader("Color", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SetNextItemWidth(-1);
            ImGui::Combo("##color_mode", &color_mode, "Family\0Owner height (m)\0Owner speed (m/s)\0");
            if (color_mode) {
                ImGui::Checkbox("Automatic range", &auto_range);
                if (!auto_range) {
                    ImGui::DragFloat("Minimum", &scalar_min, 0.01f);
                    ImGui::DragFloat("Maximum", &scalar_max, 0.01f);
                }
                ImVec2 p = ImGui::GetCursorScreenPos();
                float w = ImGui::GetContentRegionAvail().x;
                auto* draw = ImGui::GetWindowDrawList();
                for (int i = 0; i < 64; ++i) {
                    auto c = visualization::ScalarColor(i / 63.f);
                    draw->AddRectFilled({p.x + w * i / 64, p.y}, {p.x + w * (i + 1) / 64, p.y + 12},
                                        ImGui::ColorConvertFloat4ToU32({c[0], c[1], c[2], 1}));
                }
                ImGui::Dummy({w, 14});
                ImGui::Text("%.4g  ...  %.4g", scalar_min, scalar_max);
            }
        }
        if (ImGui::CollapsingHeader("Families", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::SmallButton("Show all"))
                visible.fill(true);
            ImGui::SameLine();
            if (ImGui::SmallButton("Hide all"))
                visible.fill(false);
            ImGui::BeginChild("family_list", {0, 130}, ImGuiChildFlags_Borders);
            for (size_t i = 0; i < family_counts.size(); ++i)
                if (family_counts[i]) {
                    ImGui::PushID(static_cast<int>(i));
                    auto family = static_cast<family_t>(i);
                    auto it = family_colors.find(family);
                    auto c = it == family_colors.end() ? defaultFamilyColor(family) : it->second;
                    float rgb[]{c.r / 255.f, c.g / 255.f, c.b / 255.f};
                    if (ImGui::ColorEdit3("##color", rgb, ImGuiColorEditFlags_NoInputs))
                        family_colors[family] = {static_cast<std::uint8_t>(rgb[0] * 255),
                                                 static_cast<std::uint8_t>(rgb[1] * 255),
                                                 static_cast<std::uint8_t>(rgb[2] * 255), 255};
                    ImGui::SameLine();
                    std::string label = "Family " + std::to_string(i) + " (" + std::to_string(family_counts[i]) + ")";
                    ImGui::Checkbox(label.c_str(), &visible[i]);
                    ImGui::PopID();
                }
            ImGui::EndChild();
        }
        if (ImGui::CollapsingHeader("Selection", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (selected < frame.positions.size()) {
                auto p = frame.positions[selected];
                ImGui::Text("Owner %u  |  Family %u", selected, static_cast<unsigned int>(frame.families[selected]));
                ImGui::Text("%s %zu", selected_sphere ? "Sphere" : "Triangle", selected_geometry);
                ImGui::Text("Position  %.5g, %.5g, %.5g", p.x, p.y, p.z);
                if (selected < frame.velocities.size()) {
                    auto v = frame.velocities[selected];
                    ImGui::Text("Velocity  %.5g, %.5g, %.5g", v.x, v.y, v.z);
                }
                auto q = frame.orientations[selected];
                ImGui::Text("Quaternion (x y z w)");
                ImGui::Text("%.4g  %.4g  %.4g  %.4g", q.x, q.y, q.z, q.w);
                if (ImGui::SmallButton("Clear selection"))
                    clearSelection();
            } else
                ImGui::TextDisabled("Right-click geometry to inspect.");
        }
        ImGui::Separator();
        ImGui::TextDisabled("Left drag: orbit  |  Middle drag: pan");
        ImGui::TextDisabled("Wheel: zoom  |  Right click: select");
        ImGui::TextDisabled("F: frame all  |  Shift+F: selected");
        if (ImGui::Button("Screenshot [F12]"))
            screenshot = "deme-screenshot.png";
        ImGui::End();
    }
};

DEMVisualizer::DEMVisualizer(std::function<DEMVisualizationScene()> scene,
                             std::function<void(DEMVisualizationFrame&, bool)> frame)
    : m_impl(std::make_unique<Impl>(std::move(scene), std::move(frame))) {}
DEMVisualizer::~DEMVisualizer() {
    Close();
}

void DEMVisualizer::Initialize() {
    auto& v = *m_impl;
    if (v.initialized)
        return;
    if (IsWindowReady())
        DEME_ERROR("Only one DEMVisualizer window can be initialized in a process at a time.");
    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_RESIZABLE);
    InitWindow(v.width, v.height, v.title.c_str());
    if (!IsWindowReady())
        DEME_ERROR("DEMVisualizer could not create a window. Check DISPLAY/WSLg and OpenGL availability.");
    v.initialized = true;
    SetWindowMinSize(640, 480);
    ::SetTargetFPS(v.target_fps);
    v.ui_context = ImGui::CreateContext();
    auto& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.BackendPlatformName = "deme_raylib";
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImFontConfig font_config;
    font_config.SizePixels = 16.f;
    io.Fonts->AddFontDefault(&font_config);
    ImGui::StyleColorsDark();
    auto& style = ImGui::GetStyle();
    style.WindowPadding = {16, 16};
    style.FramePadding = {7, 5};
    style.ItemSpacing = {8, 8};
    style.FrameRounding = 4;
    style.WindowBorderSize = 0;
    style.Colors[ImGuiCol_WindowBg] = {0.075f, 0.09f, 0.12f, 1};
    style.Colors[ImGuiCol_Header] = {0.15f, 0.22f, 0.29f, 1};
    style.Colors[ImGuiCol_Button] = {0.13f, 0.3f, 0.39f, 1};
    if (!ImGui_ImplOpenGL3_Init("#version 330"))
        DEME_ERROR("Could not initialize the visualizer UI renderer.");
    v.ui_renderer_ready = true;
    v.renderer.Initialize();
    v.scene_loaded = false;
}

bool DEMVisualizer::Run() const {
    return m_impl->initialized && !WindowShouldClose();
}

// Capture at a synchronized solver boundary. UI redraw while paused follows exactly the same non-blocking path.
void DEMVisualizer::Render() {
    auto& v = *m_impl;
    if (!v.initialized)
        DEME_ERROR("DEMVisualizer::Render requires Initialize() first.");
    if (IsWindowMinimized()) {
        PollInputEvents();
        WaitTime(0.02);
        return;
    }
    ImGui::SetCurrentContext(v.ui_context);
    v.capture_frame(v.frame, v.color_mode == 2 || v.selected != NULL_BODYID);
    if (!v.scene_loaded || v.scene.revision != v.frame.revision) {
        bool first = !v.scene_loaded;
        v.scene = v.capture_scene();
        // Analytical owners and combined masters have no drawable geometry; exclude them from legends and filters.
        v.owner_categories.assign(v.frame.positions.size(), 0);
        for (const auto& s : v.scene.spheres)
            v.owner_categories.at(s.owner) |= 1;
        for (const auto& t : v.scene.triangles)
            v.owner_categories.at(t.owner) |= 2;
        v.renderer.SetScene(v.scene, v.resolution);
        v.scene_loaded = true;
        v.clearSelection();
        if (first && !v.custom_camera)
            v.fit(false);
    }
    v.updateColors();
    visualization::BeginInputFrame();
    ImGui_ImplOpenGL3_NewFrame();
    ImGui::NewFrame();
    v.sidebar();
    auto& io = ImGui::GetIO();
    if (!io.WantCaptureKeyboard) {
        if (IsKeyPressed(KEY_SPACE))
            SetPaused(!v.paused);
        if (IsKeyPressed(KEY_PERIOD))
            RequestStep();
        if (IsKeyPressed(KEY_F))
            v.fit(IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT));
        if (IsKeyPressed(KEY_F12))
            v.screenshot = "deme-screenshot.png";
    }
    if (!io.WantCaptureMouse && GetMouseX() < v.viewportWidth())
        visualization::Navigate(v.camera);
    // Mode may change in the UI; fetch velocities immediately rather than displaying a height-colored speed frame.
    if (v.color_mode == 2 && v.frame.velocities.size() != v.frame.positions.size())
        v.capture_frame(v.frame, true);
    v.updateColors();
    v.renderer.SetFrame(v.frame, v.colors);
    if (!io.WantCaptureMouse && IsMouseButtonPressed(MOUSE_BUTTON_RIGHT))
        PickAt(GetMouseX(), GetMouseY());
    BeginDrawing();
    ClearBackground(toRayColor(v.background));
    int render_width = std::max(1, v.viewportWidth() * GetRenderWidth() / std::max(GetScreenWidth(), 1));
    rlViewport(0, 0, render_width, GetRenderHeight());
    v.renderer.Draw(v.camera, render_width, GetRenderHeight(), v.render_spheres, v.render_triangles, v.wireframe,
                    v.selected);
    if (v.grid)
        visualization::DrawReferenceGrid(v.camera, render_width, GetRenderHeight());
    rlViewport(0, 0, GetRenderWidth(), GetRenderHeight());
    ImGui::Render();
    rlDrawRenderBatchActive();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    if (!v.screenshot.empty()) {
        // raylib's TakeScreenshot strips directory components; ExportImage preserves the caller's full path.
        Image image = LoadImageFromScreen();
        bool saved = image.data && ExportImage(image, v.screenshot.c_str());
        UnloadImage(image);
        if (!saved)
            DEME_WARNING("Could not save visualizer screenshot to %s.", v.screenshot.c_str());
        v.screenshot.clear();
    }
    EndDrawing();
}

void DEMVisualizer::Close() {
    if (!m_impl || !m_impl->initialized)
        return;
    auto& v = *m_impl;
    ImGui::SetCurrentContext(v.ui_context);
    v.renderer.Close();
    if (v.ui_renderer_ready)
        ImGui_ImplOpenGL3_Shutdown();
    v.ui_renderer_ready = false;
    ImGui::DestroyContext(v.ui_context);
    v.ui_context = nullptr;
    CloseWindow();
    v.initialized = false;
    v.scene_loaded = false;
}

void DEMVisualizer::SetWindowSize(int width, int height) {
    if (width <= 0 || height <= 0)
        DEME_ERROR("DEMVisualizer window dimensions must be positive.");
    m_impl->width = width;
    m_impl->height = height;
    if (m_impl->initialized)
        ::SetWindowSize(width, height);
}
void DEMVisualizer::SetWindowTitle(const std::string& title) {
    m_impl->title = title;
    if (m_impl->initialized)
        ::SetWindowTitle(title.c_str());
}
void DEMVisualizer::SetTargetFPS(int fps) {
    if (fps <= 0)
        DEME_ERROR("DEMVisualizer target FPS must be positive (got %d).", fps);
    m_impl->target_fps = fps;
    if (m_impl->initialized)
        ::SetTargetFPS(fps);
}
void DEMVisualizer::SetCameraPosition(float3 position) {
    if (!finite(visualization::ToVector(position)))
        DEME_ERROR("Camera position must be finite.");
    m_impl->camera.position = visualization::ToVector(position);
    m_impl->custom_camera = true;
}
void DEMVisualizer::SetCameraTarget(float3 target) {
    if (!finite(visualization::ToVector(target)))
        DEME_ERROR("Camera target must be finite.");
    m_impl->camera.target = visualization::ToVector(target);
    m_impl->custom_camera = true;
}
void DEMVisualizer::SetBackgroundColor(DEMVisualizerColor color) {
    m_impl->background = color;
}
void DEMVisualizer::SetFamilyColor(family_t family, DEMVisualizerColor color) {
    m_impl->family_colors[family] = color;
}
void DEMVisualizer::SetRenderSpheres(bool render) {
    m_impl->render_spheres = render;
}
void DEMVisualizer::SetRenderTriangles(bool render) {
    m_impl->render_triangles = render;
}
bool DEMVisualizer::IsRenderingSpheres() const {
    return m_impl->render_spheres;
}
bool DEMVisualizer::IsRenderingTriangles() const {
    return m_impl->render_triangles;
}
bool DEMVisualizer::ShouldStep() {
    if (!m_impl->paused)
        return true;
    bool step = m_impl->step_requested;
    m_impl->step_requested = false;
    return step;
}
void DEMVisualizer::SetPaused(bool paused) {
    m_impl->paused = paused;
    m_impl->step_requested = false;
}
bool DEMVisualizer::IsPaused() const {
    return m_impl->paused;
}
void DEMVisualizer::RequestStep() {
    m_impl->paused = true;
    m_impl->step_requested = true;
}
void DEMVisualizer::SetFamilyVisible(family_t family, bool visible) {
    m_impl->visible[family] = visible;
}
void DEMVisualizer::SetColorMode(DEMVisualizerColorMode mode) {
    if (mode < DEMVisualizerColorMode::FAMILY || mode > DEMVisualizerColorMode::SPEED)
        DEME_ERROR("Unknown visualizer color mode.");
    m_impl->color_mode = static_cast<int>(mode);
}
void DEMVisualizer::FrameAll() {
    if (m_impl->initialized)
        m_impl->fit(false);
}
void DEMVisualizer::FrameSelected() {
    if (m_impl->initialized)
        m_impl->fit(true);
}

// IDs encode geometry indices, not owner IDs. Several selected components can therefore refer to the same owner.
void DEMVisualizer::PickAt(int x, int y) {
    auto& v = *m_impl;
    if (!v.initialized || !v.scene_loaded)
        return;
    int w = std::max(1, v.viewportWidth() * GetRenderWidth() / std::max(GetScreenWidth(), 1));
    int px = x * GetRenderWidth() / std::max(GetScreenWidth(), 1);
    int py = y * GetRenderHeight() / std::max(GetScreenHeight(), 1);
    auto id = v.renderer.Pick(v.camera, w, GetRenderHeight(), px, py, v.render_spheres, v.render_triangles);
    v.clearSelection();
    if (!id)
        return;
    size_t index = id - 1;
    if (index < v.scene.spheres.size()) {
        v.selected = v.scene.spheres[index].owner;
        v.selected_geometry = index;
        v.selected_sphere = true;
    } else if (index - v.scene.spheres.size() < v.scene.triangles.size()) {
        v.selected_geometry = index - v.scene.spheres.size();
        v.selected = v.scene.triangles[v.selected_geometry].owner;
        v.selected_sphere = false;
    }
}
bodyID_t DEMVisualizer::GetSelectedOwner() const {
    return m_impl->selected;
}
size_t DEMVisualizer::GetSelectedGeometryID() const {
    return m_impl->selected_geometry;
}
bool DEMVisualizer::IsSelectedSphere() const {
    return m_impl->selected != NULL_BODYID && m_impl->selected_sphere;
}
void DEMVisualizer::RequestScreenshot(const std::string& path) {
    if (path.empty())
        DEME_ERROR("Screenshot path must not be empty.");
    m_impl->screenshot = path;
}

}  // namespace deme
