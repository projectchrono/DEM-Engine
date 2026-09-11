// Copyright (c) 2026, SBEL GPU Development Team
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <algorithm>
#include <array>
#include <cmath>

#include <imgui.h>
#include <raylib.h>
#include <rlgl.h>
#include "RayMathCompat.h"

namespace deme {
namespace visualization {

inline Vector3 ToVector(float3 p) {
    return {p.x, p.y, p.z};
}

// Use the same projection for geometry and reference lines, including non-square viewports and small DEM units.
inline Matrix Projection(const Camera3D& camera, int width, int height) {
    double distance = Vector3Distance(camera.position, camera.target);
    return MatrixPerspective(camera.fovy * DEG2RAD, static_cast<double>(width) / std::max(height, 1),
                             std::max(0.00001, distance * 0.0001), std::max(1000.0, distance * 100.0));
}

// DEME is Z-up. A distance-scaled XY grid provides context without uploading simulation geometry through raylib.
inline void DrawReferenceGrid(const Camera3D& camera, int width, int height) {
    float spacing =
        std::pow(10.f, std::floor(std::log10(std::max(Vector3Distance(camera.position, camera.target), 1.e-5f))) - 1);
    float cx = std::floor(camera.target.x / spacing) * spacing;
    float cy = std::floor(camera.target.y / spacing) * spacing;
    BeginMode3D(camera);
    rlSetMatrixProjection(Projection(camera, width, height));
    for (int i = -10; i <= 10; ++i) {
        DrawLine3D({cx + i * spacing, cy - 10 * spacing, 0}, {cx + i * spacing, cy + 10 * spacing, 0},
                   {57, 65, 79, 255});
        DrawLine3D({cx - 10 * spacing, cy + i * spacing, 0}, {cx + 10 * spacing, cy + i * spacing, 0},
                   {57, 65, 79, 255});
    }
    DrawLine3D({0, 0, 0}, {spacing * 3, 0, 0}, {220, 95, 95, 255});
    DrawLine3D({0, 0, 0}, {0, spacing * 3, 0}, {95, 200, 125, 255});
    DrawLine3D({0, 0, 0}, {0, 0, spacing * 3}, {95, 150, 240, 255});
    EndMode3D();
}

// Owner quaternions are x,y,z,w. This also handles components of combined owners using their reimposed poses.
inline Vector3 WorldPoint(float3 local, float3 position, float4 orientation) {
    return Vector3Add(ToVector(position),
                      Vector3RotateByQuaternion(
                          ToVector(local), Quaternion{orientation.x, orientation.y, orientation.z, orientation.w}));
}

// A small platform bridge keeps ImGui independent of raylib's private GLFW window. Render backend is upstream ImGui.
inline void BeginInputFrame() {
    auto& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(static_cast<float>(GetScreenWidth()), static_cast<float>(GetScreenHeight()));
    io.DisplayFramebufferScale = ImVec2(static_cast<float>(GetRenderWidth()) / std::max(GetScreenWidth(), 1),
                                        static_cast<float>(GetRenderHeight()) / std::max(GetScreenHeight(), 1));
    io.DeltaTime = std::max(GetFrameTime(), 0.001f);
    auto p = GetMousePosition();
    io.AddMousePosEvent(p.x, p.y);
    io.AddFocusEvent(IsWindowFocused());
    io.AddMouseButtonEvent(0, IsMouseButtonDown(MOUSE_BUTTON_LEFT));
    io.AddMouseButtonEvent(1, IsMouseButtonDown(MOUSE_BUTTON_RIGHT));
    io.AddMouseButtonEvent(2, IsMouseButtonDown(MOUSE_BUTTON_MIDDLE));
    auto wheel = GetMouseWheelMoveV();
    io.AddMouseWheelEvent(wheel.x, wheel.y);
    io.AddKeyEvent(ImGuiMod_Ctrl, IsKeyDown(KEY_LEFT_CONTROL) || IsKeyDown(KEY_RIGHT_CONTROL));
    io.AddKeyEvent(ImGuiMod_Shift, IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT));
    io.AddKeyEvent(ImGuiMod_Alt, IsKeyDown(KEY_LEFT_ALT) || IsKeyDown(KEY_RIGHT_ALT));
    io.AddKeyEvent(ImGuiMod_Super, IsKeyDown(KEY_LEFT_SUPER) || IsKeyDown(KEY_RIGHT_SUPER));
    const int keys[] = {KEY_TAB,    KEY_LEFT, KEY_RIGHT,  KEY_UP,     KEY_DOWN,      KEY_PAGE_UP, KEY_PAGE_DOWN,
                        KEY_HOME,   KEY_END,  KEY_INSERT, KEY_DELETE, KEY_BACKSPACE, KEY_SPACE,   KEY_ENTER,
                        KEY_ESCAPE, KEY_A,    KEY_C,      KEY_V,      KEY_X,         KEY_Y,       KEY_Z};
    const ImGuiKey mapped[] = {ImGuiKey_Tab,       ImGuiKey_LeftArrow, ImGuiKey_RightArrow, ImGuiKey_UpArrow,
                               ImGuiKey_DownArrow, ImGuiKey_PageUp,    ImGuiKey_PageDown,   ImGuiKey_Home,
                               ImGuiKey_End,       ImGuiKey_Insert,    ImGuiKey_Delete,     ImGuiKey_Backspace,
                               ImGuiKey_Space,     ImGuiKey_Enter,     ImGuiKey_Escape,     ImGuiKey_A,
                               ImGuiKey_C,         ImGuiKey_V,         ImGuiKey_X,          ImGuiKey_Y,
                               ImGuiKey_Z};
    for (size_t i = 0; i < sizeof(keys) / sizeof(keys[0]); ++i)
        io.AddKeyEvent(mapped[i], IsKeyDown(keys[i]));
    for (int c = GetCharPressed(); c; c = GetCharPressed())
        io.AddInputCharacter(static_cast<unsigned int>(c));
}

// Z-up orbit camera with distance-scaled panning. Clamping pitch prevents the up vector becoming singular.
inline void Navigate(Camera3D& camera) {
    Vector3 offset = Vector3Subtract(camera.position, camera.target);
    float distance = std::max(Vector3Length(offset), 1.e-5f);
    Vector2 delta = GetMouseDelta();
    float yaw = std::atan2(offset.y, offset.x);
    float pitch = std::asin(std::clamp(offset.z / distance, -1.f, 1.f));
    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        yaw -= delta.x * 0.006f;
        pitch = std::clamp(pitch + delta.y * 0.006f, -1.55f, 1.55f);
    }
    if (IsMouseButtonDown(MOUSE_BUTTON_MIDDLE)) {
        Vector3 forward = Vector3Normalize(Vector3Negate(offset));
        Vector3 right = Vector3Normalize(Vector3CrossProduct(forward, camera.up));
        Vector3 up = Vector3CrossProduct(right, forward);
        float scale = distance * 0.0015f;
        camera.target = Vector3Add(
            camera.target, Vector3Add(Vector3Scale(right, -delta.x * scale), Vector3Scale(up, delta.y * scale)));
    }
    distance = std::clamp(distance * std::exp(-GetMouseWheelMove() * 0.12f), 1.e-5f, 1.e12f);
    camera.position =
        Vector3Add(camera.target, {distance * std::cos(pitch) * std::cos(yaw),
                                   distance * std::cos(pitch) * std::sin(yaw), distance * std::sin(pitch)});
}

// Perceptually ordered blue/cyan/yellow ramp, shared by owner colors and the scalar legend.
inline std::array<float, 4> ScalarColor(float t) {
    t = std::clamp(t, 0.f, 1.f);
    const Vector3 a{0.16f, 0.28f, 0.75f}, b{0.1f, 0.8f, 0.75f}, c{1.f, 0.85f, 0.2f};
    auto color = t < 0.5f ? Vector3Lerp(a, b, t * 2) : Vector3Lerp(b, c, (t - 0.5f) * 2);
    return {color.x, color.y, color.z, 1.f};
}

}  // namespace visualization
}  // namespace deme
