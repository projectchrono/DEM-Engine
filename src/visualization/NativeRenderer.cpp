// Copyright (c) 2026, SBEL GPU Development Team
// SPDX-License-Identifier: BSD-3-Clause
#include "NativeRenderer.h"

#include <climits>
#include <cstddef>
#include <limits>

#include "core/utils/Logger.hpp"
#include <glad.h>
#include "RayMathCompat.h"
#include <rlgl.h>

namespace deme {
namespace visualization {
namespace {

// Explicit attribute locations are shared by both VAOs. Identity is integer data to preserve all 32 owner/element bits.
const char* vertex_shader = R"glsl(#version 330
layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;
layout(location=2) in vec4 component;
layout(location=3) in uvec2 identity;
uniform mat4 mvp;
uniform samplerBuffer owners;
uniform bool sphere;
uniform uint selected;
out vec3 worldNormal;
flat out vec4 tint;
flat out uint element;
vec3 rotateQ(vec4 q, vec3 p) { return p + 2.0 * cross(q.xyz, cross(q.xyz,p) + q.w*p); }
void main() {
    int index = int(identity.x)*3;
    vec3 p = texelFetch(owners,index).xyz;
    vec4 q = texelFetch(owners,index+1);
    tint = texelFetch(owners,index+2);
    if (identity.x == selected) tint.rgb = mix(tint.rgb,vec3(1.0,0.78,0.2),0.65);
    vec3 local = sphere ? component.xyz + component.w*position : position;
    gl_Position = mvp * vec4(p + rotateQ(q,local),1.0);
    worldNormal = rotateQ(q,normal);
    element = identity.y;
}
)glsl";

const char* fragment_shader = R"glsl(#version 330
in vec3 worldNormal;
flat in vec4 tint;
flat in uint element;
uniform bool picking;
out vec4 color;
void main() {
    if (tint.a <= 0.0) discard;
    if (picking) {
        color = vec4(float(element & 255u),float((element >> 8u)&255u),
                     float((element >> 16u)&255u),float((element >> 24u)&255u))/255.0;
    } else {
        vec3 n = normalize(worldNormal);
        if (!gl_FrontFacing) n = -n;
        float light = 0.34 + 0.66*max(dot(n,normalize(vec3(0.4,-0.5,0.8))),0.0);
        color = vec4(tint.rgb*light,tint.a);
    }
}
)glsl";

struct Vertex {
    Vector3 position, normal;
};
struct Instance {
    float x, y, z, radius;
    std::uint32_t owner, element;
};
struct MeshVertex {
    Vector3 position, normal;
    std::uint32_t owner, element;
};

Vector3 vector(float3 v) {
    return {v.x, v.y, v.z};
}

// Bind interleaved position/normal attributes, leaving the integer identity layout to the caller.
void vertexAttributes(size_t stride) {
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, static_cast<int>(stride), nullptr);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, static_cast<int>(stride), reinterpret_cast<void*>(sizeof(Vector3)));
}

}  // namespace

void NativeRenderer::Initialize() {
    shader = LoadShaderFromMemory(vertex_shader, fragment_shader);
    if (!shader.id || shader.id == rlGetShaderIdDefault()) {
        shader = {};
        DEME_ERROR("Native visualizer requires OpenGL 3.3 and could not compile its geometry shaders.");
    }
    glGenVertexArrays(1, &sphere_vao);
    glGenBuffers(1, &sphere_vertices);
    glGenBuffers(1, &sphere_instances);
    glGenVertexArrays(1, &mesh_vao);
    glGenBuffers(1, &mesh_vertices);
    glGenBuffers(1, &mesh_indices);
    glGenBuffers(1, &owner_buffer);
    glGenTextures(1, &owner_texture);
}

void NativeRenderer::Close() {
    if (pick_target.id)
        UnloadRenderTexture(pick_target);
    pick_target = {};
    if (shader.id)
        UnloadShader(shader);
    shader = {};
    glDeleteVertexArrays(1, &sphere_vao);
    glDeleteVertexArrays(1, &mesh_vao);
    const unsigned int buffers[] = {sphere_vertices, sphere_instances, mesh_vertices, mesh_indices, owner_buffer};
    glDeleteBuffers(5, buffers);
    glDeleteTextures(1, &owner_texture);
    sphere_vao = mesh_vao = sphere_vertices = sphere_instances = mesh_vertices = mesh_indices = owner_buffer =
        owner_texture = 0;
    owner_capacity = 0;
}

// Build shared sphere topology and static component instances. Facet vertices retain their own flat normal and ID.
void NativeRenderer::SetScene(const DEMVisualizationScene& scene, int resolution) {
    if (scene.spheres.size() > INT_MAX || scene.triangles.size() > INT_MAX / 3 ||
        scene.spheres.size() + scene.triangles.size() >= std::numeric_limits<std::uint32_t>::max()) {
        DEME_ERROR("Visualization geometry exceeds the OpenGL draw/index capacity.");
    }
    Mesh sphere = GenMeshSphere(1.0f, resolution, resolution * 2);
    std::vector<Vertex> vertices;
    vertices.reserve(sphere.triangleCount * 3);
    for (int i = 0; i < sphere.triangleCount * 3; ++i) {
        int v = sphere.indices ? sphere.indices[i] : i;
        vertices.push_back({{sphere.vertices[v * 3], sphere.vertices[v * 3 + 1], sphere.vertices[v * 3 + 2]},
                            {sphere.normals[v * 3], sphere.normals[v * 3 + 1], sphere.normals[v * 3 + 2]}});
    }
    UnloadMesh(sphere);
    sphere_vertex_count = static_cast<int>(vertices.size());
    sphere_count = static_cast<int>(scene.spheres.size());
    std::vector<Instance> instances;
    instances.reserve(scene.spheres.size());
    for (size_t i = 0; i < scene.spheres.size(); ++i) {
        const auto& s = scene.spheres[i];
        instances.push_back({s.offset.x, s.offset.y, s.offset.z, s.radius, s.owner, static_cast<std::uint32_t>(i + 1)});
    }
    glBindVertexArray(sphere_vao);
    glBindBuffer(GL_ARRAY_BUFFER, sphere_vertices);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.data(), GL_STATIC_DRAW);
    vertexAttributes(sizeof(Vertex));
    glBindBuffer(GL_ARRAY_BUFFER, sphere_instances);
    glBufferData(GL_ARRAY_BUFFER, instances.size() * sizeof(Instance), instances.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(Instance), nullptr);
    glVertexAttribDivisor(2, 1);
    glEnableVertexAttribArray(3);
    glVertexAttribIPointer(3, 2, GL_UNSIGNED_INT, sizeof(Instance), reinterpret_cast<void*>(offsetof(Instance, owner)));
    glVertexAttribDivisor(3, 1);

    std::vector<MeshVertex> mesh;
    std::vector<std::uint32_t> indices;
    mesh.reserve(scene.triangles.size() * 3);
    indices.reserve(scene.triangles.size() * 3);
    for (size_t i = 0; i < scene.triangles.size(); ++i) {
        const auto& t = scene.triangles[i];
        Vector3 a = vector(t.a), b = vector(t.b), c = vector(t.c);
        Vector3 normal = Vector3Normalize(Vector3CrossProduct(Vector3Subtract(b, a), Vector3Subtract(c, a)));
        for (auto p : {a, b, c}) {
            indices.push_back(static_cast<std::uint32_t>(mesh.size()));
            mesh.push_back({p, normal, t.owner, static_cast<std::uint32_t>(scene.spheres.size() + i + 1)});
        }
    }
    mesh_index_count = static_cast<int>(indices.size());
    glBindVertexArray(mesh_vao);
    glBindBuffer(GL_ARRAY_BUFFER, mesh_vertices);
    glBufferData(GL_ARRAY_BUFFER, mesh.size() * sizeof(MeshVertex), mesh.data(), GL_STATIC_DRAW);
    vertexAttributes(sizeof(MeshVertex));
    glEnableVertexAttribArray(3);
    glVertexAttribIPointer(3, 2, GL_UNSIGNED_INT, sizeof(MeshVertex),
                           reinterpret_cast<void*>(offsetof(MeshVertex, owner)));
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh_indices);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(std::uint32_t), indices.data(), GL_STATIC_DRAW);
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

// The only per-frame geometry upload is O(owners), independent of component/facet counts.
void NativeRenderer::SetFrame(const DEMVisualizationFrame& frame, const std::vector<std::array<float, 4>>& colors) {
    const size_t count = frame.positions.size();
    int max_texels = 0;
    glGetIntegerv(GL_MAX_TEXTURE_BUFFER_SIZE, &max_texels);
    if (count > static_cast<size_t>(max_texels) / 3 || count > INT_MAX / 3) {
        DEME_ERROR("Visualization needs %zu owner texels; this OpenGL device supports %d.", count * 3, max_texels);
    }
    owner_data.resize(count * 3);
    for (size_t i = 0; i < count; ++i) {
        auto p = frame.positions[i];
        auto q = frame.orientations[i];
        owner_data[i * 3] = {p.x, p.y, p.z, 0};
        owner_data[i * 3 + 1] = {q.x, q.y, q.z, q.w};
        owner_data[i * 3 + 2] = colors[i];
    }
    glBindBuffer(GL_TEXTURE_BUFFER, owner_buffer);
    size_t bytes = owner_data.size() * sizeof(owner_data[0]);
    if (bytes > owner_capacity || !owner_capacity) {
        owner_capacity = std::max(bytes, sizeof(owner_data[0]));
        glBufferData(GL_TEXTURE_BUFFER, owner_capacity, nullptr, GL_DYNAMIC_DRAW);
    }
    if (bytes)
        glBufferSubData(GL_TEXTURE_BUFFER, 0, bytes, owner_data.data());
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_BUFFER, owner_texture);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F, owner_buffer);
    glBindTexture(GL_TEXTURE_BUFFER, 0);
    glBindBuffer(GL_TEXTURE_BUFFER, 0);
}

// Draw directly with OpenGL; flush raylib first and restore the state its overlay drawing expects afterwards.
void NativeRenderer::Draw(const Camera3D& camera,
                          int width,
                          int height,
                          bool spheres,
                          bool triangles,
                          bool wireframe,
                          bodyID_t selected,
                          bool picking) {
    rlDrawRenderBatchActive();
    float distance = Vector3Distance(camera.position, camera.target);
    double near_clip = std::max(0.00001, static_cast<double>(distance) * 0.0001);
    Matrix projection = MatrixPerspective(camera.fovy * DEG2RAD, static_cast<double>(width) / std::max(height, 1),
                                          near_clip, std::max(1000.0, static_cast<double>(distance) * 100.0));
    Matrix mvp = MatrixMultiply(GetCameraMatrix(camera), projection);
    glUseProgram(shader.id);
    glUniformMatrix4fv(glGetUniformLocation(shader.id, "mvp"), 1, GL_FALSE, MatrixToFloat(mvp));
    glUniform1i(glGetUniformLocation(shader.id, "owners"), 0);
    glUniform1i(glGetUniformLocation(shader.id, "picking"), picking);
    glUniform1ui(glGetUniformLocation(shader.id, "selected"), selected);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_BUFFER, owner_texture);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    if (picking) {
        glDisable(GL_BLEND);
    } else {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    }
    // DEM mesh facets must remain visible from either side. The fragment shader flips back-face lighting normals.
    glDisable(GL_CULL_FACE);
    if (wireframe && !picking)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    if (spheres && sphere_count) {
        glUniform1i(glGetUniformLocation(shader.id, "sphere"), 1);
        glBindVertexArray(sphere_vao);
        glDrawArraysInstanced(GL_TRIANGLES, 0, sphere_vertex_count, sphere_count);
    }
    if (triangles && mesh_index_count) {
        glUniform1i(glGetUniformLocation(shader.id, "sphere"), 0);
        glBindVertexArray(mesh_vao);
        glDrawElements(GL_TRIANGLES, mesh_index_count, GL_UNSIGNED_INT, nullptr);
    }
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_BUFFER, 0);
    glUseProgram(0);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glEnable(GL_CULL_FACE);
}

// A separate single-sample target preserves exact IDs even when the visible window uses MSAA.
std::uint32_t
NativeRenderer::Pick(const Camera3D& camera, int width, int height, int x, int y, bool spheres, bool triangles) {
    if (width <= 0 || height <= 0 || x < 0 || y < 0 || x >= width || y >= height)
        return 0;
    if (pick_target.texture.width != width || pick_target.texture.height != height) {
        if (pick_target.id)
            UnloadRenderTexture(pick_target);
        pick_target = LoadRenderTexture(width, height);
        if (!pick_target.id)
            DEME_ERROR("Unable to allocate visualizer picking framebuffer.");
    }
    BeginTextureMode(pick_target);
    ClearBackground({0, 0, 0, 0});
    GLboolean dither = glIsEnabled(GL_DITHER);
    glDisable(GL_DITHER);
    Draw(camera, width, height, spheres, triangles, false, NULL_BODYID, true);
    unsigned char pixel[4]{};
    glReadPixels(x, height - 1 - y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, pixel);
    if (dither)
        glEnable(GL_DITHER);
    EndTextureMode();
    return std::uint32_t(pixel[0]) | (std::uint32_t(pixel[1]) << 8) | (std::uint32_t(pixel[2]) << 16) |
           (std::uint32_t(pixel[3]) << 24);
}

}  // namespace visualization
}  // namespace deme
