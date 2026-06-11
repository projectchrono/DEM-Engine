// Copyright (c) 2021, SBEL GPU Development Team
// SPDX-License-Identifier: BSD-3-Clause

#include <DEM/API.h>

#include <iostream>

using namespace deme;

int main() {
    constexpr int kNumQuads = 200;
    constexpr int kNumTriangles = 2 * kNumQuads;
    constexpr float kDx = 0.01f;
    constexpr float kWidth = 0.1f;
    constexpr float kPenetration = 0.001f;
    constexpr float kStepSize = 1e-6f;

    DEMSolver sim;
    sim.SetVerbosity("ERROR");
    sim.InstructBoxDomainDimension(5.f, 5.f, 5.f);
    sim.SetGravitationalAcceleration(make_float3(0.f));
    sim.SetMeshUniversalContact(true);
    sim.SetExpandSafetyType("auto");

    auto mat = sim.LoadMaterial({{"E", 1e7}, {"nu", 0.3}, {"CoR", 0.3}, {"mu", 0.3}});
    sim.AddBCPlane(make_float3(0.f), make_float3(0.f, 0.f, 1.f), mat);

    DEMMesh strip;
    strip.m_vertices.reserve(2 * (kNumQuads + 1));
    strip.m_face_v_indices.reserve(kNumTriangles);
    for (int i = 0; i <= kNumQuads; ++i) {
        const float x = (i - kNumQuads / 2) * kDx;
        strip.m_vertices.push_back(make_float3(x, -0.5f * kWidth, -kPenetration));
        strip.m_vertices.push_back(make_float3(x, 0.5f * kWidth, -kPenetration));
    }
    for (int i = 0; i < kNumQuads; ++i) {
        const int bottom = 2 * i;
        const int top = bottom + 1;
        const int next_bottom = bottom + 2;
        const int next_top = bottom + 3;
        // This ordering makes the triangle adjacency graph a path with diameter kNumTriangles - 1.
        strip.m_face_v_indices.push_back(make_int3(bottom, next_bottom, top));
        strip.m_face_v_indices.push_back(make_int3(next_bottom, next_top, top));
    }
    strip.nTri = kNumTriangles;
    strip.SetMaterial(mat);
    strip.SetMass(1.f);
    strip.SetMOI(make_float3(1.f));
    auto strip_handle = sim.AddMesh(strip);
    strip_handle->SetFamily(1);

    sim.SetInitTimeStep(kStepSize);
    sim.Initialize();
    sim.DoDynamics(kStepSize);

    const size_t patch_contacts = sim.GetNumContacts();
    if (patch_contacts != 1) {
        std::cerr << "FAIL: expected one island for a connected " << kNumTriangles << "-triangle contact strip, got "
                  << patch_contacts << std::endl;
        return 1;
    }

    std::cout << "PASS: connected " << kNumTriangles << "-triangle contact strip produced one island." << std::endl;
    return 0;
}
