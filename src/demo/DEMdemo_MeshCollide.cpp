//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// A small-scale demo that tests meshed particle collisions.
// =============================================================================

#include <core/ApiVersion.h>
#include <core/utils/ThreadManager.h>
#include <DEM/API.h>
#include <DEM/utils/Samplers.hpp>

#include <filesystem>
#include <cstdio>
#include <time.h>
#include <filesystem>

using namespace deme;
using namespace std::filesystem;

int main() {
    DEMSolver DEMSim;
    DEMSim.SetVerbosity("INFO");
    // DEMSim.UseFrictionlessHertzianModel();
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    DEMSim.SetContactOutputContent({"OWNER", "FORCE", "POINT", "NORMAL", "TORQUE", "CNT_WILDCARD"});
    DEMSim.InstructBoxDomainDimension(10, 10, 4);
    // Note!! If you want meshes to have contacts, set this to true!!
    // If not, meshes will not have contacts with each other or analytical boundaries (but still have contacts with
    // clumps).
    DEMSim.SetMeshUniversalContact(true);

    // Special material: has a cohesion param
    auto mat_type_1 = DEMSim.LoadMaterial({{"E", 1e8}, {"nu", 0.3}, {"CoR", 0.8}, {"mu", 0.3}, {"Crr", 0.01}});
    auto mat_type_2 = DEMSim.LoadMaterial({{"E", 1e8}, {"nu", 0.4}, {"CoR", 0.6}, {"mu", 0.3}, {"Crr", 0.01}});
    // If you don't have this line, then CoR between thw 2 materials will take average when they are in contact
    DEMSim.SetMaterialPropertyPair("CoR", mat_type_1, mat_type_2, 0.6);

    // Add a bottom plane
    DEMSim.AddBCPlane(make_float3(0, 0, -1.8), make_float3(0, 0, 1), mat_type_2);

    auto particle1 = DEMSim.AddWavefrontMeshObject((GET_DATA_PATH() / "mesh/cube.obj").string(), mat_type_1);
    particle1->SetFamily(0);
    particle1->SetInitPos(make_float3(-0.4, 0, -1.25));
    particle1->SetMass(10000.);
    particle1->SetMOI(make_float3(2000., 2000., 2000.));
    auto tracker1 = DEMSim.Track(particle1);

    auto particle2 = DEMSim.AddWavefrontMeshObject((GET_DATA_PATH() / "mesh/cube.obj").string(), mat_type_1);
    particle2->SetFamily(0);
    particle2->SetInitPos(make_float3(0.4, 0, 0));
    particle2->SetMass(10000.);
    particle2->SetMOI(make_float3(2000., 2000., 2000.));
    auto tracker2 = DEMSim.Track(particle2);

    float step_time = 1e-5;
    DEMSim.SetInitTimeStep(step_time);
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, -9.81));
    DEMSim.SetExpandSafetyType("auto");
    // DEMSim.DisableAdaptiveBinSize();
    // DEMSim.SetInitBinSize(10);
    DEMSim.Initialize();

    // Ready simulation
    path out_dir = current_path();
    out_dir /= "DemoOutput_MeshCollide";
    create_directory(out_dir);

    unsigned int ID1 = tracker1->GetOwnerIDs()[0];
    unsigned int ID2 = tracker2->GetOwnerIDs()[0];

    float frame_time = 1e-2;
    int frame = 0;
    int frame_step = (int)(frame_time / step_time);
    double final_time = 0.5;
    for (int i = 0; i < (int)(final_time / step_time); i++) {
        if (i % frame_step == 0) {
            frame++;
            std::cout << "Frame: " << frame << std::endl;

            char meshfilename[100];
            sprintf(meshfilename, "DEMdemo_mesh_%04d.vtk", frame);
            DEMSim.WriteMeshFile(out_dir / meshfilename);

            // Test if family changer works
            float3 pos1 = tracker1->Pos();
            float3 pos2 = tracker2->Pos();

            // Test getting all contact force pairs concerning different trackers
            std::vector<float3> forces_mesh, points_mesh;
            // GetOwnerContactForces is another way to query the contact forces, if you don't want to write them to
            // files. If a contact involves at least one of the owner IDs provided as the first arg of
            // GetOwnerContactForces, it will be outputted. Note if a contact involves two IDs of the user-provided
            // list, then the force for that contact will be given as the force experienced by whichever owner that
            // appears earlier in the ID list. DEMSim.GetOwnerContactForces({ID1, ID2}, points_mesh, forces_mesh);

            std::cout << "----------------------------------------" << std::endl;
            std::cout << "Particle 1 Z coord is " << pos1.z << std::endl;
            std::cout << "Particle 2 Z coord is " << pos2.z << std::endl;
            if (points_mesh.size() > 0) {
                std::cout << "Two meshes collide, one contact is at (" << points_mesh[0].x << ", " << points_mesh[0].y
                          << ", " << points_mesh[0].z << ")." << std::endl;
            }
            DEMSim.ShowMemStats();
            std::cout << "----------------------------------------" << std::endl;
        }

        DEMSim.DoDynamics(step_time);
    }

    DEMSim.ShowThreadCollaborationStats();
    DEMSim.ShowTimingStats();
    std::cout << "DEMdemo_MeshCollide exiting..." << std::endl;
    return 0;
}
