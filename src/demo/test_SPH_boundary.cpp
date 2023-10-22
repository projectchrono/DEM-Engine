//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// Fracture
// =============================================================================

#include <DEM/API.h>
#include <DEM/HostSideHelpers.hpp>
#include <DEM/utils/Samplers.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>

using namespace deme;

const double math_PI = 3.1415927;

int main() {
    DEMSolver DEMSim;
    DEMSim.UseFrictionalHertzianModel();
    DEMSim.SetVerbosity(INFO);
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    DEMSim.SetOutputContent(OUTPUT_CONTENT::XYZ | OUTPUT_CONTENT::VEL);
    DEMSim.EnsureKernelErrMsgLineNum();

    srand(7001);
    DEMSim.SetCollectAccRightAfterForceCalc(true);
    DEMSim.SetErrorOutAvgContacts(150);

    // Scale factor
    float scaling = 1.f;

    // total number of random clump templates to generate

    double radius = 0.12 / 8.0 / 2.0;
    double density = 1000;

    int totalSpheres = 3550;

    int num_template = 1;

    float plane_bottom = -0.35;

    auto mat_type_walls = DEMSim.LoadMaterial({{"E", 10e9}, {"nu", 0.3}, {"CoR", 0.60}, {"mu", 0.04}, {"Crr", 0.00}});

    auto mat_type_particles =
        DEMSim.LoadMaterial({{"E", 1.0e9}, {"nu", 0.35}, {"CoR", 0.85}, {"mu", 0.0}, {"Crr", 0.0}});

    DEMSim.SetMaterialPropertyPair("CoR", mat_type_walls, mat_type_particles, 0.50);
    DEMSim.SetMaterialPropertyPair("Crr", mat_type_walls, mat_type_particles, 0.05);
    DEMSim.SetMaterialPropertyPair("mu", mat_type_walls, mat_type_particles, 0.30);

    // Make ready for simulation
    float step_size = 2.0e-6;
    DEMSim.InstructBoxDomainDimension({-0.5, 0.5}, {-0.5, 0.5}, {-0.5, 1.50});
    DEMSim.InstructBoxDomainBoundingBC("top_open", mat_type_walls);
    DEMSim.SetInitTimeStep(step_size);
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, -9.81));
    // Max velocity info is generally just for the solver's reference and the user do not have to set it. The solver
    // wouldn't take into account a vel larger than this when doing async-ed contact detection: but this vel won't
    // happen anyway and if it does, something already went wrong.
    DEMSim.SetMaxVelocity(25.);
    DEMSim.SetInitBinSize(radius * 5);

    // Loaded meshes are by-default fixed
    auto fixed = DEMSim.AddWavefrontMeshObject("../data/granularFlow/float_1.obj", mat_type_walls);

    fixed->Scale(1 * 1.0);
    fixed->SetFamily(10);
    std::string shake_pattern_xz = " 0.01 * sin( 50 * 2 * deme::PI * t)";
    DEMSim.SetFamilyPrescribedLinVel(10, shake_pattern_xz, shake_pattern_xz, shake_pattern_xz);

    // Make an array to store these generated clump templates
    std::vector<std::shared_ptr<DEMClumpTemplate>> clump_types;

    double maxRadius = 0;

    for (int i = 0; i < num_template; i++) {
        std::vector<float> radii;
        std::vector<float3> relPos;
        std::vector<std::shared_ptr<DEMMaterial>> mat;

        double radiusMin = radius;

        float3 tmp;
        tmp.x = 0;
        tmp.y = 0;
        tmp.z = 0;

        relPos.push_back(tmp);
        mat.push_back(mat_type_particles);
        radii.push_back(radiusMin);

        double c = radiusMin;  // smaller dim of the ellipse
        double b = radiusMin;
        double a = radiusMin;

        float mass = 4.0 / 3.0 * PI * a * b * c * density;
        float3 MOI = make_float3(1.f / 5.f * mass * (b * b + c * c), 1.f / 5.f * mass * (a * a + c * c),
                                 1.f / 5.f * mass * (b * b + a * a));
        std::cout << a << " chosen moi ..." << a / radius << std::endl;

        auto clump_ptr = DEMSim.LoadClumpType(mass, MOI, radii, relPos, mat_type_particles);
        // clump_ptr->AssignName("fsfs");
        clump_types.push_back(clump_ptr);
    }

    std::filesystem::path out_dir = std::filesystem::current_path();
    out_dir += "/test_SPH";
    remove_all(out_dir);
    create_directory(out_dir);

    unsigned int currframe = 0;
    unsigned int curr_step = 0;
    float settle_frame_time = 0.004;

    char filename[200], meshfile[200];

    float shift_xyz = 1.0 * (radius) * 2.0;
    float x = 0;
    float y = 0;

    float z = shift_xyz / 2;  // by default we create beads at 0
    double emitterZ = 0.5;
    unsigned int actualTotalSpheres = 0;

    // Some inspectors
    auto max_z_finder = DEMSim.CreateInspector("clump_max_z");

    //DEMSim.SetFamilyExtraMargin(1, 0.0 * radius);

    DEMSim.SetInitTimeStep(step_size);
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0.00, -9.81));
    DEMSim.Initialize();
    DEMSim.DisableContactBetweenFamilies(20, 1);
    std::cout << "Initial number of contacts: " << DEMSim.GetNumContacts() << std::endl;

    int frame = 0;
    bool generate = true;
    bool initialization = true;
    double timeTotal = 0;
    double consolidation = true;

    while (initialization) {
        std::vector<std::shared_ptr<DEMClumpTemplate>> input_pile_template_type;
        std::vector<float3> input_pile_xyz;
        PDSampler sampler(shift_xyz);

        bool generate = (plane_bottom + shift_xyz / 2 > emitterZ) ? false : true;

        if (generate) {
            float sizeZ = (frame == 0) ? 1.30 : 0.00;
            float sizeX = 0.20;
            float z = plane_bottom + shift_xyz + sizeZ / 2.0;

            float3 center_xyz = make_float3(0, 0, z);

            // std::cout << "level of particles position ... " << center_xyz.z << std::endl;

            auto heap_particles_xyz = sampler.SampleCylinderZ(center_xyz, sizeX / 2, sizeZ / 2.0);
            unsigned int num_clumps = heap_particles_xyz.size();
            // std::cout << "number of particles at this level ... " << num_clumps << std::endl;

            for (unsigned int i = actualTotalSpheres; i < actualTotalSpheres + num_clumps; i++) {
                input_pile_template_type.push_back(clump_types.at(i % num_template));
            }

            input_pile_xyz.insert(input_pile_xyz.end(), heap_particles_xyz.begin(), heap_particles_xyz.end());

            auto the_pile = DEMSim.AddClumps(input_pile_template_type, input_pile_xyz);
            the_pile->SetVel(make_float3(-0.00, 0.0, -0.90));
            the_pile->SetFamily(100);

            DEMSim.UpdateClumps();

            std::cout << "Total num of particles: " << (int)DEMSim.GetNumClumps() << std::endl;
            actualTotalSpheres = (int)DEMSim.GetNumClumps();
        }
        timeTotal += settle_frame_time;
        // std::cout << "Total runtime: " << timeTotal << "s; settling for: " << settle_frame_time << std::endl;
        // std::cout << "maxZ is: " << max_z_finder->GetValue() << std::endl;

        initialization = (actualTotalSpheres < totalSpheres) ? true : false;

        if (generate && !(frame % 1)) {
            std::cout << "frame : " << frame << std::endl;
            sprintf(filename, "%s/DEMdemo_settling_%04d.csv", out_dir.c_str(), frame);
            DEMSim.WriteSphereFile(std::string(filename));
            sprintf(meshfile, "%s/DEMdemo_mesh.vtk", out_dir.c_str());
            DEMSim.WriteMeshFile(std::string(meshfile));
            // DEMSim.ShowThreadCollaborationStats();
            frame++;
        }

        DEMSim.DoDynamicsThenSync(settle_frame_time);

        plane_bottom = max_z_finder->GetValue();
    }
    std::cout << "Initialization done with : " << actualTotalSpheres << "particles" << std::endl;

    float sim_end = 1.0;
    unsigned int fps = 20;
    float frame_time = 1.0 / fps;

    std::cout << "Output at " << fps << " FPS" << std::endl;
    std::cout << "time output " << frame_time << "" << std::endl;

    unsigned int frame_count = 0;
    unsigned int step_count = 0;

    frame = 0;

    sprintf(meshfile, "%s/DEMdemo_funnel_%04d.vtk", out_dir.c_str(), frame);
    DEMSim.WriteMeshFile(std::string(meshfile));
    sprintf(filename, "%s/DEMdemo_settling_%04d.csv", out_dir.c_str(), frame++);
    DEMSim.WriteSphereFile(std::string(filename));

    bool status_1 = true;
    bool status_2 = false;

    // DEMSim.DisableContactBetweenFamilies(10, 1);
    float totalRunTime = 0.0f;

    // Simulation loop
    for (int i = 0; totalRunTime < sim_end; i++) {
        totalRunTime += frame_time;
        char filename[200];
        char meshname[200];
        char cnt_filename[200];
        std::cout << "Outputting frame: " << frame_time << std::endl;
        sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), frame_count);
        sprintf(meshname, "%s/DEMdemo_mesh_%04d.vtk", out_dir.c_str(), frame_count);
        DEMSim.WriteSphereFile(std::string(filename));
        DEMSim.WriteMeshFile(std::string(meshname));
        frame_count++;
        DEMSim.ShowThreadCollaborationStats();
        // std::cout << "Initial number of contacts: " << DEMSim.GetNumContacts() << std::endl;
        if (totalRunTime > 0.9 && status_1) {
            DEMSim.DoDynamicsThenSync(0);
            std::cout << "gate is in motion from: " << totalRunTime << " s" << std::endl;
            std::cout << "and it will stop in : " << totalRunTime << " s" << std::endl;
            DEMSim.ChangeFamily(10, 0);
            status_1 = false;
        }
        DEMSim.DoDynamics(frame_time);
    }

    DEMSim.ShowTimingStats();
    std::cout << "Fracture demo exiting..." << std::endl;
    return 0;
}
