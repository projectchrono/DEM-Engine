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

int main() {
    DEMSolver DEMSim;
    DEMSim.UseFrictionalHertzianModel();
    DEMSim.SetVerbosity(INFO);
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    DEMSim.SetOutputContent(OUTPUT_CONTENT::XYZ | OUTPUT_CONTENT::VEL);
    DEMSim.EnsureKernelErrMsgLineNum();

    srand(7001);
    DEMSim.SetCollectAccRightAfterForceCalc(true);
    // DEMSim.SetErrorOutAvgContacts(150);

    // Scale factor
    float scaling = 1.f;

    // total number of random clump templates to generate

    double radius = (1.00/2); // 
    double density = 1000;

    // int totalSpheres = 3845;  // particles for dp5
    int totalSpheres = 12000;  // particles for dp6
    // int totalSpheres = 200;
    // int totalSpheres=30900; //dp10

    int num_template = 1;

    float plane_bottom = 0.30;

    auto mat_type_walls = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.00}, {"mu", 0.10}, {"Crr", 0.00}});

    auto mat_type_particles =
        DEMSim.LoadMaterial({{"E", 1.0e9}, {"nu", 0.35}, {"CoR", 0.00}, {"mu", 0.10}, {"Crr", 0.0}});

    DEMSim.SetMaterialPropertyPair("CoR", mat_type_walls, mat_type_particles, 0.00);
    DEMSim.SetMaterialPropertyPair("Crr", mat_type_walls, mat_type_particles, 0.00);
    DEMSim.SetMaterialPropertyPair("mu", mat_type_walls, mat_type_particles, 0.10);

    // Make ready for simulation
    float step_size = 2.0e-6;
    DEMSim.InstructBoxDomainDimension({-30.0, 30.0}, {-10, 10}, {-10.0, 140.0});
    DEMSim.InstructBoxDomainBoundingBC("top_open", mat_type_walls);
    DEMSim.SetInitTimeStep(step_size);
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, -9.81));
    // Max velocity info is generally just for the solver's reference and the user do not have to set it. The solver
    // wouldn't take into account a vel larger than this when doing async-ed contact detection: but this vel won't
    // happen anyway and if it does, something already went wrong.
    DEMSim.SetMaxVelocity(25.);
    DEMSim.SetInitBinSize(radius * 5);

    // Loaded meshes are by-default fixed
    auto fixed = DEMSim.AddWavefrontMeshObject("../data/my/wavestar2fill.obj", mat_type_walls);
    // float_96_2fill
    fixed->Scale(10 * 1.0);
    fixed->SetFamily(10);
    float ang = PI;
    float3 move = make_float3(0.00, 0.00, 0);  // z
    float4 rot = make_float4(cos(ang / 2), 0, 0, sin(ang / 2));

    fixed->Move(move, rot);
    std::string shake_pattern_xz = " 0.01 * sin( 30 * 2 * deme::PI * t)";
    DEMSim.SetFamilyPrescribedLinVel(10, shake_pattern_xz, shake_pattern_xz, shake_pattern_xz);
    DEMSim.SetFamilyPrescribedLinVel(11, "0", "0", "0");

    auto top_plane = DEMSim.AddWavefrontMeshObject("../data/granularFlow/cylinder.obj", mat_type_walls);
    top_plane->SetInitPos(make_float3(0, 0, 05.50));
    top_plane->SetMass(1.);
    top_plane->Scale(make_float3(5, 5, 0.1));
    top_plane->SetFamily(20);

    // auto funnel = DEMSim.AddWavefrontMeshObject(GetDEMEDataFile("mesh/funnel.obj"), mat_type_walls);
    // funnel->Scale(0.007);
    // move = make_float3(0.00, 0.00, 2.35);  // z
    // ang = PI;
    // rot = make_float4(cos(ang / 2), 0, 0, sin(ang / 2));

    // funnel->Move(move, rot);
    // funnel->SetFamily(10);

    DEMSim.SetFamilyPrescribedLinVel(20, "0", "0", to_string_with_precision(0));
    DEMSim.SetFamilyPrescribedLinVel(21, "0", "0", to_string_with_precision(-radius * 2 / 1.0));
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
    out_dir += "/Filling_Geo_FOWT_wavestar6";
    remove_all(out_dir);
    create_directory(out_dir);

    unsigned int currframe = 0;
    unsigned int curr_step = 0;
    float settle_frame_time = 0.004;

    unsigned int actualTotalSpheres = 0;

    // Some inspectors
    auto max_z_finder = DEMSim.CreateInspector("clump_max_z");

    // DEMSim.SetFamilyExtraMargin(1, 0.0 * radius);

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
    float shift_xyz = 1.02 * (radius) * 2.0;

    std::vector<std::shared_ptr<DEMClumpTemplate>> input_pile_template_type;
    std::vector<float3> input_pile_xyz;
    HCPSampler sampler(shift_xyz);

    float x = 0.0;
    float y = -0.0;

    float sizeZ = plane_bottom + 80.40;
    float sizeX = 2.00;
    float z = plane_bottom + shift_xyz + sizeZ / 2.0; 

    float3 center_xyz = make_float3(x, y, z);

    // std::cout << "level of particles position ... " << center_xyz.z << std::endl;

    auto heap_particles_xyz = sampler.SampleBox(center_xyz, make_float3(sizeX / 2, sizeX / 2, sizeZ / 2.0));
    unsigned int num_clumps = heap_particles_xyz.size();

    std::cout << "Total num of particles: " << heap_particles_xyz.size() << std::endl;
    for (unsigned int i = 0; i < heap_particles_xyz.size(); i++) {
        input_pile_template_type.push_back(clump_types.at(i % num_template));
    }

    input_pile_xyz.insert(input_pile_xyz.end(), heap_particles_xyz.begin(), heap_particles_xyz.end());

    auto the_pile = DEMSim.AddClumps(input_pile_template_type, input_pile_xyz);
    the_pile->SetVel(make_float3(-0.00, 0.0, -0.90));
    the_pile->SetFamily(1);

    DEMSim.UpdateClumps();

    std::cout << "Total num of particles: " << (int)DEMSim.GetNumClumps() << std::endl;
    actualTotalSpheres = (int)DEMSim.GetNumClumps();

    // std::cout << "Total runtime: " << timeTotal << "s; settling for: " << settle_frame_time << std::endl;
    // std::cout << "maxZ is: " << max_z_finder->GetValue() << std::endl;

    std::cout << "Initialization done with : " << actualTotalSpheres << "particles" << std::endl;

    float sim_end = 40.0;
    unsigned int fps = 10;
    float frame_time = 1.0 / fps;

    std::cout << "Output at " << fps << " FPS" << std::endl;
    std::cout << "time output " << frame_time << "" << std::endl;

    unsigned int frame_count = 0;
    unsigned int step_count = 0;

    frame = 0;

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
        std::cout << "Outputting frame: " << frame_count << std::endl;
        sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), frame_count);
        sprintf(meshname, "%s/DEMdemo_mesh_%04d.vtk", out_dir.c_str(), frame_count);
        DEMSim.WriteSphereFile(std::string(filename));
        DEMSim.WriteMeshFile(std::string(meshname));
        frame_count++;
        DEMSim.ShowThreadCollaborationStats();
        // std::cout << "Initial number of contacts: " << DEMSim.GetNumContacts() << std::endl;
        if (totalRunTime > 8 && status_1) {
            DEMSim.DoDynamicsThenSync(0);
            std::cout << "gate is in motion from: " << totalRunTime << " s" << std::endl;
            std::cout << "and it will stop in : " << totalRunTime << " s" << std::endl;

            DEMSim.ChangeFamily(20, 21);
            DEMSim.EnableContactBetweenFamilies(21, 1);
            status_1 = false;
        }

        if (totalRunTime > 5 && status_2) {
            DEMSim.DoDynamicsThenSync(0);
            std::cout << "gate is in motion from: " << totalRunTime << " s" << std::endl;
            std::cout << "and it will stop in : " << totalRunTime << " s" << std::endl;
            // DEMSim.ChangeFamily(21, 20);
            DEMSim.ChangeFamily(10, 11);

            status_2 = false;
        }

        DEMSim.DoDynamics(frame_time);
    }

    DEMSim.ShowTimingStats();
    std::cout << "Fracture demo exiting..." << std::endl;
    return 0;
}
