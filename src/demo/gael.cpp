//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#include <core/ApiVersion.h>
#include <core/utils/ThreadManager.h>
#include <DEM/API.h>
#include <DEM/HostSideHelpers.hpp>
#include <DEM/utils/Samplers.hpp>

#include <cstdio>
#include <chrono>
#include <filesystem>
#include <map>
#include <random>
#include <cmath>

// =============================================================================
// In GRCPrep demo series, we try to prepare a sample of the GRC simulant, which
// are supposed to be used for extraterrestrial rover mobility simulations. It is
// made of particles of various sizes and shapes following a certain distribution.
// In Part1, it creates several batches of clumps and let them settle at the bottom
// of the domain.
// =============================================================================

using namespace deme;
using namespace std::filesystem;
// =============================================================================
// PARAMETER DEFINITION
// =============================================================================

double E_terrain = 1e8, nu_terrain = 0.3, CoR_terrain = 0.3, mu_terrain = 0.1, Crr_terrain = 0.0;
int srandValue = 759;
// double world_x_size = 1.10, world_y_size = 0.30, world_z_size = 1.00;//chrono simulation dimensions
double world_x_size = 1.11, world_y_size = 0.31, world_z_size = 2.0;  // chrono simulation dimensions
float sampleheight = 0.06;                                            // for the generation of the random material
double bottom = -sampleheight;
float size_z_batch = 3 * sampleheight;
// location of bottom boundary plane
// Define the terrain particle templates
// Default values for GPR1 example
// Calculate its mass and MOI
float terrain_density = 2.320e3;  // 2.6e3;
float targetMass = 0.60 * world_x_size * world_y_size * sampleheight * terrain_density;

// Coarse sand
// Define radius
// std::vector<double> radius = {0.004,0.0025,0.002,0.0016,0.001,0.00071,0.0005};
//  Instatiate particles with a probability that is in line with their weight distribution.
// std::vector<double> weight_perc = {0.01732, 0.03013, 0.05466, 0.16078, 0.14146, 0.15697, 0.43868};
// Fine gravel
// std::vector<double> radius      = {0.008*0.5,0.0063*0.5,0.0056*0.5,0.00475*0.5,0.004*0.5,0.002*0.5,0.001*0.5};
// std::vector<double> weight_perc = {0.0042, 0.0423, 0.13191, 0.20089, 0.54679, 0.04379, 0.02889};
std::vector<double> radius = {0.008 * 0.5, 0.0063 * 0.5, 0.0056 * 0.5, 0.00475 * 0.5, 0.004 * 0.5, 0.002 * 0.5};
std::vector<double> weight_perc = {0.0042, 0.0423, 0.13191, 0.20089, 0.54679, 0.04379 + 0.02889};
std::vector<int> particlePerClump = {2, 2, 2, 2, 2, 1, 1};

// Make ready for simulation
float step_size = 2e-6;
float time_end = 10.0;
float paddingFactor = 3.0;  // distance between particles
float toleranceFactor = 0.98;
float settletoleranceFactor = 0.02;  // m
float gravity = -9.81;

int main() {
    DEMSolver DEMSim;
    DEMSim.SetVerbosity(INFO);
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    // DEMSim.SetOutputContent(OUTPUT_CONTENT::FAMILY);
    DEMSim.SetOutputContent(OUTPUT_CONTENT::XYZ);
    DEMSim.SetOutputContent(OUTPUT_CONTENT::VEL);
    srand(srandValue);

    // Define materials
    auto mat_type_terrain = DEMSim.LoadMaterial(
        {{"E", E_terrain}, {"nu", nu_terrain}, {"CoR", CoR_terrain}, {"mu", mu_terrain}, {"Crr", Crr_terrain}});

    // Define the simulation world
    DEMSim.InstructBoxDomainDimension({-world_x_size / 1.5, world_x_size / 1.5},
                                      {-world_y_size / 1.5, world_y_size / 1.5}, {2.0 * bottom, world_z_size});
    // Add 5 bounding planes around the simulation world, and leave the top open
    // DEMSim.InstructBoxDomainBoundingBC("top_open", mat_type_terrain);
    // DEMSim.AddBCPlane(make_float3(0, 0, bottom), make_float3(0, 0, 1), mat_type_terrain);

    auto top_plane = DEMSim.AddWavefrontMeshObject("../data/mesh/box.obj", mat_type_terrain);
    top_plane->SetInitPos(make_float3(0, 0, bottom - 0.010));
    top_plane->SetMass(1.);
    top_plane->Scale(make_float3(world_x_size / 2.0, world_y_size / 2.0, 2.00));
    top_plane->SetFamily(10);
    DEMSim.SetFamilyFixed(10);

    std::string shake_pattern_xz = " 0.002 * sin( 50 * 2 * deme::PI * t)";
    std::string shake_pattern_y = " 0.002 * sin( 200 * 2 * deme::PI * t)";
    DEMSim.SetFamilyPrescribedLinVel(9, shake_pattern_xz, shake_pattern_xz, shake_pattern_y);
    // Scale the template we just created
    // Then load it to system
    // 11 types of spheres, diameter from 0.25cm to 0.35cm
    std::vector<std::shared_ptr<DEMClumpTemplate>> ground_particle_templates;

    for (int i = 0; i < radius.size(); i++) {
        std::vector<float> radii;
        std::vector<float3> relPos;
        std::vector<std::shared_ptr<DEMMaterial>> mat;

        double radiusMax = radius[i];
        double radiusMin = 8.0 / 8.0 * radiusMax;
        double eccentricity = 3.0 / 8.0 * radiusMax;
        /// particle one of the two-particle clump
        float3 tmp = make_float3(0, 0, 0);
        double a = 0, b = 0, c = 0;

        if (particlePerClump[i] == int(1)) {
            relPos.push_back(tmp);
            mat.push_back(mat_type_terrain);
            radii.push_back(radiusMax);

            a = radiusMax;
            b = radiusMax;
            c = radiusMax;
            std::cout << "A clump with only one particle" << std::endl;
        } else if (particlePerClump[i] == int(2)) {
            tmp.x = -1 * eccentricity / 2.0;
            tmp.y = 0;
            tmp.z = 0;
            relPos.push_back(tmp);
            mat.push_back(mat_type_terrain);
            radii.push_back(radiusMin);

            double x = eccentricity / 2.0;
            double y = 0;
            double z = 0;
            tmp.x = x;
            tmp.y = y;
            tmp.z = z;
            relPos.push_back(tmp);
            mat.push_back(mat_type_terrain);
            radii.push_back(radiusMin);

            c = radiusMin;  // smaller dim of the ellipse
            b = radiusMin;
            a = radiusMax;
            std::cout << "A clump with two particles" << std::endl;
        }

        float mass = 4.0 / 3.0 * deme::PI * a * b * c * terrain_density;
        float3 MOI = make_float3(1.f / 5.f * mass * (b * b + c * c), 1.f / 5.f * mass * (a * a + c * c),
                                 1.f / 5.f * mass * (b * b + a * a));

        auto clump_ptr = DEMSim.LoadClumpType(mass, MOI, radii, relPos, mat_type_terrain);
        // clump_ptr->AssignName("fsfs");
        ground_particle_templates.push_back(clump_ptr);
    }

    // for (int ii = 0; ii < radius.size(); ii++)
    // {
    //     float sphere_mass = radius[ii] * radius[ii] * radius[ii] * terrain_density * 4 / 3 * PI;
    //     ground_particle_templates.push_back(DEMSim.LoadSphereType(sphere_mass,radius[ii], mat_type_terrain));
    // }

    // Assign names to the templates to be able to load them (also scale can be applied here see)
    // https://github.com/projectchrono/DEM-Engine/blob/main/src/demo/DEMdemo_GRCPrep_Part1.cpp

    for (int ii = 0; ii < radius.size(); ii++) {
        std::shared_ptr<DEMClumpTemplate>& my_template = ground_particle_templates.at(ii);
        // Note the mass and MOI are also scaled in the process, automatically. But if you are not happy with this, you
        // can always manually change mass and MOI afterwards.
        // my_template->Scale(scales.at(i));
        // Give these templates names, 0000, 0001 etc.
        char t_name[20];
        sprintf(t_name, "%04d", ii);
        my_template->AssignName(std::string(t_name));
    }

    std::vector<double> grain_perc;
    for (int i = 0; i < radius.size(); i++) {
        grain_perc.push_back(weight_perc.at(i) / std::pow(radius.at(i), 3));
    }
    {
        double tmp = vector_sum(grain_perc);
        std::for_each(grain_perc.begin(), grain_perc.end(), [tmp](double& p) { p /= tmp; });
        std::cout << "Percentage of grains add up to " << vector_sum(grain_perc) << std::endl;
    }
    std::random_device r;
    std::default_random_engine e1(r());
    // Distribution that defines different weights (17, 10, etc.) for numbers.
    std::discrete_distribution<int> discrete_dist(grain_perc.begin(), grain_perc.end());

    // Sampler to use
    HCPSampler sampler(radius.at(0) * paddingFactor);

    // Make ready for simulation
    DEMSim.SetInitTimeStep(step_size);
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, gravity));
    // Max velocity info is generally just for the solver's reference and the user do not have to set it. The solver
    // wouldn't take into account a vel larger than this when doing async-ed contact detection: but this vel won't
    // happen anyway and if it does, something already went wrong.
    DEMSim.SetMaxVelocity(15.);  // Normally is set to 15
    // Error out vel is used to force the simulation to abort when something goes wrong.
    DEMSim.SetErrorOutVelocity(50.);
    // DEMSim.SetExpandSafetyMultiplier(1.2);
    DEMSim.SetInitBinNumTarget(1e7);
    DEMSim.Initialize();

    path out_dir = current_path();
    out_dir += "/Output";
    create_directory(out_dir);
    unsigned int currframe = 0;
    unsigned int curr_step = 0;

    float sample_halfheight = size_z_batch / 2;
    float sample_halfwidth_x = (world_x_size * toleranceFactor) / 2;
    float sample_halfwidth_y = (world_y_size * toleranceFactor) / 2;
    float offset_z = size_z_batch / 2 + settletoleranceFactor;
    float settle_frame_time = 0.02;
    float settle_batch_time = 0.3;

    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    // 0.25e6
    // float particle_number = 2784051;//fine sand(?)
    // float particle_number = 120000;

    auto max_z_finder = DEMSim.CreateInspector("clump_max_z");
    auto max_vel_finder = DEMSim.CreateInspector("clump_max_absv");
    auto totalMass = DEMSim.CreateInspector("clump_mass");
    float height = 0;
    float current_z = bottom;
    float mass;

    // while (DEMSim.GetNumClumps() < particle_number) {
    bool consolidationEnds = false;

    while (!consolidationEnds) {
        if (mass < targetMass && current_z < (world_z_size - 2 * offset_z)) {
            // DEMSim.ClearCache(); // Clearing cache is no longer needed
            float3 sample_center = make_float3(0, 0, offset_z + current_z);
            std::vector<std::shared_ptr<DEMClumpTemplate>> heap_template_in_use;
            std::vector<unsigned int> heap_family;
            // Sample and add heap particles
            auto heap_particles_xyz = sampler.SampleBox(
                sample_center, make_float3(sample_halfwidth_x, sample_halfwidth_y, sample_halfheight));
            for (unsigned int i = 0; i < heap_particles_xyz.size(); i++) {
                int ind = std::round(discrete_dist(e1));
                heap_template_in_use.push_back(ground_particle_templates.at(ind));
                heap_family.push_back(ind);
            }
            auto heap_particles = DEMSim.AddClumps(heap_template_in_use, heap_particles_xyz);
            // Give ground particles a small initial velocity so they `collapse' at the start of the simulation
            heap_particles->SetVel(make_float3(0.00, 0, -0.05));
            heap_particles->SetFamilies(heap_family);
            DEMSim.UpdateClumps();
            std::cout << "Current number of clumps: " << DEMSim.GetNumClumps() << std::endl;
        }
        // Allow for some settling
        // Must DoDynamicsThenSync (not DoDynamics), as adding entities to the simulation is only allowed at a
        // sync-ed point of time.
        float maxvel = 0;

        std::cout << "Frame: " << currframe << std::endl;
        char filename[200], meshfilename[200], cnt_filename[200];
        sprintf(filename, "%s/consolidation_output_%04d.csv", out_dir.c_str(), currframe);
        sprintf(meshfilename, "%s/DEMdemo_mesh_%04d.vtk", out_dir.c_str(), currframe);

        DEMSim.WriteSphereFile(std::string(filename));
        DEMSim.WriteMeshFile(std::string(meshfilename));
        DEMSim.DoDynamicsThenSync(settle_frame_time);
        current_z = max_z_finder->GetValue();
        maxvel = max_vel_finder->GetValue();
        mass = totalMass->GetValue();

        if (maxvel < 0.15 && mass > targetMass) {
            consolidationEnds = true;
            DEMSim.ChangeFamily(10, 9);
            std::cout << "Consolidating for one second" << std::endl;
            auto max_z_finder = DEMSim.CreateInspector("clump_max_z");
            auto min_z_finder = DEMSim.CreateInspector("clump_min_z");
            for (float t = 0; t < 1.00; t += 0.025) {
                std::cout << "Frame: " << currframe << std::endl;
                char filename[200], meshfilename[200], cnt_filename[200];
                sprintf(filename, "%s/consolidation_output_%04d.csv", out_dir.c_str(), currframe);
                sprintf(meshfilename, "%s/DEMdemo_mesh%04d.csv.vtk", out_dir.c_str(), currframe);

                DEMSim.WriteSphereFile(std::string(filename));
                DEMSim.WriteMeshFile(std::string(meshfilename));
                currframe++;
                float terrain_max_z = max_z_finder->GetValue();
                float terrain_min_z = min_z_finder->GetValue();
                std::cout << "Consolidation: " << terrain_max_z - terrain_min_z << std::endl;
                DEMSim.DoDynamics(0.025);
            }
            break;
        }

        currframe++;
        std::cout << "Current z: " << current_z << " with max vel: " << maxvel << std::endl;
        std::cout << "Mass so far is: " << mass << " targeting: " << targetMass << std::endl;

        // DEMSim.ShowThreadCollaborationStats();

        // Create a file after sucessfully settling each batch in case the simulation crashes and needs to restart
        // char cp3_filename[200];
        // char cnt3_filename[200];
        // sprintf(cp3_filename, "%s/clump_at%1.3f.csv", out_dir.c_str(), height);
        // DEMSim.WriteClumpFile(std::string(cp3_filename));
        // sprintf(cnt3_filename, "%s/contact_at%1.3f.csv", out_dir.c_str(), height);
        // DEMSim.WriteContactFile(std::string(cnt3_filename));
    }

    std::cout << "End of particle creation" << std::endl;

    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_sec = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << time_sec.count() << " seconds (wall time) to finish the simulation" << std::endl;
    std::string tagFile = "cons" + std::to_string(world_x_size);

    char cp_filename[200];
    sprintf(cp_filename, "%s/GRC_%s_Final.csv", out_dir.c_str(), tagFile.c_str());
    DEMSim.WriteClumpFile(std::string(cp_filename));

    char cnt_filename[200];
    sprintf(cnt_filename, "%s/Contact_pairs_%s_Final.csv", out_dir.c_str(), tagFile.c_str());
    DEMSim.WriteContactFile(std::string(cnt_filename));

    DEMSim.ShowThreadCollaborationStats();
    DEMSim.ClearThreadCollaborationStats();

    std::cout << "DEMdemo exiting..." << std::endl;
    return 0;
}