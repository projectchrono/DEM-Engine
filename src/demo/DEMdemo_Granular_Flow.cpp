//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// A repose angle test. Particles flow through a mesh-represented funnel and form
// a pile that has an apparent angle.
// =============================================================================

#include <core/ApiVersion.h>
#include <core/utils/ThreadManager.h>
#include <DEM/API.h>
#include <DEM/HostSideHelpers.hpp>
#include <DEM/utils/Samplers.hpp>

#include <cstdio>
#include <chrono>
#include <filesystem>
#include <random>

using namespace deme;
using namespace std::filesystem;

int main() {
    DEMSolver DEMSim;
    DEMSim.UseFrictionalHertzianModel();
    DEMSim.SetVerbosity(INFO);
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    DEMSim.EnsureKernelErrMsgLineNum();

    srand(7001);
    DEMSim.SetCollectAccRightAfterForceCalc(true);
    DEMSim.SetErrorOutAvgContacts(31);

    //DEMSim.SetExpandSafetyAdder(0.5);

    // Scale factor
    float scaling = 1.f;

    // total number of random clump templates to generate

    double radius = 0.003300 * scaling /2.0 ;
    double density = 1410;

    int nSphere = 10000;

    float min_rad = 0.97 * radius;
    float max_rad = 0.97 * radius;

    float max_relpos = 0.05 * radius;
    float min_relpos = 0.02 * radius;

    int num_template = 10000;

    float plane_bottom = 0.01f * scaling;
    float funnel_bottom = 0.02f * scaling;
    float funnel_outlet = 0.080f * scaling;
    float funnel_slope = 1.0 / 8.0 * 3.14;

    double tilt = 3.141592 / 6.0; 

    double gateOpen = 0.120;
    double gateSpeed = 2.0;

    auto mat_type_bottom = DEMSim.LoadMaterial({{"E", 50e9}, {"nu", 0.3}, {"CoR", 0.60}});
    auto mat_type_flume = DEMSim.LoadMaterial({{"E", 50e9}, {"nu", 0.3}, {"CoR", 0.60}});
    auto mat_type_walls = DEMSim.LoadMaterial({{"E", 10e9}, {"nu", 0.3}, {"CoR", 0.90}, {"mu", 0.04}, {"Crr", 0.04}});
    auto mat_type_particles =
        DEMSim.LoadMaterial({{"E", 2.7e9}, {"nu", 0.35}, {"CoR", 0.83}, {"mu", 0.40}, {"Crr", 0.010}});
    DEMSim.SetMaterialPropertyPair("CoR", mat_type_walls, mat_type_particles, 0.5);
    DEMSim.SetMaterialPropertyPair("Crr", mat_type_walls, mat_type_particles, 0.02);

    DEMSim.SetMaterialPropertyPair("CoR", mat_type_flume, mat_type_particles, 0.7);    //it is supposed to be
    DEMSim.SetMaterialPropertyPair("Crr", mat_type_flume, mat_type_particles, 0.001);   // plexiglass
    DEMSim.SetMaterialPropertyPair("mu", mat_type_flume, mat_type_particles, 0.15);

    DEMSim.SetMaterialPropertyPair("CoR", mat_type_bottom, mat_type_particles, 0.7);    //it is supposed to be
    DEMSim.SetMaterialPropertyPair("Crr", mat_type_bottom, mat_type_particles, 0.000);   // bakelite
    DEMSim.SetMaterialPropertyPair("mu", mat_type_bottom, mat_type_particles, 0.05);
    

    /*
    // First create clump type 0 for representing the ground
    float ground_sp_r = 0.02;
    auto template_ground = DEMSim.LoadSphereType(0.5, ground_sp_r, mat_type_walls);
    */
    // Make ready for simulation
    float step_size =1.0e-6;
    DEMSim.InstructBoxDomainDimension({-0.20, 2.50}, {-0.12, 0.12}, {-0.20, 1.55});
    DEMSim.InstructBoxDomainBoundingBC("top_open", mat_type_walls);
    DEMSim.SetInitTimeStep(step_size);
    DEMSim.SetGravitationalAcceleration(make_float3(9.81 * std::sin(tilt), 0, -9.81 * std::cos(tilt)));
    // Max velocity info is generally just for the solver's reference and the user do not have to set it. The solver
    // wouldn't take into account a vel larger than this when doing async-ed contact detection: but this vel won't
    // happen anyway and if it does, something already went wrong.
    DEMSim.SetMaxVelocity(25.);
    DEMSim.SetInitBinSize(min_rad * 6);

    // Loaded meshes are by-default fixed
   auto flume = DEMSim.AddWavefrontMeshObject("../data/granularFlow/flume.obj", mat_type_flume);    
   auto gateFixed = DEMSim.AddWavefrontMeshObject("../data/granularFlow/gateFixed.obj", mat_type_flume); 
   auto hopper = DEMSim.AddWavefrontMeshObject("../data/granularFlow/hopper.obj", mat_type_flume);  
   auto bottom = DEMSim.AddWavefrontMeshObject("../data/granularFlow/bottom.obj", mat_type_bottom);
    


    auto gate = DEMSim.AddWavefrontMeshObject("../data/granularFlow/gate.obj", mat_type_flume);

    flume->SetFamily(10);
    gateFixed->SetFamily(10);
    hopper->SetFamily(10);
    bottom->SetFamily(10);
    gate->SetFamily(3);

    std::string shake_pattern_xz = " 0.06 * sin( 300 * 2 * deme::PI * t)";
    std::string shake_pattern_y = " 0.06 * sin( 30 * 2 * deme::PI * t)";

    DEMSim.SetFamilyFixed(1);
    DEMSim.SetFamilyFixed(3);
    DEMSim.SetFamilyPrescribedLinVel(4, "0", "0", to_string_with_precision(gateSpeed));  
    DEMSim.SetFamilyPrescribedLinVel(10, shake_pattern_xz, shake_pattern_y, shake_pattern_xz);

    auto max_z_finder = DEMSim.CreateInspector("clump_max_z");
    auto min_z_finder = DEMSim.CreateInspector("clump_min_z");
    auto total_mass_finder = DEMSim.CreateInspector("clump_mass");
    auto max_v_finder = DEMSim.CreateInspector("clump_max_absv");

    // Make an array to store these generated clump templates
    std::vector<std::shared_ptr<DEMClumpTemplate>> clump_types;
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(radius,radius*0.05);
    double maxRadius= 0;

    for (int i = 0; i < num_template; i++) {

        std::vector<float> radii;
        std::vector<float3> relPos;
        std::vector<std::shared_ptr<DEMMaterial>> mat;

        double radiusMax = distribution(generator);
        double radiusMin = 3.0 / 4.0 * radiusMax;
        double eccentricity = 1.0 / 4.0 * radiusMax;

        radii.push_back(radiusMin);
        float3 tmp;
        tmp.x = 0;
        tmp.y = 0;
        tmp.z = 0;
        relPos.push_back(tmp);
        mat.push_back(mat_type_particles);

        double x = eccentricity;
        double y = 0;
        double z = 0;
        tmp.x = x;
        tmp.y = y;
        tmp.z = z;
        relPos.push_back(tmp);
        mat.push_back(mat_type_particles);

        radii.push_back(radiusMin);

        double c = radiusMin;  // smaller dim of the ellipse
        double b = radiusMin;
        double a = radiusMax;

        float mass = 4.0 / 3.0 * 3.141592 * a * b * c * density;
        float3 MOI = make_float3(1.f / 5.f * mass * (b * b + c * c), 1.f / 5.f * mass * (a * a + c * c),
                                 1.f / 5.f * mass * (b * b + a * a));
        std::cout << x << " chosen moi ..." << a / radius << std::endl;
        // LoadClumpType returns a shared_ptr that points to this template so you may modify it. Also, material can be
        // vector or a material shared ptr, and in the latter case it will just be applied to all component spheres this
        // clump has.
        maxRadius = (radiusMax > maxRadius) ? radiusMax : maxRadius;
        auto clump_ptr = DEMSim.LoadClumpType(mass, MOI, radii, relPos, mat_type_particles);
        // clump_ptr->AssignName("fsfs");
        clump_types.push_back(clump_ptr);
    }

    unsigned int currframe = 0;
    unsigned int curr_step = 0;
    double settle_frame_time = 0.01;
    // Track the projectile
    path out_dir = current_path();
    out_dir += "/DemoOutput_Flume";
    create_directory(out_dir);
    char filename[200], meshfile[200];

    float shift_xyz = 1.0* (maxRadius) * 2.0;
    float x = 0;
    float y = 0;
    float yW = funnel_outlet;
    float z = shift_xyz/2;  // by default we create beads at 0
    double emitterZ= 0.80;
    unsigned int totalSpheres =0;
    
    DEMSim.Initialize();
    
    int frame=0;
    bool generate =true;
    bool initialization=true;
    double timeTotal=0;
    double consolidation=true;


    sprintf(meshfile, "%s/DEMdemo_funnel_%04d.vtk", out_dir.c_str(), frame);
    DEMSim.WriteMeshFile(std::string(meshfile));

    while (initialization) {
        DEMSim.ClearCache();
        std::vector<std::shared_ptr<DEMClumpTemplate>> input_pile_template_type;
        std::vector<float3> input_pile_xyz;
        PDSampler sampler(shift_xyz);
           
        bool generate =(plane_bottom+0.20/2>emitterZ)? false:true;

        if (generate){   
                 
        
        float sizeZ=(frame==0)? 0.60 : 0.10;
        float sizeX=(frame==0)? 0.49 : 0.40;        
        float z= plane_bottom+shift_xyz+sizeZ/2.0;
        yW=(z>0.50)? funnel_outlet+(z - 0.50)*std::tan(tilt)/3:funnel_outlet; 

        float3 center_xyz = make_float3(0.01+sizeX/2, 0, z);
        float3 size_xyz = make_float3((sizeX - shift_xyz) / 2.0, (yW - shift_xyz) / 2.0, sizeZ/2.0);

        std::cout << "level of particles position ... " << center_xyz.z << std::endl;
        
        auto heap_particles_xyz = sampler.SampleBox(center_xyz, size_xyz);
        unsigned int num_clumps = heap_particles_xyz.size();
        std::cout << "number of particles at this level ... " << num_clumps << std::endl;

        for (unsigned int i = 0; i < num_clumps; i++) {
            input_pile_template_type.push_back(clump_types.at(i % num_template));
        }

        input_pile_xyz.insert(input_pile_xyz.end(), heap_particles_xyz.begin(), heap_particles_xyz.end());
        
   
    
    auto the_pile = DEMSim.AddClumps(input_pile_template_type, input_pile_xyz);
    the_pile->SetVel(make_float3(-0.30, 0, -0.80));
    the_pile->SetFamily(100);
    
    DEMSim.UpdateClumps();

    std::cout << "Total num of particles: " << (int)DEMSim.GetNumClumps() << std::endl;
    totalSpheres=(int)DEMSim.GetNumClumps();
    // Generate initial clumps for piling
        }
        timeTotal+=settle_frame_time;
    std::cout << "Total runtime: " << timeTotal << "s; settling for: " << settle_frame_time << std::endl;
    std::cout << "maxZ is: " << max_z_finder->GetValue() << std::endl;


    initialization= (totalSpheres < 200000)? true : false;


    if (generate) {
        
        std::cout << "frame : " << frame << std::endl;
        sprintf(filename, "%s/DEMdemo_settling_%04d.csv", out_dir.c_str(), frame);
        DEMSim.WriteSphereFile(std::string(filename));
        //DEMSim.ShowThreadCollaborationStats();
        frame--;
        }

    DEMSim.DoDynamicsThenSync(settle_frame_time);

    plane_bottom=max_z_finder->GetValue();

        if (!initialization){
            
            for(int i = 0; i < (int)(0.4/settle_frame_time); i++){
                DEMSim.DoDynamicsThenSync(settle_frame_time);
                sprintf(filename, "%s/DEMdemo_settling_%04d.csv", out_dir.c_str(), i);
                DEMSim.WriteSphereFile(std::string(filename));
                std::cout << "consolidating for "<<  i*settle_frame_time  <<  "s " << std::endl;
            }
        }

    }


    


    double k = 4.0 / 3.0 * 10e9 * std::pow(radius / 2.0, 0.5f);
    double m = 4.0 / 3.0 * 3.141592 * std::pow(radius, 3);
    double dt_crit = 0.64 * std::pow(m / k, 0.5f);

    std::cout << "dt critical is: " << dt_crit << std::endl;

    float timeStep = 1e-5;
    int numStep = 7.0 / timeStep;
    int timeOut = 0.01 / timeStep;
    int gateMotion = (gateOpen / gateSpeed) / timeStep;
    std::cout << "Frame: " << timeOut << std::endl;
    frame = 0;
    
    
    DEMSim.WriteMeshFile(std::string(meshfile));
    char cnt_filename[200];
    //sprintf(cnt_filename, "%s/Contact_pairs_1_.csv", out_dir.c_str());
    sprintf(meshfile, "%s/DEMdemo_funnel_%04d.vtk", out_dir.c_str(), frame);

    bool status = true;
    bool stopGate = true;
           
    for (int i = 0; i < numStep; i++) {

        DEMSim.DoDynamics(timeStep);

        if (!(i % timeOut) || i == 0) {
            sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), frame);            
            sprintf(meshfile, "%s/DEMdemo_mesh_%04d.vtk", out_dir.c_str(), frame);

            DEMSim.WriteMeshFile(std::string(meshfile));
            DEMSim.WriteSphereFile(std::string(filename));
            
            std::cout << "Frame: " << frame << std::endl;
            std::cout << "Elapsed time: " << timeStep * i << std::endl;
            // DEMSim.ShowThreadCollaborationStats();
            
            frame++;
        }

        if ((i > (timeOut * 2)) && status) {
            std::cout << "gate is in motion from: " << timeStep * i << " s" << std::endl;
            DEMSim.ChangeFamily(10, 1);
            DEMSim.ChangeFamily(3, 4);
            status = false;
        }

        if ((i >= (timeOut * (2) + gateMotion -1)) && stopGate) {
            std::cout << "gate has stopped at: " << timeStep * i << " s" << std::endl;
            DEMSim.ChangeFamily(4, 3);
            stopGate = false;
        }
    }

    DEMSim.ShowTimingStats();
    DEMSim.ClearTimingStats();

    std::cout << "DEMdemo_Repose exiting..." << std::endl;
    return 0;
}