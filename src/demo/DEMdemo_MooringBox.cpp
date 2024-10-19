//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// Mooring line case
// =============================================================================

#include <DEM/API.h>
#include <DEM/HostSideHelpers.hpp>
#include <DEM/utils/Samplers.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

class MotionSolver {
  public:
    // Constructor
    MotionSolver(double mass, double damping, double stiffness, double dt)
        : m(mass), c(damping), k(stiffness), dt(dt) {}

    MotionSolver(Eigen::Matrix3d mass, Eigen::Matrix3d damping, Eigen::Matrix3d stiffness, double dt)
        : M(mass), C(damping), K(stiffness), dt(dt) {}

    // first-order ODEs
    // dx/dt = v
    // dv/dt = (1/m) * (F(t) - c * v - k * x)
  private:
    Eigen::Matrix3d M;  // mass
    Eigen::Matrix3d C;  // damping coefficient
    Eigen::Matrix3d K;  // stiffness
    double m;
    double c;
    double k;
    double dt;  // time step
  public:
    void equation_of_motion(double t,
                            const std::vector<double>& state,
                            std::vector<double>& derivatives,
                            double force) {
        double x = state[0];
        double v = state[1];
        derivatives[0] = v;
        derivatives[1] = (1 / m) * (force - c * v - k * (x + 0.08));
    }

    // void equation_of_motion(double t,
    //                         const Eigen::Vector3d& statePos,
    //                         const Eigen::Vector3d& stateVel,
    //                         Eigen::Vector3d& derivatives,
    //                         Eigen::Vector3d& force) {
    //     double x = state[0];
    //     double v = state[1];
    //     derivatives[0] = v;
    //     derivatives[1] = (1 / m) * (force - c * v - k * (x + 0.08));
    // }
    // Runge-Kutta 4th order step
    void rk4_step(double t, std::vector<double>& state, double force) {
        std::vector<double> k1(2), k2(2), k3(2), k4(2), temp_state(2);

        // k1
        equation_of_motion(t, state, k1, force);

        // k2
        temp_state[0] = state[0] + 0.5 * dt * k1[0];
        temp_state[1] = state[1] + 0.5 * dt * k1[1];
        equation_of_motion(t + 0.5 * dt, temp_state, k2, force);

        // k3
        temp_state[0] = state[0] + 0.5 * dt * k2[0];
        temp_state[1] = state[1] + 0.5 * dt * k2[1];
        equation_of_motion(t + 0.5 * dt, temp_state, k3, force);

        // k4
        temp_state[0] = state[0] + dt * k3[0];
        temp_state[1] = state[1] + dt * k3[1];
        equation_of_motion(t + dt, temp_state, k4, force);

        // Update state with RK4 formula
        state[0] += (dt / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]);
        state[1] += (dt / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]);
    }

    // Solve the system using RK4 - it only advance one time step at a time
    void solve(std::vector<double>& state, double force) {
        // Initial state [x, v]

        double t = dt;

        rk4_step(t, state, force);
    }
};

// external force function
double harmonic_force(double t) {
    double amplitude = 4.0;
    double frequency = 1. / 1.8;
    return amplitude * sin(2 * M_PI * frequency * t);
}

double evaluate_force(double z0, double L, double k, double H, double omega, double t) {
    double rho = 1000;
    double g = 9.81;

    // double F1 = rho * g * z0 * L;

    double F = k * rho * g * H * exp(k * z0) * (sin(k * L - omega * t) + sin(omega * t));

    return F;
}

double TheoryLength(double T, double d);
double eta_s(double x, double t, double H, double L, double T);
double press2(double x, double t, double z, double H, double L, double T);
double verticalForce(double time, double H, double L, double T, double size, double draft);
double horizontalForce(double time,
                       double waveHeight,
                       double waveLength,
                       double wavePeriod,
                       double size,
                       double draft,
                       double postion_X);

using namespace deme;

int main() {
    DEMSolver DEMSim;
    DEMSim.SetVerbosity(INFO);
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    DEMSim.SetOutputContent(OUTPUT_CONTENT::VEL | FAMILY);
    DEMSim.SetMeshOutputFormat(MESH_FORMAT::VTK);
    DEMSim.SetContactOutputContent(DEME_POINT | OWNER | FORCE | CNT_WILDCARD);

    Eigen::Matrix3d m = Eigen::Matrix3d::Random();
    m = (m + Eigen::Matrix3d::Constant(1.2)) * 50;
    std::cout << "m =" << std::endl << m << std::endl;
    Eigen::Vector3d v(1, 2, 3);

    std::cout << "m * v =" << std::endl << m * v << std::endl;

    DEMSim.SetErrorOutAvgContacts(20);
    // DEMSim.SetForceCalcThreadsPerBlock(256);
    //  E, nu, CoR, mu, Crr...
    auto mat_type_container =
        DEMSim.LoadMaterial({{"E", 10e9}, {"nu", 0.3}, {"CoR", 0.1}, {"mu", 0.50}, {"Crr", 0.10}});
    auto mat_type_particle = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.20}, {"CoR", 0.1}, {"mu", 0.50}, {"Crr", 0.05}});
    // If you don't have this line, then values will take average between 2 materials, when they are in contact
    DEMSim.SetMaterialPropertyPair("CoR", mat_type_container, mat_type_particle, 0.20);
    DEMSim.SetMaterialPropertyPair("Crr", mat_type_container, mat_type_particle, 0.50);
    DEMSim.SetMaterialPropertyPair("mu", mat_type_container, mat_type_particle, 0.50);
    // We can specify the force model using a file.
    auto my_force_model = DEMSim.ReadContactForceModel("ForceModelMooringPosition.cu");

    // Those following lines are needed. We must let the solver know that those var names are history variable etc.
    my_force_model->SetMustHaveMatProp({"E", "nu", "CoR", "mu", "Crr"});
    my_force_model->SetMustPairwiseMatProp({"CoR", "mu", "Crr"});
    // Pay attention to the extra per-contact wildcard `unbroken' here.
    my_force_model->SetPerContactWildcards({"delta_time", "delta_tan_x", "delta_tan_y", "delta_tan_z",
                                            "innerInteraction", "initialLength", "restLength", "tension"});

    float world_size = 10;
    float container_diameter = 0.06;
    float terrain_density = 1.200e3 * 2;
    float sphere_rad = 0.003;

    float step_size = 1e-6;
    float fact_radius = 2.0;

    DEMSim.InstructBoxDomainDimension({-3, 3}, {-1, 1}, {-1.0, 1});
    // No need to add simulation `world' boundaries, b/c we'll add a cylinderical container manually
    DEMSim.InstructBoxDomainBoundingBC("all", mat_type_container);
    // DEMSim.SetInitBinSize(sphere_rad * 5);
    //  Now add a cylinderical boundary along with a bottom plane
    double bottom = -0;
    double top = 0.10;

    // Creating the two clump templates we need, which are just spheres
    std::shared_ptr<DEMClumpTemplate> templates_terrain;

    templates_terrain =
        DEMSim.LoadSphereType(sphere_rad * sphere_rad * sphere_rad * 4 / 3 * 1.0e3 * PI, sphere_rad, mat_type_particle);

    auto data_xyz = DEMSim.ReadClumpXyzFromCsv("../data/my/CatenaryBody.csv");
    std::vector<float3> input_xyz;

    std::vector<std::shared_ptr<DEMClumpTemplate>> input_pile_template_type;
    std::cout << data_xyz.size() << " Data points are loaded from the external list." << std::endl;

    for (unsigned int i = 0; i < (data_xyz.size()); i++) {
        char t_name[20];
        sprintf(t_name, "%d", i);

        auto this_type_xyz = data_xyz[std::string(t_name)];
        input_xyz.insert(input_xyz.end(), this_type_xyz.begin(), this_type_xyz.end());

        input_pile_template_type.push_back(templates_terrain);
    }
    auto allParticles = DEMSim.AddClumps(input_pile_template_type, input_xyz);
    allParticles->SetFamily(1);

    auto data_xyz_anchor = DEMSim.ReadClumpXyzFromCsv("../data/my/CatenaryAnchors.csv");
    std::vector<float3> input_xyz_2;

    std::vector<std::shared_ptr<DEMClumpTemplate>> input_pile_template_type_2;
    std::cout << data_xyz_anchor.size() << " Data points are loaded from the external list." << std::endl;

    for (unsigned int i = 0; i < (data_xyz_anchor.size()); i++) {
        char t_name[20];
        sprintf(t_name, "%d", i);

        auto this_type_xyz = data_xyz_anchor[std::string(t_name)];
        input_xyz_2.insert(input_xyz_2.end(), this_type_xyz.begin(), this_type_xyz.end());

        input_pile_template_type_2.push_back(templates_terrain);
    }

    auto allParticles_2 = DEMSim.AddClumps(input_pile_template_type_2, input_xyz_2);
    allParticles_2->SetFamily(2);
    DEMSim.SetFamilyFixed(2);

    float massFloater = 3.16;
    auto data_xyz_fairlead = DEMSim.ReadClumpXyzFromCsv("../data/my/CatenaryFairlead.csv");
    std::vector<std::shared_ptr<DEMClumpTemplate>> clump_cylinder;
    // Then load it to system
    {  // initialize cylinder clump
        std::vector<float3> relPos;
        std::vector<float> radii;
        std::vector<std::shared_ptr<DEMMaterial>> mat;
        for (int i = 0; i < data_xyz_fairlead.size(); i++) {
            char t_name[20];
            sprintf(t_name, "%d", i);

            auto tmp = data_xyz_fairlead[std::string(t_name)];
            relPos.insert(relPos.end(), tmp.begin(), tmp.end());

            mat.push_back(mat_type_particle);
            radii.push_back(sphere_rad);
        }

        float Ixx = 1.f / 2.f * massFloater;
        float Iyy = Ixx;
        float3 MOI = make_float3(Ixx, Iyy, Iyy);

        auto clump_ptr = DEMSim.LoadClumpType(massFloater, MOI, radii, relPos, mat_type_particle);
        // clump_ptr->AssignName("fsfs");
        clump_cylinder.push_back(clump_ptr);
    }
    std::cout << "Total num of clumps: " << clump_cylinder.size() << std::endl;
    std::vector<float3> input_pile_xyz;
    input_pile_xyz.insert(input_pile_xyz.end(), make_float3(0.0, 0, -0.0126));

    auto the_pile = DEMSim.AddClumps(clump_cylinder, input_pile_xyz);
    the_pile->SetFamily(3);
    DEMSim.SetFamilyFixed(3);
    auto anchoring_track = DEMSim.Track(the_pile);

    float zPos = 0.0;
    std::string buoyancy = to_string_with_precision(0.20 * 0.20 * 0.08 * 1000 * 9.81 / massFloater);
    std::string xMot = "";
    //  DEMSim.AddFamilyPrescribedAcc(3, "0", "0", buoyancy);

    std::cout << "Total num of particles: " << the_pile->GetNumClumps() << std::endl;
    std::cout << "Total num of spheres: " << the_pile->GetNumSpheres() << std::endl;

    auto top_plane = DEMSim.AddWavefrontMeshObject("../data/my/cube.obj", mat_type_container);
    top_plane->SetInitPos(make_float3(0, 0, 0.0));
    top_plane->SetMass(1.);
    top_plane->Scale(make_float3(0.2, 0.2, 0.132));
    top_plane->SetFamily(10);
    // DEMSim.SetFamilyFixed(10);
    auto phantom_track = DEMSim.Track(top_plane);

    auto bottom_plane = DEMSim.AddWavefrontMeshObject("../data/my/cylinder.obj", mat_type_container);
    bottom_plane->SetInitPos(make_float3(0, 0, -0.50 - sphere_rad - 0.001));
    bottom_plane->SetMass(1.);
    bottom_plane->Scale(make_float3(2, 1, 0.001));
    bottom_plane->SetFamily(20);
    DEMSim.SetFamilyFixed(20);

    std::cout << "Total num of particles: " << allParticles->GetNumClumps() << std::endl;

    std::filesystem::path out_dir = std::filesystem::current_path();
    std::string nameOutFolder = "R" + std::to_string(sphere_rad) + "_Int" + std::to_string(fact_radius) + "";
    out_dir += "/DemoOutput_MooringLine";
    remove_all(out_dir);
    create_directory(out_dir);

    // Some inspectors

    DEMSim.SetFamilyExtraMargin(1, fact_radius * sphere_rad);
    DEMSim.SetFamilyExtraMargin(2, fact_radius * sphere_rad);
    DEMSim.SetFamilyExtraMargin(3, fact_radius * sphere_rad);

    DEMSim.SetInitTimeStep(step_size);
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0.00, 1 * -9.81));
    DEMSim.Initialize();
    // DEMSim.DisableContactBetweenFamilies(20, 1);
    std::cout << "Initial number of contacts: " << DEMSim.GetNumContacts() << std::endl;

    float sim_end = 50;

    unsigned int fps = 100;
    float time_out = 0.05;
    unsigned int datafps = int(1.0 / time_out);
    unsigned int modfpsGeo = fps / datafps;
    float frame_time = 1.0 / fps;
    std::cout << "Output at " << fps << " FPS" << std::endl;
    unsigned int out_steps = (unsigned int)(1.0 / (datafps * step_size));
    unsigned int frame_count = 0;
    unsigned int step_count = 0;

    // DEMSim.DisableContactBetweenFamilies(10, 1);

    double L0 = 0.0;
    double stress = 0.0;
    std::string nameOutFile = "data_R" + std::to_string(sphere_rad) + "_Int" + std::to_string(fact_radius) + ".csv";
    std::ofstream csvFile(nameOutFile);

    DEMSim.SetFamilyContactWildcardValueBoth(1, "initialLength", 0.0);
    DEMSim.SetFamilyContactWildcardValueBoth(1, "innerInteraction", 0.0);
    DEMSim.SetFamilyContactWildcardValueBoth(2, "initialLength", 0.0);
    DEMSim.SetFamilyContactWildcardValueBoth(2, "innerInteraction", 0.0);
    DEMSim.SetFamilyContactWildcardValueBoth(3, "initialLength", 0.0);
    DEMSim.SetFamilyContactWildcardValueBoth(3, "innerInteraction", 0.0);
    std::cout << "Contacts now: " << DEMSim.GetNumContacts() << std::endl;
    DEMSim.DoDynamicsThenSync(0.0);
    DEMSim.SetFamilyContactWildcardValueBoth(1, "innerInteraction", 2.0);
    DEMSim.SetFamilyContactWildcardValue(1, 2, "innerInteraction", 2.0);
    DEMSim.SetFamilyContactWildcardValue(1, 3, "innerInteraction", 2.0);

    std::cout << "Contacts now: " << DEMSim.GetNumContacts() << std::endl;
    DEMSim.MarkFamilyPersistentContactEither(1);
    DEMSim.MarkFamilyPersistentContact(1, 2);
    DEMSim.MarkFamilyPersistentContact(1, 3);

    DEMSim.DoDynamicsThenSync(0.0);

    DEMSim.DisableContactBetweenFamilies(1, 10);
    DEMSim.DisableContactBetweenFamilies(3, 10);

    fact_radius=3.0;

    DEMSim.SetFamilyExtraMargin(1, fact_radius * sphere_rad);
    //DEMSim.SetFamilyExtraMargin(2, fact_radius * sphere_rad);
    DEMSim.SetFamilyExtraMargin(3, fact_radius * sphere_rad);

    std::cout << "Establishing inner forces: " << frame_count << std::endl;

    float3 position = anchoring_track->Pos();

    // Simulation loop

    float hydroStiffness = 1000.0 * 9.81 * 0.20 * 0.20;
    float damping = 0.05 * 2 * sqrt(hydroStiffness * massFloater);
    MotionSolver solverZ(massFloater * 1.30, damping, hydroStiffness, frame_time);
    MotionSolver solverX(1.40 * massFloater, damping, 0.0 * hydroStiffness, frame_time);
    std::vector<double> stateZ = {0.0, 0.0};
    std::vector<double> stateX = {0.0, 0.0};

    double waveLength = 3.57;

    for (float time = 0; time < sim_end; time += frame_time) {
        // DEMSim.ShowThreadCollaborationStats();

        auto temp = anchoring_track->ContactAcc();
        // double forceZ = -(0.20 * 0.20 * (-0.08 + stateZ[0]) * 1000 * 9.81) + temp.z * massFloater;
        //  force += evaluate_force(0.080 + state[0], 0.20, 2 * PI / 3.56, 0.12, 2 * PI / 1.80, time);
        double draft = -0.08 + stateZ[0];
        double forceZ = verticalForce(time, 0.12, waveLength, 1.60, 0.20, draft)+temp.z * massFloater;
        double eta = eta_s(0, time, 0.12, waveLength, 1.60);

        // forceZ += harmonic_force(time);
        double forceX = horizontalForce(time, 0.12, waveLength, 1.60, 0.20, draft, -0.10+stateX[0]);
        forceX -= horizontalForce(time, 0.12, waveLength, 1.60, 0.20, draft, 0.10+stateX[0]);
        forceX+=temp.x * massFloater;

        solverZ.solve(stateZ, forceZ);
        solverX.solve(stateX, forceX);
        // std::cout << "t = " << time << ", x = " << state[0] << ", v = " << state[1] << std::endl;
        std::cout << "forceX = " << float(forceX) << " N forceZ = " << float(forceZ) << " N free surface " << eta
                  << std::endl;

        anchoring_track->SetPos(position + make_float3(stateX[0], 0, stateZ[0]));
        anchoring_track->SetVel(make_float3(stateX[1], 0, stateZ[1]));
        float3 phantom_position = anchoring_track->Pos();
        float4 phantom_quat = anchoring_track->OriQ();
        phantom_track->SetPos(phantom_position);
        phantom_track->SetOriQ(phantom_quat);

        
        if (frame_count % modfpsGeo == 0 || frame_count == 0) {
            char filename[200];
            char meshname[200];
            char cnt_filename[200];
            // std::cout << "Contacts now: " << DEMSim.GetNumContacts() << std::endl;
            std::cout << "time: " << time << " s" << std::endl;
            std::cout << "Force: " << temp.x * massFloater << std::endl;
            sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), frame_count / modfpsGeo);
            sprintf(meshname, "%s/DEMdemo_mesh_%04d.vtk", out_dir.c_str(), frame_count / modfpsGeo);
            sprintf(cnt_filename, "%s/DEMdemo_contact_%04d.csv", out_dir.c_str(), frame_count / modfpsGeo);

            DEMSim.WriteSphereFile(std::string(filename));
            DEMSim.WriteMeshFile(std::string(meshname));
            DEMSim.WriteContactFile(std::string(cnt_filename));
        }
        frame_count++;
        DEMSim.DoDynamicsThenSync(frame_time);
    }
    csvFile.close();
    DEMSim.ShowTimingStats();
    std::cout << "Fracture demo exiting..." << std::endl;
    return 0;
}

double verticalForce(double time, double H, double L, double T, double size, double draft) {
    double dx = size / 20.0;
    double force = 0.0;
    for (double l = 0; l <= size; l += dx) {
        double eta = eta_s(l, time, H, L, T);
        double z = draft + eta;
        force += press2(l, time, z, H, L, T) * dx * size;
    }
    return force;
}

double horizontalForce(double time,
                       double waveHeight,
                       double waveLength,
                       double wavePeriod,
                       double size,
                       double draft,
                       double postion_X) {
    double eta = eta_s(postion_X, time, waveHeight, waveLength, wavePeriod);
    double z = draft + eta;
    double dz = abs(z) / 20.0;
    double force = 0.0;
    for (double l = draft; l <= eta; l += dz) {
        force += press2(postion_X, time, l, waveHeight, waveLength, wavePeriod) * dz * size;
    }
    return force;
}

// Function to calculate the wavelength using the dispersion relation
double TheoryLength(double T, double d) {
    double g = 9.81;
    return T * std::sqrt(g * d);
}

// Surface elevation function
double eta_s(double x, double t, double H, double L, double T) {
    const double d = 0.50;
    return H / 2.0 * std::cos(2.0 * M_PI * x / L - 2.0 * M_PI * t / T) +
           M_PI * H * H / 8.0 / L * std::cosh(2 * M_PI * d / L) / std::pow(std::sinh(2.0 * M_PI * d / L), 3) *
               (2.0 + std::cosh(4.0 * M_PI * d / L)) * std::cos(4.0 * M_PI * x / L - 4.0 * M_PI * t / T);
}

// Wave-induced pressure function
double press2(double x, double t, double z, double H, double L, double T) {
    const double rho = 1000;
    const double g = 9.81;
    const double d = 0.50;
    return rho * g * H / 2.0 * std::cosh(2 * M_PI * (z + d) / L) / std::cosh(2.0 * M_PI * d / L) *
               std::cos(2 * M_PI * x / L - 2 * M_PI * t / T) -
           rho * g * z +
           3.0 / 8.0 * rho * g * M_PI * H * H / L * std::tanh(2 * M_PI * d / L) /
               std::pow(std::sinh(2 * M_PI * d / L), 2) *
               (std::cosh(4 * M_PI * (z + d) / L) / std::pow(std::sinh(2 * M_PI * d / L), 2) - 1.0 / 3.0) *
               std::cos(4 * M_PI * x / L - 4 * M_PI * t / T) -
           1.0 / 8.0 * rho * g * M_PI * H * H / L * std::tanh(2 * M_PI * d / L) /
               std::pow(std::sinh(2 * M_PI * d / L), 2) * (std::cosh(4 * M_PI * (z + d) / L) - 1);
}