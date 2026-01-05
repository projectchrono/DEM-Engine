// Streamfinish demo (both container AND workpiece are OBJ meshes)
// Rotierender Behälter (Z‑Achse) + stationäres Werkstück nach Einfahren
// Parameter per CLI: --rpm <double>, --ball_d_mm <double>, --container_R <m>, --fill_H <m>,
//                    --container_H <m>, --insert_depth <m>, --insert_time <s>,
//                    --workpiece <obj path>, --container_mesh <obj path>
// Hinweis: Kein analytischer Fallback – beide Geometrien werden als Mesh geladen.

#include <core/ApiVersion.h>
#include <core/utils/ThreadManager.h>
#include <DEM/API.h>
#include <DEM/HostSideHelpers.hpp>
#include <DEM/utils/Samplers.hpp>

#include <filesystem>
#include <cstdio>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

using namespace deme;

using std::string;

static double optAsDouble(int argc, char** argv, const string& key, double defval) {
    for (int i = 1; i < argc - 1; ++i) {
        if (string(argv[i]) == key) return std::stod(argv[i + 1]);
    }
    return defval;
}
static string optAsString(int argc, char** argv, const string& key, const string& defval) {
    for (int i = 1; i < argc - 1; ++i) {
        if (string(argv[i]) == key) return string(argv[i + 1]);
    }
    return defval;
}

int main(int argc, char** argv) {
    DEMSolver DEMSim;

    // ---------------- Params (CLI with sensible defaults) ----------------
    const float MESH_MM2M = 1.0f / 1000.0f;  // mm -> m

    const double rpm            = optAsDouble(argc, argv, "--rpm", 50.0);
    const double ball_d_mm      = optAsDouble(argc, argv, "--ball_d_mm", 12.8);
    const double container_R    = optAsDouble(argc, argv, "--container_R", 0.525);  // [m]
    const double container_H    = optAsDouble(argc, argv, "--container_H", 0.5);  // [m]
    const double fill_H         = optAsDouble(argc, argv, "--fill_H", 0.24);       // [m]
    const double insert_depth   = optAsDouble(argc, argv, "--insert_depth", 0.2);  // [m]
    const double insert_time    = optAsDouble(argc, argv, "--insert_time", 0.5);    // [s]
    const string workpiece_obj  = optAsString(argc, argv, "--workpiece", (GET_DATA_PATH() / "mesh/workpiece.obj").string());
    const string container_obj  = optAsString(argc, argv, "--container_mesh", (GET_DATA_PATH() / "mesh/container.obj").string());

    if (workpiece_obj.empty() || container_obj.empty()) {
        std::cerr << "Both --workpiece and --container_mesh must be provided (OBJ files)." << std::endl;
        return 2;
    }

    const double ball_r         = 0.5 * ball_d_mm * 1e-3; // mm -> m
    const double omega          = rpm * 2.0 * M_PI / 60.0; // rad/s
    const double v_ins          = insert_depth / insert_time; // m/s (downwards)
    const double frame_dt = optAsDouble(argc, argv, "--frame_dt", 0.01);         // [s] Zeit zwischen Frames
    const string out_dir  = optAsString(argc, argv, "--out_dir", "output_streamfinish");  // Ausgabeverzeichnis


    // ---------------- Materials ----------------
    auto mat_container = DEMSim.LoadMaterial({{"E", 1e8}, {"nu", 0.30}, {"CoR", 0.5}, {"mu", 0.50}, {"Crr", 0.0}});
    auto mat_media     = DEMSim.LoadMaterial({{"E", 1e8}, {"nu", 0.30}, {"CoR", 0.5}, {"mu", 0.20}, {"Crr", 0.0}});
    DEMSim.SetMaterialPropertyPair("mu", mat_container, mat_media, 0.50);

    // ---------------- World / domain ----------------
    const float m = 0.01f;                          // Puffer
    const float worldX = 2.f * float(container_R) + 2.f * m;   // ≈ 1.05 + 0.02
    const float worldY = 2.f * float(container_R) + 2.f * m;
    const float worldZ = float(container_H) + 10.f * m;         // 0.50 + 0.1 = 0.6
    DEMSim.InstructBoxDomainDimension(worldX, worldY, worldZ);
    DEMSim.InstructBoxDomainBoundingBC("all", mat_media);
    DEMSim.SetGravitationalAcceleration(make_float3(0.f, 0.f, -9.81f));
    DEMSim.SetInitTimeStep(5e-6);

    // Optional: add top/bottom planes to confine the media in Z (bleiben stationär)
    // auto bc = DEMSim.AddExternalObject();
    // bc->AddPlane(make_float3(0, 0, -0.5f * worldZ), make_float3(0, 0,  1), mat_container); // bottom, Normal nach +Z
    // bc->AddPlane(make_float3(0, 0,  0.5f * worldZ), make_float3(0, 0, -1), mat_container); // top,    Normal nach −Z


    // ---------------- Container (mesh, rotating about global Z) ----------------
    const unsigned FAM_CONTAINER = 10;
    auto container = DEMSim.AddWavefrontMeshObject(container_obj, mat_container);
    // Annahme: Container-OBJ ist um Ursprung herum modelliert (Z‑Achse entlang global Z),
    container->Scale(MESH_MM2M);
    // Positioniere so, dass Container „auf dem Boden“ steht
    container->SetInitPos(make_float3(0.f, 0.f, 0.f));
    container->SetFamily(FAM_CONTAINER);
    // Reine Rotation um Z, Translation fixieren (keine Drift)
    DEMSim.SetFamilyPrescribedLinVel(FAM_CONTAINER, "0", "0", "0");
    DEMSim.SetFamilyPrescribedAngVel(FAM_CONTAINER, "0", "0", std::to_string(omega));

    // ---------------- Media (spheres) ----------------
    const double rho_media = 2600.0; // kg/m^3 (Beispiel: Korund)
    const double mass = rho_media * (4.0 / 3.0) * M_PI * ball_r * ball_r * ball_r;
    auto sphere_type = DEMSim.LoadSphereType(float(mass), float(ball_r), mat_media);

    // Packen: HCP in Z-Zylinder (Füllhöhe)
    HCPSampler sampler(float(2.05f * ball_r));
    // Ziel: z ∈ [0, fill_H]  → Mittelpunkt bei z = 0.5*fill_H, halbe Höhe = 0.5*fill_H
    const float3 fill_center = make_float3(0.f, 0.f, 0.5f * float(fill_H));
    const float  r_fill      = float(container_R - 1.1f * ball_r);  // kleiner Rand zur Wand
    const float  h_half      = 0.5f * float(fill_H);
    auto xyz = sampler.SampleCylinderZ(fill_center, r_fill, h_half);
    std::vector<float3> locs;
    locs.reserve(xyz.size());
    for (const auto& p : xyz) {
        locs.push_back(p);
    }
    DEMSim.AddClumps(
        std::vector<std::shared_ptr<DEMClumpTemplate>>(locs.size(), sphere_type),
        locs
    );

    // ---------------- Workpiece (mesh: insert then hold) ----------------
    const unsigned FAM_WORKPIECE = 20;
    auto workpiece = DEMSim.AddWavefrontMeshObject(workpiece_obj, mat_container);
    workpiece->Scale(MESH_MM2M);
    const float z_surface = -0.5f * worldZ + float(fill_H);
    workpiece->SetInitPos(make_float3(0.425f, 0.f, float(insert_depth)));
    workpiece->SetFamily(FAM_WORKPIECE);

    // Zeitgesteuerte Einfahrt: v = -v_ins für t < t_ins, sonst 0
    const string pre_code =
        string("const float t_ins=") + std::to_string(insert_time) + ";" +
        string("const float v_ins=") + std::to_string(v_ins) + ";" +
        string("const float v = (t < t_ins) ? -v_ins : 0.0f;");
    DEMSim.SetFamilyPrescribedLinVel(FAM_WORKPIECE, "0", "0", "v", true, pre_code);


    // ---------------- Output & run ----------------
    DEMSim.SetVerbosity(STEP_METRIC);
    // Force the solver to error out if something went crazy. A good practice to add them, but not necessary.
    DEMSim.SetErrorOutVelocity(20.);
    DEMSim.SetErrorOutAvgContacts(50);

    DEMSim.Initialize();
    // --- Output-Konfiguration (Mixer-Stil) ---
    std::filesystem::create_directories(out_dir);

    // Formate wie im Mixer: Clumps als CSV, Mesh als VTK
    DEMSim.SetOutputFormat("CSV");
    DEMSim.SetOutputContent({"XYZ", "VEL"});     // ggf. erweitern: "VEL","ANG_VEL","FAMILY"
    DEMSim.SetMeshOutputFormat("VTK");

    // Frame-Schreiber (Clumps + Mesh)
    auto write_frame = [&](int f) {
        char path[512];

        std::snprintf(path, sizeof(path), "%s/clumps_%06d.csv", out_dir.c_str(), f);
        DEMSim.WriteClumpFile(std::string(path));   // <— HIER

        std::snprintf(path, sizeof(path), "%s/mesh_%06d.vtk", out_dir.c_str(), f);
        DEMSim.WriteMeshFile(std::string(path));    // <— UND HIER

        // Optional (groß!): Kontakte
        // std::snprintf(path, sizeof(path), "%s/contacts_%06d.csv", out_dir.c_str(), f);
        // DEMSim.WriteContactFile(path);
    };

    // Startzustand (Frame 0)
    int frame = 0;
    write_frame(frame++);
    // SIM-Driver
    const double t_end = insert_time + 2.0;
    const double dt_ts = DEMSim.GetTimeStepSize();   // tatsächliches ∆t aus der Engine
    double next_frame  = frame_dt;
    auto start = std::chrono::high_resolution_clock::now();

    const int steps_per_frame = std::max(1, int(std::round(frame_dt / dt_ts)));
    int step = 0;
    while (DEMSim.GetSimTime() < t_end) {
        DEMSim.DoStepDynamics();
        if ((++step) % steps_per_frame == 0) write_frame(frame++);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt = end - start;
    std::cout << dt.count() << " s wall time for streamfinish demo (all-mesh)" << std::endl;

    DEMSim.ShowTimingStats();
    DEMSim.ShowMemStats();
    return 0;
}
