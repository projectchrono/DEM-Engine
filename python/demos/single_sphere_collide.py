"""Two colliding spheres over meshes, ported from DEMdemo_SingleSphereCollide.cpp."""

import argparse
import math
from pathlib import Path


def main():
    """Configure cohesion, insert a second sphere at runtime, and write synchronized frames."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("DemoOutput_SingleSphereCollide"))
    parser.add_argument("--smoke-test", action="store_true", help="Run 0.3 s, including the sphere collision")
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration must be finite and positive")
    import deme

    solver = deme.DEMSolver([args.device])
    solver.SetVerbosity("ERROR")
    solver.SetOutputFormat("VTK")
    solver.SetContactOutputContent(["OWNER", "FORCE", "POINT", "NORMAL", "TORQUE", "CNT_WILDCARD"])
    solver.SetPersistentContact(True)
    solver.SetMeshUniversalContact(True)
    sphere_material = solver.LoadMaterial(
        {"E": 1e9, "nu": 0.3, "CoR": 0.8, "mu": 0.3, "Crr": 0.01, "Cohesion": 50.0})
    mesh_material = solver.LoadMaterial(
        {"E": 2e9, "nu": 0.4, "CoR": 0.6, "mu": 0.3, "Crr": 0.01, "Cohesion": 50.0})
    solver.SetMaterialPropertyPair("CoR", sphere_material, mesh_material, 0.6)
    solver.SetMaterialPropertyPair("Cohesion", sphere_material, mesh_material, 100.0)
    sphere = solver.LoadSphereType(11728.0, 1.0, sphere_material)
    first = solver.AddClumps(sphere, [[-1.2, 0, 0]])
    first.SetVel([[1, 0, 0]])
    first.SetFamily(0)
    first_tracker = solver.Track(first)

    # The upper mesh moves freely; the lower mesh is a fixed support.
    plane_file = deme.GetDEMEDataFile("mesh/plane_20by20.obj")
    plane = solver.AddWavefrontMeshObject(plane_file, mesh_material)
    plane.SetInitPos([0, 0, -1.25])
    plane.SetMass(10000.0)
    support = solver.AddWavefrontMeshObject(plane_file, mesh_material)
    support.SetInitPos([0, 0, -1.5])
    support.SetFamily(100)
    solver.SetFamilyFixed(100)
    energy = solver.CreateInspector("clump_kinetic_energy")

    # This packaged force model needs both contact history and pairwise properties.
    # Installed kernels sit alongside the data directory; use an absolute path
    # because the C++ convenience search may still point at the build machine.
    model_file = deme.GetDEMEDataFile("../kernel/DEMUserScripts/ForceModelWithCohesion.cu")
    force = solver.ReadContactForceModel(model_file)
    force.SetPerContactWildcards({"delta_time", "delta_tan_x", "delta_tan_y", "delta_tan_z"})
    force.SetMustPairwiseMatProp({"CoR", "mu", "Crr", "Cohesion"})
    solver.SetTimeStepSize(2e-5)
    solver.SetGravitationalAcceleration([0, 0, -9.8])
    solver.SetCDUpdateFreq(10)
    solver.SetMaxVelocity(6.0)
    solver.SetExpandSafetyType("auto")
    solver.SetExpandSafetyMultiplier(1.2)
    solver.SetIntegrator("centered_difference")
    print("Initializing CUDA kernels (a cold cache can take tens of minutes)...", flush=True)
    solver.Initialize()

    # UpdateClumps is the current Python binding of the C++ Update operation.
    # It incorporates new clumps without rebuilding the simulation from scratch.
    second = solver.AddClumps(sphere, [[1.2, 0, 0]])
    second.SetVel([[-1, 0, 0]])
    second.SetFamily(1)
    second_tracker = solver.Track(second)
    solver.UpdateClumps()
    print("Running sphere collision...", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    duration = 0.3 if args.smoke_test else args.duration
    frame_time = 0.01
    for frame in range(math.ceil(duration / frame_time) + 1):
        if frame:
            solver.MarkPersistentContact()
            solver.DoDynamicsThenSync(min(frame_time, duration - (frame - 1) * frame_time))
        solver.WriteSphereFile(str(args.output_dir / f"spheres_{frame:04d}.vtk"))
        solver.WriteMeshFile(str(args.output_dir / f"meshes_{frame:04d}.vtk"))
        solver.WriteContactFile(str(args.output_dir / f"contacts_{frame:04d}.csv"))
    positions = (first_tracker.Pos(), second_tracker.Pos())
    if not all(math.isfinite(value) for position in positions for value in position):
        raise RuntimeError("Non-finite sphere position")
    print(f"Finished: positions={positions}, kinetic energy={energy.GetValue():.6g}")
    print(f"Output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
