"""Rotate a drum containing spheres and ellipsoids (DEMdemo_Centrifuge.cpp)."""

import argparse
import math
from pathlib import Path


def main():
    """Build six grain templates and prescribe rotation of an analytical drum."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--particle-scale", type=float, default=0.05, help="Ellipsoid short semi-axis in meters")
    parser.add_argument("--output-dir", type=Path, default=Path("DemoOutput_Centrifuge"))
    parser.add_argument("--smoke-test", action="store_true", help="Use larger grains and run 0.03 s")
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration must be finite and positive")
    if not math.isfinite(args.particle_scale) or not 0 < args.particle_scale <= 0.1:
        parser.error("--particle-scale must be in (0, 0.1]")
    import deme

    solver = deme.DEMSolver([args.device])
    solver.SetVerbosity("ERROR")
    solver.SetOutputFormat("CSV")
    solver.SetOutputContent(["FAMILY"])
    sand = solver.LoadMaterial({"E": 1e9, "nu": 0.3, "CoR": 0.6, "mu": 0.5, "Crr": 0.01})
    drum_material = solver.LoadMaterial({"E": 2e9, "nu": 0.3, "CoR": 0.6, "mu": 0.5, "Crr": 0.01})
    solver.SetMaterialPropertyPair("mu", sand, drum_material, 0.5)
    scale = 0.1 if args.smoke_test else args.particle_scale
    mass = 2600 * 4 * math.pi * 2 / 3
    ellipsoid = solver.LoadClumpType(mass, [mass, mass, 0.4 * mass],
                                    deme.GetDEMEDataFile("clumps/ellipsoid_2_1_1.csv"), sand)
    # Scale already updates geometry, mass (s^3), and inertia (s^5).
    ellipsoid.Scale(scale)
    templates = []
    for density_index in range(3):
        grain = solver.Duplicate(ellipsoid)
        multiplier = 1.5**density_index
        grain.SetMass(ellipsoid.Mass() * multiplier)
        grain.SetMOI([value * multiplier for value in ellipsoid.MOI()])
        templates.extend([grain, solver.LoadSphereType(grain.Mass(), 2**(1 / 3) * scale, sand)])

    drum = solver.AddExternalObject()
    drum.AddCylinder([0, 0, 0], [0, 0, 1], 2.0, drum_material)
    drum.AddPlane([0, 0, -0.47], [0, 0, 1], drum_material)
    drum.AddPlane([0, 0, 0.47], [0, 0, -1], drum_material)
    drum.SetMass(1.0)
    drum.SetMOI([13 / 12, 13 / 12, 2.0])
    drum.SetFamily(100)
    # Keep the drum centered while prescribing its rotation about the cylinder axis.
    solver.SetFamilyPrescribedLinVel(100, "0", "0", "0", False)
    solver.SetFamilyPrescribedAngVel(100, "0", "0", "6.0")
    tracker = solver.Track(drum)
    # Keep even the corner grains clear of the cylindrical wall.
    half_width = min(2 / 1.5, (2 - 2.01 * scale) / math.sqrt(2))
    # The sphere approximation extends 2.08 * scale along z, slightly beyond
    # the nominal ellipsoid semi-axis; 2.1 leaves a small end-cap clearance.
    positions = deme.DEMBoxGridSampler([0, 0, 0], [half_width, half_width, 0.47 - 2.1 * scale],
                                     scale * 2**(1 / 3) * 2.1, scale * 2**(1 / 3) * 2.1, scale * 4.2)
    grains = solver.AddClumps([templates[i % 6] for i in range(len(positions))], positions)
    grains.SetFamilies([(i % 6) // 2 for i in range(len(positions))])
    speed = solver.CreateInspector("clump_max_absv")
    solver.InstructBoxDomainDimension(5.0, 5.0, 5.0)
    solver.SetTimeStepSize(5e-6)
    solver.SetGravitationalAcceleration([0, 0, -9.81])
    solver.SetExpandSafetyType("auto")
    # Broad-phase padding must include the analytical surface speed: omega * radius.
    solver.SetExpandSafetyAdder(12.0)
    print("Initializing CUDA kernels (a cold cache can take tens of minutes)...", flush=True)
    solver.Initialize()
    print("Running rotating drum...", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    duration = 0.03 if args.smoke_test else args.duration
    frame_time = 0.05
    for frame in range(math.ceil(duration / frame_time) + 1):
        if frame:
            solver.DoDynamicsThenSync(min(frame_time, duration - (frame - 1) * frame_time))
        solver.WriteSphereFile(str(args.output_dir / f"grains_{frame:04d}.csv"))
        solver.WriteAnalyticalFile(str(args.output_dir / f"drum_{frame:04d}.vtk"))
    max_speed = speed.GetValue()
    if not math.isfinite(max_speed):
        raise RuntimeError("Non-finite grain speed")
    print(f"Finished: {len(positions)} grains, maximum speed={max_speed:.6g} m/s")
    torque = tuple(a * b for a, b in zip(tracker.ContactAngAccLocal(), [13 / 12, 13 / 12, 2.0]))
    print(f"Drum contact torque (local): {torque}")
    print(f"Output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
