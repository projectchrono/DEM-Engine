"""Drop a meshed projectile into a small granular bed (DEMdemo_BallDrop.cpp)."""

import argparse
import math
from pathlib import Path
import random


def main():
    """Settle a reproducible bed, release a tracked mesh, and measure penetration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--duration", type=float, default=0.5, help="Time after releasing the projectile, in seconds")
    parser.add_argument("--settle-time", type=float, default=0.5)
    parser.add_argument("--drop-height", type=float, default=0.05)
    parser.add_argument("--density", type=float, default=7800.0, help="Projectile density in kg/m^3")
    parser.add_argument("--output-dir", type=Path, default=Path("DemoOutput_BallDrop"))
    parser.add_argument("--smoke-test", action="store_true", help="Use a coarser bed and short settling/drop stages")
    args = parser.parse_args()
    for name in ("duration", "settle_time", "density", "drop_height"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    import deme

    solver = deme.DEMSolver([args.device])
    solver.SetVerbosity("ERROR")
    solver.SetOutputFormat("CSV")
    material = solver.LoadMaterial({"E": 7e7, "nu": 0.24, "CoR": 0.9, "mu": 0.3, "Crr": 0.0})
    solver.InstructBoxDomainDimension([-0.04, 0.04], [-0.04, 0.04], [0, 0.8])
    solver.InstructBoxDomainBoundingBC("top_open", material)

    # A seeded mixture on an HCP lattice avoids random initial overlaps.
    grain_radius = 0.005 if args.smoke_test else 0.0025
    radii = [grain_radius * (0.85 + 0.03 * i) for i in range(6)]
    templates = [solver.LoadSphereType(2500 * 4 * math.pi * r**3 / 3, r, material) for r in radii]
    positions = deme.DEMBoxHCPSampler([0, 0, 0.02], [0.04 - grain_radius * 1.01] * 2 +
                                    [0.02 - grain_radius * 1.01], 2.01 * grain_radius)
    rng = random.Random(4150)
    bed = solver.AddClumps([rng.choice(templates) for _ in positions], positions)
    bed.SetFamily(0)
    top = solver.CreateInspector("clump_max_z")

    radius = 0.0127
    mass = args.density * 4 * math.pi * radius**3 / 3
    projectile = solver.AddWavefrontMeshObject(deme.GetDEMEDataFile("mesh/sphere.obj"), material)
    projectile.Scale(radius)
    projectile.SetMass(mass)
    projectile.SetMOI([0.4 * mass * radius**2] * 3)
    projectile.SetInitPos([0, 0, 0.6])
    projectile.SetFamily(2)
    solver.SetFamilyFixed(2)
    solver.DisableContactBetweenFamilies(0, 2)
    tracker = solver.Track(projectile)
    solver.SetTimeStepSize(2e-6)
    solver.SetGravitationalAcceleration([0, 0, -9.81])
    solver.SetMaxVelocity(30.0)
    print("Initializing CUDA kernels (a cold cache can take tens of minutes)...", flush=True)
    solver.Initialize()
    print("Settling the bed...", flush=True)
    solver.DoDynamicsThenSync(0.02 if args.smoke_test else args.settle_time)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    solver.WriteClumpFile(str(args.output_dir / "settled_bed.csv"))
    bed_top = top.GetValue()
    solver.ChangeFamily(2, 0)
    tracker.SetPos([0, 0, bed_top + radius + (0.001 if args.smoke_test else args.drop_height)])
    print("Dropping the projectile...", flush=True)
    duration = 0.04 if args.smoke_test else args.duration
    frame_time = 0.01
    for frame in range(math.ceil(duration / frame_time) + 1):
        if frame:
            solver.DoDynamicsThenSync(min(frame_time, duration - (frame - 1) * frame_time))
        solver.WriteSphereFile(str(args.output_dir / f"bed_{frame:04d}.csv"))
        solver.WriteMeshFile(str(args.output_dir / f"projectile_{frame:04d}.vtk"))
    position = tracker.Pos()
    if not all(math.isfinite(value) for value in position):
        raise RuntimeError("Non-finite projectile position")
    print(f"Finished: {len(positions)} grains, penetration={bed_top - (position[2] - radius):.6g} m")
    print(f"Output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
