"""Run a minimal one-sphere DEME simulation and print its final position."""

import deme


# A one-element device list places both DEME workers on logical CUDA device 0.
solver = deme.DEMSolver([0])

material = solver.LoadMaterial(
    {
        "E": 1.0e7,
        "nu": 0.3,
        "CoR": 0.5,
        "mu": 0.4,
        "Crr": 0.0,
    }
)

# Explicit spans make the global origin and floor location unambiguous.
solver.InstructBoxDomainDimension(
    (-0.5, 0.5),
    (-0.5, 0.5),
    (0.0, 1.0),
)
solver.InstructBoxDomainBoundingBC("top_open", material)

sphere_type = solver.LoadSphereType(0.01, 0.025, material)
sphere_batch = solver.AddClumps(sphere_type, [[0.0, 0.0, 0.5]])
sphere_tracker = solver.Track(sphere_batch)

solver.SetGravitationalAcceleration([0.0, 0.0, -9.81])
solver.SetInitTimeStep(1.0e-5)
solver.Initialize()

# Synchronize before querying tracked state on the host.
solver.DoDynamicsThenSync(0.01)
print("Sphere position:", sphere_tracker.Pos())
