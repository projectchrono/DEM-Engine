"""Retrieve tracked DEME positions directly into a CuPy allocation."""

import cupy as cp
import deme


solver = deme.DEMSolver([0])
material = solver.LoadMaterial(
    {"E": 1.0e7, "nu": 0.3, "CoR": 0.5, "mu": 0.4, "Crr": 0.0}
)
solver.InstructBoxDomainDimension(1.0, 1.0, 1.0)
sphere_type = solver.LoadSphereType(0.01, 0.025, material)
initial_positions = [[0.0, 0.0, 0.2], [0.0, 0.0, 0.4]]
sphere_batch = solver.AddClumps(sphere_type, initial_positions)
tracker = solver.Track(sphere_batch)
solver.SetInitTimeStep(1.0e-5)
solver.Initialize()
solver.DoDynamicsThenSync(0.001)

destination_device = 0
owner_count = len(initial_positions)
with cp.cuda.Device(destination_device):
    # A float3 result requires a C-contiguous (owner_count, 3) float32 array.
    positions = cp.empty((owner_count, 3), dtype=cp.float32)
    tracker.PositionsToDevice(
        positions.data.ptr,
        owner_count,  # Capacity is float3 elements, not bytes or scalar floats.
        destination_device,
    )
    print(positions)
