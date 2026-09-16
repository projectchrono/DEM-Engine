"""Time DEME initialization with an explicitly configured persistent Jitify cache."""

import os
from pathlib import Path
from time import perf_counter


cache_value = os.environ.get("DEME_PERSISTENT_JITIFY_CACHE")
if not cache_value:
    raise SystemExit("Set DEME_PERSISTENT_JITIFY_CACHE to a user-owned cache file before running this example")
if cache_value.lower() in {"0", "1", "false", "true", "off", "on", "no", "yes"}:
    raise SystemExit("Use an explicit cache filename for a reproducible cold-versus-warm timing comparison")

# Set the environment before importing DEME so the same script also covers future initialization-order changes.
import deme


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
solver.InstructBoxDomainDimension((-0.5, 0.5), (-0.5, 0.5), (0.0, 1.0))
solver.InstructBoxDomainBoundingBC("top_open", material)
sphere_type = solver.LoadSphereType(0.01, 0.025, material)
solver.AddClumps(sphere_type, [[0.0, 0.0, 0.5]])
solver.SetInitTimeStep(1.0e-5)

start = perf_counter()
solver.Initialize()
elapsed = perf_counter() - start

cache_path = Path(cache_value).expanduser()
cache_size = cache_path.stat().st_size if cache_path.is_file() else 0
print(f"Initialize: {elapsed:.3f} s")
print(f"Cache: {cache_path} ({cache_size} bytes)")
