# Python demos

These scripts use `import deme` and the data and kernels shipped with the Python
wheel. Install the package into your active environment:

```sh
python -m pip install 'deme[cuda12]'
```

The `cuda12` extra supplies CUDA 12 libraries and headers
for Linux, including WSL2. A compatible NVIDIA GPU driver is still required.
First-time solver initialization compiles CUDA kernels and can take tens of
minutes for these scenes. Cached runs are much faster; a short smoke duration
does not reduce this compilation work.

Run these commands from a repository checkout. The scripts resolve simulation
assets through the installed package, so they also work from another working
directory when invoked by their absolute path. No additional Python packages are
needed beyond the DEME installation.

| Script | C++ source | What it demonstrates |
| --- | --- | --- |
| `single_sphere_collide.py` | `src/demo/DEMdemo_SingleSphereCollide.cpp` | Two spheres, moving and fixed meshes, cohesive contact, pairwise material properties, runtime insertion with `UpdateClumps()`, tracking and contact output |
| `ball_drop.py` | `src/demo/DEMdemo_BallDrop.cpp` | A polydisperse bed, meshed projectile, settling, family-based release, and penetration measurement |
| `centrifuge.py` | `src/demo/DEMdemo_Centrifuge.cpp` | Ellipsoid clumps and equal-mass spheres at three densities, an analytical drum with end caps, prescribed rotation, and contact torque |
| `jitify_cache_timing.py` | — | Kernel compilation cache timing; requires the `DEME_PERSISTENT_JITIFY_CACHE` environment variable |

```sh
python python/demos/single_sphere_collide.py
python python/demos/ball_drop.py --density 7800 --drop-height 0.05
python python/demos/centrifuge.py --duration 2
```

The three physics demos run headlessly. Use `--device 0` to select a GPU,
`--duration SECONDS` to change the simulated duration, and `--output-dir PATH`
to choose an output directory. Existing files with the same names are overwritten.
Use `--help` for all options; it works without loading CUDA.

## Scope of the ports

- **Single sphere collision:** retains the C++ demo's sphere masses, radii, initial
  velocities, mesh positions, cohesive material parameters, and time step. The
  second sphere is added after `Initialize()` via `UpdateClumps()` (the current
  Python alias for the C++ `Update()` operation). Redundant API regression exercises
  and the interactive viewer are omitted to keep the example readable.
- **Ball drop:** runs one impact instead of the C++ parameter sweep. The default
  bed is 8 cm wide and initially 4 cm deep, with grain radii 2.125–2.5 mm on a
  seeded HCP mixture. The projectile radius (12.7 mm), material parameters, and
  time step follow C++. Adjust `--settle-time` as needed; the default 0.5 s is a
  demonstration setting, not a guarantee of an equilibrated bed. The reported
  penetration is measured at the final requested time, not necessarily at rest.
- **Centrifuge:** keeps the 2 m drum radius, 1 m nominal height, 6 rad/s rotation,
  and C++ time step. The default particle scale is 0.05 m instead of 0.01 m to
  reduce particle count. End caps share the drum owner, so reported torque is
  the total for the cylinder and caps. The initial grid leaves clearance for the
  ellipsoids. To use the C++ particle scale and duration:

  ```sh
  python python/demos/centrifuge.py --particle-scale 0.01 --duration 20
  ```

## Short checks

```sh
python python/demos/single_sphere_collide.py --smoke-test
python python/demos/ball_drop.py --smoke-test
python python/demos/centrifuge.py --smoke-test
```

These options override the duration (and, for the bed/drum, particle size). The
sphere check runs 0.3 s to include the sphere collision. The ball-drop check uses
0.02 s of settling followed by a 0.04 s drop from 1 mm above the bed. The drum
check uses 0.1 m particles and runs 0.03 s, long enough for the lowest
ellipsoids to reach the lower cap. Each script checks a final position or maximum speed for non-finite values. These are setup/runtime checks, not validation of
settled-bed behavior or centrifugal segregation, and still require kernel
compilation on a cold cache.

## Viewing output

The collision and ball-drop demos write frames every 0.01 s; the centrifuge
writes every 0.05 s. Frame zero is the initial state for that stage. The final
interval is shortened when the requested duration is not an exact multiple.

Open the VTK sequences in ParaView. For `spheres_*.vtk`, apply **Glyph**, choose a
sphere, scale by the `r` field, and use a scale factor of 2 if the glyph source
has the default radius of 0.5. Meshes and analytical geometry
are separate VTK sequences. The bed and centrifuge write particle CSV files
with component positions and radii; use a CSV reader and **Table To Points**
followed by sphere glyphs. Contact CSV files contain the requested owner IDs,
forces, points, normals, torques, and history fields.
