# SBEL Chrono DEM-Engine

**Now version 3: GPU-accelerated discrete element simulation built for performance, with C++ and Python APIs.**

DEM-Engine (DEME) simulates granular materials using one or two NVIDIA GPUs.
This branch supports sphere clumps, mesh particles, analytical boundaries,
rigid combined owners, and customizable contact force models. It also provides
an interactive visualizer and host/device data access for co-simulation.

<p>
  <img width="380" src="https://i.imgur.com/DKGlM14.jpg" alt="DEM-Engine granular simulation">
  <img width="380" src="https://i.imgur.com/Pt74UFM.gif" alt="DEM-Engine demo animation">
  
</p>

<p>
  <img width="460" src="https://i.imgur.com/A3utANi.gif" alt="DEM-Engine simulation animation">
  <img width="300" src="https://i.imgur.com/4R25TPX.gif" alt="DEM-Engine demo animation">
</p>

## What's new in DEME 3?

**[Get started with DEME 3: installation guide](docs/installation.rst)** —
Python packages, C++ source builds, and system requirements.

- **Mesh contact:** mesh–mesh collisions and a new clump–mesh scheme that combines
  triangle contributions into patch/island contacts before evaluating forces.
- **Rigid combined bodies:** group members into one rigid assembly, replacing
  geometry-wildcard-based constructions with member-level controls.
- **On-device coupling:** exchange state and forces directly with other GPU packages.
- **Persistent kernel caching:** reuse compiled kernels across compatible repeated
  runs to reduce initialization time.
- **Interactive visualization and expanded Python workflows.**

**When to stay with DEME 2:** DEME 3 currently supports only NVIDIA GPUs and may
use more memory. If you need non-NVIDIA GPU support, or your application does
not need mesh–mesh contact or the new aggregated clump–mesh contact scheme,
consider staying with **DEME 2.4.2**, the final DEME 2 release.

For C++, use the upstream [v2.4.2 tag](https://github.com/projectchrono/DEM-Engine/tree/v2.4.2):

```bash
git clone --branch v2.4.2 --recurse-submodules https://github.com/projectchrono/DEM-Engine.git DEM-Engine-2.4.2
```

For pyDEME, explicitly pin the Python distribution: `python -m pip install "deme==2.4.2"`.
Use version 2.4.2's installation requirements and examples for either route.
See [DEME 3 features and migration considerations](docs/deme3-new-features.rst)
for details, including the contact-model changes and memory tradeoffs.

## Why use DEME?

DEME is designed for large granular simulations where particle shape, contact
physics, and computational cost matter. Typical applications include mixing,
hopper flow, soil penetration, wheel–terrain interaction, and granular impact.

- **Complex particle shapes.** Represent grains with clumped spheres or mesh
  particles, and build rigid assemblies with combined owners. DEME supports
  **mesh–mesh contact**, allowing mesh particles to collide with one another.
- **Custom contact physics.** Define your own contact force models, including
  cohesion, electrostatic interactions, and bonds that can break. Material
  properties and per-contact variables let you tailor the model to your problem.
- **GPU performance.** Use one or two NVIDIA GPUs, including consumer and data
  center hardware. As an illustrative benchmark from the main-branch README,
  one million three-sphere clumps simulated for one million timesteps takes
  around one hour on RTX 3080s. Runtime depends on the geometry, contact
  model, and simulation settings.
- **Control over the simulation.** Prescribe motion, extract forces, and update
  geometry to model processes such as mesh deformation or grain breakage.
  The examples show how to supply these behaviors through the API.
- **On-device co-simulation.** Exchange simulation state and forces directly
  with other GPU-based packages through device buffers, avoiding CPU round trips
  for the exchanged data. Host data access also supports coupling to solvers
  such as [Chrono](https://github.com/projectchrono/chrono) for multibody dynamics
  or other physics.
- **C++ and Python workflows.** Start with Python or integrate the C++ library
  into an application. The C++ API follows a Chrono-like design, and the
  interactive visualizer helps inspect simulations as they run.

## Start here

| Task | Documentation |
| --- | --- |
| Install Python, build C++, or install the C++ library | [Installation](docs/installation.rst) |
| Run a first simulation | [Quickstart](docs/quickstart.rst) · [Python example](docs/python/quickstart.rst) |
| Understand owners, families, frames, and runtime setup | [Core concepts](docs/concepts.rst) |
| Use mesh particles, templates, and combined bodies | [Mesh particles](docs/mesh-particles.rst) |
| Find a demo to adapt | [Examples](docs/examples.rst) · [C++ sources](src/demo) |
| Use Python | [Python guide](docs/python/index.rst) · [Demos](python/demos/README.md) · [API reference](docs/python/reference.rst) |
| Look up the C++ API | [C++ reference](docs/cpp-api/index.rst) |
| Select GPUs or exchange simulation data | [Device selection](docs/python/device-selection.rst) · [Data access](docs/python/data-access.rst) |
| Visualize results | [Interactive visualization and ParaView](docs/visualization.rst) |
| Diagnose installation or runtime errors | [Troubleshooting](docs/troubleshooting.rst) |
| Build or host the documentation website | [Build and preview](docs/README.md) · [Hosting](docs/developer/hosting.rst) |
| Contribute or cite DEME | [Project information](docs/project.rst) · [Architecture](docs/developer/architecture.rst) |

The [documentation index](docs/index.rst) collects the complete guide. The C++
reference is rendered from Doxygen by the documentation build; build the site
to browse C++ declarations and Python documentation together.

## Python in brief

On a supported Linux or WSL2 host with a CUDA 12.9-compatible NVIDIA driver:

```bash
python -m pip install "deme[cuda12]"
```

The `cuda12` extra installs CUDA runtime libraries, NVRTC, and headers through pip;
no system CUDA Toolkit installation is needed for Python wheels. This setup applies
only to the Python extension; standalone C++ applications keep their normal CUDA
configuration. Use plain `pip install deme` to use an existing toolkit.

```python
import deme

solver = deme.DEMSolver()
```

After installation, run a Python demo from the repository root:

```bash
python python/demos/single_sphere_collide.py --smoke-test
```

This headless example simulates two colliding spheres over meshes and writes
visualization files. The first run may take time to compile CUDA kernels.
See the [Python demos](python/demos/README.md) for more examples, command-line
options, and instructions for viewing their output.

See [installation requirements](docs/installation.rst) for wheel compatibility
and source builds. New scripts should use `import deme`; `import DEME` remains
a compatibility alias. Features in this checkout may be newer than a released wheel.

## Community and license

[Demo videos](https://uwmadison.app.box.com/s/u4m9tee3k1vizf097zkq3rgv54orphyv) ·
[Project Chrono forum](https://groups.google.com/g/projectchrono) ·
[Contributors](docs/CONTRIBUTORS.md) · [BSD-3-Clause license](LICENSE.md) ·
[Citation](#citation)

## Citation

If you use DEME in your research, please cite the
[DEM-Engine design and usage paper](https://doi.org/10.1016/j.cpc.2024.109196):

```bibtex
@article{zhang_2024_deme,
title = {Chrono {DEM-Engine}: A Discrete Element Method dual-{GPU} simulator with customizable contact forces and element shape},
journal = {Computer Physics Communications},
volume = {300},
pages = {109196},
year = {2024},
issn = {0010-4655},
doi = {https://doi.org/10.1016/j.cpc.2024.109196},
author = {Ruochun Zhang and Bonaventura Tagliafierro and Colin {Vanden Heuvel} and Shlok Sabarwal and Luning Bakke and Yulong Yue and Xin Wei and Radu Serban and Dan Negruţ},
keywords = {Discrete Element Method, GPU computing, Physics-based simulation, Scientific package, BSD3 open-source},
}
```

For the clump-based granular solver and its application to rover dynamics, see
[the granular simulation paper](https://doi.org/10.1007/s00366-023-01921-9):

```bibtex
@article{ruochunGRC-DEM2023,
      title={A {GPU}-accelerated simulator for the {DEM} analysis of granular systems composed of clump-shaped elements}, 
      author={Ruochun Zhang and Colin {Vanden Heuvel} and Alexander Schepelmann and Arno Rogg and Dimitrios Apostolopoulos and Samuel Chandler and Radu Serban and Dan Negrut},
      year={2024},
      journal={Engineering with Computers},
      doi={https://doi.org/10.1007/s00366-023-01921-9}
}
```
