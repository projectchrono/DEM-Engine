# DEM-Engine

**GPU-accelerated discrete element simulation with C++ and Python APIs.**

DEM-Engine (DEME) simulates granular materials using one or two NVIDIA GPUs.
This branch supports sphere clumps, mesh particles, analytical boundaries,
rigid combined owners, and customizable contact force models. It also provides
an interactive visualizer and host/device data access for co-simulation.

<p>
  <img width="380" src="https://i.imgur.com/DKGlM14.jpg" alt="DEM-Engine granular simulation">
  <img width="380" src="https://i.imgur.com/A3utANi.gif" alt="DEM-Engine simulation animation">
</p>

<p>
  <img width="380" src="https://i.imgur.com/YOEbAd8.gif" alt="DEM-Engine demo animation">
  <img width="380" src="https://i.imgur.com/4R25TPX.gif" alt="DEM-Engine demo animation">
</p>

## Why use DEME?

DEME is designed for large granular simulations where particle shape, contact
physics, and computational cost matter. Typical applications include mixing,
hopper flow, soil penetration, wheel–terrain interaction, and granular impact.

- **Complex particle shapes.** Represent grains with clumped spheres or mesh
  particles, and build rigid assemblies with combined owners.
- **Custom contact physics.** Define your own contact force models, including
  cohesion, electrostatic interactions, and bonds that can break. Material
  properties and per-contact variables let you tailor the model to your problem.
- **GPU performance.** Use one or two NVIDIA GPUs, including consumer and data
  center hardware. As an illustrative benchmark from the main-branch README,
  one million three-sphere clumps simulated for one million timesteps takes
  around one hour on two RTX 3080s. Runtime depends on the geometry, contact
  model, and simulation settings.
- **Control over the simulation.** Prescribe motion, extract forces, and update
  geometry to model processes such as mesh deformation or grain breakage.
  The examples show how to supply these behaviors through the API.
- **Co-simulation.** Couple DEME to other solvers, such as
  [Chrono](https://github.com/projectchrono/chrono), for multibody dynamics or
  other physics. Host and GPU data access support exchanging state and forces.
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
configuration. Use plain `pip install deme` to
use an existing toolkit. Preview builds from `Mesh_Particles_Py` are published as
`deme3` (`pip install "deme3[cuda12]"`), with the same `import deme` namespace.

```python
import deme

solver = deme.DEMSolver()
```

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
