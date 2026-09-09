DEM-Engine documentation
========================

DEM-Engine (DEME) is a GPU-accelerated discrete element method solver with a
Chrono-like C++ API and Python bindings distributed as ``deme``.

This documentation covers installation, the simulation model, the Python and
C++ interfaces, and the constraints that matter when running JIT-compiled CUDA
kernels.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   quickstart
   concepts
   mesh-particles
   examples
   visualization
   troubleshooting

.. toctree::
   :maxdepth: 2
   :caption: Interfaces

   python/index
   cpp-api/index

.. toctree::
   :maxdepth: 2
   :caption: Development

   developer/architecture
   developer/documentation
   developer/packaging
   developer/hosting
   project
