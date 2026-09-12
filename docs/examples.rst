Examples and co-simulation
==========================

Run a C++ demo
--------------

From the repository root, after configuring as described in :doc:`installation`:

.. code-block:: console

   cmake --build build --target DEMdemo_SingleSphereCollide --config Release
   cd build
   ./bin/DEMdemo_SingleSphereCollide

On Windows, use ``bin/Release/DEMdemo_SingleSphereCollide.exe`` for a
multi-configuration Release build. Demos can create output files in the working
directory. Read each source file's introductory comments for inputs, output,
and any prerequisite terrain preparation.

``SingleSphereCollide``, ``MeshCollide``, and ``MeshFalling`` use the interactive
viewer when it is enabled. See :doc:`visualization` for responsive loop patterns,
the recommended PoC/small-scale scope, and simulation-throughput tradeoffs.

Choose a starting point
-----------------------

All demo sources live under ``src/demo/``; the target names below have the
prefix ``DEMdemo_``. The checked-in ``src/demo/CMakeLists.txt`` lists the
available demo targets.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Topic
     - Demo targets (omit the ``DEMdemo_`` prefix here)
   * - Installation check
     - ``SingleSphereCollide``
   * - Clump-based mixing
     - ``Mixer``
   * - Mesh particle contact and motion
     - ``MeshCollide``, ``MeshFalling``, ``DrumCubes``, ``MixerCubes``
   * - Analytical and meshed cylinder contacts
     - ``HopperSphereCylinder``, ``HopperSphereMeshedCylinder``
   * - Prescribed boundary motion
     - ``Centrifuge``, ``Sieve``
   * - Granular experiments
     - ``BallDrop``, ``ConePenetration``, ``RotatingDrum``, ``Repose``, ``Plow``
   * - Terrain preparation and wheel tests
     - ``GRCPrep_Part1``, ``GRCPrep_Part2``, ``GRCPrep_Part3``, ``WheelDP``,
       ``WheelDPSimplified``, ``WheelSlopeSlip``
   * - Additional properties and custom force models
     - ``Indentation``, ``Electrostatic``, ``FractureBox``
   * - Prescribed mesh deformation and force extraction
     - ``FlexibleMesh``
   * - Nonstandard applications
     - ``GameOfLife``

``WheelDP`` needs a prepared terrain checkpoint; ``WheelDPSimplified`` is a
starting point without that prerequisite. ``Electrostatic`` illustrates forces
that act without physical overlap. ``FractureBox`` uses contact wildcards for
bonds and breakage. ``FlexibleMesh`` prescribes deformation; it does not include
a solid-mechanics solver.

For Python, start with :doc:`python/quickstart` and the runnable files under
``docs/python/examples/``. Use examples from this checkout with its matching
extension: older scripts on the historical ``pyDEME_demo`` branch can use
different APIs.

Co-simulation
-------------

DEME handles granular dynamics and exposes forces and body state to external
solvers. Coupled applications can use another solver for joints, motors,
controllers, or deformable-body mechanics. See :doc:`python/data-access` for
host and CUDA-array exchange and :doc:`installation` for installing the C++
library.

The historical ``feature/DEME`` branch of ``projectchrono/chrono-projects``
contains Chrono coupling examples. Those examples target their corresponding
Chrono/DEME revisions and may need porting to this branch. Their CMake setup
uses ``Chrono_DIR``, ``ENABLE_PROJECTS=ON``, ``ENABLE_DEME_TESTS=ON``, and
``DEME_DIR`` after both libraries are installed.
