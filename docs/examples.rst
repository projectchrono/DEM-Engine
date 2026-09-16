Examples and co-simulation
==========================

Run a C++ demo
--------------

From the repository root, after configuring as described in `Installation <installation.rst>`__:

.. code-block:: console

   cmake --build build --target DEMdemo_SingleSphereCollide --config Release
   cd build
   ./bin/DEMdemo_SingleSphereCollide

On Windows, use ``bin/Release/DEMdemo_SingleSphereCollide.exe`` for a
multi-configuration Release build. Demos can create output files in the working
directory. Read each source file's introductory comments for inputs, output,
and any prerequisite terrain preparation.

``SingleSphereCollide``, ``MeshCollide``, and ``MeshFalling`` use the interactive
viewer when it is enabled. See `Interactive visualization <visualization.rst>`__ for responsive loop patterns,
the recommended PoC/small-scale scope, and simulation-throughput tradeoffs.

Python demos
------------

After installing ``deme``, run a first demo from the repository root:

.. code-block:: console

   python python/demos/single_sphere_collide.py --smoke-test

The three physics demos run headlessly and accept ``--device``, ``--duration``,
``--output-dir``, and ``--smoke-test``. Use ``--help`` to inspect their options
without loading CUDA. Cold-cache initialization can take much longer than the
short simulated duration. They write output files for visualization; see
`the Python demos guide <../python/demos/README.md>`__ for commands,
porting scope, and output instructions.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Script (under ``python/demos/``)
     - What it does
   * - `single_sphere_collide.py <../python/demos/single_sphere_collide.py>`__
     - Collides two spheres over meshes with cohesive contact; inserts the second
       sphere after initialization and writes contact and geometry output.
   * - `ball_drop.py <../python/demos/ball_drop.py>`__
     - Settles a polydisperse spherical bed, releases a meshed projectile, and
       measures penetration at the requested final time. Runs one impact rather
       than the C++ parameter sweep.
   * - `centrifuge.py <../python/demos/centrifuge.py>`__
     - Rotates an analytical drum containing ellipsoids and equal-mass spheres
       at three densities; reports contact torque. Uses larger particles than
       the C++ case by default to reduce the particle count.
   * - `jitify_cache_timing.py <../python/demos/jitify_cache_timing.py>`__
     - Times initialization of a one-sphere setup and reports the persistent
       Jitify header-cache file size. Requires ``DEME_PERSISTENT_JITIFY_CACHE``
       to be an explicit filename. Compare separate runs; compiled-kernel cache
       state also affects the timing.

For smaller API-specific examples, `Python quickstart <python/quickstart.rst>`__ runs a sphere drop,
while `Retrieving simulation data <python/data-access.rst>`__ demonstrates CuPy and Warp device buffers.
The ``docs/python/examples/`` directory also includes explicit GPU selection.
Use these examples with a matching version of the installed extension.

C++ demo catalog
----------------

The following catalog covers every demo target in ``src/demo/CMakeLists.txt``.
Target names have the prefix ``DEMdemo_``; links below download the corresponding
source under ``src/demo/``. Choose ``SingleSphereCollide`` for a first run,
``MeshCollide`` for mesh contacts, or ``Mixer`` for a larger application.

First simulations and contact inspection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Demo (without ``DEMdemo_``)
     - What it does
   * - `SingleSphereCollide <../src/demo/DEMdemo_SingleSphereCollide.cpp>`__
     - A small sphere-collision setup with mesh boundaries, cohesive material properties, runtime insertion, and contact-force inspection. A starting point for the API and interactive viewer.
   * - `MeshCollide <../src/demo/DEMdemo_MeshCollide.cpp>`__
     - Collides two cube meshes above an analytical plane, demonstrating mesh universal contact, tracking, and contact-force retrieval.
   * - `MeshFalling <../src/demo/DEMdemo_MeshFalling.cpp>`__
     - Drops a mixture of meshed boxes, spheres, cones, and cylinders onto a plane to exercise mesh–mesh and mesh–plane contacts.
   * - `ContactChain <../src/demo/DEMdemo_ContactChain.cpp>`__
     - Applies an external load to the topmost particle of a granular arrangement to examine contact-chain propagation.
   * - `TestPack <../src/demo/DEMdemo_TestPack.cpp>`__
     - Contains rolling-up-an-incline, falling-ellipsoid, and stacked-sphere validation cases. The current main function runs the incline case; the other calls are commented out.
   * - `TestRestart <../src/demo/DEMdemo_TestRestart.cpp>`__
     - Loads example contact pairs and contact wildcards from CSV, attaches them to a sphere batch, and duplicates batches to exercise contact-history initialization.

Flow, mixing, and particle generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Demo (without ``DEMdemo_``)
     - What it does
   * - `Repose <../src/demo/DEMdemo_Repose.cpp>`__
     - Releases particles through a mesh funnel to form a pile for an angle-of-repose experiment.
   * - `Repose2D <../src/demo/DEMdemo_Repose2D.cpp>`__
     - An angle-of-repose variant using ForceModel2D.cu and particle centers sampled in the X–Z plane.
   * - `Plow <../src/demo/DEMdemo_Plow.cpp>`__
     - Moves a bowl-shaped plow through granular material using prescribed motion.
   * - `Sieve <../src/demo/DEMdemo_Sieve.cpp>`__
     - Prescribes back-and-forth motion of a clump-built sieve so sufficiently small particles can fall through.
   * - `RotatingDrum <../src/demo/DEMdemo_RotatingDrum.cpp>`__
     - Rotates a drum represented by clumped spheres with ellipsoidal grains inside to study the granular free-surface slope.
   * - `Centrifuge <../src/demo/DEMdemo_Centrifuge.cpp>`__
     - Rotates an analytical container containing grains of different shapes and densities; family labels distinguish the populations.
   * - `DrumCubes <../src/demo/DEMdemo_DrumCubes.cpp>`__
     - Rotates an analytical cylindrical drum and lids containing cube mesh particles, demonstrating repeated mesh templates and mesh contacts.
   * - `DrumCubesSmall <../src/demo/DEMdemo_DrumCubesSmall.cpp>`__
     - A smaller cube-particle setup using 10 mm cubes. The current code uses rotating planar side walls rather than the cylindrical wall described in its introductory comment.
   * - `ResponseAngleMesh <../src/demo/DEMdemo_ResponseAngleMesh.cpp>`__
     - Loads mesh-particle templates from STL and rotates a meshed drum with analytical end caps to study the surface response angle.
   * - `Mixer <../src/demo/DEMdemo_Mixer.cpp>`__
     - Prescribes rotation of a meshed bladed mixer through three-sphere clump particles inside a cylindrical chamber.
   * - `MixerCubes <../src/demo/DEMdemo_MixerCubes.cpp>`__
     - Replaces the mixer demo’s clumps with cube mesh particles, enabling contacts with the mixer, chamber, and other cubes.
   * - `HopperSphereCylinder <../src/demo/DEMdemo_HopperSphereCylinder.cpp>`__
     - Settles and discharges a mixture of spheres and cylinder-shaped clumps through a mesh funnel after opening its gate.
   * - `HopperSphereMeshedCylinder <../src/demo/DEMdemo_HopperSphereMeshedCylinder.cpp>`__
     - Discharges spheres mixed with low-poly mesh cylinders by disabling a mesh plug’s contacts. Includes a small self-checking --smoke-test mode.
   * - `PolydisperseGeneration <../src/demo/DEMdemo_PolydisperseGeneration.cpp>`__
     - Settles a spherical bed, inspects its height, and writes a clump checkpoint. The proposed runtime particle-enlargement loop is currently commented out.
   * - `Shake <../src/demo/DEMdemo_Shake.cpp>`__
     - Shakes a jar of particles to explore changes in packing and bulk density.

Impact and soil characterization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Demo (without ``DEMdemo_``)
     - What it does
   * - `BallDrop <../src/demo/DEMdemo_BallDrop.cpp>`__
     - Drops a meshed ball into a loose granular bed over a sweep of projectile densities and drop heights.
   * - `BallDrop2D <../src/demo/DEMdemo_BallDrop2D.cpp>`__
     - A ball-impact variant using ForceModel2D.cu, with a settling stage followed by release of a tracked mesh projectile.
   * - `PlateSinkage <../src/demo/DEMdemo_PlateSinkage.cpp>`__
     - Prepares and saves a cohesionless bed, reloads its positions and orientations with a cohesive model, then presses a circular plate into it to record pressure versus sinkage.
   * - `ConePenetration <../src/demo/DEMdemo_ConePenetration.cpp>`__
     - Compresses a bed of clumped particles before performing a cone-penetration test; demonstrates stepwise control of the compressor.
   * - `Indentation <../src/demo/DEMdemo_Indentation.cpp>`__
     - Compresses a granular sample and uses contact-neighbor information and custom properties to examine its strain distribution.

Terrain preparation and wheel mobility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Demo (without ``DEMdemo_``)
     - What it does
   * - `GRCPrep_Part1 <../src/demo/DEMdemo_GRCPrep_Part1.cpp>`__
     - Generates and settles batches of GRC simulant particles with a distribution of sizes and shapes; writes the initial bed for Part 2.
   * - `GRCPrep_Part2 <../src/demo/DEMdemo_GRCPrep_Part2.cpp>`__
     - Reads Part 1 output and replicates the settled particles to create a thicker bed. This is a large, potentially multimillion-particle case.
   * - `GRCPrep_Part3 <../src/demo/DEMdemo_GRCPrep_Part3.cpp>`__
     - Reads Part 2 particle positions and orientations, builds the terrain, compresses it with a plane, and saves the final bed.
   * - `RoverWheel <../src/demo/DEMdemo_RoverWheel.cpp>`__
     - Drives a sphere-clump wheel with prescribed angular velocity over a bed of ellipsoidal clumps; demonstrates wheel principal-frame setup.
   * - `WheelDP <../src/demo/DEMdemo_WheelDP.cpp>`__
     - Runs a drawbar-pull test with a meshed Curiosity wheel and prepared GRC terrain. Requires GRC_3e6.csv from the GRC preparation sequence in the working directory.
   * - `WheelDPSimplified <../src/demo/DEMdemo_WheelDPSimplified.cpp>`__
     - Builds its own three-sphere-clump terrain and prescribes wheel translation and rotation for one slip case, measuring terrain forces without a GRC checkpoint prerequisite.
   * - `WheelSlopeSlip <../src/demo/DEMdemo_WheelSlopeSlip.cpp>`__
     - Studies slip versus slope using a sphere-clump representation of a Viper-style wheel. Requires the prepared GRC_3e6.csv terrain checkpoint.

Custom physics and coupling building blocks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Demo (without ``DEMdemo_``)
     - What it does
   * - `Electrostatic <../src/demo/DEMdemo_Electrostatic.cpp>`__
     - Uses rigid combined groups for charged particles and a segmented mesh rod. Owner charge wildcards and a custom force model demonstrate spatially varying electrostatic interactions.
   * - `FractureBox <../src/demo/DEMdemo_FractureBox.cpp>`__
     - Breaks a concrete bar using a custom force model with interparticle bonds and contact-history variables.
   * - `FlexibleMesh <../src/demo/DEMdemo_FlexibleMesh.cpp>`__
     - Retrieves mesh-node data and prescribes deformation while interacting with grains. Illustrates the data exchange needed for flexible-body coupling; it does not include a structural solver.
   * - `SolarSystem <../src/demo/DEMdemo_SolarSystem.cpp>`__
     - Uses a custom nonlocal gravitational force to model the Sun and planets. Demonstrates interactions without physical overlap; broadly expanded interaction ranges can create many candidate pairs.
   * - `GameOfLife <../src/demo/DEMdemo_GameOfLife.cpp>`__
     - Implements cellular-automaton evolution with DEME’s APIs to illustrate uses beyond granular mechanics.

Run the GRC preparation stages in order from the same working directory so
later stages can find earlier output directories. Before running ``WheelDP``
or ``WheelSlopeSlip``, place the final terrain checkpoint where the source
expects ``GRC_3e6.csv``. These large terrain cases are not installation smoke
tests. Modular regression tests live separately under ``src/demo/ModularTests``.

Co-simulation
-------------

DEME handles granular dynamics and exposes forces and body state to external
solvers, including direct CUDA-buffer exchange with other GPU packages.
Coupled applications can use another solver for joints, motors,
controllers, or deformable-body mechanics. See `Retrieving simulation data <python/data-access.rst>`__ for
host and CUDA-array exchange and `Installation <installation.rst>`__ for installing the C++
library.

The historical ``feature/DEME`` branch of ``projectchrono/chrono-projects``
contains Chrono coupling examples. Those examples target their corresponding
Chrono/DEME revisions and may need porting to this branch. Their CMake setup
uses ``Chrono_DIR``, ``ENABLE_PROJECTS=ON``, ``ENABLE_DEME_TESTS=ON``, and
``DEME_DIR`` after both libraries are installed.
