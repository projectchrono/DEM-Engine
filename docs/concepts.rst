Core concepts
=============

For more information about DEME's design concepts, see the
`DEM-Engine paper in Computer Physics Communications (2024)
<https://doi.org/10.1016/j.cpc.2024.109196>`_. This page summarizes the concepts
used by the current API; the paper provides the background on the solver's
design and computational approach.

Setup and runtime
-----------------

DEME caches materials, templates, geometry, family prescriptions, output
choices, and force-model inputs before ``DEMSolver::Initialize``. Initialization
allocates device state and prepares runtime-compiled kernels. Avoid rebuilding
setup data in high-frequency dynamics calls.

Dynamic and kinematic workers
-----------------------------

DEME uses two cooperating workers:

``dT``
   Integrates body state and evaluates contact forces.

``kT``
   Performs collision detection and supplies potential contact pairs.

Their asynchronous handoff is part of the solver's performance model. User code
should use the public solver API rather than directly synchronizing worker data.

Owners and members
------------------

An owner represents a simulated body. Members are the collision geometries
belonging to that owner. Clumps, analytical objects, meshes, and combined owners
have different setup and tracking behavior. A combined owner preserves rigid
relative motion among its members.

Families and prescriptions
--------------------------

Families group owners for contact rules and prescribed motion. Prescriptions
can constrain position, velocity, orientation, or acceleration. Their order
relative to integration and user-added acceleration matters.

Frames and quaternions
----------------------

User-facing positions and linear velocities are generally in the global frame.
Some angular quantities are stored in the owner's local principal frame.
Quaternion values use ``(x, y, z, w)`` ordering. Check each API description
before combining local and global quantities.

Runtime CUDA compilation
------------------------

DEME uses NVRTC and Jitify to compile kernels and user force models at runtime.
An installed package therefore needs compatible CUDA headers, NVRTC, packaged
kernel sources, and a supported NVIDIA driver—not merely a loadable Python
extension.
