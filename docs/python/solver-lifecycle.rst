Solver lifecycle
================

A Python simulation follows four phases. Keeping this boundary explicit avoids
rebuilding setup data during time stepping and prevents access to state that
does not exist yet.

1. Construct the solver
-----------------------

Choose worker device placement when constructing ``DEMSolver``. Device
placement cannot be changed later. See `CUDA device selection <device-selection.rst>`__.

2. Describe the system
----------------------

Before initialization:

* load materials and clump or mesh templates;
* set the domain and boundary conditions;
* add clumps, meshes, or external objects;
* configure gravity, the force model, timestep, output, and family behavior;
* create trackers for batches or objects whose state will be queried.

Setup methods return native-backed handles. Retain the solver for at least as
long as any of these handles.

3. Initialize
-------------

Call ``Initialize()`` once after setup:

.. code-block:: python

   solver.SetInitTimeStep(1.0e-5)
   solver.Initialize()

Initialization resolves cached topology and prepares runtime/JIT CUDA data.
Methods documented as post-initialization operations must not be used earlier.

Python processes use the same persistent Jitify header cache as C++ programs.
Set ``DEME_PERSISTENT_JITIFY_CACHE`` in the shell that launches Python, or set
``os.environ["DEME_PERSISTENT_JITIFY_CACHE"]`` before importing ``deme`` and
calling ``Initialize()``. Use an explicit, user-owned filename; unset the
variable to retain normal non-persistent Jitify behavior.

To compare cold and warm initialization in separate Python processes, run
``python/demos/jitify_cache_timing.py`` from the repository root twice with the
same cache filename. Remove the cache before the first run so that it measures
header discovery rather than reuse.

4. Advance, synchronize, and retrieve
--------------------------------------

``DoDynamics(duration)`` advances for a duration and may return before all
worker activity needed by a subsequent host query is synchronized.
``DoDynamicsThenSync(duration)`` is the convenient choice when the next action
reads or modifies simulation state. ``DoStepDynamics()`` advances one timestep.

Trackers provide scalar and bulk host access:

.. code-block:: python

   solver.DoDynamicsThenSync(0.01)
   first_position = tracker.Pos()
   all_positions = tracker.Positions()

They can also write directly to caller-owned CUDA allocations; see
`Retrieving simulation data <data-access.rst>`__.

Output methods may queue work. Call ``WaitForPendingOutput()`` before relying
on all requested files being complete. The solver destructor also waits for
pending output, but explicit synchronization makes program behavior clearer.
