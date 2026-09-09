Python API overview
===================

DEME currently preserves its established C++-style method names in Python.
This page maps common tasks to the principal bound objects; runtime docstrings
and the :doc:`../cpp-api/index` provide signature-level detail.

Core objects
------------

``DEMSolver``
   Owns workers and simulation state. It loads setup data, configures the
   solver, initializes, advances time, creates trackers and inspectors, and
   writes output.

``DEMMaterial``
   Stores material properties loaded through ``DEMSolver.LoadMaterial``.

``DEMClumpTemplate`` and ``DEMClumpBatch``
   Describe clump topology and a group of instances. A batch returned by
   ``AddClumps`` can be passed to ``Track``.

``DEMMesh`` and ``DEMExternObj``
   Represent mesh particles/bodies and analytical external geometry. See
   :doc:`../mesh-particles` for mesh templates and contact scope.

``Tracker`` (native ``DEMTracker``)
   Reads or modifies a tracked batch/object after initialization. See
   :doc:`data-access`.

``DEMInspector``
   Queries aggregate or spatial simulation properties.

``DEMForceModel``
   Selects or customizes the contact force calculation and its material or
   wildcard requirements.

``DEMVisualizer``
   Displays the current solver state on each explicit ``Render`` call without
   advancing the simulation. See :doc:`../visualization`.

Common task map
---------------

.. list-table::
   :header-rows: 1

   * - Task
     - Starting methods
   * - Select GPUs
     - ``DEMSolver(...)``, ``GetGPUDeviceIDs``
   * - Load material
     - ``LoadMaterial``
   * - Define the domain
     - ``InstructBoxDomainDimension``, ``InstructBoxDomainBoundingBC``
   * - Create particles
     - ``LoadSphereType``, clump-template loaders, ``AddClumps``
   * - Create geometry
     - ``LoadMeshType``, mesh/external-object add methods, ``AddBCPlane``
   * - Enable mesh–mesh and mesh–analytical contacts
     - ``SetMeshUniversalContact(True)`` during setup
   * - Create rigid combined bodies
     - ``LoadCombinedClumpType``, ``LoadCombinedMeshType``, ``AddCombinedFromTemplate``
   * - Configure motion
     - gravity, family prescriptions, timestep methods
   * - Start simulation
     - ``Initialize``
   * - Advance simulation
     - ``DoDynamics``, ``DoDynamicsThenSync``, ``DoStepDynamics``
   * - Inspect or control objects
     - ``Track`` and ``Tracker`` methods
   * - Retrieve CUDA data
     - ``Tracker`` methods ending in ``ToDevice``
   * - Write output
     - ``WriteSphereFile``, mesh/contact output methods,
       ``WaitForPendingOutput``
   * - Render interactively
     - ``DEMVisualizer``, ``Render``, ``SetRenderSpheres``,
       ``SetRenderTriangles``

Discovering installed signatures
--------------------------------

The pybind11 extension exposes method signatures and docstrings directly:

.. code-block:: python

   import deme

   help(deme.DEMSolver)
   help(deme.Tracker)
   print(deme.DEMSolver.Initialize.__doc__)

Static type stubs and automatically generated Python API pages are not yet
shipped. Consequently, IDE completion may be incomplete even though a method
is available at runtime.
