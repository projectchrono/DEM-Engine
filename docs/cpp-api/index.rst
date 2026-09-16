C++ API reference
=================

The reference below is generated from the documented public C++ headers. Worker
internals and CUDA implementation headers are intentionally excluded.

Solver
------

.. doxygenclass:: deme::DEMSolver
   :project: DEME
   :members:
   :protected-members:

Tracking and inspection
-----------------------

.. doxygenclass:: deme::DEMTracker
   :project: DEME
   :members:

.. doxygenclass:: deme::DEMInspector
   :project: DEME
   :members:

Materials and force models
--------------------------

.. doxygenclass:: deme::DEMMaterial
   :project: DEME
   :members:

.. doxygenclass:: deme::DEMForceModel
   :project: DEME
   :members:

Geometry and initialization
---------------------------

.. doxygenclass:: deme::DEMInitializer
   :project: DEME
   :members:

.. doxygenclass:: deme::DEMClumpBatch
   :project: DEME
   :members:

.. doxygenclass:: deme::DEMExternObj
   :project: DEME
   :members:

.. doxygenclass:: deme::DEMMesh
   :project: DEME
   :members:

``deme::DEMMeshConnected`` is a compatibility type alias for
``deme::DEMMesh`` and therefore does not have a separate Doxygen class page.

Sampling
--------

.. doxygenclass:: deme::PDSampler
   :project: DEME
   :members:

.. doxygenclass:: deme::GridSampler
   :project: DEME
   :members:

.. doxygenclass:: deme::HCPSampler
   :project: DEME
   :members:
