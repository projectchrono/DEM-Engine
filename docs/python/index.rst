Python interface
================

The Python package exposes DEM-Engine's solver, setup objects, trackers, and
CUDA-aware data access. Start with the quickstart, then use the focused guides
for device placement and result retrieval.

.. toctree::
   :maxdepth: 2

   quickstart
   solver-lifecycle
   device-selection
   data-access
   api-overview
   reference

Import and compatibility
------------------------

Use the lowercase package in new code:

.. code-block:: python

   import deme

``import DEME`` re-exports the same API for backward compatibility. The
capitalized spelling should not be used in new examples.

Version
-------

The runtime version is available as:

.. parsed-literal::

   assert deme.__version__ == "|release|"

API conventions
---------------

The current bindings intentionally retain the C++ method names, such as
``SetGravitationalAcceleration`` and ``DoDynamics``. Pythonic snake-case aliases
and type stubs are planned, but are not yet part of the stable interface.

CUDA vector types exposed by the API are generally accepted as three- or
four-element Python sequences and returned as tuples. Quaternion ordering is
``(x, y, z, w)``.

Objects returned by solver setup methods frequently own or reference native
simulation state. Keep the solver alive for as long as trackers, inspectors, or
other solver-owned handles are in use.

Reference status
----------------

The `Python API overview <api-overview.rst>`__ inventories the main Python objects and points to the
task-oriented guides. The extension also contains runtime pybind11 docstrings;
use ``help(deme.DEMSolver)`` or ``help(deme.Tracker)`` in an installed
environment. The generated `C++ API reference <../cpp-api/index.rst>`__ remains the exhaustive
low-level reference for APIs shared with C++.
