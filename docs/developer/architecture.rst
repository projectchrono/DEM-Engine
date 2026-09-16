Developer architecture
======================

Public API boundary
-------------------

Public operations enter through ``DEMSolver`` and related setup objects. New
simulation state normally follows this path:

#. public API or cached initializer data;
#. dynamic and kinematic worker storage;
#. device-data pointer binding;
#. host-to-device transfer;
#. JIT substitutions or kernel arguments;
#. runtime update behavior.

Missing one stage can produce failures only after initialization or only on a
second GPU.

CUDA and JIT kernels
--------------------

Kernel launch argument types must match exactly. Validate and explicitly narrow
host values at the launch boundary. Treat block size as correctness-sensitive:
large JIT kernels can exceed register or launch limits even when a smaller
kernel accepts the same configuration.

Synchronization
---------------

The dT/kT streams, events, and handoff buffers implement the intended
synchronization. Avoid adding device-wide synchronization to hot paths. When
cross-device data is transferred, preserve the lifetime of the source storage
until the destination operation has completed.

Adding Python bindings
----------------------

When exposing a public API:

#. bind the method in ``src/DEM/python/bindings.cpp``;
#. use explicit overload casts;
#. provide parameter names and a practical docstring;
#. document units, frame conventions, and object lifetime;
#. add an installed-wheel test, not only an in-tree import test;
#. update the Python guide when behavior differs from C++.
