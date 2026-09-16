CUDA device selection
=====================

DEME runs a dynamic worker (``dT``) and a kinematic/contact-detection worker
(``kT``). Their logical CUDA devices are selected when ``DEMSolver`` is
constructed.

Count-based selection
---------------------

.. code-block:: python

   import deme

   solver = deme.DEMSolver(1)  # dT and kT both use logical device 0

``DEMSolver()`` is equivalent to ``DEMSolver(2)``. When at least two devices
are visible, it assigns dT to device 0 and kT to device 1. If only one is
visible, DEME warns and assigns both workers to device 0. Only counts 1 and 2
are accepted.

Explicit logical device IDs
---------------------------

Use a list to choose devices rather than relying on the first visible devices:

.. literalinclude:: examples/device_selection.py
   :language: python
   :linenos:

A one-element list assigns both workers to that device. In a two-element list,
the first ID is assigned to dT and the second to kT. Repeated IDs are valid, so
``DEMSolver([2, 2])`` places both workers on logical device 2.

``GetGPUDeviceIDs()`` returns ``[dT_device, kT_device]`` and is useful for
logging or choosing a destination device for direct data retrieval.

Logical versus physical IDs
---------------------------

IDs are CUDA logical device IDs as seen by this process. Environment-level
masking and reordering such as ``CUDA_VISIBLE_DEVICES`` therefore affect the
numbering. For example:

.. code-block:: console

   CUDA_VISIBLE_DEVICES=3,1 python my_simulation.py

Inside that process, logical device 0 denotes physical device 3 and logical
device 1 denotes physical device 1. An explicit ID must be non-negative and
smaller than the number of visible devices; invalid IDs fail during
construction.

The constructor allocates CUDA resources. Select visibility before starting
Python, and construct the solver only after the intended CUDA environment is
in place.
