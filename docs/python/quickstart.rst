Python quickstart
=================

Install DEME in an environment with a compatible NVIDIA driver and CUDA
runtime, then verify the package and version:

.. code-block:: console

   python -c "import deme; print(deme.__version__)"

The following complete example creates a material, a bounded domain, and one
spherical clump, then advances the simulation:

.. literalinclude:: examples/sphere_drop.py
   :language: python
   :linenos:

Run it with:

.. code-block:: console

   python docs/python/examples/sphere_drop.py

The solver constructor initializes CUDA worker resources, so even importing
successfully is not sufficient to run a simulation without a visible,
supported NVIDIA GPU. See `CUDA device selection <device-selection.rst>`__ when the process can see more
than one GPU.

The example uses ``DoDynamicsThenSync`` because the position is read
immediately afterward. For longer simulations, asynchronous ``DoDynamics``
calls can overlap host work; synchronize before reading results or exiting.

More Python demos
-----------------

The repository's ``python/demos`` directory includes ports of
``DEMdemo_SingleSphereCollide``, ``DEMdemo_BallDrop``, and
``DEMdemo_Centrifuge``. They demonstrate cohesive contact, runtime particle
insertion, a meshed projectile, and a rotating analytical drum:

.. code-block:: console

   python python/demos/single_sphere_collide.py --smoke-test
   python python/demos/ball_drop.py --smoke-test
   python python/demos/centrifuge.py --smoke-test

Run from a checkout with the Python package installed. Each script supports
``--help``, ``--device``, ``--duration``, and ``--output-dir``. See
``python/demos/README.md`` for full runs, output visualization, and differences
from the larger C++ examples. Smoke runs still require a GPU and first-time
kernel compilation.

Where to go next
----------------

* `Solver lifecycle <solver-lifecycle.rst>`__ explains which operations belong before and after
  ``Initialize()``.
* `Retrieving simulation data <data-access.rst>`__ covers host-returning tracker methods and direct retrieval
  into CUDA arrays.
* `Python API overview <api-overview.rst>`__ maps common tasks to the Python objects and methods that
  implement them.
