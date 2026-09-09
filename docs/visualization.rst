Interactive visualization
=========================

``DEMVisualizer`` is a lightweight, step-wise viewer built with the bundled
raylib submodule. It is enabled in ordinary C++ builds and Python wheels by
default. The viewer does not advance simulation time: each call to ``Render``
synchronously captures and displays the solver state at that moment.

C++ usage
---------

.. code-block:: cpp

   #include <DEM/API.h>
   #include <DEM/utils/DEMVisualizer.h>

   deme::DEMSolver solver;
   // Configure the solver, add geometry, then initialize it.
   solver.Initialize();

   deme::DEMVisualizer visualizer(solver);
   visualizer.Initialize();

   while (visualizer.Run()) {
       solver.DoDynamicsThenSync(1.0 / 60.0);
       visualizer.Render();
   }

Python usage
------------

.. code-block:: python

   import deme

   solver = deme.DEMSolver(1)
   # Configure the solver, add geometry, then initialize it.
   solver.Initialize()

   visualizer = deme.DEMVisualizer(solver)
   visualizer.Initialize()

   while visualizer.Run():
       solver.DoDynamicsThenSync(1.0 / 60.0)
       visualizer.Render()

Spheres and triangles are both rendered by default. They can be controlled
independently before or during the visualization loop:

.. code-block:: python

   visualizer.SetRenderSpheres(False)
   visualizer.SetRenderTriangles(True)

``Render`` performs synchronous device-to-host data movement. Call it at the
desired display frequency rather than at every solver timestep. Do not call
``Render`` concurrently with ``DoDynamics`` from another thread.

Linux build requirements
------------------------

The source build uses raylib's bundled GLFW backend and therefore requires
the standard X11 and OpenGL development headers. On Debian or Ubuntu:

.. code-block:: console

   sudo apt-get install libx11-dev libxrandr-dev libxinerama-dev \
       libxcursor-dev libxi-dev libgl1-mesa-dev

The GitHub wheel workflow installs the equivalent packages in its manylinux
container. At runtime, a graphical display and OpenGL implementation must be
available. WSL users should run under WSLg or configure ``DISPLAY`` to an X
server.

The visualizer can be omitted for a headless source build with:

.. code-block:: console

   cmake -S . -B build -DDEME_BUILD_VISUALIZER=OFF

Postprocessing with ParaView
----------------------------

Load output ``.vtk`` mesh files directly. For component-sphere CSV output from
``WriteSphereFile``, apply **Table To Points** using the ``X``, ``Y``, and ``Z``
columns. Then apply **Glyph**, choose **Sphere**, select ``r`` as the scale
array, set the scale factor to 2 (radii become diameters), and choose
**All Points**. Reduce sphere resolution when visualizing large datasets.
