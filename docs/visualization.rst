Interactive visualization
=========================

``DEMVisualizer`` is a native OpenGL 3.3 simulation inspector with a Dear ImGui
sidebar and raylib window/input support. It is enabled in ordinary C++ builds
and Python wheels by default. Existing visualizer calls remain supported.
The viewer does not advance simulation time: each call to ``Render``
synchronously captures and displays the solver state at that moment.

Recommended scope
-----------------

Use the interactive viewer for **proof-of-concept (PoC) work and small-scale
simulations**: checking geometry placement, inspecting motion, and exploring
solver settings. It is not recommended as an always-on viewer for large
production runs or throughput benchmarks. For those workloads, run without
interactive rendering, write output at a suitable interval, and inspect it
later in ParaView. There is no universal particle-count cutoff: mesh complexity,
sphere detail, hardware, and display frequency all affect the cost.

Sphere components use a shared instanced mesh. Triangle geometry stays in
persistent indexed buffers in owner-local coordinates. Each display frame
uploads owner poses and colors, rather than reconstructing component positions
and triangle vertices on the CPU. Geometry uploads occur after initialization,
``Update()``, mesh deformation (including wear), or a sphere-detail change.

Writing a responsive visualization loop
---------------------------------------

``Render()`` handles mouse and keyboard input as well as drawing. The target FPS
is a frame-rate limit, not a background event loop: it cannot make the window
respond while your application is inside a blocking solver call or file write.
A simulation-time interval such as ``DoDynamicsThenSync(1.0 / 60.0)`` can take
seconds of wall time to compute; it does not imply a 60 Hz display.

Use short dynamics advances and a **wall-clock** display deadline, independently
of simulation-time file-output intervals. The examples below advance one solver
timestep per call. They check the deadline between calls, keep drawing while
paused, and consume ``ShouldStep()`` exactly once per intended advance. A paused
single-step therefore advances one timestep. Check for window closure again
after rendering so closing the viewer does not advance another step.

Schedule the next deadline **after rendering finishes**. The renderer's FPS
limiter may sleep for a whole frame; scheduling from the start can make every
subsequent timestep immediately trigger another render, leaving little time
for physics. A 16 ms work interval below is a starting point, not a guaranteed
60 FPS rate. Increase it to favor simulation throughput. Keep your physical
timestep unchanged when tuning the display frequency.

C++ usage
~~~~~~~~~

After configuring and initializing ``solver``:

.. code-block:: cpp

   #include <chrono>
   #include "DEM/API.h"
   #include "DEM/utils/DEMVisualizer.h"

   deme::DEMVisualizer visualizer(solver);
   visualizer.SetTargetFPS(60);
   visualizer.Initialize();

   using Clock = std::chrono::steady_clock;
   auto next_display = Clock::now();
   const auto work_interval = std::chrono::milliseconds(16);
   // Match the solver's internal float timestep; a slightly larger double
   // duration could request an extra step. This example uses a fixed timestep.
   const float dt = static_cast<float>(solver.GetTimeStepSize());

   while (visualizer.Run()) {
       if (visualizer.IsPaused() || Clock::now() >= next_display) {
           visualizer.Render();
           next_display = Clock::now() + work_interval;
       }
       if (!visualizer.Run())
           break;
       if (visualizer.ShouldStep())
           solver.DoDynamics(dt);
   }
   solver.DoDynamicsThenSync(0.0);  // Join both workers at the end.
   visualizer.Close();

Python usage
~~~~~~~~~~~~

After configuring and initializing ``solver``:

.. code-block:: python

   import time
   import numpy as np
   import deme

   visualizer = deme.DEMVisualizer(solver)
   visualizer.SetTargetFPS(60)
   visualizer.Initialize()

   next_display = time.monotonic()
   work_interval = 0.016
   # Round to the internal float timestep before passing a Python float.
   # This example uses a fixed timestep.
   dt = float(np.float32(solver.GetTimeStepSize()))

   try:
       while visualizer.Run():
           if visualizer.IsPaused() or time.monotonic() >= next_display:
               visualizer.Render()
               next_display = time.monotonic() + work_interval
           if not visualizer.Run():
               break
           if visualizer.ShouldStep():
               solver.DoDynamics(dt)
   finally:
       solver.DoDynamicsThenSync(0.0)
       visualizer.Close()

``DoDynamics`` returns after the dynamic worker finishes the requested advance;
these loops capture its state on the same application thread afterwards. They
do not need to reset both workers with ``DoDynamicsThenSync`` at every display
refresh. Do not run ``Render`` concurrently with dynamics from another thread.
For a finite simulation, also add your simulation-end condition to the loop.
If you change the solver timestep at runtime, refresh ``dt`` accordingly.

Spheres and triangles are both rendered by default. They can be controlled
independently before or during the visualization loop:

.. code-block:: python

   visualizer.SetRenderSpheres(False)
   visualizer.SetRenderTriangles(True)

Performance tradeoffs
---------------------

Persistent geometry reduces rebuilding work, but the viewer is not free:

* Each ``Render`` synchronously transfers owner state to the CPU, prepares
  colors, and uploads frame data to OpenGL. Speed coloring and inspecting a
  selected owner also request velocity data. Paused redraws still capture state.
* Sphere detail and mesh facet counts affect drawing cost and GPU memory use.
  Geometry changes, including mesh deformation and wear, trigger geometry
  uploads. Hiding geometry reduces drawing work but does not eliminate the
  frame capture for all owners.
* Frequent calls across the solver boundary, particularly from Python, add
  overhead. Drawing, state transfers, and the FPS limiter share the application's
  time with simulation. More responsive interaction can mean a substantially
  slower overall run, even with few particles.
* A long individual solver step, expensive output operation, or large frame
  transfer can still stall input. The wall-clock deadline is checked only when
  control returns to the application; rendering is not asynchronous.

Measure representative runs with and without calls to ``Render`` using the
same physics and output settings. Compare wall time per simulated second,
not just the viewer's displayed FPS. In one development run,
``SingleSphereCollide`` took about 287 seconds with output-frame-only rendering
and 521 seconds with frequent interactive updates (about 1.8 times as long).
That illustrates the tradeoff, not a portable benchmark or expected slowdown
for other scenes or machines.

For PoC work, start with a small scene, reduce sphere detail, and lengthen the
work interval if throughput matters more than camera responsiveness. Pausing
is useful for inspection, but still incurs redraw costs. For large or long
runs, omit the viewer loop and use file output and postprocessing instead.
Removing the viewer at build time is optional; see the headless build option
below. Moving rendering to a background thread is not a supported shortcut.

Inspection controls
-------------------

* Left drag: orbit around the camera target. Middle drag: pan. Wheel: zoom.
* ``F``: frame visible geometry. ``Shift+F``: frame the selected owner.
* Right click: select a sphere component or mesh triangle using depth-tested
  geometry IDs. The sidebar reports its owner, family, position, orientation,
  and linear velocity. Hidden geometry cannot be selected.
* ``Space``: pause/resume. ``.``: request one application-defined dynamics advance while
  paused (one solver timestep in the loops above). **The application must honor** ``ShouldStep()`` **as shown above.**
  It consumes one pending step request; call it once per intended advance.
* ``F12`` or the screenshot button: save ``deme-screenshot.png`` in the current
  directory. ``RequestScreenshot(path)`` saves the next rendered frame elsewhere.
* ``Esc``: close the window. The three existing visualizer demos continue their
  simulation/output after closing the window.

The sidebar controls sphere/mesh visibility, wireframe, the XY reference grid,
sphere detail, per-family visibility/colors, and coloring by owner height or
linear speed. Scalar modes include an automatic or manually specified range
and a color legend. Height uses the owner center, not individual component
centers. Values are in the simulation's length/time units (the labels assume SI).
Analytical boundaries currently remain outside this viewer's geometry interface;
use analytical VTK output for those surfaces.

These controls also have C++/Python entry points: ``SetPaused``, ``RequestStep``,
``SetFamilyVisible``, ``SetColorMode``, ``FrameAll``, ``FrameSelected``, and
``PickAt``. Python color modes are ``deme.VisualizerColorMode.FAMILY``, ``HEIGHT``,
and ``SPEED``; C++ uses ``deme::DEMVisualizerColorMode``. ``PickAt`` uses logical
window coordinates and the last rendered frame. ``GetSelectedOwner`` returns
``NULL_BODYID`` (the maximum 32-bit unsigned value) when nothing is selected;
``GetSelectedGeometryID`` returns ``SIZE_MAX`` in that case. Selected geometry
indices are global sphere or triangle indices, distinguished by
``IsSelectedSphere()``.

Camera settings supplied before initialization are preserved; otherwise the
first frame automatically fits the scene. The camera, frame buffers, and GUI
must be used on the thread owning the window. Only one native viewer window
may be open in a process. Pause does not block ``Render``; keep rendering to
handle input while paused. The existing visualizer demos illustrate display
updates between solver calls, independently of file-output intervals. For
custom applications, use the loop pattern above and check ``ShouldStep()``
before every advance, including advances between display deadlines.

Scene/frame interface
---------------------

``GetVisualizationScene()`` returns local sphere offsets/radii, local triangle
vertices, owner mappings, and a geometry revision. Cache it until the revision
in ``GetVisualizationFrame(frame, include_velocities=false)`` changes.
``DEMVisualizationFrame`` contains reusable vectors indexed by owner ID:
positions, quaternions (x, y, z, w), families, and optional velocities. Python
uses ``deme.VisualizationFrame()`` and the same solver methods. Python field
access converts vectors to Python values; the native viewer uses the reusable
C++ buffers directly.

Capture scene/frame data only at a synchronized solver boundary. Runtime family
or rigid-pose changes update the frame without invalidating geometry. Runtime
mesh changes through solver/tracker APIs invalidate the scene even when the
facet count is unchanged. ``GetVisualizationSnapshot`` remains available for
consumers needing expanded world-space geometry.

This implementation uses synchronous host state transfer. CUDA/OpenGL interop,
asynchronous rendering, sphere impostors, contact overlays, and force dragging
are not part of this viewer version.

Verification
------------

``DEMTest_VisualizationScene`` checks local-to-world transforms, combined-owner
members, family/velocity updates, deformation invalidation, and insertion.
``DEMTest_Visualizer`` additionally requires CUDA and a display; it exercises
rendering, occlusion-aware picking, filters, stepping requests, window resize,
and close/reopen. Pass a PNG filename to save a frame for inspection::

   ./build/bin/DEMTest_Visualizer /tmp/deme-visualizer.png

The existing ``DEMdemo_SingleSphereCollide``, ``DEMdemo_MeshCollide``, and
``DEMdemo_MeshFalling`` continue to use the viewer and honor pause/single-step.

Linux build requirements
------------------------

Dear ImGui v1.91.9b is vendored with its MIT license; builds require no additional
download for the UI. The source build uses raylib's bundled GLFW backend and requires
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
