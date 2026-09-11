Interactive visualization
=========================

``DEMVisualizer`` is a native OpenGL 3.3 simulation inspector with a Dear ImGui
sidebar and raylib window/input support. It is enabled in ordinary C++ builds
and Python wheels by default. Existing visualizer calls remain supported.
The viewer does not advance simulation time: each call to ``Render``
synchronously captures and displays the solver state at that moment.

Sphere components use a shared instanced mesh. Triangle geometry stays in
persistent indexed buffers in owner-local coordinates. Each display frame
uploads owner poses and colors, rather than reconstructing component positions
and triangle vertices on the CPU. Geometry uploads occur after initialization,
``Update()``, mesh deformation (including wear), or a sphere-detail change.

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
       visualizer.Render();
       if (visualizer.ShouldStep()) {
           solver.DoDynamicsThenSync(1.0 / 60.0);
       }
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
       visualizer.Render()
       if visualizer.ShouldStep():
           solver.DoDynamicsThenSync(1.0 / 60.0)

Spheres and triangles are both rendered by default. They can be controlled
independently before or during the visualization loop:

.. code-block:: python

   visualizer.SetRenderSpheres(False)
   visualizer.SetRenderTriangles(True)

``Render`` performs synchronous device-to-host data movement. Call it at the
desired display frequency rather than at every solver timestep. Do not call
``Render`` concurrently with ``DoDynamics`` from another thread.

Inspection controls
-------------------

* Left drag: orbit around the camera target. Middle drag: pan. Wheel: zoom.
* ``F``: frame visible geometry. ``Shift+F``: frame the selected owner.
* Right click: select a sphere component or mesh triangle using depth-tested
  geometry IDs. The sidebar reports its owner, family, position, orientation,
  and linear velocity. Hidden geometry cannot be selected.
* ``Space``: pause/resume. ``.``: request one displayed simulation interval while
  paused. **The application must honor** ``ShouldStep()`` **as shown above.**
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
handle input while paused. The existing C++ demos check a wall-clock display deadline between individual
solver steps, independently of file-output intervals. Their paused single-step
control advances one solver timestep. Custom applications should also keep
dynamics calls short and service the viewer regularly: a long blocking dynamics
call prevents camera and UI input from being processed, regardless of target FPS.

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
