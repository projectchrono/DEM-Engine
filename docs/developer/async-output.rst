Asynchronous output: ownership and synchronization
==================================================

Scope and current status
------------------------

This page describes file-output ownership and synchronization requirements.
It separates protections already implemented from the contract intended for all
writers. Asynchronous output does not make the solver API safe for arbitrary
concurrent calls from multiple application threads.

Current writer ownership
------------------------

.. list-table::
   :header-rows: 1
   :widths: 23 42 35

   * - Writer
     - Background inputs
     - Protection and remaining limitations
   * - ``WriteContactFile``
     - Owned ``ContactInfoContainer``, copied output flags and wildcard names.
     - Snapshot completed before return. Subsequent dynamics cannot change the
       writer's inputs. Includes the CSV fallback for binary output and the
       potential-pair convenience API.
   * - ``WriteAnalyticalFile``
     - Owned vector of transformed analytical components; copied domain bounds
       and tessellation resolution.
     - Snapshots inputs before launching the writer.
   * - ``WriteSphereFile``
     - Shared dT host arrays and output metadata, accessed through the solver.
     - CSV, binary fallback and VTK still depend on those inputs remaining
       unchanged until the writer finishes.
   * - ``WriteClumpFile``
     - Shared dT host arrays and output metadata; copied numeric precision.
     - CSV and binary fallback still depend on shared inputs remaining unchanged.
   * - ``WriteMeshFile``
     - Shared host arrays, mesh caches and output metadata; some options copied.
     - VTK, STL and PLY do not yet own complete snapshots. Refreshing a cache
       before launch does not isolate it from later mutation.

See the implementations in `APIPublic.cpp <../../src/DEM/APIPublic.cpp>`__, the
host readers in `dT.cpp <../../src/DEM/dT.cpp>`__, and the snapshot-only contact
formatter in `ContactOutput.hpp <../../src/DEM/utils/ContactOutput.hpp>`__.

Execution and synchronization
-----------------------------

The solver has one pending output thread. Each file-output method first calls
``WaitForPendingOutput()``, which joins the previous writer. Output jobs therefore
serialize with each other; they do not each run on an independent concurrent
writer thread.

The shared-array methods then download the necessary GPU values into the solver's
existing host storage and launch a thread capturing ``this``. Formatting and disk
writing take place on that thread. Capturing ``this`` copies a pointer, not the
arrays. Local vectors or string streams constructed by the writer do not protect
the shared inputs while the writer is constructing them.

The snapshot methods instead finish reading solver state on the calling thread
and hand owned data to the background thread. Contact output performs filtering,
ID resolution and coordinate transformations while constructing this snapshot;
CSV formatting and disk I/O remain asynchronous.

``DoDynamics()`` does not join pending output. ``DoDynamicsThenSync()`` synchronizes
the simulation workers, not the file writer. Some mutating operations, including
``Update()`` and the mesh-deformation APIs, explicitly wait for output. Those are
local safeguards, not a general lock around every host access or mutation.

Data categories and treatment
-----------------------------

Device arrays
~~~~~~~~~~~~~

File writers use CPU-readable data downloaded before launch. Once a transfer has
completed, changing a separate device allocation does not itself change its host
mirror. This is why overlapping GPU dynamics and output can work when host inputs
remain stable. Transfer completion alone does not guarantee that the host mirror
will remain unchanged afterward.

Shared host mirrors
~~~~~~~~~~~~~~~~~~~

The host portions of ``DualArray`` objects belong to the solver. A later download,
host-side setter, resize or storage replacement can change the same memory a
writer is reading. Resizing can also invalidate references or pointers. A host
mirror is not an output-job snapshot merely because it is separate from device
memory. Shared storage requires protection until the last reader finishes.

Counts, mappings and configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Counts, owner/geometry mappings, family filters, output flags, wildcard names and
coordinate-conversion parameters are part of the output's inputs too. Arrays and
metadata must describe the same frame. Copying a count alone cannot protect an
array that can be overwritten or reallocated; copying an array alone cannot
protect interpretation through mutable mappings or settings.

Contact output resolves these dependencies before launch and copies its
column configuration. Its background formatter has no solver access. The shared
sphere, clump and mesh writers still consult solver metadata while formatting.

Mesh caches and setup geometry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``WriteMeshFile()`` synchronizes deformed mesh caches before starting its writer,
but the writer still reads solver-owned meshes and owner transforms. Treat vertex
and connectivity vectors, patch data and any output attributes as dependencies.
Setup-time data may be shared only if its lifetime and immutability are guaranteed
for the entire write. Do not infer that guarantee merely because a vector usually
stays unchanged during dynamics. Direct mutation of exposed setup/cache objects
also needs to be considered when reviewing a writer.

Job-owned snapshots
~~~~~~~~~~~~~~~~~~~

A complete snapshot owns the values needed to interpret and serialize a frame.
It must not retain pointers, references or views into mutable solver storage.
Contact output holds an independently populated container through a const shared
pointer; analytical output moves its component vector into the writer. These
jobs keep their storage alive until formatting and writing finish.

Contact snapshot construction
-----------------------------

``WriteContactFile()`` calls ``generateContactInfo()`` before launching the
writer and captures the resulting owned container, output flags and wildcard
names. Both normal CSV and the binary-to-CSV fallback use the same snapshot-only
formatter. Invalid contact types are rejected during snapshot construction.

Remaining risks and caller guidance
-----------------------------------

Sphere, clump and mesh output have not yet been converted to complete snapshots.
Their shared inputs can cause inconsistent frames, invalid indexing or invalid
memory access if changed concurrently. This is an ownership risk identified from
the implementation, not a claim that every ordinary dynamics call reproduces a
failure in each writer. Stable owner counts and GPU-only updates can leave their
host inputs undisturbed, and some APIs already wait before mutation.

Until those writers are isolated, callers can explicitly finish output before
resuming dynamics or performing operations that could change its shared inputs:

.. code-block:: cpp

   solver.WriteMeshFile("frame.vtk");  // Same precaution for sphere/clump output.
   solver.WaitForPendingOutput();
   solver.DoDynamicsThenSync(duration);

Contact and analytical output do not need that wait solely to protect their
snapshots from subsequent dynamics. All asynchronous outputs still need a
completion wait before the application reads, checks, deletes or otherwise
consumes the destination file. An owned snapshot guarantees input stability;
it does not make the file complete when the API returns.

Required contract and follow-up work
------------------------------------

Every background writer must either own its complete inputs or have explicit
synchronization protecting all shared inputs for their entire use. The preferred
follow-up is to give sphere, clump and mesh output complete owned snapshots,
including any metadata needed for serialization. These writers currently retain
the shared-storage behavior described above.

Build snapshots while the relevant solver state is stable, then pass only those
snapshots to standalone formatting code. Preserve file layouts, filtering,
coordinate conventions and format fallbacks. Snapshot creation adds foreground
CPU work, but should leave formatting and disk writing asynchronous and avoid
unnecessary device-wide synchronization in dynamics. Assess memory use for large
outputs, including snapshot and formatting buffers.

Ownership does not solve every output concern. File-open/write failures and
exceptions on the background thread need a separate error-propagation policy;
the snapshot mechanism does not provide one. Completion waits, solver lifetime and the
supported application-thread calling model also remain part of the contract.

Regression coverage
-------------------

`DEMTest_ContactOutputSnapshot.cpp <../../src/demo/ModularTests/DEMTest_ContactOutputSnapshot.cpp>`__
uses a POSIX FIFO to block the writer opening its destination until after dynamics
changes the contact list. It compares the delayed output byte-for-byte with a
reference from the requested frame. Coverage includes contact-list growth, CSV,
binary fallback with potential pairs, force filtering to zero rows and contact
wildcards. The FIFO test skips on Windows; the production snapshot mechanism is
platform-independent.

The test deliberately has no completion wait between queuing the delayed contact
output and advancing dynamics. Its other waits are required to finish the
reference file before reading it and to join the delayed writer after draining
the FIFO. Removing those waits would not be a valid test of snapshot safety.

Future writer conversions should receive equivalent delayed-output tests covering
their mutable arrays, metadata and caches. Do not use sleeps as the only mechanism
for provoking a race, and do not treat contact-test success as coverage of the
remaining shared-array writers.
