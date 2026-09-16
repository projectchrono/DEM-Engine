DEME 3 features and migration considerations
============================================

DEME 3 adds mesh-particle contacts, rigid combined bodies, direct GPU data
exchange, persistent compiled-kernel caching, and interactive visualization.
The sections below explain when these capabilities are useful and what to
consider when moving an existing DEME 2 application.

Mesh particles and aggregated mesh contacts
-------------------------------------------

DEME 3 supports mesh–mesh contacts, allowing triangle meshes to represent
interacting grains as well as external bodies. Enable mesh contacts with other
meshes and analytical geometry using ``SetMeshUniversalContact(true)`` before
initialization. Its default is false; see `Mesh particles and combined owners <mesh-particles.rst>`__ for setup.

The clump–mesh contact scheme also changes. Rather than evaluating each
sphere–triangle contribution as an independent force interaction, the mesh
contact pipeline can combine triangle contributions into patch/island contacts
and evaluate the force model using the resulting contact geometry. The patch
force kernels consume aggregated area, normal, penetration, and contact-point
data. This provides a combined treatment of a mesh's effect on a clump instead
of treating every contacted triangle independently.

The sphere side still identifies an individual component sphere. A clump with
multiple spheres, or a mesh with distinct patches/contact islands, can therefore
have multiple resulting contacts. Aggregation does not mean exactly one contact
per clump–mesh pair, and primitive geometry checks are still needed.

``SetSimplePatchCombination`` selects grouping by patch-ID pair; the more
involved contact-island handling distinguishes contact regions.
``SetStablePatchIslandIDs`` controls stabilizing flooded island identities
across detection steps. Consult `C++ API reference <cpp-api/index.rst>`__ for these controls rather
than assuming all neighboring triangles are always merged into one contact.

This changes the contact representation used by force models and contact
history. Revalidate representative forces, torques, and bulk behavior when
porting a DEME 2 simulation, especially if its custom model relied on separate
triangle contacts. See `Internal type codes for debugging <developer/type-codes.rst>`__ when inspecting primitive
and patch contact arrays.

Using DEME 2-style mesh patches within DEME 3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Call ``SetDEME2MeshBehavior(true)`` **before loading meshes or mesh templates**
to assign every triangle of subsequently loaded meshes to its own patch. This
is useful when comparing an existing triangle-based contact setup with DEME 3's
normal patch assignments. With a solver, material, and mesh file already
available, use this setup fragment:

.. code-block:: cpp

   solver.SetDEME2MeshBehavior(true);
   auto mesh_type = solver.LoadMeshType("surface.obj", material);
   auto mesh = solver.AddMeshFromTemplate(mesh_type);

The Python equivalent is:

.. code-block:: python

   solver.SetDEME2MeshBehavior(True)
   mesh_type = solver.LoadMeshType("surface.obj", material)
   mesh = solver.AddMeshFromTemplate(mesh_type)

The default is false. Calling ``SetDEME2MeshBehavior(false)`` leaves patch
assignments unchanged for subsequent loads; it does not merge triangles back
together or undo changes on already loaded meshes/templates. In particular,
set the option before ``LoadMeshType`` when using templates.

This controls mesh patch assignment only. It neither switches to the DEME 2
solver nor restores its memory footprint or GPU backends, and it does not
replace ``SetMeshUniversalContact``. Validate forces and motion when using it
for migration comparisons.

Rigid combined bodies and geometry-wildcard migration
-----------------------------------------------------

Combined bodies provide the replacement workflow for constructions that used
geometry wildcards to distinguish parts of a body. Define members as clump or
mesh templates, assign their properties at the member-owner level, and group
them using ``LoadCombinedClumpType`` or ``LoadCombinedMeshType`` followed by
``AddCombinedFromTemplate``. Member-relative positions and orientations remain
fixed as the group moves. Owner wildcards can carry member-specific custom
properties; contact wildcards remain available for interaction history.

Geometry-wildcard-based constructions were difficult to manage because force
interactions were resolved for individual geometry pairs. Two bodies with
``n`` and ``m`` primitives can expose up to ``n * m`` candidate pairs (quadratic
when both have ``n`` primitives), with the realized contacts depending on
geometry and collision filtering. This can produce far more interactions and
per-contact state than an application intended. Wildcards themselves do not
create geometric overlap; assigning data to each geometry does not turn those
geometries into one aggregate interaction either.

Combined bodies make the ownership and rigid grouping explicit. Forces and
moments from the members contribute to the group's motion, and contacts among
members of the same group are suppressed by default. The
``SetAllowIntraCombinedOwnerContacts`` control can change that policy. Grouping
does not eliminate external primitive-pair detection or guarantee one contact
per combined-body pair. Patch aggregation and rigid grouping solve different
parts of the problem.

Migration is a modeling change, not a mechanical rename of geometry wildcards:
choose member boundaries and move custom properties to the appropriate owner or
contact variables. Combined bodies are rigid assemblies, not joints or
structural deformation models. See `Mesh particles and combined owners <mesh-particles.rst>`__ and `Core concepts <concepts.rst>`__.

On-device coupling
------------------

DEME 3 can transfer state and contact-force data directly into CUDA device
buffers for consumption by other GPU packages. Device-input APIs also support
updating simulation state from GPU buffers. This avoids a CPU round trip for
the exchanged data and enables coupling with GPU-based dynamics or continuum
solvers. Host-based access remains available.

Direct device exchange is not a promise of zero-copy access or asynchronous
execution: APIs can copy data and synchronize. Follow the pointer, capacity,
device-selection, and lifetime requirements in `Retrieving simulation data <python/data-access.rst>`__ and
the C++ reference. The Python guide includes CuPy and Warp examples.

Persistent compiled-kernel caching
----------------------------------

The runtime compiler stores serialized kernel instantiations on disk and reuses
compatible entries across processes. Repeated runs can therefore skip kernel
compilation and reduce startup time. A cold cache or changed specialization
still requires compilation; this feature does not increase the speed of the
simulation timesteps themselves.

Cache keys include generated source, compilation flags/options, kernel and
template specialization, API version, CUDA/NVRTC versions, and GPU architecture.
A matching cache entry is loaded; a missing or unreadable serialized entry is
compiled again. Source substitutions or a different toolchain/device can require
new entries. Keep the cache in a persistent writable directory, for example:

.. code-block:: console

   mkdir -p "$HOME/.cache/deme/kernels"
   export DEME_JIT_CACHE_DIR="$HOME/.cache/deme/kernels"

Without this override, DEME selects ``jit_cache`` under its runtime data path
when that path exists, otherwise ``dem-jit`` in the temporary directory. See
``src/core/utils/JitHelper.cpp`` for cache selection and key construction.

This is separate from ``DEME_PERSISTENT_JITIFY_CACHE``, the opt-in cache of
CUDA header sources used during Jitify discovery. That header cache has its own
path and invalidation considerations; see `Troubleshooting <troubleshooting.rst>`__. Clear affected
caches after changing included headers if their contents are not otherwise
reflected in the cache key.

Visualization and Python workflows
----------------------------------

The interactive visualizer displays the current simulation state when asked to
render, making it useful for inspecting geometry and motion during a run. See
`Interactive visualization <visualization.rst>`__ for examples and display requirements.

Python exposes the new mesh, combined-body, visualization, and device-data
capabilities. Start with `Python quickstart <python/quickstart.rst>`__, then consult
`Python API overview <python/api-overview.rst>`__ and the Python demos under ``python/demos/``. Python
bindings and customizable contact physics also existed in DEME 2; their presence
alone is not a DEME 3 addition.

When to stay with DEME 2
------------------------

DEME 3 currently requires NVIDIA GPUs and CUDA; it does not yet provide a
non-NVIDIA GPU backend. If you need the AMD HIP/ROCm source-build path available
in DEME 2, stay with that version. This does not imply AMD support in the
published Python wheels; follow the selected version's backend requirements.

DEME 3 may also use more memory. Its mesh-contact pipeline carries primitive
and patch contact data, aggregation intermediates, and history/identity state.
The overhead depends on the scene and settings; there is no fixed memory ratio
between versions. Turning off mesh–mesh contacts does not necessarily remove
all of that overhead or restore DEME 2's clump–mesh force treatment.

If you do not need mesh–mesh contact or the new aggregated clump–mesh scheme,
DEME 2 may remain a suitable choice, particularly for a memory-constrained or
already validated application. Compare representative workloads before moving
an established pipeline, rather than assuming identical forces or memory use.

Use the final DEME 2 release: 2.4.2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For C++, use the pinned
`v2.4.2 source tree <https://github.com/projectchrono/DEM-Engine/tree/v2.4.2>`_
from the upstream repository. A separate checkout preserves an existing DEME 3
working tree:

.. code-block:: console

   git clone --branch v2.4.2 --recurse-submodules https://github.com/projectchrono/DEM-Engine.git DEM-Engine-2.4.2
   cd DEM-Engine-2.4.2

Cloning a tag creates a detached checkout, which is fine for building. Follow
the README in that checkout for the appropriate CUDA or HIP build instructions.
Do not track the moving ``main`` branch if you intend to remain on 2.4.2.

For pyDEME, the Python distribution name is ``deme``. Explicitly install the
2.4.2 version, preferably in a separate environment from DEME 3:

.. code-block:: console

   python -m pip install "deme==2.4.2"

Keep ``deme==2.4.2`` in your application's requirements file to retain the pin.
Use 2.4.2's Python/platform/CUDA requirements and examples, rather than assuming
DEME 3's ``cuda12`` extra or new APIs are available in DEME 2. An unpinned install
or upgrade can select a newer major version.
