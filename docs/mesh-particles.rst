Mesh particles and combined owners
==================================

Contact scope
-------------

This branch supports meshes as particles as well as external bodies. Enable
mesh contacts with other meshes and analytical objects during setup:

.. code-block:: cpp

   solver.SetMeshUniversalContact(true);

The default is false: meshes then contact clumps only. This option replaces
the older README's blanket statement that mesh–mesh and mesh–analytical contacts
are unsupported. Family contact rules and geometry-specific contact settings
still apply. See :doc:`cpp-api/index` for the public API and :doc:`examples`
for mesh collision and hopper examples.

Mesh templates
--------------

``LoadMeshType`` loads a reusable mesh template without adding a body to the
simulation. It accepts a filename with an optional material, or a ``DEMMesh&``.
``AddMeshFromTemplate`` copies that template into a simulation instance and
preserves its template identity for template-level mass/MOI jitification.
Configure shared geometry and physical properties before instantiation.

The following setup fragment assumes an existing solver and material, and a
mesh asset with mass and principal moments of inertia computed for its scale:

.. code-block:: cpp

   auto mesh_type = solver.LoadMeshType("particle.obj", material);
   mesh_type->SetMass(mass);
   mesh_type->SetMOI(principal_moi);
   auto first = solver.AddMeshFromTemplate(mesh_type, make_float3(-2.f, 0.f, 0.f));
   auto second = solver.AddMeshFromTemplate(mesh_type, make_float3(2.f, 0.f, 0.f));
   first->SetFamily(0);
   second->SetFamily(0);

The C++ position parameter is ``float3`` and defaults to the origin. Create
instances individually; there is no vector-position overload or batch overload
for this method. Loading a template once avoids repeated file loading, but
instances still copy mesh setup data; this API does not promise zero-copy
geometry storage. Changes to a template do not retroactively edit its copies.

``Duplicate(mesh)`` copies a mesh and adds the copy to the solver. Use
``SetInitPos`` on the returned object during setup to place it. For newly loaded
standalone meshes, prefer ``AddMesh``; ``AddWavefrontMeshObject`` is a legacy API.
Resolve templates and instance properties before ``Initialize()``. Consult the
setup/runtime and ``Update()`` rules before adding bodies to a running solver.

Rigid combined bodies
---------------------

``LoadCombinedClumpType`` and ``LoadCombinedMeshType`` define groups with fixed
member-relative positions and orientations. ``AddCombinedFromTemplate`` places
a group at a global pose. The selected master member provides the reference
frame. Combined bodies preserve rigid relative motion; they do not implement
joints or deformable multibody dynamics. See :doc:`concepts` and the generated
C++ and Python references for units, quaternion ordering, and tracker behavior.

Mesh patches
------------

``DEMMesh::SetPatchLocations`` supplies locations relative to the mesh's implicit
center of mass, one entry per patch. ``ComputePatchLocations`` can derive them:
a single patch uses the origin, while multiple patches use averages of their
triangle centroids. These are setup quantities, not global body positions.
The C++ reference documents patch geometry and contact controls.
