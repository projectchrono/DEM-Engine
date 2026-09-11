Internal type codes for debugging
=================================

Use these tables when inspecting contact arrays, owner state, or debugger
output. The source of truth is ``src/DEM/Defines.h``; storage types are defined
in ``src/DEM/VariableTypes.h``. Values describe this checkout and should not be
assumed identical in older releases. Prefer symbolic constants in application
code.

Contact types
-------------

``contact_t`` is an unsigned 8-bit integer. These codes appear in
``contactTypePrimitive`` and ``contactTypePatch``. They describe the geometry
pair, not the owner IDs or the position of a contact in an array.

.. csv-table::
   :header: "Decimal", "Hex", "Internal name", "Geometry A", "Geometry B"
   :widths: 8 8 44 20 20

   0, 0x00, ``NOT_A_CONTACT``, None, None
   17, 0x11, ``SPHERE_SPHERE_CONTACT``, Sphere, Sphere
   18, 0x12, ``SPHERE_TRIANGLE_CONTACT``, Sphere, Triangle
   20, 0x14, ``SPHERE_ANALYTICAL_CONTACT``, Sphere, Analytical component
   34, 0x22, ``TRIANGLE_TRIANGLE_CONTACT``, Triangle, Triangle
   36, 0x24, ``TRIANGLE_ANALYTICAL_CONTACT``, Triangle, Analytical component

``SPHERE_MESH_CONTACT`` is a legacy alias for ``SPHERE_TRIANGLE_CONTACT``
(value 18), not an additional contact type. ``NUM_SUPPORTED_CONTACT_TYPES``
is 5: it is a count, not the maximum code. ``ALL_CONTACT_TYPES`` lists the five
supported pairs; ``isSupportedContactType`` validates a code.

Encoding and decoding
~~~~~~~~~~~~~~~~~~~~~

The high four bits hold geometry A's type, and the low four bits hold geometry
B's type:

.. code-block:: text

   contact_code = (geometry_A << 4) | geometry_B
   geometry_A = contact_code >> 4
   geometry_B = contact_code & 0x0F

For example, 36 is ``0x24``: triangle (2) against analytical component (4).
Order matters: ``0x21`` is not a supported substitute for ``0x12``. Packing two
geometry codes does not by itself make the resulting pair supported.

The C++ helpers are ``encodeType``, ``decodeTypeA``, and ``decodeTypeB``. Their
packing width depends on the template argument types. When decoding an
integer from a log, cast it to ``deme::contact_t`` first so the helpers use
four-bit fields rather than fields sized for ``int``:

.. code-block:: cpp

   const auto type = static_cast<deme::contact_t>(36);
   const auto type_a = deme::decodeTypeA(type);  // GEO_T_TRIANGLE
   const auto type_b = deme::decodeTypeB(type);  // GEO_T_ANALYTICAL

Geometry and owner codes
------------------------

``geoType_t`` and ``ownerType_t`` are unsigned 8-bit integers. A geometry is a
collision component belonging to an owner; a clump can contain many spheres,
and a mesh can contain many triangles.

.. csv-table::
   :header: "Value", "Geometry code", "Internal owner code", "Meaning"
   :widths: 8 30 32 30

   0, ``NOT_A_GEO``, ``NOT_A_OWNER``, No geometry / no owner type
   1, ``GEO_T_SPHERE``, ``OWNER_T_CLUMP``, Sphere component / clump owner
   2, ``GEO_T_TRIANGLE``, ``OWNER_T_MESH``, Triangle component / mesh owner
   4, ``GEO_T_ANALYTICAL``, ``OWNER_T_ANALYTICAL``, Analytical component / analytical owner

The analytical code is 4, not 3; these values also participate in bitwise
operations. ``ownerTypes`` stores the internal owner codes. Combined owners
have no separate ``OWNER_T_COMBINED`` code: rigid grouping does not introduce
a fourth geometry or owner type. See :doc:`../concepts` for owners and members.

Do not confuse these codes with ``OWNER_TYPE``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The separate C++ ``enum class OWNER_TYPE``, also exposed in Python, has the
following underlying values. It is not numerically interchangeable with the
internal ``OWNER_T_*`` constants:

.. csv-table::
   :header: "Enum", "Enum value", "Corresponding internal code", "Internal value"
   :widths: 35 10 40 15

   ``OWNER_TYPE::CLUMP``, 0, ``OWNER_T_CLUMP``, 1
   ``OWNER_TYPE::ANALYTICAL``, 1, ``OWNER_T_ANALYTICAL``, 4
   ``OWNER_TYPE::MESH``, 2, ``OWNER_T_MESH``, 2

For Python enum access, use ``deme.OWNER_TYPE.CLUMP`` and the analogous names.
Check the field or API's declared type before interpreting a numeric value.

Analytical geometry subtypes
----------------------------

``objType_t`` is an unsigned 8-bit integer identifying the shape of an
analytical component. All these shapes have geometry code
``GEO_T_ANALYTICAL`` (4); their subtype is not packed into the contact code.

.. csv-table::
   :header: "Value", "Internal name", "Shape"
   :widths: 10 50 40

   0, ``ANAL_OBJ_TYPE_PLANE``, Plane
   1, ``ANAL_OBJ_TYPE_PLATE``, Plate
   2, ``ANAL_OBJ_TYPE_CYL_INF``, Infinite cylinder
   3, ``ANAL_OBJ_TYPE_CONE_INF``, Infinite cone
   4, ``ANAL_OBJ_TYPE_CONE``, Finite cone

This table lists declared subtype codes; it does not assert that every shape
supports every contact path. In particular, subtype 0 means a plane, not an
invalid geometry. The separate normal-direction flag uses
``ENTITY_NORMAL_INWARD = 0`` and ``ENTITY_NORMAL_OUTWARD = 1``.

Reading IDs and debugger output
-------------------------------

* Type codes are distinct from owner IDs, geometry IDs, and family numbers.
  An owner ID of 0 may be valid even though the owner *type* code 0 is a sentinel.
* Primitive and patch contacts use the same contact type encoding but different
  geometry indexing. ``DEME_GET_GEO_OWNER_ID`` maps triangles through
  ``ownerTriMesh``; ``DEME_GET_PATCH_OWNER_ID`` maps mesh patches through
  ``ownerPatchMesh``. Do not use a mesh patch index as a triangle index.
* Unsigned 8-bit values may print as characters in C++ streams. Cast a type
  code to ``unsigned int`` when printing its numeric value.
