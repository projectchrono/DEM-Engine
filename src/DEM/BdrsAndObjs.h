//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef DEME_BOUNDARIES
#define DEME_BOUNDARIES

#include <vector>
#include <string>
#include <limits>
#include <iostream>
#include <sstream>
#include <array>
#include <cmath>
#include <cstdint>

#include "../kernel/DEMHelperKernels.cuh"
#include "Defines.h"
#include "Structs.h"
#include "../core/utils/CudaAllocator.hpp"
#include "utils/HostSideHelpers.hpp"

namespace deme {

/// External object type
/// Note all of them are "shell" objects, not solid objects. If you need a solid cylinder, use one CYLINDER as
/// the side plus 2 CIRCLE as the ends to emulate it. Please be sure to set OUTWARD CYLINDER normal in this case.
enum class OBJ_COMPONENT { PLANE, SPHERE, PLATE, CIRCLE, CYL, CYL_INF, CONE, CONE_INF, TRIANGLE };

/// Sphere
struct DEMSphereParams_t {
    float radius;
    objNormal_t normal;
};

/// Cone side. `cone_tip` is the apex, `dir` points toward the base, and `slope` is radial growth per axial distance.
struct DEMConeParams_t {
    float3 cone_tip;
    float3 dir;
    float slope;
    float hmax;
    float hmin;
    objNormal_t normal;
};

/// Infinite Plane defined by point in plane and normal
struct DEMPlaneParams_t {
    float3 normal;
    float3 position;
};

/// Customized finite Plate defined by center of the plate, normal and y dim
struct DEMPlateParams_t {
    float3 normal;
    float3 center;
    float h_dim_x;
    float h_dim_y;
};

/// Infinite Z-aligned cylinder
struct DEMCylinderParams_t {
    float3 center;
    float3 dir;
    float radius;
    objNormal_t normal;
};

/// Validate the geometric parameters shared by infinite and bounded analytical cones.
inline void assertConeInputs(const float3 axis,
                             const float slope,
                             const float hmin,
                             const float hmax,
                             const std::string& function_name) {
    if (!std::isfinite(axis.x) || !std::isfinite(axis.y) || !std::isfinite(axis.z) || length(axis) <= DEME_TINY_FLOAT) {
        DEME_ERROR("%s's axis argument must be a finite, non-zero direction.", function_name.c_str());
    }
    if (!std::isfinite(slope) || slope <= 0.0f) {
        DEME_ERROR("%s's slope argument must be finite and positive.", function_name.c_str());
    }
    if (!std::isfinite(hmin) || !std::isfinite(hmax) || hmin < 0.0f || hmax <= hmin) {
        DEME_ERROR("%s's axial bounds must be finite and satisfy 0 <= hmin < hmax.", function_name.c_str());
    }
}

/// API-(Host-)side struct that holds cached user-input external objects
class DEMExternObj : public DEMInitializer {
  public:
    DEMExternObj() { obj_type = OWNER_TYPE::ANALYTICAL; }
    ~DEMExternObj() {}

    // Component object types. This means the types of each component, and is different from obj_type.
    std::vector<OBJ_COMPONENT> types;
    // Component object materials
    std::vector<std::shared_ptr<DEMMaterial>> materials;
    // Family code (used in prescribing its motions etc.)
    unsigned int family_code = RESERVED_FAMILY_NUM;  ///< Means it defaults to the "fixed" family

    // // The coordinate of the CoM of this external object, in the frame where all its components' properties are
    // // reported. This is usually all-0 (meaning you should define the object's components in its CoM frame to begin
    // // with), but it can be user-specified.
    // float3 CoM = make_float3(0);
    // // CoM frame's orientation quaternion in the frame which is used to report all its components' properties.
    // Usually
    // // unit quaternion.
    // float4 CoM_oriQ = make_float4(0, 0, 0, 1);

    // Obj's CoM initial position
    float3 init_pos = make_float3(0);
    // Obj's initial orientation quaternion
    float4 init_oriQ = make_float4(0, 0, 0, 1);
    // Obj's mass (huge by default)
    float mass = 1e6;
    // Obj's MOI (huge by default)
    float3 MOI = make_float3(1e6);

    union DEMAnalEntParams {
        DEMPlateParams_t plate;
        DEMPlaneParams_t plane;
        DEMCylinderParams_t cyl;
        DEMConeParams_t cone;
    };
    std::vector<DEMAnalEntParams> entity_params;

    /// Define object contact family number
    void SetFamily(const unsigned int code) {
        if (code > std::numeric_limits<family_t>::max()) {
            DEME_ERROR(
                "An external object is instructed to have family number %u, which is larger than the max "
                "allowance %u",
                code, std::numeric_limits<family_t>::max());
        }
        family_code = code;
    }

    /// Set mass
    void SetMass(float mass) { this->mass = mass; }
    /// Get mass
    float GetMass() const { return mass; }
    /// Set MOI (in principal frame)
    void SetMOI(float3 MOI) { this->MOI = MOI; }
    void SetMOI(const std::vector<float>& MOI) {
        assertThreeElements(MOI, "SetMOI", "MOI");
        SetMOI(make_float3(MOI[0], MOI[1], MOI[2]));
    }
    /// Get MOI (in principal frame)
    float3 GetMOI() const { return MOI; }

    /// @brief Set the initial quaternion for this object (before simulation initializes).
    /// @param rotQ Initial quaternion.
    void SetInitQuat(const float4 rotQ) { init_oriQ = rotQ; }
    void SetInitQuat(const std::vector<float>& rotQ) {
        assertFourElements(rotQ, "SetInitQuat", "rotQ");
        SetInitQuat(make_float4(rotQ[0], rotQ[1], rotQ[2], rotQ[3]));
    }

    /// @brief Set the initial position for this object (before simulation initializes).
    /// @param displ Initial position.
    void SetInitPos(const float3 displ) { init_pos = displ; }
    void SetInitPos(const std::vector<float>& displ) {
        assertThreeElements(displ, "SetInitPos", "displ");
        SetInitPos(make_float3(displ[0], displ[1], displ[2]));
    }

    /// Add a plane with infinite size
    void AddPlane(const float3 pos, const float3 normal, const std::shared_ptr<DEMMaterial>& material) {
        types.push_back(OBJ_COMPONENT::PLANE);
        materials.push_back(material);
        DEMAnalEntParams params;
        params.plane.position = pos;
        float3 unit_normal = normalize(normal);
        params.plane.normal = unit_normal;
        entity_params.push_back(params);
    }
    void AddPlane(const std::vector<float>& pos,
                  const std::vector<float>& normal,
                  const std::shared_ptr<DEMMaterial>& material) {
        assertThreeElements(pos, "AddPlane", "pos");
        assertThreeElements(normal, "AddPlane", "normal");
        AddPlane(make_float3(pos[0], pos[1], pos[2]), make_float3(normal[0], normal[1], normal[2]), material);
    }

    /*
    /// Add a plate with finite size.
    /// Assuming the normal you specified is the z-direction and that normal vector originates from the pos point you
    /// input. Then specify the dimensions along x- and y-axes to define the plate's area.
    void AddPlate(const float3 pos,
                  const float3 normal,
                  const float xdim,
                  const float ydim,
                  const std::shared_ptr<DEMMaterial>& material) {
        types.push_back(OBJ_COMPONENT::PLATE);
        materials.push_back(material);
        DEMAnalEntParams params;
        params.plate.center = pos;
        float3 unit_normal = normalize(normal);
        params.plate.normal = unit_normal;
        params.plate.h_dim_x = xdim / 2.0;
        params.plate.h_dim_y = ydim / 2.0;
        entity_params.push_back(params);
    }
    */

    /// Add a z-axis-aligned cylinder of infinite length
    void AddZCylinder(const float3 pos,
                      const float rad,
                      const std::shared_ptr<DEMMaterial>& material,
                      const objNormal_t normal = ENTITY_NORMAL_INWARD) {
        types.push_back(OBJ_COMPONENT::CYL_INF);
        materials.push_back(material);
        DEMAnalEntParams params;
        params.cyl.center = pos;
        params.cyl.radius = rad;
        params.cyl.dir = make_float3(0, 0, 1);
        params.cyl.normal = normal;
        entity_params.push_back(params);
    }
    void AddZCylinder(const std::vector<float>& pos,
                      const float rad,
                      const std::shared_ptr<DEMMaterial>& material,
                      const objNormal_t normal = ENTITY_NORMAL_INWARD) {  // ENTITY_NORMAL_INWARD is 0, and objNormal_t
                                                                          // is an integer (for Shlok)
        assertThreeElements(pos, "AddZCylinder", "pos");
        AddZCylinder(make_float3(pos[0], pos[1], pos[2]), rad, material, normal);
    }

    /// Add a cylinder of infinite length, which is along a user-specific axis
    void AddCylinder(const float3 pos,
                     const float3 axis,
                     const float rad,
                     const std::shared_ptr<DEMMaterial>& material,
                     const objNormal_t normal = ENTITY_NORMAL_INWARD) {
        types.push_back(OBJ_COMPONENT::CYL_INF);
        materials.push_back(material);
        DEMAnalEntParams params;
        params.cyl.center = pos;
        params.cyl.radius = rad;
        params.cyl.dir = normalize(axis);
        params.cyl.normal = normal;
        entity_params.push_back(params);
    }
    void AddCylinder(const std::vector<float>& pos,
                     const std::vector<float>& axis,
                     const float rad,
                     const std::shared_ptr<DEMMaterial>& material,
                     const objNormal_t normal = ENTITY_NORMAL_INWARD) {
        assertThreeElements(pos, "AddCylinder", "pos");
        assertThreeElements(axis, "AddCylinder", "axis");
        AddCylinder(make_float3(pos[0], pos[1], pos[2]), make_float3(axis[0], axis[1], axis[2]), rad, material, normal);
    }

    /// Add an analytical single-nappe cone side extending indefinitely from `tip` along `axis`.
    void AddCone(const float3 tip,
                 const float3 axis,
                 const float slope,
                 const std::shared_ptr<DEMMaterial>& material,
                 const objNormal_t normal = ENTITY_NORMAL_INWARD) {
        assertConeInputs(axis, slope, 0.0f, DEME_HUGE_FLOAT, "AddCone");
        types.push_back(OBJ_COMPONENT::CONE_INF);
        materials.push_back(material);
        DEMAnalEntParams params;
        params.cone.cone_tip = tip;
        params.cone.dir = normalize(axis);
        params.cone.slope = slope;
        params.cone.hmin = 0.0f;
        params.cone.hmax = DEME_HUGE_FLOAT;
        params.cone.normal = normal;
        entity_params.push_back(params);
    }
    void AddCone(const std::vector<float>& tip,
                 const std::vector<float>& axis,
                 const float slope,
                 const std::shared_ptr<DEMMaterial>& material,
                 const objNormal_t normal = ENTITY_NORMAL_INWARD) {
        assertThreeElements(tip, "AddCone", "tip");
        assertThreeElements(axis, "AddCone", "axis");
        AddCone(make_float3(tip[0], tip[1], tip[2]), make_float3(axis[0], axis[1], axis[2]), slope, material, normal);
    }

    /// Add a cone or frustum side clipped to hmin <= dot(point - tip, axis) <= hmax; caps are not included.
    void AddConeSegment(const float3 tip,
                        const float3 axis,
                        const float slope,
                        const float hmin,
                        const float hmax,
                        const std::shared_ptr<DEMMaterial>& material,
                        const objNormal_t normal = ENTITY_NORMAL_INWARD) {
        assertConeInputs(axis, slope, hmin, hmax, "AddConeSegment");
        types.push_back(OBJ_COMPONENT::CONE);
        materials.push_back(material);
        DEMAnalEntParams params;
        params.cone.cone_tip = tip;
        params.cone.dir = normalize(axis);
        params.cone.slope = slope;
        params.cone.hmin = hmin;
        params.cone.hmax = hmax;
        params.cone.normal = normal;
        entity_params.push_back(params);
    }
    void AddConeSegment(const std::vector<float>& tip,
                        const std::vector<float>& axis,
                        const float slope,
                        const float hmin,
                        const float hmax,
                        const std::shared_ptr<DEMMaterial>& material,
                        const objNormal_t normal = ENTITY_NORMAL_INWARD) {
        assertThreeElements(tip, "AddConeSegment", "tip");
        assertThreeElements(axis, "AddConeSegment", "axis");
        AddConeSegment(make_float3(tip[0], tip[1], tip[2]), make_float3(axis[0], axis[1], axis[2]), slope, hmin, hmax,
                       material, normal);
    }
};

// DEM mesh object
class DEMMesh : public DEMInitializer {
  private:
    void assertPatchLength(size_t len, const std::string name) {
        if (len != nPatches) {
            DEME_ERROR(
                "%s input argument must have length %u (not %zu), same as the number of convex patches in the "
                "mesh.",
                name.c_str(), nPatches, len);
        }
    }

    void assertTriLength(size_t len, const std::string name) {
        if (nTri == 0) {
            DEME_WARNING(
                "The settings at the %s call were applied to 0 mesh facet.\nPlease consider using "
                "%s only after loading the mesh file, because mesh utilities are supposed to provide per-facet "
                "control of your mesh, so we need to know the mesh first.",
                name.c_str(), name.c_str());
        }
        if (len != nTri) {
            DEME_ERROR("%s input argument must have length %zu (not %zu), same as the number of triangles in the mesh.",
                       name.c_str(), nTri, len);
        }
    }

  public:
    // Number of triangle facets in the mesh
    size_t nTri = 0;

    // Owner number in DEM simulation
    bodyID_t owner = NULL_BODYID;

    // Position in the m_meshes array
    unsigned int cache_offset = 0;

    std::vector<float3> m_vertices;
    std::vector<float3> m_normals;
    std::vector<float3> m_UV;
    std::vector<float3> m_colors;

    std::vector<int3> m_face_v_indices;
    std::vector<int3> m_face_n_indices;
    std::vector<int3> m_face_uv_indices;
    std::vector<int3> m_face_col_indices;

    /// @brief Get the coordinates of the vertices of this mesh.
    /// @return A reference to the vertices data vector (of float3) of the mesh.
    std::vector<float3>& GetCoordsVertices() { return m_vertices; }
    /// @brief Get the coordinates of the vertices of this mesh.
    /// @return N (number of vertices) by 3 matrix.
    std::vector<std::vector<float>> GetCoordsVerticesAsVectorOfVectors();

    std::vector<float3>& GetCoordsNormals() { return m_normals; }
    std::vector<float3>& GetCoordsUV() { return m_UV; }
    std::vector<float3>& GetCoordsColors() { return m_colors; }

    /// @brief Get the vertices number of all the triangles of this mesh.
    /// @return A reference to the vertices number data vector (of int3) of the mesh.
    std::vector<int3>& GetIndicesVertexes() { return m_face_v_indices; }
    /// @brief Get the vertices number of all the triangles of this mesh.
    /// @return N (number of vertices) by 3 matrix.
    std::vector<std::vector<int>> GetIndicesVertexesAsVectorOfVectors();

    std::vector<int3>& GetIndicesNormals() { return m_face_n_indices; }
    std::vector<int3>& GetIndicesUV() { return m_face_uv_indices; }
    std::vector<int3>& GetIndicesColors() { return m_face_col_indices; }

    // Material types for each mesh facet
    std::vector<std::shared_ptr<DEMMaterial>> materials;
    bool isMaterialSet = false;
    // Family code (used in prescribing its motions etc.)
    unsigned int family_code = RESERVED_FAMILY_NUM;  ///< Means it defaults to the "fixed" family

    // // The coordinate of the CoM of this meshed object, in the frame where all the mesh's node coordinates are
    // // reported. This is usually all-0 (meaning you should define the object's components in its CoM frame to begin
    // // with), but it can be user-specified.
    // float3 CoM = make_float3(0);
    // // CoM frame's orientation quaternion in the frame which is used to report all the mesh's node coordinates.
    // // It is usually unit quaternion.
    // float4 CoM_oriQ = make_float4(0, 0, 0, 1);

    // Mesh's CoM initial position
    float3 init_pos = make_float3(0);
    // Mesh's initial orientation quaternion
    float4 init_oriQ = make_float4(0, 0, 0, 1);
    // Mesh's mass
    float mass = 1.f;
    // Mesh's MOI
    float3 MOI = make_float3(1.f);
    // Whether mass/MOI were explicitly specified by the user.
    bool mass_specified = false;
    bool moi_specified = false;
    // API-level mesh template identity. Mesh instances created from LoadMeshType/AddMeshFromTemplate keep this mark so
    // mass/MOI jitification can treat repeated mesh particles like repeated clump templates.
    size_t mesh_template_mark = NULL_MESH_TEMPLATE_MARK;
    // If true, this mesh is treated as a shell surface with finite thickness in mesh utilities.
    bool is_shell = false;
    // Physical shell thickness (full thickness, not half-thickness), in simulation length unit.
    float shell_thickness = 0.f;

    std::string filename;  ///< file string if loading an obj file

    // If true, when the mesh is initialized into the system, it will re-order the nodes of each triangle so that the
    // normals derived from right-hand-rule are the same as the normals in the mesh file
    bool use_mesh_normals = false;
    // If true, this mesh is treated as convex for contact island reduction.
    bool is_convex = false;
    // If true, this mesh is never selected as the winner side for island labeling.
    bool never_winner = false;

    DEMMesh() { obj_type = OWNER_TYPE::MESH; }
    DEMMesh(std::string input_file) {
        LoadWavefrontMesh(input_file);
        obj_type = OWNER_TYPE::MESH;
    }
    DEMMesh(std::string input_file, const std::shared_ptr<DEMMaterial>& mat) {
        LoadWavefrontMesh(input_file);
        SetMaterial(mat);
        obj_type = OWNER_TYPE::MESH;
    }
    ~DEMMesh() {}

    /// Load a triangle mesh saved as an STL file (ASCII or binary)
    bool LoadSTLMesh(std::string input_file, bool load_normals = true);

    /// Load a triangle mesh saved as a PLY file (ASCII, triangulated or polygonal)
    bool LoadPLYMesh(std::string input_file, bool load_normals = true);

    /// Load a triangle mesh saved as a Wavefront .obj file
    bool LoadWavefrontMesh(std::string input_file, bool load_normals = true, bool load_uv = false);

    /// Write the specified meshes in a Wavefront .obj file
    static void WriteWavefront(const std::string& filename, std::vector<DEMMesh>& meshes);

    /// Utility function for merging multiple meshes.
    static DEMMesh Merge(std::vector<DEMMesh>& meshes);

    /// Get the number of triangles already added to this mesh
    size_t GetNumTriangles() const { return nTri; }

    /// Get the number of nodes in the mesh
    size_t GetNumNodes() const { return m_vertices.size(); }

    /// Instruct that when the mesh is initialized into the system, it will re-order the nodes of each triangle so that
    /// the normals derived from right-hand-rule are the same as the normals in the mesh file
    void UseNormals(bool use = true) { use_mesh_normals = use; }
    /// Mark this mesh as convex for contact reduction purposes.
    void SetConvex(bool convex = true) { is_convex = convex; }
    /// Query whether this mesh is marked convex.
    bool IsConvex() const { return is_convex; }
    /// Prevent this mesh from ever being chosen as the winner side in island labeling.
    void SetNeverWinner(bool never = true) { never_winner = never; }
    /// Query whether this mesh is marked as never-winner.
    bool IsNeverWinner() const { return never_winner; }
    /// Treat this mesh as a shell surface with finite thickness. Thickness must be finite and non-negative.
    void SetShellThickness(float thickness) {
        if (!std::isfinite(thickness) || thickness < 0.f) {
            DEME_ERROR("Shell thickness must be finite and non-negative (got %.9g).", thickness);
        }
        shell_thickness = thickness;
        is_shell = thickness > DEME_TINY_FLOAT;
    }
    /// Disable shell mode (fallback to zero-thickness triangle surface behavior).
    void DisableShell() {
        is_shell = false;
        shell_thickness = 0.f;
    }
    /// Query whether this mesh is configured as a shell.
    bool IsShell() const { return is_shell; }
    /// Get full shell thickness.
    float GetShellThickness() const { return shell_thickness; }
    /// Get half shell thickness (used internally by kernels).
    float GetShellHalfThickness() const { return (is_shell && shell_thickness > 0.f) ? 0.5f * shell_thickness : 0.f; }

    /// Access the n-th triangle in mesh
    DEMTriangle GetTriangle(size_t index) const {  // No need to wrap (for Shlok)
        return DEMTriangle(m_vertices[m_face_v_indices[index].x], m_vertices[m_face_v_indices[index].y],
                           m_vertices[m_face_v_indices[index].z]);
    }

    /// Clear all data
    void Clear() {
        this->m_vertices.clear();
        this->m_normals.clear();
        this->m_UV.clear();
        this->m_colors.clear();
        this->m_face_v_indices.clear();
        this->m_face_n_indices.clear();
        this->m_face_uv_indices.clear();
        this->m_face_col_indices.clear();
        this->m_patch_ids.clear();
        this->m_patch_locations.clear();
        this->nPatches = 1;
        this->patches_explicitly_set = false;
        this->patch_locations_explicitly_set = false;
        this->owner = NULL_BODYID;
        this->mesh_template_mark = NULL_MESH_TEMPLATE_MARK;
    }

    /// Set mass.
    void SetMass(float mass) {
        this->mass = mass;
        this->mass_specified = true;
    }
    /// Get mass.
    float GetMass() const { return mass; }
    /// Set MOI (in principal frame).
    void SetMOI(float3 MOI) {
        this->MOI = MOI;
        this->moi_specified = true;
    }
    /// Set MOI (in principal frame).
    void SetMOI(const std::vector<float>& MOI) {
        assertThreeElements(MOI, "SetMOI", "MOI");
        SetMOI(make_float3(MOI[0], MOI[1], MOI[2]));
    }
    /// Get MOI (in principal frame).
    float3 GetMOI() const { return MOI; }
    /// Set mesh family number.
    void SetFamily(unsigned int num) { this->family_code = num; }

    /// @brief Set material types for the mesh. The input vector should have the same length as the number of patches in
    /// the mesh, and each element is the material for that patch.
    /// @details This allows you to set different materials for different patches of the mesh, which can be useful if
    /// your mesh has multiple convex patches with different material properties.
    void SetMaterial(const std::vector<std::shared_ptr<DEMMaterial>>& input) {
        assertPatchLength(input.size(), "SetMaterial");
        materials = input;
        isMaterialSet = true;
    }
    /// @brief Set material types for the mesh. Using this method makes a uniform-materialed mesh.
    void SetMaterial(const std::shared_ptr<DEMMaterial>& input) {
        SetMaterial(std::vector<std::shared_ptr<DEMMaterial>>(nPatches, input));
    }

    /// Compute volume, centroid and MOI in CoM frame (unit density).
    /// For shells (`SetShellThickness`), uses a centered-thickening shell model: volume = surface area * thickness.
    void ComputeMassProperties(double& volume, float3& center, float3& inertia) const;
    /// Compute volume, centroid and full inertia tensor in CoM frame (unit density).
    /// `inertia_products` stores tensor terms (Ixy, Iyz, Izx).
    void ComputeMassProperties(double& volume, float3& center, float3& inertia, float3& inertia_products) const;
    /// Check if mesh is watertight (closed, manifold). Returns true if no boundary/non-manifold edges.
    bool IsWatertight(size_t* boundary_edges = nullptr, size_t* nonmanifold_edges = nullptr) const;

    /*
    /// Create a map of neighboring triangles, vector of:
    /// [Ti TieA TieB TieC]
    /// (the free sides have triangle id = -1).
    /// Return false if some edge has more than 2 neighboring triangles
    bool ComputeNeighbouringTriangleMap(std::vector<std::array<int, 4>>& tri_map) const;
    */

    /// @brief Give the meshed object an initial rotation, before the simulation starts.
    void SetInitQuat(const float4 rotQ) { init_oriQ = rotQ; }
    void SetInitQuat(const std::vector<float>& rotQ) {
        assertFourElements(rotQ, "SetInitQuat", "rotQ");
        SetInitQuat(make_float4(rotQ[0], rotQ[1], rotQ[2], rotQ[3]));
    }

    /// @brief Transform the meshed object so it gets to its initial position, before the simulation starts.
    void SetInitPos(const float3 displ) { init_pos = displ; }
    void SetInitPos(const std::vector<float>& displ) {
        assertThreeElements(displ, "SetInitPos", "displ");
        SetInitPos(make_float3(displ[0], displ[1], displ[2]));
    }

    /// If this mesh's component triangles are not reported by the user in its centroid and principal system,
    /// then the user needs to call this method immediately to report this mesh's volume centroid and principal axes,
    /// and nodes will be adjusted by this call so that the mesh's frame is its centroid and principal system.
    void InformCentroidPrincipal(float3 center, float4 prin_Q) {
        // Getting to Centroid and Principal is a translation then a rotation (local), so the undo order to undo
        // translation then rotation
        for (auto& node : m_vertices) {
            applyFrameTransformGlobalToLocal(node, center, prin_Q);
        }
        for (auto& normal : m_normals) {
            applyOriQToVector3(normal, make_float4(-prin_Q.x, -prin_Q.y, -prin_Q.z, prin_Q.w));
            const float n_len = length(normal);
            if (n_len > DEME_TINY_FLOAT) {
                normal /= n_len;
            }
        }
    }
    void InformCentroidPrincipal(const std::vector<float>& center, const std::vector<float>& prin_Q) {
        assertThreeElements(center, "InformCentroidPrincipal", "center");
        assertFourElements(prin_Q, "InformCentroidPrincipal", "prin_Q");
        InformCentroidPrincipal(make_float3(center[0], center[1], center[2]),
                                make_float4(prin_Q[0], prin_Q[1], prin_Q[2], prin_Q[3]));
    }

    /// The opposite of InformCentroidPrincipal, and it is another way to align this mesh's coordinate system with its
    /// centroid and principal system: rotate then move this mesh, so that at the end of this operation, the mesh's
    /// frame is its centroid and principal system.
    void Move(float3 vec, float4 rot_Q) {
        for (auto& node : m_vertices) {
            applyFrameTransformLocalToGlobal(node, vec, rot_Q);
        }
        for (auto& normal : m_normals) {
            applyOriQToVector3(normal, rot_Q);
            const float n_len = length(normal);
            if (n_len > DEME_TINY_FLOAT) {
                normal /= n_len;
            }
        }
    }
    void Move(const std::vector<float>& vec, const std::vector<float>& rot_Q) {
        assertThreeElements(vec, "Move", "vec");
        assertFourElements(rot_Q, "Move", "rot_Q");
        Move(make_float3(vec[0], vec[1], vec[2]), make_float4(rot_Q[0], rot_Q[1], rot_Q[2], rot_Q[3]));
    }

    /// Mirror all points in the mesh about a plane. If this changes the mass properties of this mesh, it is the user's
    /// responsibility to reset them.
    void Mirror(float3 plane_point, float3 plane_normal) {
        plane_normal = normalize(plane_normal);
        for (auto& node : m_vertices) {
            float3 node2plane = plane_point - node;
            float proj = dot(node2plane, plane_normal);
            // If proj is negative, we need to go along the neg dir of plane normal anyway; if proj is positive, we need
            // to go along the positive dir of the plane anyway
            node += 2 * proj * plane_normal;
        }
        // The nodal normal also changes. Although, we don't need it in general.
        for (auto& normal : m_normals) {
            float proj = dot(normal, plane_normal);
            // Different from mirroring nodes
            normal -= 2 * proj * plane_normal;
        }
        // Mirroring will change the order of the facet nodes, so RHR becomes LHR. We have to account for that.
        for (auto& face_v_indices : m_face_v_indices) {
            auto tmp = face_v_indices.y;
            face_v_indices.y = face_v_indices.z;
            face_v_indices.z = tmp;
        }
        for (auto& face_n_indices : m_face_n_indices) {
            auto tmp = face_n_indices.y;
            face_n_indices.y = face_n_indices.z;
            face_n_indices.z = tmp;
        }
        for (auto& face_uv_indices : m_face_uv_indices) {
            auto tmp = face_uv_indices.y;
            face_uv_indices.y = face_uv_indices.z;
            face_uv_indices.z = tmp;
        }
        for (auto& face_col_indices : m_face_col_indices) {
            auto tmp = face_col_indices.y;
            face_col_indices.y = face_col_indices.z;
            face_col_indices.z = tmp;
        }
    }
    void Mirror(const std::vector<float>& plane_point, const std::vector<float>& plane_normal) {
        assertThreeElements(plane_point, "Mirror", "plane_point");
        assertThreeElements(plane_normal, "Mirror", "plane_normal");
        Mirror(make_float3(plane_point[0], plane_point[1], plane_point[2]),
               make_float3(plane_normal[0], plane_normal[1], plane_normal[2]));
    }

    /// @brief Scale all geometry component of this mesh.
    void Scale(float s) {
        // Never let mass become negative.
        assertPositive(s, "Scale", "s");
        for (auto& node : m_vertices) {
            node *= s;
        }
        if (is_shell) {
            shell_thickness *= s;
        }
        double double_s = (double)std::abs(s);
        mass *= double_s * double_s * double_s;
        MOI *= double_s * double_s * double_s * double_s * double_s;
    }
    /// @brief Scale all geometry component of this mesh. Specify x, y, z respectively.
    void Scale(float3 s) {
        // Never let mass become negative.
        assertPositive(s.x, "Scale", "s");
        assertPositive(s.y, "Scale", "s");
        assertPositive(s.z, "Scale", "s");
        for (auto& node : m_vertices) {
            node = node * s;
        }
        // Really just an estimate. The user should reset mass properties manually afterwards.
        double prod = (double)s.x * (double)s.y * (double)s.z;
        mass *= prod;
        MOI.x *= prod * s.x * s.x;  // Square, so always positive
        MOI.y *= prod * s.y * s.y;
        MOI.z *= prod * s.z * s.z;
    }
    void Scale(const std::vector<float>& s) {
        assertThreeElements(s, "Scale", "s");
        Scale(make_float3(s[0], s[1], s[2]));
    }

    ////////////////////////////////////////////////////////
    // Mesh patch information for convex patch splitting
    ////////////////////////////////////////////////////////
    // Patch ID for each triangle facet (defaults to 0 for all triangles, assuming convex mesh)
    std::vector<patchID_t> m_patch_ids;
    // Number of patches in this mesh
    unsigned int nPatches = 1;
    // Whether patch information has been explicitly set (either computed or manually supplied)
    bool patches_explicitly_set = false;
    // Relative location (to CoM) of each patch (vector of length nPatches)
    std::vector<float3> m_patch_locations;
    // Whether patch locations have been explicitly set
    bool patch_locations_explicitly_set = false;

    /// @brief Manually set the patch IDs for each triangle.
    /// @details Allows user to manually specify which patch each triangle belongs to. This is useful when
    /// the user has pre-computed patch information or wants to define patches based on custom criteria.
    /// @param patch_ids Vector of patch IDs, one for each triangle. Must have the same length as the number
    /// of triangles in the mesh. Patch IDs should be non-negative integers starting from 0.
    void SetPatchIDs(const std::vector<patchID_t>& patch_ids);

    /// @brief Assign every triangle to its own patch.
    /// @details Patch IDs are assigned in triangle order as 0, 1, ..., GetNumTriangles() - 1. This must be called after
    /// loading or constructing the mesh, and the triangle count must fit in the patchID_t representation.
    void SetEachTriangleAsPatch();

    /// @brief Get the patch ID for each triangle.
    /// @return Vector of patch IDs (one per triangle). By default, all triangles are in patch 0 (assuming convex mesh).
    const std::vector<patchID_t>& GetPatchIDs() const { return m_patch_ids; }

    /// @brief Get the number of patches in the mesh.
    /// @return Number of patches. Default is 1 (assuming convex mesh).
    unsigned int GetNumPatches() const { return nPatches; }

    /// @brief Check if patch information has been explicitly set.
    /// @return True if patches have been computed via SplitIntoConvexPatches() or set via SetPatchIDs(), false if using
    /// default (single patch).
    bool ArePatchesExplicitlySet() const { return patches_explicitly_set; }

    /// @brief Set the relative location (to CoM) of each patch.
    /// @details Allows user to manually specify the location of each patch relative to the mesh's center of mass.
    /// @param patch_locations Vector of locations (float3), one for each patch. Must have the same length as the number
    /// of patches in the mesh.
    void SetPatchLocations(const std::vector<float3>& patch_locations) {
        assertPatchLength(patch_locations.size(), "SetPatchLocations");
        m_patch_locations = patch_locations;
        patch_locations_explicitly_set = true;
    }

    /// @brief Get the relative location (to CoM) of each patch.
    /// @return Vector of locations (one per patch). Will be automatically calculated at initialization if not
    /// explicitly set.
    const std::vector<float3>& GetPatchLocations() const { return m_patch_locations; }

    /// @brief Check if patch locations have been explicitly set.
    /// @return True if locations have been set via SetPatchLocations(), false if they will be auto-calculated.
    bool ArePatchLocationsExplicitlySet() const { return patch_locations_explicitly_set; }

    /// @brief Compute patch locations relative to the implicit CoM of the mesh.
    /// @details For single patch: returns (0,0,0). For multiple patches: returns average of triangle centroids per
    /// patch.
    /// @return Vector of locations (one per patch).
    std::vector<float3> ComputePatchLocations() const;

    enum class PatchQualityLevel : uint8_t { SAFE = 0, WARN = 1, CRITICAL = 2 };

    enum class PatchConstraintStatus : uint8_t { SATISFIED = 0, TOO_MANY_UNMERGEABLE = 1, TOO_FEW_UNSPLITTABLE = 2 };

    struct PatchQualityPatch {
        PatchQualityLevel level = PatchQualityLevel::SAFE;
        float worst_angle_deg = 0.0f;
        float coherence_r = 1.0f;
        unsigned int n_tris = 0;
        unsigned int hard_crossings = 0;
        unsigned int concave_crossings = 0;
        unsigned int unoriented_edges = 0;
    };

    struct PatchQualityReport {
        PatchQualityLevel overall = PatchQualityLevel::SAFE;
        PatchConstraintStatus constraint_status = PatchConstraintStatus::SATISFIED;
        unsigned int achieved_patches = 0;
        unsigned int requested_min = 1;
        unsigned int requested_max = std::numeric_limits<unsigned int>::max();
        std::vector<PatchQualityPatch> per_patch;
    };

    struct PatchQualityOptions {
        float safe_r = 0.85f;
        float warn_r = 0.65f;
        float warn_worst_angle_margin_deg = 5.0f;
        bool hard_crossings_are_critical = true;
        bool concave_crossings_are_critical = false;
        unsigned int unoriented_warn_threshold = 10;
    };

    struct PatchSplitOptions {
        // `hard_angle_deg` passed to SplitIntoConvexPatches is always the maximum local neighbor angle.
        // `soft_angle_deg` and `patch_normal_max_deg` add optional stricter checks while preserving the legacy default.
        float soft_angle_deg = -1.0f;
        float patch_normal_max_deg = -1.0f;
        bool block_concave_edges = false;
        float concave_allow_deg = 0.0f;
        unsigned int patch_min = 1;
        unsigned int patch_max = std::numeric_limits<unsigned int>::max();
        bool seed_largest_first = true;
    };

    /// @brief Split the mesh into connected patches based on face-normal and optional quality constraints.
    /// @details The default overload preserves the original angle-threshold region-growing behavior. The extended
    /// overload can also fill a quality report and apply optional concavity/patch-normal checks.
    unsigned int SplitIntoConvexPatches(float hard_angle_deg,
                                        const PatchSplitOptions& opt,
                                        PatchQualityReport* out_report,
                                        const PatchQualityOptions& qopt);
    unsigned int SplitIntoConvexPatches(float hard_angle_deg = 30.0f) {
        return SplitIntoConvexPatches(hard_angle_deg, PatchSplitOptions(), nullptr, PatchQualityOptions());
    }
    unsigned int SplitIntoConvexPatches(float hard_angle_deg, const PatchSplitOptions& opt) {
        return SplitIntoConvexPatches(hard_angle_deg, opt, nullptr, PatchQualityOptions());
    }
    unsigned int SplitIntoConvexPatches(float hard_angle_deg,
                                        const PatchSplitOptions& opt,
                                        PatchQualityReport* out_report) {
        return SplitIntoConvexPatches(hard_angle_deg, opt, out_report, PatchQualityOptions());
    }

    ////////////////////////////////////////////////////////
    // Some geo wildcard-related stuff
    ////////////////////////////////////////////////////////
    // Initial geometry wildcard that all triangles should have (one value per triangle facet)
    std::unordered_map<std::string, std::vector<float>> geo_wildcards;
    // Can be used to save mem after initialization
    void ClearWildcards() { deallocate_array(geo_wildcards); }
    void SetGeometryWildcards(const std::unordered_map<std::string, std::vector<float>>& wildcards) {
        if (wildcards.begin()->second.size() != nTri) {
            DEME_ERROR(
                "Input geometry wildcard arrays in a SetGeometryWildcards call must all have the same size as "
                "the number of triangles in this mesh.\nHere, the input array has length %zu but this mesh has %zu "
                "triangles.",
                wildcards.begin()->second.size(), nTri);
        }
        geo_wildcards = wildcards;
    }
    void AddGeometryWildcard(const std::string& name, const std::vector<float>& vals) {
        if (vals.size() != nTri) {
            DEME_ERROR(
                "Input geometry wildcard array in a AddGeometryWildcard call must have the same size as the "
                "number of triangles in this mesh.\nHere, the input array has length %zu but this mesh has %zu "
                "triangles.",
                vals.size(), nTri);
        }
        geo_wildcards[name] = vals;
    }
    void AddGeometryWildcard(const std::string& name, float val) {
        AddGeometryWildcard(name, std::vector<float>(nTri, val));
    }
};

// Backward compatibility alias
using DEMMeshConnected = DEMMesh;

// Template-level rigid-group definition for combining owner templates with fixed relative poses.
class DEMCombinedTemplate {
  public:
    OWNER_TYPE member_type = OWNER_TYPE::CLUMP;
    size_t master_member = 0;
    std::vector<std::shared_ptr<DEMClumpTemplate>> clump_templates;
    std::vector<std::shared_ptr<DEMMesh>> mesh_templates;
    std::vector<float3> rel_pos;
    std::vector<float4> rel_oriQ;
    unsigned int load_order = 0;
};

// Runtime metadata for a batch of combined template instantiations.
// Each element in the per-instance vectors corresponds to one instantiation request.
class DEMCombinedInstances {
  public:
    std::shared_ptr<DEMCombinedTemplate> type;
    // Flattened member initializer handles: n_instances blocks, each with template member count entries.
    std::vector<std::shared_ptr<DEMInitializer>> member_objs;
    std::vector<bodyID_t> member_owner_ids;
    std::vector<float> member_mass;
    std::vector<float3> member_moi;
    // Per-instantiation equivalent mass/MOI.
    std::vector<float> master_equiv_mass;
    std::vector<float3> master_equiv_moi;
    std::vector<bodyID_t> master_owner_ids;
    bool owners_resolved = false;
    size_t n_instances = 0;

    /// Get total number of member owners in this combined batch.
    size_t GetNumOwners() const { return member_objs.size(); }

    /// Add an owner wildcard to all member owners in this combined batch with the same value.
    void AddOwnerWildcard(const std::string& name, float val) {
        for (auto& obj : member_objs) {
            auto batch = std::dynamic_pointer_cast<DEMClumpBatch>(obj);
            if (!batch) {
                DEME_ERROR(
                    "DEMCombinedInstances::AddOwnerWildcard encountered a member that is not a clump batch.\n"
                    "Owner wildcards are only supported for combined clump owners.");
            }
            batch->AddOwnerWildcard(name, val);
        }
    }

    /// Add an owner wildcard to all member owners in this combined batch with per-owner values.
    void AddOwnerWildcard(const std::string& name, const std::vector<float>& vals) {
        if (vals.size() != member_objs.size()) {
            DEME_ERROR(
                "Input owner wildcard array in a DEMCombinedInstances::AddOwnerWildcard call must have the same "
                "size as the number of member owners.\nHere, the input array has length %zu but this combined "
                "batch has %zu member owners.",
                vals.size(), member_objs.size());
        }
        for (size_t i = 0; i < member_objs.size(); i++) {
            auto batch = std::dynamic_pointer_cast<DEMClumpBatch>(member_objs[i]);
            if (!batch) {
                DEME_ERROR(
                    "DEMCombinedInstances::AddOwnerWildcard encountered a member that is not a clump batch.\n"
                    "Owner wildcards are only supported for combined clump owners.");
            }
            batch->AddOwnerWildcard(name, vals[i]);
        }
    }
};

}  // namespace deme

#endif
