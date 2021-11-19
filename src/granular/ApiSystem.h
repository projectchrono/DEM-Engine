//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#pragma once

#include <vector>
#include <set>
#include <cfloat>

#include <core/ApiVersion.h>
#include <granular/PhysicsSystem.h>
#include <core/utils/ManagedAllocator.hpp>
#include <core/utils/ThreadManager.h>
#include <core/utils/GpuManager.h>
#include <core/utils/Macros.h>
#include <helper_math.cuh>
#include <granular/GranularDefines.h>

namespace sgps {

// class DEMSolver_impl;
// class DEMKinematicThread;
// class DEMDynamicThread;
// class ThreadManager;

class DEMSolver {
  public:
    DEMSolver(float rad);
    virtual ~DEMSolver();

    // Instruct the dimension of the ``world'', as well as the origin point of this ``world''. On initialization, this
    // info will be used to figure out how to assign the num of voxels in each direction. If your ``useful'' domain is
    // not box-shaped, then define a box that contains your domian. O is the coordinate of the left-bottom-front point
    // of your simulation ``world''.
    void InstructBoxDomainDimension(float x, float y, float z, float3 O = make_float3(0));

    // Explicitly instruct the number of voxels (as 2^{x,y,z}) along each direction, as well as the smallest unit length
    // l. This is usually for test purposes, and will overwrite other size-related definitions of the big domain.
    void InstructBoxDomainNumVoxel(unsigned char x,
                                   unsigned char y,
                                   unsigned char z,
                                   float len_unit = 1e-10f,
                                   float3 O = make_float3(0));

    // Set gravity
    void SetGravitationalAcceleration(float3 g);
    // Set a constant time step size
    void SetTimeStepSize(double ts_size);
    // TODO: Implement an API that allows setting ts size through a list

    // A convenient call that sets the origin of your coordinate system to be in the dead center of your simulation
    // ``world''. Useful especially you feel like having this ``world'' large to safely hold everything, and don't quite
    // care about the amount of accuracy lost by not fine-tuning the ``world'' size.
    // Returns the coordinate of the left-bottom-front point of your simulation ``world'' after this operation.
    float3 CenterCoordSys();

    /// Set the ratio by which the radii of the spheres are expanded for the purpose of contact detection (safe, and
    /// creates false positives)
    void SetExpandFactor(float beta);

    // Load possible clump types into the API-level cache.
    // Return the index of the clump type just loaded.
    unsigned int LoadClumpType(float mass,
                               float3 moi,
                               const std::vector<float>& sp_radii,
                               const std::vector<float3>& sp_locations_xyz,
                               const std::vector<unsigned int>& sp_material_ids);
    // TODO: need to overload with (vec_distinctSphereRadiiOffset_default_t spheres_component_type, vec_float3
    // location). If this method is called then corresponding sphere_types must have been defined via LoadSphereType.

    // a simplified version of LoadClumpType: it's just a one-sphere clump
    unsigned int LoadClumpSimpleSphere(float mass, float radius, unsigned int material_id);

    // Load possible materials into the API-level cache
    // Return the index of the material type just loaded
    unsigned int LoadMaterialType(float density, float E);

    // Load input clumps (topology types and initial locations) on a per-pair basis
    void SetClumps(const std::vector<unsigned int>& types, const std::vector<float3>& xyz);

    // Return the voxel ID of a clump by its numbering
    voxelID_t GetClumpVoxelID(unsigned int i) const;

    // Write current simulation status to a file
    // Write overlapping spheres, not clumps. It makes the file larger, but less trouble to visualize. Use for test
    // purposes only.
    void WriteFileAsSpheres(const std::string& outfilename) const;

    int Initialize();

    int LaunchThreads();

    // Copy the cached sim params to the GPU-accessible managed memory, so that they are picked up from the next ts of
    // simulation. Usually used when you want to change simulation parameters after the system is already Intialized.
    void UpdateSimParams();

    /*
      protected:
        DEMSolver() : m_sys(nullptr) {}
        DEMSolver_impl* m_sys;
    */

  private:
    // This is the cached material information.
    // It will be massaged into the managed memory upon Initialize().
    struct DEMMaterial {
        float density;
        float E;
    };
    std::vector<DEMMaterial> m_sp_materials;

    // This is the cached clump structure information.
    // It will be massaged into kernels upon Initialize.
    std::vector<float> m_template_mass;
    std::vector<float3> m_template_moi;
    std::vector<std::vector<unsigned int>> m_template_sp_mat_ids;
    std::vector<std::vector<float>> m_template_sp_radii;
    std::vector<std::vector<float3>> m_template_sp_relPos;

    /*
    // Dan and Ruochun decided NOT to extract unique input values.
    // Instead, we trust users: we simply store all clump template info users give.
    // So this unique-value-extractor block is disabled and commented.

    // unique clump masses derived from m_template_mass
    std::set<float> m_template_mass_types;
    std::vector<unsigned int> m_template_mass_type_offset;
    // unique sphere radii types derived from m_template_sp_radii
    std::set<float> m_template_sp_radii_types;
    std::vector<std::vector<distinctSphereRadiiOffset_default_t>> m_template_sp_radii_type_offset;
    // unique sphere (local) location types derived from m_template_sp_relPos
    // std::set<float3, float3_less_than> m_clumps_sp_location_types;
    std::set<float3> m_clumps_sp_location_types;
    std::vector<std::vector<distinctSphereRelativePositions_default_t>> m_clumps_sp_location_type_offset;
    */

    // ``World'' size along X dir
    float m_boxX = 0.f;
    // ``World'' size along Y dir
    float m_boxY = 0.f;
    // ``World'' size along Z dir
    float m_boxZ = 0.f;
    // Origin of the ``world''
    float3 m_boxLBF = make_float3(0);
    // Number of voxels in the X direction, expressed as a power of 2
    unsigned char nvXp2;
    // Number of voxels in the Y direction, expressed as a power of 2
    unsigned char nvYp2;
    // Number of voxels in the Z direction, expressed as a power of 2
    unsigned char nvZp2;
    // Gravitational acceleration
    float3 G;
    // Actual (double-precision) size of a voxel
    double m_voxelSize;
    // Time step size
    double m_ts_size;
    // The length unit. Any XYZ we report to the user, is under the hood a multiple of this l.
    float l = FLT_MAX;
    // The edge length of a bin (for contact detection)
    double m_binSize;
    // Sphere radii inflation ratio (for safer contact detection)
    float m_expand_factor = 1.0f;

    // Total number of spheres
    size_t nSpheresGM;
    // Total number of clump bodies
    size_t nClumpBodies;

    float sphereUU;

    // Whether the number of voxels and length unit l is explicitly given by the user.
    bool explicit_nv_override = false;
    // Whether the GPU-side systems have been initialized
    bool sys_initialized = false;

    // Right now, the following two are integrated into one, in nDistinctClumpComponents
    // unsigned int nDistinctSphereRadii_computed;
    // unsigned int nDistinctSphereRelativePositions_computed;

    unsigned int nDistinctClumpComponents_computed;
    unsigned int nDistinctClumpBodyTopologies_computed;
    unsigned int nMatTuples_computed;

    // cached state vectors such as the types and locations of the initial clumps to fill the sim domain with
    std::vector<unsigned int> m_input_clump_types;
    std::vector<float3> m_input_clump_xyz;

    int updateFreq = 1;
    int timeDynamicSide = 1;
    int timeKinematicSide = 1;
    int nDynamicCycles = 5;

    GpuManager* dTkT_GpuManager;
    ThreadManager* dTkT_InteractionManager;
    DEMKinematicThread* kT;
    DEMDynamicThread* dT;

    int generateJITResources();
    // Figure out the unit length l and corresponding numbers of voxels along each direction, based on domain size X, Y,
    // Z
    void figureOutNV();
    // Set the default bin (for contact detection) size to be the same of the smallest sphere
    void decideDefaultBinSize();
    // Transfer cached sim params to dT (and kT?)
    void transferSimParams();
    // Wait for kT and dT until they are done with the work and a signal is give by them, then the ApiSystem can go on.
    void waitOnThreads();
    // Allocate and populate kT dT managed arrays.
    void initializeArrays();
    // Pack array pointers to a struct so they can be easily used as kernel arguments
    void packDataPointers();
    // Warn users if the data types defined in GranularDefines.h do not blend well with the user inputs (such as when
    // the user inputs a huge amount of clump templates).
    void validateUserInputs();
};

}  // namespace sgps
