# Patch Location Implementation Summary

## Overview
This document describes the implementation of patch location data in the DEM-Engine mesh system. The patch location represents the relative position (to the implicit CoM of the mesh) of each patch in a mesh.

## Problem Statement
The requirement was to add a data member and associated methods to DEMMesh to store the relative location (XYZ) of mesh patches. This data should:
- Be a vector of length `nPatches`
- Allow explicit user setting
- Auto-compute if not explicitly set:
  - Single patch: location is (0,0,0) (same as CoM)
  - Multiple patches: average of all triangle centroids per patch
- Be transferred to dT at initialization
- Be stored in dT similar to `relPosNode1`, `relPosNode2`, `relPosNode3`

## Implementation Details

### 1. DEMMesh Class Changes (`src/DEM/BdrsAndObjs.h`)

#### New Data Members:
```cpp
// Relative location (to CoM) of each patch (vector of length nPatches)
std::vector<float3> m_patch_locations;
// Whether patch locations have been explicitly set
bool patch_locations_explicitly_set = false;
```

#### New Methods:
```cpp
// Set patch locations explicitly
void SetPatchLocations(const std::vector<float3>& patch_locations);

// Get patch locations
const std::vector<float3>& GetPatchLocations() const;

// Check if patch locations were explicitly set
bool ArePatchLocationsExplicitlySet() const;

// Compute patch locations (auto-calculation logic)
std::vector<float3> ComputePatchLocations() const;
```

### 2. Automatic Patch Location Computation (`src/DEM/MeshUtils.cpp`)

The `ComputePatchLocations()` method implements the following logic:

#### Single Patch:
- Returns location (0,0,0) - representing the mesh's CoM

#### Multiple Patches:
- Computes the centroid of each triangle
- Averages all triangle centroids belonging to the same patch
- Includes bounds checking to prevent out-of-bounds access

```cpp
std::vector<float3> DEMMesh::ComputePatchLocations() const {
    std::vector<float3> patch_locations(nPatches, make_float3(0, 0, 0));

    if (nPatches == 1) {
        // Single patch: location is just CoM (0,0,0)
        return patch_locations;
    }

    // Multiple patches: compute average of triangle centroids per patch
    // ... (implementation details in MeshUtils.cpp)
    
    return patch_locations;
}
```

### 3. dT Storage (`src/DEM/dT.h`)

Added a new DualArray to store patch locations in dT:

```cpp
// Relative position (to mesh CoM) of each mesh patch
DualArray<float3> relPosPatch = DualArray<float3>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
```

This follows the same pattern as:
- `relPosNode1` - relative position of triangle vertex 1
- `relPosNode2` - relative position of triangle vertex 2
- `relPosNode3` - relative position of triangle vertex 3

### 4. DEMDataDT Structure Update (`src/DEM/Defines.h`)

Added pointer to the patch location data:

```cpp
struct DEMDataDT {
    // ... existing members ...
    float3* relPosNode1;
    float3* relPosNode2;
    float3* relPosNode3;
    float3* relPosPatch;  // NEW: Patch locations
    materialsOffset_t* patchMaterialOffset;
    // ...
};
```

### 5. Data Transfer During Initialization (`src/DEM/dT.cpp`)

#### Array Initialization:
```cpp
// Resize to the number of mesh patches
DEME_DUAL_ARRAY_RESIZE(ownerPatchMesh, nMeshPatches, 0);
DEME_DUAL_ARRAY_RESIZE(patchMaterialOffset, nMeshPatches, 0);
DEME_DUAL_ARRAY_RESIZE(relPosPatch, nMeshPatches, make_float3(0));  // NEW
```

#### Data Population:
```cpp
// Populate patch locations for this mesh
// If explicitly set, use those; otherwise compute them
std::vector<float3> this_mesh_patch_locations;
if (input_mesh_objs.at(i)->patch_locations_explicitly_set) {
    this_mesh_patch_locations = input_mesh_objs.at(i)->m_patch_locations;
} else {
    this_mesh_patch_locations = input_mesh_objs.at(i)->ComputePatchLocations();
}
// Store the patch locations in the global array
for (size_t patch_idx = p_start; patch_idx < p; patch_idx++) {
    relPosPatch[nExistingMeshPatches + patch_idx] = this_mesh_patch_locations[patch_idx - p_start];
}
```

#### Device Binding:
```cpp
// In packDataPointers()
relPosPatch.bindDevicePointer(&(granData->relPosPatch));

// In data transfer
relPosPatch.toDeviceAsync(streamInfo.stream);
```

## Usage Examples

### Example 1: Automatic Computation (Single Patch)
```cpp
auto mesh = std::make_shared<DEMMesh>();
mesh->LoadWavefrontMesh("my_convex_mesh.obj");
// mesh has 1 patch by default
auto locations = mesh->ComputePatchLocations();
// locations[0] will be (0, 0, 0)
```

### Example 2: Automatic Computation (Multiple Patches)
```cpp
auto mesh = std::make_shared<DEMMesh>();
mesh->LoadWavefrontMesh("my_complex_mesh.obj");
mesh->SplitIntoConvexPatches(30.0f);  // Split into multiple patches
auto locations = mesh->ComputePatchLocations();
// locations[i] will be the average centroid of all triangles in patch i
```

### Example 3: Explicit Setting
```cpp
auto mesh = std::make_shared<DEMMesh>();
mesh->LoadWavefrontMesh("my_mesh.obj");
mesh->SplitIntoConvexPatches(30.0f);

// Manually specify patch locations
std::vector<float3> my_locations = {
    make_float3(0.5, 0.5, 0.0),
    make_float3(-0.5, -0.5, 0.0)
};
mesh->SetPatchLocations(my_locations);
```

## Testing

A test file `test_patch_locations.cpp` has been created to verify:
1. Single patch meshes have location (0,0,0)
2. Multi-patch meshes correctly compute average centroids
3. Explicitly set locations are correctly stored and retrieved

## Integration with Existing Code

The implementation is designed to be:
- **Non-intrusive**: Does not modify existing mesh functionality
- **Backward compatible**: Default behavior remains unchanged
- **Consistent**: Follows existing patterns (similar to `relPosNode1/2/3`)
- **Efficient**: Computed only once at initialization

## Future Work

While the data is now stored in dT and transferred to the device, it is not yet used in simulation kernels. Future work may include:
- Using patch locations in contact force calculations
- Using patch locations for visualization
- Using patch locations for collision detection optimization

## Files Modified

1. `src/DEM/BdrsAndObjs.h` - DEMMesh class definition
2. `src/DEM/MeshUtils.cpp` - Patch location computation implementation
3. `src/DEM/dT.h` - dT data storage
4. `src/DEM/dT.cpp` - Data initialization and transfer
5. `src/DEM/Defines.h` - DEMDataDT structure

## Files Added

1. `test_patch_locations.cpp` - Test file for patch location functionality
2. `PATCH_LOCATION_IMPLEMENTATION.md` - This documentation file
