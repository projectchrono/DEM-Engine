# Mesh Template Functionality

This document describes the new mesh template functionality added to DEM-Engine, which allows users to efficiently create multiple mesh particle instances from a single template.

## Overview

Previously, meshes were added directly to the simulation using `AddWavefrontMeshObject()`. Now, similar to the existing clump template system, users can:

1. Load a mesh as a **template** (not immediately in the simulation)
2. **Instantiate** multiple mesh particles from the template at different locations
3. **Duplicate** existing mesh objects

## New API Methods

### LoadMeshType()

Loads a mesh as a template without adding it to the simulation.

```cpp
// Load from file with material
auto mesh_template = DEMSim.LoadMeshType(
    "path/to/mesh.obj",
    mat_type,
    true,  // load_normals
    false  // load_uv
);

// Load from file without material
auto mesh_template = DEMSim.LoadMeshType(
    "path/to/mesh.obj",
    true,  // load_normals
    false  // load_uv
);

// Load from DEMMeshConnected object
DEMMeshConnected mesh;
mesh.LoadWavefrontMesh("path/to/mesh.obj");
mesh.SetMaterial(mat_type);
auto mesh_template = DEMSim.LoadMeshType(mesh);
```

### AddMeshFromTemplate()

Creates a mesh instance from a template at a specified location.

```cpp
// Using float3
auto mesh1 = DEMSim.AddMeshFromTemplate(mesh_template, make_float3(0, 0, 0));

// Using vector
std::vector<float> pos = {1.0, 2.0, 3.0};
auto mesh2 = DEMSim.AddMeshFromTemplate(mesh_template, pos);
```

After instantiation, you can set additional properties:

```cpp
mesh1->SetFamily(0);
mesh1->SetMass(1000.);
mesh1->SetMOI(make_float3(200., 200., 200.));
```

### Duplicate()

Creates a deep copy of an existing mesh object.

```cpp
auto mesh_copy = DEMSim.Duplicate(original_mesh);
mesh_copy->SetInitPos(make_float3(5, 0, 0));
```

## Usage Example

```cpp
#include <DEM/API.h>

int main() {
    DEMSolver DEMSim;
    DEMSim.SetVerbosity("INFO");
    
    // Load material
    auto mat_type = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.6}});
    
    // Load mesh as template (not yet in simulation)
    auto mesh_template = DEMSim.LoadMeshType(
        "data/mesh/cube.obj",
        mat_type,
        true,   // load_normals
        false   // load_uv
    );
    
    // Create multiple instances at different locations
    auto mesh1 = DEMSim.AddMeshFromTemplate(mesh_template, make_float3(-2, 0, 0));
    mesh1->SetFamily(0);
    mesh1->SetMass(1000.);
    mesh1->SetMOI(make_float3(200., 200., 200.));
    
    auto mesh2 = DEMSim.AddMeshFromTemplate(mesh_template, make_float3(2, 0, 0));
    mesh2->SetFamily(0);
    mesh2->SetMass(1000.);
    mesh2->SetMOI(make_float3(200., 200., 200.));
    
    // Duplicate an existing mesh
    auto mesh3 = DEMSim.Duplicate(mesh1);
    mesh3->SetInitPos(make_float3(0, 2, 0));
    
    // Initialize and run simulation
    DEMSim.Initialize();
    // ... simulation code ...
    
    return 0;
}
```

## Benefits

1. **Memory Efficiency**: Load mesh geometry once, instantiate many times
2. **Cleaner Code**: Separates template definition from instantiation
3. **Consistency**: Similar workflow to existing clump template system
4. **Flexibility**: Easy to create copies with modified properties

## Comparison with Clump Templates

| Feature | Clump Templates | Mesh Templates |
|---------|----------------|----------------|
| Template Loading | `LoadClumpType()` | `LoadMeshType()` |
| Instantiation | `AddClumps()` | `AddMeshFromTemplate()` |
| Batch Creation | Yes (multiple at once) | No (one at a time) |
| Duplication | `Duplicate()` | `Duplicate()` |

## Notes

- Mesh templates are stored separately from simulation meshes
- Template storage is never cleared during `ClearCache()` calls
- One-by-one instantiation (no batch creation like clumps)
- Each instance is independent and can have different properties
- Compatible with all existing mesh features (tracking, modification, etc.)

## Demo

See `test_mesh_template.cpp` for a complete working example demonstrating:
- Loading mesh templates
- Creating multiple instances
- Duplicating meshes
- Running a simulation with template-based meshes
