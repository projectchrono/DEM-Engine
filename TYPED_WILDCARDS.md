# Typed Wildcard Arrays in DEM-Engine

## Overview

DEM-Engine now supports multiple data types for wildcard arrays, allowing you to associate different types of properties with contacts, owners, and geometries. Previously, all wildcards were stored as `float` arrays. Now you can use:

- **FLOAT** - Standard floating-point values (default, backward compatible)
- **UINT8** - Unsigned 8-bit integers (0-255)
- **BOOL** - Boolean values using `notStupidBool_t`

This enhancement enables more efficient memory usage and better semantic representation of properties like flags, counters, or state indicators.

## Backward Compatibility

All existing code continues to work without changes. The default type for wildcards is `FLOAT`, so:

```cpp
// This continues to work as before - uses FLOAT type by default
my_force_model->SetPerContactWildcards({"delta_time", "delta_tan_x", "delta_tan_y", "delta_tan_z"});
```

## Using Typed Wildcards

### New API Methods

Three new methods allow explicit type specification:

```cpp
void SetPerContactWildcardsWithTypes(const std::unordered_map<std::string, WILDCARD_TYPE>& wildcards_with_types);
void SetPerOwnerWildcardsWithTypes(const std::unordered_map<std::string, WILDCARD_TYPE>& wildcards_with_types);
void SetPerGeometryWildcardsWithTypes(const std::unordered_map<std::string, WILDCARD_TYPE>& wildcards_with_types);
```

### Example Usage

#### Contact Wildcards with Mixed Types

```cpp
#include <DEM/API.h>

// Create a force model
auto my_force_model = DEMSim.DefineForceModel();

// Define contact wildcards with different types
std::unordered_map<std::string, WILDCARD_TYPE> contact_wildcards = {
    {"delta_time", WILDCARD_TYPE::FLOAT},      // Time elapsed (float)
    {"delta_tan_x", WILDCARD_TYPE::FLOAT},     // Tangential displacement X
    {"delta_tan_y", WILDCARD_TYPE::FLOAT},     // Tangential displacement Y
    {"delta_tan_z", WILDCARD_TYPE::FLOAT},     // Tangential displacement Z
    {"contact_broken", WILDCARD_TYPE::BOOL},   // Boolean flag for broken contacts
    {"collision_count", WILDCARD_TYPE::UINT8}  // Counter for number of collisions (0-255)
};

my_force_model->SetPerContactWildcardsWithTypes(contact_wildcards);
```

#### Owner Wildcards with Boolean Flags

```cpp
// Define owner wildcards
std::unordered_map<std::string, WILDCARD_TYPE> owner_wildcards = {
    {"cohesion_strength", WILDCARD_TYPE::FLOAT},  // Cohesion parameter
    {"is_damaged", WILDCARD_TYPE::BOOL},          // Damage state flag
    {"damage_level", WILDCARD_TYPE::UINT8}        // Damage level (0-255)
};

my_force_model->SetPerOwnerWildcardsWithTypes(owner_wildcards);
```

#### Geometry Wildcards with Charge Values

```cpp
// Define geometry wildcards for electrostatic simulation
std::unordered_map<std::string, WILDCARD_TYPE> geo_wildcards = {
    {"charge", WILDCARD_TYPE::FLOAT},          // Electric charge
    {"is_conductor", WILDCARD_TYPE::BOOL},     // Conductor flag
    {"material_id", WILDCARD_TYPE::UINT8}      // Material identifier
};

my_force_model->SetPerGeometryWildcardsWithTypes(geo_wildcards);
```

## Accessing Typed Wildcards in Custom Force Models

### Device-Side Macros

When writing custom force models (CUDA kernels), use these macros to access wildcards:

```cpp
// For FLOAT wildcards - direct access
float delta_time = DEME_CONTACT_WILDCARD_FLOAT(simParams, contactWildcards, 0, myContactID);

// For UINT8 wildcards
uint8_t collision_count = DEME_CONTACT_WILDCARD_UINT8(simParams, contactWildcards, 5, myContactID);

// For BOOL wildcards
notStupidBool_t is_broken = DEME_CONTACT_WILDCARD_BOOL(simParams, contactWildcards, 4, myContactID);

// Generic macro that automatically casts based on type info (returns float)
float value = DEME_CONTACT_WILDCARD(simParams, contactWildcards, wcNum, myContactID);
```

Similar macros exist for owner and geometry wildcards:
- `DEME_OWNER_WILDCARD_FLOAT/UINT8/BOOL`
- `DEME_GEO_WILDCARD_FLOAT/UINT8/BOOL`

## Benefits

1. **Memory Efficiency**: Using `UINT8` or `BOOL` types reduces memory usage compared to `float`
   - `UINT8`: 1 byte vs 4 bytes (75% reduction)
   - `BOOL`: 1 byte vs 4 bytes (75% reduction)

2. **Semantic Clarity**: Boolean flags and integer counters are more clearly expressed
   - `is_damaged` as BOOL is clearer than storing 0.0/1.0 as float
   - `damage_level` as UINT8 naturally ranges from 0-255

3. **Performance**: Smaller data types can improve cache efficiency on GPU

## Implementation Details

### Internal Storage

Internally, all wildcard arrays are now stored as `char*` (scratch_t*) arrays. The actual interpretation of the data is controlled by type information stored in `DEMSimParams`:

```cpp
struct DEMSimParams {
    // ...
    uint8_t contactWildcardTypes[DEME_MAX_WILDCARD_NUM];
    uint8_t ownerWildcardTypes[DEME_MAX_WILDCARD_NUM];
    uint8_t geoWildcardTypes[DEME_MAX_WILDCARD_NUM];
    // ...
};
```

### Type Casting

When accessing wildcard data, the system automatically casts the byte array to the appropriate pointer type:

```cpp
// For FLOAT type (4 bytes per element)
float* float_array = (float*)wildcard_base_ptr;
float value = float_array[index];

// For UINT8 type (1 byte per element)
uint8_t* uint8_array = (uint8_t*)wildcard_base_ptr;
uint8_t value = uint8_array[index];

// For BOOL type (1 byte per element)
notStupidBool_t* bool_array = (notStupidBool_t*)wildcard_base_ptr;
notStupidBool_t value = bool_array[index];
```

## Migration Guide

### From Old API to New API

If you want to leverage the new type system:

**Before:**
```cpp
my_force_model->SetPerContactWildcards({"delta_time", "contact_flag"});
```

**After:**
```cpp
std::unordered_map<std::string, WILDCARD_TYPE> contact_wildcards = {
    {"delta_time", WILDCARD_TYPE::FLOAT},
    {"contact_flag", WILDCARD_TYPE::BOOL}  // More efficient than float
};
my_force_model->SetPerContactWildcardsWithTypes(contact_wildcards);
```

## Limitations

1. Maximum of `DEME_MAX_WILDCARD_NUM` (currently 16) wildcards of each category
2. Supported types are limited to FLOAT, UINT8, and BOOL
3. User-provided input data (in clump batches) is still stored as float and converted during initialization

## Future Enhancements

Potential future improvements:
- Support for additional types (int32, double, etc.)
- Direct support for typed input data in clump batches
- Automatic type inference from initial values
