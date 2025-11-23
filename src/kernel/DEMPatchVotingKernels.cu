// DEM patch-based voting kernels for mesh contact correction
#include <DEM/Defines.h>
#include <DEMHelperKernels.cuh>

// Kernel to compute weighted normals (normal * area) for voting
// Also prepares the area values for reduction
__global__ void prepareWeightedNormalsForVoting(deme::DEMDataDT* granData,
                                                 float3* weightedNormals,
                                                 double* areas,
                                                 deme::contactPairs_t startOffset,
                                                 deme::contactPairs_t count) {
    deme::contactPairs_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        deme::contactPairs_t myContactID = startOffset + idx;
        
        // Get the contact normal from contactForces
        float3 normal = granData->contactForces[myContactID];
        
        // Extract the area (double) from contactPointGeometryB (stored as float3)
        float3 areaStorage = granData->contactPointGeometryB[myContactID];
        double area = float3StorageToDouble(areaStorage);
        
        // Compute weighted normal (normal * area)
        weightedNormals[idx].x = normal.x * area;
        weightedNormals[idx].y = normal.y * area;
        weightedNormals[idx].z = normal.z * area;
        
        // Store area for reduction
        areas[idx] = area;
    }
}

// Kernel to normalize the voted normals by dividing by total area and scatter to output
// If total area is 0, set result to (0,0,0)
__global__ void normalizeAndScatterVotedNormals(deme::patchIDPair_t* originalKeys,
                                                 deme::patchIDPair_t* uniqueKeys,
                                                 float3* votedWeightedNormals,
                                                 double* totalAreas,
                                                 float3* output,
                                                 size_t* numUniqueKeys,
                                                 deme::contactPairs_t startOffset,
                                                 deme::contactPairs_t count) {
    deme::contactPairs_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        // Get the key for this contact
        deme::patchIDPair_t myKey = originalKeys[idx];
        
        // Find the corresponding unique key (linear search - could be optimized with binary search if sorted)
        size_t numUnique = *numUniqueKeys;
        float3 votedNormal = make_float3(0.0f, 0.0f, 0.0f);
        
        for (size_t i = 0; i < numUnique; i++) {
            if (uniqueKeys[i] == myKey) {
                double totalArea = totalAreas[i];
                if (totalArea > 0.0) {
                    // Normalize by dividing by total area
                    votedNormal.x = votedWeightedNormals[i].x / totalArea;
                    votedNormal.y = votedWeightedNormals[i].y / totalArea;
                    votedNormal.z = votedWeightedNormals[i].z / totalArea;
                }
                // else: votedNormal remains (0,0,0)
                break;
            }
        }
        
        // Write to output at the correct position
        output[startOffset + idx] = votedNormal;
    }
}
