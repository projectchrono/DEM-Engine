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
        weightedNormals[idx] = make_float3(normal.x * area, normal.y * area, normal.z * area);

        // Store area for reduction
        areas[idx] = area;
    }
}

// Device function for binary search to find key index
__device__ inline size_t binarySearchKey(deme::patchIDPair_t* keys, size_t count, deme::patchIDPair_t target) {
    size_t left = 0;
    size_t right = count;

    while (left < right) {
        size_t mid = left + (right - left) / 2;
        if (keys[mid] < target) {
            left = mid + 1;
        } else if (keys[mid] > target) {
            right = mid;
        } else {
            return mid;
        }
    }
    return count;  // Not found
}

// Kernel to normalize the voted normals by dividing by total area and scatter to output
// If total area is 0, set result to (0,0,0)
// Assumes uniqueKeys are sorted (CUB's ReduceByKey maintains sort order)
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

        // Find the corresponding unique key using binary search
        size_t numUnique = *numUniqueKeys;
        size_t keyIdx = binarySearchKey(uniqueKeys, numUnique, myKey);

        float3 votedNormal = make_float3(0.0f, 0.0f, 0.0f);
        if (keyIdx < numUnique) {
            double totalArea = totalAreas[keyIdx];
            if (totalArea > 0.0) {
                // Normalize by dividing by total area (use reciprocal multiplication for efficiency)
                double invTotalArea = 1.0 / totalArea;
                votedNormal.x = votedWeightedNormals[keyIdx].x * invTotalArea;
                votedNormal.y = votedWeightedNormals[keyIdx].y * invTotalArea;
                votedNormal.z = votedWeightedNormals[keyIdx].z * invTotalArea;
            }
            // else: votedNormal remains (0,0,0)
        }

        // Write to output at the correct position
        output[startOffset + idx] = votedNormal;
    }
}
