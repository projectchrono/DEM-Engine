//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#pragma once

#include <granular/GranularDefines.h>
#include <core/utils/ManagedAllocator.hpp>

namespace sgps {
// Structs defined here will be used by some host classes in DEM.
// NOTE: Data structs here need to be those complex ones (such as needing to include ManagedAllocator.hpp), which may
// not be jitifiable.

/// <summary>
/// DEMSolverStateData contains information that pertains the DEM solver, at a certain point in time. It also contains
/// space allocated as system scratch pad and as thread temporary arrays.
/// </summary>
class DEMSolverStateData {
  private:
    size_t* pMaxNumberSpheresInAnyBin;
    size_t* pTotalBinSphereTouchPairs;  ///< Used in kT storing the number of total bin--sphere contact pairs

    /// The vector used by CUB or by anybody else that needs scratch space.
    /// Please pay attention to the type the vector stores.
    std::vector<scratch_t, ManagedAllocator<scratch_t>> deviceScratchSpace;

    /// The vector used by threads when they need temporary arrays (very typically, for storing arrays outputted by cub
    /// scan or reduce operations). If more than one temporary storage arrays are needed, use pointers to break this
    /// array into pieces.
    std::vector<scratch_t, ManagedAllocator<scratch_t>> threadTempVector;

    /// current integration time step
    float crntStepSize;  // DN: needs to be brought here from GranParams
    float crntSimTime;   // DN: needs to be brought here from GranParams
  public:
    DEMSolverStateData() {
        cudaMallocManaged(&pMaxNumberSpheresInAnyBin, sizeof(size_t));
        cudaMallocManaged(&pTotalBinSphereTouchPairs, sizeof(size_t));
    }
    ~DEMSolverStateData() {
        cudaFree(pMaxNumberSpheresInAnyBin);
        cudaFree(pTotalBinSphereTouchPairs);
    }

    /// return raw pointer to swath of device memory that is at least "sizeNeeded" large
    inline scratch_t* allocateScratchSpace(size_t sizeNeeded) {
        if (deviceScratchSpace.size() < sizeNeeded) {
            deviceScratchSpace.resize(sizeNeeded);
        }
        return deviceScratchSpace.data();
    }

    inline scratch_t* allocateTempVector(size_t sizeNeeded) {
        if (threadTempVector.size() < sizeNeeded) {
            threadTempVector.resize(sizeNeeded);
        }
        return threadTempVector.data();
    }

    /// store the number of total bin--sphere contact pairs
    inline void setTotalBinSphereTouchPairs(size_t n) { *pTotalBinSphereTouchPairs = n; }

    /// return the number of total bin--sphere contact pairs
    inline size_t getTotalBinSphereTouchPairs() { return *pTotalBinSphereTouchPairs; }
};

}  // namespace sgps