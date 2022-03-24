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
/// space allocated as system scratch pad.
/// </summary>
class DEMSolverStateData {
  private:
    size_t* pMaxNumberSpheresInAnyBin;
    size_t* pTotalBinSphereTouchPairs;  ///< Used in kT storing the number of total bin--sphere contact pairs

    /// vector of unsigned int that lives on the device; used by CUB or by anybody else that needs scrap space.
    /// Please pay attention to the type the vector stores.
    std::vector<scratch_t, ManagedAllocator<scratch_t>> deviceScratchSpace;

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

    /// store the number of total bin--sphere contact pairs
    inline void setTotalBinSphereTouchPairs(size_t n) { *pTotalBinSphereTouchPairs = n; }

    /// return the number of total bin--sphere contact pairs
    inline size_t getTotalBinSphereTouchPairs() { return *pTotalBinSphereTouchPairs; }
};

}  // namespace sgps