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
/// DEMSolverStateData contains information that pertains the DEM solver dT thread, at a certain point in time. It also
/// contains space allocated as system scratch pad and as thread temporary arrays.
/// </summary>
class DEMSolverStateDataDT {
  private:
    size_t* pForceCollectionRuns;  ///< Cub output of how many cub runs it executed for collecting forces (dT)

    // The vector used by CUB or by anybody else that needs scratch space.
    // Please pay attention to the type the vector stores.
    std::vector<scratch_t, ManagedAllocator<scratch_t>> threadScratchSpace;

    // The vectors used by threads when they need temporary arrays (very typically, for storing arrays outputted by cub
    // scan or reduce operations).
    std::vector<scratch_t, ManagedAllocator<scratch_t>> threadTempVector1;
    std::vector<scratch_t, ManagedAllocator<scratch_t>> threadTempVector2;
    std::vector<scratch_t, ManagedAllocator<scratch_t>> threadTempVector3;

    // The vectors used to cache some array (typically the result of some pre- or post-processing) which can potentially
    // be used across iterations.
    std::vector<scratch_t, ManagedAllocator<scratch_t>> threadCachedOwner;
    std::vector<scratch_t, ManagedAllocator<scratch_t>> threadCachedMass;
    std::vector<scratch_t, ManagedAllocator<scratch_t>> threadCachedMOI;

    /// current integration time step
    float crntStepSize;  // DN: needs to be brought here from GranParams
    float crntSimTime;   // DN: needs to be brought here from GranParams
  public:
    DEMSolverStateDataDT() { cudaMallocManaged(&pForceCollectionRuns, sizeof(size_t)); }
    ~DEMSolverStateDataDT() { cudaFree(pForceCollectionRuns); }

    // Return raw pointer to swath of device memory that is at least "sizeNeeded" large
    inline scratch_t* allocateScratchSpace(size_t sizeNeeded) {
        if (threadScratchSpace.size() < sizeNeeded) {
            threadScratchSpace.resize(sizeNeeded);
        }
        return threadScratchSpace.data();
    }

    // Better way to write this???
    inline scratch_t* allocateTempVector1(size_t sizeNeeded) {
        if (threadTempVector1.size() < sizeNeeded) {
            threadTempVector1.resize(sizeNeeded);
        }
        return threadTempVector1.data();
    }
    inline scratch_t* allocateTempVector2(size_t sizeNeeded) {
        if (threadTempVector2.size() < sizeNeeded) {
            threadTempVector2.resize(sizeNeeded);
        }
        return threadTempVector2.data();
    }
    inline scratch_t* allocateTempVector3(size_t sizeNeeded) {
        if (threadTempVector3.size() < sizeNeeded) {
            threadTempVector3.resize(sizeNeeded);
        }
        return threadTempVector3.data();
    }

    inline scratch_t* allocateCachedOwner(size_t sizeNeeded) {
        if (threadCachedOwner.size() < sizeNeeded) {
            threadCachedOwner.resize(sizeNeeded);
        }
        return threadCachedOwner.data();
    }
    inline scratch_t* allocateCachedMass(size_t sizeNeeded) {
        if (threadCachedMass.size() < sizeNeeded) {
            threadCachedMass.resize(sizeNeeded);
        }
        return threadCachedMass.data();
    }
    inline scratch_t* allocateCachedMOI(size_t sizeNeeded) {
        if (threadCachedMOI.size() < sizeNeeded) {
            threadCachedMOI.resize(sizeNeeded);
        }
        return threadCachedMOI.data();
    }

    // inline void setForceCollectionRuns(size_t n) { *pForceCollectionRuns = n; }
    inline size_t getForceCollectionRuns() { return *pForceCollectionRuns; }
    inline size_t* getForceCollectionRunsPointer() { return pForceCollectionRuns; }
};

/// <summary>
/// DEMSolverStateData contains information that pertains the DEM solver kT thread, at a certain point in time. It also
/// contains space allocated as system scratch pad and as thread temporary arrays.
/// </summary>
class DEMSolverStateDataKT {
  private:
    size_t* pNumBinSphereTouchPairs;  ///< Used in kT storing the number of total bin--sphere contact pairs
    size_t* pNumActiveBins;           ///< Used in kT storing the number of total "active" bins (in contact detection)

    // The vector used by CUB or by anybody else that needs scratch space.
    // Please pay attention to the type the vector stores.
    std::vector<scratch_t, ManagedAllocator<scratch_t>> threadScratchSpace;

    // The vectors used by threads when they need temporary arrays (very typically, for storing arrays outputted by cub
    // scan or reduce operations).
    std::vector<scratch_t, ManagedAllocator<scratch_t>> threadTempVector1;
    std::vector<scratch_t, ManagedAllocator<scratch_t>> threadTempVector2;
    std::vector<scratch_t, ManagedAllocator<scratch_t>> threadTempVector3;
    std::vector<scratch_t, ManagedAllocator<scratch_t>> threadTempVector4;
    // In theory you can keep going and invent more vectors here. But I feel these I have here are just enough for me to
    // use conveniently.

  public:
    DEMSolverStateDataKT() {
        cudaMallocManaged(&pNumBinSphereTouchPairs, sizeof(size_t));
        cudaMallocManaged(&pNumActiveBins, sizeof(size_t));
    }
    ~DEMSolverStateDataKT() {
        cudaFree(pNumBinSphereTouchPairs);
        cudaFree(pNumActiveBins);
    }

    // Return raw pointer to swath of device memory that is at least "sizeNeeded" large
    inline scratch_t* allocateScratchSpace(size_t sizeNeeded) {
        if (threadScratchSpace.size() < sizeNeeded) {
            threadScratchSpace.resize(sizeNeeded);
        }
        return threadScratchSpace.data();
    }

    // Better way to write this???
    inline scratch_t* allocateTempVector1(size_t sizeNeeded) {
        if (threadTempVector1.size() < sizeNeeded) {
            threadTempVector1.resize(sizeNeeded);
        }
        return threadTempVector1.data();
    }
    inline scratch_t* allocateTempVector2(size_t sizeNeeded) {
        if (threadTempVector2.size() < sizeNeeded) {
            threadTempVector2.resize(sizeNeeded);
        }
        return threadTempVector2.data();
    }
    inline scratch_t* allocateTempVector3(size_t sizeNeeded) {
        if (threadTempVector3.size() < sizeNeeded) {
            threadTempVector3.resize(sizeNeeded);
        }
        return threadTempVector3.data();
    }
    inline scratch_t* allocateTempVector4(size_t sizeNeeded) {
        if (threadTempVector4.size() < sizeNeeded) {
            threadTempVector4.resize(sizeNeeded);
        }
        return threadTempVector4.data();
    }

    inline void setNumBinSphereTouchPairs(size_t n) { *pNumBinSphereTouchPairs = n; }
    inline size_t getNumBinSphereTouchPairs() { return *pNumBinSphereTouchPairs; }
    inline size_t* getNumBinSphereTouchPairsPointer() { return pNumBinSphereTouchPairs; }

    inline void setNumActiveBins(size_t n) { *pNumActiveBins = n; }
    inline size_t getNumActiveBins() { return *pNumActiveBins; }
    inline size_t* getNumActiveBinsPointer() { return pNumActiveBins; }
};

}  // namespace sgps