//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#pragma once

#include <DEM/DEMDefines.h>
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
    // The vector used by CUB or by anybody else that needs scratch space.
    // Please pay attention to the type the vector stores.
    std::vector<scratch_t, ManagedAllocator<scratch_t>> threadScratchSpace;

    // The vectors used by threads when they need temporary arrays (very typically, for storing arrays outputted by cub
    // scan or reduce operations).
    std::vector<scratch_t, ManagedAllocator<scratch_t>> threadTempVector1;
    std::vector<scratch_t, ManagedAllocator<scratch_t>> threadTempVector2;
    std::vector<scratch_t, ManagedAllocator<scratch_t>> threadTempVector3;
    std::vector<scratch_t, ManagedAllocator<scratch_t>> threadTempVector4;
    std::vector<scratch_t, ManagedAllocator<scratch_t>> threadTempVector5;

    // The vectors used to cache some array (typically the result of some pre- or post-processing) which can potentially
    // be used across iterations.
    std::vector<scratch_t, ManagedAllocator<scratch_t>> threadCachedOwner;

  public:
    // Temp size_t variables that can be reused
    size_t* pTempSizeVar1;

    // Number of contacts in this CD step
    size_t* pNumContacts;

    DEMSolverStateDataDT() {
        cudaMallocManaged(&pTempSizeVar1, sizeof(size_t));
        cudaMallocManaged(&pNumContacts, sizeof(size_t));
        *pNumContacts = 0;
    }
    ~DEMSolverStateDataDT() {
        cudaFree(pTempSizeVar1);
        cudaFree(pNumContacts);
    }

    // Return raw pointer to swath of device memory that is at least "sizeNeeded" large
    inline scratch_t* allocateScratchSpace(size_t sizeNeeded) {
        if (threadScratchSpace.size() < sizeNeeded) {
            threadScratchSpace.resize(sizeNeeded);
        }
        return threadScratchSpace.data();
    }

    // TODO: Better way to write this???
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
    inline scratch_t* allocateTempVector5(size_t sizeNeeded) {
        if (threadTempVector5.size() < sizeNeeded) {
            threadTempVector5.resize(sizeNeeded);
        }
        return threadTempVector5.data();
    }

    inline scratch_t* allocateCachedOwner(size_t sizeNeeded) {
        if (threadCachedOwner.size() < sizeNeeded) {
            threadCachedOwner.resize(sizeNeeded);
        }
        return threadCachedOwner.data();
    }
};

/// <summary>
/// DEMSolverStateData contains information that pertains the DEM solver kT thread, at a certain point in time. It also
/// contains space allocated as system scratch pad and as thread temporary arrays.
/// </summary>
class DEMSolverStateDataKT {
  private:
    // The vector used by CUB or by anybody else that needs scratch space.
    // Please pay attention to the type the vector stores.
    std::vector<scratch_t, ManagedAllocator<scratch_t>> threadScratchSpace;

    // The vectors used by threads when they need temporary arrays (very typically, for storing arrays outputted by cub
    // scan or reduce operations).
    std::vector<scratch_t, ManagedAllocator<scratch_t>> threadTempVector1;
    std::vector<scratch_t, ManagedAllocator<scratch_t>> threadTempVector2;
    std::vector<scratch_t, ManagedAllocator<scratch_t>> threadTempVector3;
    std::vector<scratch_t, ManagedAllocator<scratch_t>> threadTempVector4;
    std::vector<scratch_t, ManagedAllocator<scratch_t>> threadTempVector5;
    std::vector<scratch_t, ManagedAllocator<scratch_t>> threadTempVector6;
    // In theory you can keep going and invent more vectors here. But I feel these I have here are just enough for me to
    // use conveniently.

  public:
    // Temp size_t variables that can be reused
    size_t* pTempSizeVar1;
    size_t* pTempSizeVar2;

    // Number of contacts in this CD step
    size_t* pNumContacts;
    // Number of contacts in the previous CD step
    size_t* pNumPrevContacts;

    DEMSolverStateDataKT() {
        cudaMallocManaged(&pNumContacts, sizeof(size_t));
        cudaMallocManaged(&pTempSizeVar1, sizeof(size_t));
        cudaMallocManaged(&pTempSizeVar2, sizeof(size_t));
        cudaMallocManaged(&pNumPrevContacts, sizeof(size_t));
        *pNumContacts = 0;
        *pNumPrevContacts = 0;
    }
    ~DEMSolverStateDataKT() {
        cudaFree(pNumContacts);
        cudaFree(pTempSizeVar1);
        cudaFree(pTempSizeVar2);
        cudaFree(pNumPrevContacts);
    }

    // Return raw pointer to swath of device memory that is at least "sizeNeeded" large
    inline scratch_t* allocateScratchSpace(size_t sizeNeeded) {
        if (threadScratchSpace.size() < sizeNeeded) {
            threadScratchSpace.resize(sizeNeeded);
        }
        return threadScratchSpace.data();
    }

    // TODO: Better way to write this???
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
    inline scratch_t* allocateTempVector5(size_t sizeNeeded) {
        if (threadTempVector5.size() < sizeNeeded) {
            threadTempVector5.resize(sizeNeeded);
        }
        return threadTempVector5.data();
    }
    inline scratch_t* allocateTempVector6(size_t sizeNeeded) {
        if (threadTempVector6.size() < sizeNeeded) {
            threadTempVector6.resize(sizeNeeded);
        }
        return threadTempVector6.data();
    }
};

struct SolverFlags {
    // Sort contact pair arrays before sending to kT
    bool should_sort_pairs = true;
    // Whether to adopt a contact force calculation strategy where a thread takes care of multiple contacts so shared
    // memory is leveraged
    // NOTE: This is not implemented
    bool use_compact_force_kernel = false;
    // This run is frictionless
    bool isFrictionless = false;
};

}  // namespace sgps