// *----------------------------------------
// GPU - Testing kernels
#include <granular/DataStructs.h>
// #include <granular/GranularDefines.h>
// #include <granular/PhysicsSystem.h> // Do not include this! It confuses JITC (But why?)

// using sgps::DEMDynamicThread;
// using sgps::DEMKinematicThread;

__global__ void dynamicTestKernel() {
    printf("Dynamic run\n");
}

__global__ void kinematicTestKernel(sgps::DEMSimParams* simParams, sgps::DEMDataKT* granData) {
    if (threadIdx.x == 0) {
        printf("A kinematic side cycle. \n");
    }
    // printf("data: %u\n", data[0]);

    if (threadIdx.x < 4) {
        // data[threadIdx.x] = 2 * data[threadIdx.x] + 1;
        // printf("%d\n", granData->locX[threadIdx.x]);
    }
}
// END of GPU Testing kernels
// *----------------------------------------
