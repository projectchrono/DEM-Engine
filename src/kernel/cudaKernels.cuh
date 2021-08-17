#include <sph/datastruct.cuh>
#include <granular/GranularDefines.h>
#include <stdio.h>

// *----------------------------------------
// GPU - Testing kernels
__global__ void dynamicTestKernel();
__global__ void kinematicTestKernel(sgps::voxelID_default_t* data);
// END of GPU Testing kernels
// *----------------------------------------

// *----------------------------------------
// SPH - Kinematic kernels
__global__ void kinematic1stPass(vector3* pos, int n, float tolerance, float radius, int* res_arr);

__global__ void kinematic2ndPass(vector3* pos,
                                 int n,
                                 int* offset,
                                 int* contact_num_arr,
                                 float tolerance,
                                 float radius,
                                 contactData* pair_data);
// END of Kinematic kernels
// *----------------------------------------

// *----------------------------------------
// SPH - Dynamic kernals
__global__ void dynamicPass(contactData* gpu_pair_data,
                            int gpu_pair_n,
                            vector3* gpu_pos,
                            vector3* gpu_vel,
                            vector3* gpu_acc,
                            float radius);
// END of Dynamic kernels
// *----------------------------------------