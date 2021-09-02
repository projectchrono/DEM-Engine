#include <kernel/cudaKernels.cuh>

// *----------------------------------------
// GPU - Testing kernels
__global__ void dynamicTestKernel() {
    printf("Dynamic run\n");
}
__global__ void kinematicTestKernel(sgps::voxelID_default_t* data) {
    if (threadIdx.x == 0) {
        printf("Kinematic run\n");
    }

    if (threadIdx.x < N_INPUT_ITEMS) {
        // data[threadIdx.x] = 2 * data[threadIdx.x] + 1;
        // printf("%d\n", data[threadIdx.x]);
    }
}
// END of GPU Testing kernels
// *----------------------------------------

// *----------------------------------------
// SPH - Kinematic kernels

__global__ void kinematic1stPass(vector3* pos, int n, float tolerance, float radius, int* res_arr) {
    // printf("in kernel\n");
    int idx = threadIdx.x;
    int count = 0;  // count total number of valid contact for the current particle

    if (idx > n) {
        res_arr[idx] = count;
        return;
    }

    for (int i = n - 1; i > idx; i--) {
        float dist2 = (pos[idx].x - pos[i].x) * (pos[idx].x - pos[i].x) +
                      (pos[idx].y - pos[i].y) * (pos[idx].y - pos[i].y) +
                      (pos[idx].z - pos[i].z) * (pos[idx].z - pos[i].z);

        if (dist2 <= (radius * 2 + tolerance) * (radius * 2 + tolerance)) {
            count++;
        }
    }

    res_arr[idx] = count;
}

__global__ void kinematic2ndPass(vector3* pos,
                                 int n,
                                 int* offset,
                                 int* contact_num_arr,
                                 float tolerance,
                                 float radius,
                                 contactData* pair_data) {
    int idx = threadIdx.x;
    if (contact_num_arr[idx] != 0) {
        int cur_idx = offset[idx];

        for (int i = n - 1; i > idx; i--) {
            float dist2 = (pos[idx].x - pos[i].x) * (pos[idx].x - pos[i].x) +
                          (pos[idx].y - pos[i].y) * (pos[idx].y - pos[i].y) +
                          (pos[idx].z - pos[i].z) * (pos[idx].z - pos[i].z);

            if (dist2 < (radius * 2 + tolerance) * (radius * 2 + tolerance)) {
                pair_data[cur_idx].contact_pair.x = idx;
                pair_data[cur_idx].contact_pair.y = i;
                cur_idx++;
            }
        }
    }
    __syncthreads();
}
// END of Kinematic kernels
// *----------------------------------------

// *----------------------------------------
// SPH - Dynamic kernals
__global__ void dynamicPass(contactData* gpu_pair_data,
                            int gpu_pair_n,
                            vector3* gpu_pos,
                            vector3* gpu_vel,
                            vector3* gpu_acc,
                            float radius) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= gpu_pair_n) {
        return;
    }

    float dir_x = gpu_pos[gpu_pair_data[idx].contact_pair.x].x - gpu_pos[gpu_pair_data[idx].contact_pair.y].x;
    float dir_y = gpu_pos[gpu_pair_data[idx].contact_pair.x].y - gpu_pos[gpu_pair_data[idx].contact_pair.y].y;
    float dir_z = gpu_pos[gpu_pair_data[idx].contact_pair.x].z - gpu_pos[gpu_pair_data[idx].contact_pair.y].z;

    float dist2 = dir_x * dir_x + dir_y * dir_y + dir_z * dir_z;
    if (dist2 < radius * radius) {
        // TODO
    }
    __syncthreads();
}
// END of Dynamic kernels
// *----------------------------------------