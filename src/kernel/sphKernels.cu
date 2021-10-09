#include <sph/datastruct.h>

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

            if (dist2 < ((radius * 2 + tolerance) * (radius * 2 + tolerance))) {
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
__global__ void dynamic1stPass(contactData* gpu_pair_data,
                               int gpu_pair_n,
                               vector3* gpu_pos,
                               vector3* gpu_vel,
                               vector3* gpu_acc,
                               bool* gpu_fix,
                               float radius) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= gpu_pair_n) {
        return;
    }

    float dir_x = gpu_pos[gpu_pair_data[idx].contact_pair.x].x - gpu_pos[gpu_pair_data[idx].contact_pair.y].x;
    float dir_y = gpu_pos[gpu_pair_data[idx].contact_pair.x].y - gpu_pos[gpu_pair_data[idx].contact_pair.y].y;
    float dir_z = gpu_pos[gpu_pair_data[idx].contact_pair.x].z - gpu_pos[gpu_pair_data[idx].contact_pair.y].z;

    float dist2 = dir_x * dir_x + dir_y * dir_y + dir_z * dir_z;

    if (dist2 < (2 * radius) * (2 * radius)) {
        float coe = 1000.f;

        if (gpu_fix[gpu_pair_data[idx].contact_pair.x] == false) {
            gpu_acc[gpu_pair_data[idx].contact_pair.x].x = dir_x * coe;
            gpu_acc[gpu_pair_data[idx].contact_pair.x].y = dir_y * coe;
            gpu_acc[gpu_pair_data[idx].contact_pair.x].z = dir_z * coe;
        }

        if (gpu_fix[gpu_pair_data[idx].contact_pair.y] == false) {
            gpu_acc[gpu_pair_data[idx].contact_pair.y].x = -dir_x * coe;
            gpu_acc[gpu_pair_data[idx].contact_pair.y].y = -dir_y * coe;
            gpu_acc[gpu_pair_data[idx].contact_pair.y].z = -dir_z * coe;
        }
    }

    __syncthreads();
}

__global__ void dynamic2ndPass(vector3* gpu_pos,
                               vector3* gpu_vel,
                               vector3* gpu_acc,
                               bool* gpu_fix,
                               int gpu_n,
                               float time_step,
                               float radius) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= gpu_n) {
        return;
    }

    if (gpu_fix[idx] == false) {
        float grav = -9.8f;
        gpu_acc[idx].z = gpu_acc[idx].z + grav;
    }

    gpu_vel[idx].x = gpu_vel[idx].x + gpu_acc[idx].x * time_step;
    gpu_vel[idx].y = gpu_vel[idx].y + gpu_acc[idx].y * time_step;
    gpu_vel[idx].z = gpu_vel[idx].z + gpu_acc[idx].z * time_step;

    gpu_pos[idx].x = gpu_pos[idx].x + gpu_vel[idx].x * time_step;
    gpu_pos[idx].y = gpu_pos[idx].y + gpu_vel[idx].y * time_step;
    gpu_pos[idx].z = gpu_pos[idx].z + gpu_vel[idx].z * time_step;

    gpu_acc[idx].x = 0.f;
    gpu_acc[idx].y = 0.f;
    gpu_acc[idx].z = 0.f;
}

__global__ void testKernel() {
    printf("test run\n");
}
// END of Dynamic kernels
// *----------------------------------------