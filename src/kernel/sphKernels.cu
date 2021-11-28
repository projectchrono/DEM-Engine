#include <sph/datastruct.h>

// *----------------------------------------
// SPH - Kinematic kernels
__global__ void IdxSweep(vector3* pos,
                         int* idx_arr,
                         int* idx_track_arr,
                         int n,
                         float d_domain_x,
                         float d_domain_y,
                         float d_domain_z,
                         int num_domain_x,
                         int num_domain_y,
                         int num_domain_z,
                         float domain_x,
                         float domain_y,
                         float domain_z) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) {
        return;
    }

    idx_track_arr[idx] = idx;

    float dx_2_0 = int(pos[idx].x - (-domain_x / 2));
    float dy_2_0 = int(pos[idx].y - (-domain_y / 2));
    float dz_2_0 = int(pos[idx].z - (-domain_z / 2));

    int x_idx = int(dx_2_0 / d_domain_x);
    int y_idx = int(dy_2_0 / d_domain_x);
    int z_idx = int(dz_2_0 / d_domain_x);

    idx_arr[idx] = z_idx * num_domain_x * num_domain_y + y_idx * num_domain_x + x_idx;
}

__global__ void kinematic1stPass(vector3* pos, int n, float tolerance, float radius, int* res_arr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int count = 0;  // count total number of valid contact for the current particle

    if (idx >= n) {
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

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
        float coe = 700.f;
        float dist = sqrt(dist2);
        float dist_inv = 1.0 / dist;
        float penetration = 2 * radius - dist;
        // fill in contact pair data with respect to the first element of the contact pair data
        gpu_pair_data[idx].contact_force.x = dir_x * coe;
        gpu_pair_data[idx].contact_force.y = dir_y * coe;
        gpu_pair_data[idx].contact_force.z = dir_z * coe;
    }

    __syncthreads();
}

__global__ void dynamic2ndPass(contactData* gpu_pair_data, int gpu_pair_n, contactData* inv_gpu_pair_data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= gpu_pair_n) {
        return;
    }

    inv_gpu_pair_data[idx].contact_pair.x = gpu_pair_data[idx].contact_pair.y;
    inv_gpu_pair_data[idx].contact_pair.y = gpu_pair_data[idx].contact_pair.x;
    inv_gpu_pair_data[idx].contact_force.x = -gpu_pair_data[idx].contact_force.x;
    inv_gpu_pair_data[idx].contact_force.y = -gpu_pair_data[idx].contact_force.y;
    inv_gpu_pair_data[idx].contact_force.z = -gpu_pair_data[idx].contact_force.z;
}

__global__ void dynamic3rdPass(int* key,
                               float* x_reduced,
                               float* y_reduced,
                               float* z_reduced,
                               int n,
                               vector3* gpu_acc) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) {
        return;
    }
    __syncthreads();

    gpu_acc[key[idx]].x = x_reduced[idx];
    gpu_acc[key[idx]].y = y_reduced[idx];
    gpu_acc[key[idx]].z = z_reduced[idx];
    __syncthreads();
}

__global__ void dynamic4thPass(vector3* gpu_pos,
                               vector3* gpu_vel,
                               vector3* gpu_acc,
                               char* gpu_fix,
                               int gpu_n,
                               float time_step,
                               float radius) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= gpu_n) {
        return;
    }

    if (gpu_fix[idx] == 0) {
        float grav = -9.8f;
        gpu_acc[idx].z = gpu_acc[idx].z + grav;

        gpu_vel[idx].x = gpu_vel[idx].x + gpu_acc[idx].x * time_step;
        gpu_vel[idx].y = gpu_vel[idx].y + gpu_acc[idx].y * time_step;
        gpu_vel[idx].z = gpu_vel[idx].z + gpu_acc[idx].z * time_step;

        gpu_pos[idx].x = gpu_pos[idx].x + gpu_vel[idx].x * time_step;
        gpu_pos[idx].y = gpu_pos[idx].y + gpu_vel[idx].y * time_step;
        gpu_pos[idx].z = gpu_pos[idx].z + gpu_vel[idx].z * time_step;
    }
    __syncthreads();

    gpu_acc[idx].x = 0.f;
    gpu_acc[idx].y = 0.f;
    gpu_acc[idx].z = 0.f;

    __syncthreads();
}

__global__ void testKernel() {
    printf("test run\n");
}
// END of Dynamic kernels
// *----------------------------------------