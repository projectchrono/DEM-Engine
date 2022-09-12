#include <sph/datastruct.h>
#include <kernel/CUDAMathHelpers.cu>
// =================================================================================================================
// ========================================= START of Kinematic kernels ============================================
// =================================================================================================================

// All helper functions
__device__ float W(float3 r, float h) {
    float invh = 1.0 / h;
    float alpha_d = 0.25 / MATH_PI * invh * invh * invh;
    float R = sqrt(r.x * r.x + r.y * r.y + r.z * r.z) * invh;
    float res = 0.0;

    if (R >= 2) {
        res = 0.0;
    } else if (R < 2 && R >= 1) {
        res = alpha_d * (2 - R) * (2 - R) * (2 - R);
    } else {
        res = alpha_d * ((2 - R) * (2 - R) * (2 - R) - 4 * (1 - R) * (1 - R) * (1 - R));
    }

    // printf("%f", res);
    return res;
}

__device__ float3 W_Grad(float3 r, float h) {
    float invh = 1.0 / h;
    float d = sqrt(r.x * r.x + r.y * r.y + r.z * r.z);
    float R = d * invh;
    float3 alpha_d = 0.75 / MATH_PI * invh * invh * invh * invh * invh * r;

    float3 coe;
    if (R < 1e-8 || R >= 2) {
        return 0.0 * r;
    } else if (R < 2 && R >= 1) {
        coe = alpha_d * (-R + 4.0 - 4.0 / R);
    } else {
        coe = alpha_d * (3.0 * R - 4.0);
    }

    return coe;
}

// =================================================================================================================
// Kinematic 1st Step, this pass identifies number of BSDs touched by each particle
// This kernel also fills the idx_track vector
// =================================================================================================================
__global__ void fillNumBSDByParticle(float3* pos_data,
                                     int* num_BSD_data,
                                     int k_n,
                                     float kernel_h,
                                     float d_domain_x,
                                     float d_domain_y,
                                     float d_domain_z,
                                     int num_domain_x,
                                     int num_domain_y,
                                     int num_domain_z,
                                     float domain_x,
                                     float domain_y,
                                     float domain_z,
                                     float buffer_width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= k_n) {
        return;
    }

    float dx_2_0 = pos_data[idx].x - (-domain_x / 2);
    float dy_2_0 = pos_data[idx].y - (-domain_y / 2);
    float dz_2_0 = pos_data[idx].z - (-domain_z / 2);

    int x_idx = (int)floor(dx_2_0 / d_domain_x);
    int y_idx = (int)floor(dy_2_0 / d_domain_y);
    int z_idx = (int)floor(dz_2_0 / d_domain_z);

    // count how many BSD the current particle belongs to
    int counter = 1;

    // we need to check 26 BDs
    for (int i = 0; i < 26; i++) {
        int x_check_idx = x_idx;
        int y_check_idx = y_idx;
        int z_check_idx = z_idx;

        switch (i) {
            case 0:
                x_check_idx = x_idx - 1;
                y_check_idx = y_idx;
                z_check_idx = z_idx;
                break;
            case 1:
                x_check_idx = x_idx - 1;
                y_check_idx = y_idx;
                z_check_idx = z_idx - 1;
                break;
            case 2:
                x_check_idx = x_idx - 1;
                y_check_idx = y_idx;
                z_check_idx = z_idx + 1;
                break;
            case 3:
                x_check_idx = x_idx - 1;
                y_check_idx = y_idx - 1;
                z_check_idx = z_idx;
                break;
            case 4:
                x_check_idx = x_idx - 1;
                y_check_idx = y_idx - 1;
                z_check_idx = z_idx - 1;
                break;
            case 5:
                x_check_idx = x_idx - 1;
                y_check_idx = y_idx - 1;
                z_check_idx = z_idx + 1;
                break;
            case 6:
                x_check_idx = x_idx - 1;
                y_check_idx = y_idx + 1;
                z_check_idx = z_idx;
                break;
            case 7:
                x_check_idx = x_idx - 1;
                y_check_idx = y_idx + 1;
                z_check_idx = z_idx - 1;
                break;
            case 8:
                x_check_idx = x_idx - 1;
                y_check_idx = y_idx + 1;
                z_check_idx = z_idx + 1;
                break;
            case 9:
                x_check_idx = x_idx + 1;
                y_check_idx = y_idx;
                z_check_idx = z_idx;
                break;
            case 10:
                x_check_idx = x_idx + 1;
                y_check_idx = y_idx;
                z_check_idx = z_idx - 1;
                break;
            case 11:
                x_check_idx = x_idx + 1;
                y_check_idx = y_idx;
                z_check_idx = z_idx + 1;
                break;
            case 12:
                x_check_idx = x_idx + 1;
                y_check_idx = y_idx - 1;
                z_check_idx = z_idx;
                break;
            case 13:
                x_check_idx = x_idx + 1;
                y_check_idx = y_idx - 1;
                z_check_idx = z_idx - 1;
                break;
            case 14:
                x_check_idx = x_idx + 1;
                y_check_idx = y_idx - 1;
                z_check_idx = z_idx + 1;
                break;
            case 15:
                x_check_idx = x_idx + 1;
                y_check_idx = y_idx + 1;
                z_check_idx = z_idx;
                break;
            case 16:
                x_check_idx = x_idx + 1;
                y_check_idx = y_idx + 1;
                z_check_idx = z_idx - 1;
                break;
            case 17:
                x_check_idx = x_idx + 1;
                y_check_idx = y_idx + 1;
                z_check_idx = z_idx + 1;
                break;
            case 18:
                x_check_idx = x_idx;
                y_check_idx = y_idx;
                z_check_idx = z_idx - 1;
                break;
            case 19:
                x_check_idx = x_idx;
                y_check_idx = y_idx;
                z_check_idx = z_idx + 1;
                break;
            case 20:
                x_check_idx = x_idx;
                y_check_idx = y_idx - 1;
                z_check_idx = z_idx;
                break;
            case 21:
                x_check_idx = x_idx;
                y_check_idx = y_idx - 1;
                z_check_idx = z_idx - 1;
                break;
            case 22:
                x_check_idx = x_idx;
                y_check_idx = y_idx - 1;
                z_check_idx = z_idx + 1;
                break;
            case 23:
                x_check_idx = x_idx;
                y_check_idx = y_idx + 1;
                z_check_idx = z_idx;
                break;
            case 24:
                x_check_idx = x_idx;
                y_check_idx = y_idx + 1;
                z_check_idx = z_idx - 1;
                break;
            case 25:
                x_check_idx = x_idx;
                y_check_idx = y_idx + 1;
                z_check_idx = z_idx + 1;
                break;
        }

        // check idx validality
        if (x_check_idx < 0 || y_check_idx < 0 || z_check_idx < 0) {
            continue;
        }

        if (x_check_idx >= num_domain_x || y_check_idx >= num_domain_y || z_check_idx >= num_domain_z) {
            continue;
        }

        // expand the SD to BSD, and check whether the particle is in the current BSD
        float BSD_x_start = x_check_idx * d_domain_x - buffer_width * kernel_h;
        float BSD_x_end = (x_check_idx + 1) * d_domain_x + buffer_width * kernel_h;
        float BSD_y_start = y_check_idx * d_domain_y - buffer_width * kernel_h;
        float BSD_y_end = (y_check_idx + 1) * d_domain_y + buffer_width * kernel_h;
        float BSD_z_start = z_check_idx * d_domain_z - buffer_width * kernel_h;
        float BSD_z_end = (z_check_idx + 1) * d_domain_z + buffer_width * kernel_h;

        if (dx_2_0 >= BSD_x_start && dx_2_0 < BSD_x_end) {
            if (dy_2_0 >= BSD_y_start && dy_2_0 < BSD_y_end) {
                if (dz_2_0 >= BSD_z_start && dz_2_0 < BSD_z_end) {
                    counter++;
                }
            }
        }
    }

    num_BSD_data[idx] = counter;
}

// =================================================================================================================
// Kinematic 2nd Step, this pass identifies the exact index of BSDs touched by each particle
// This kernel also fills the BSD_iden_idx which also identifies whether the particle is in buffer (0 is not in buffer,
// 1 is in buffer)
// =================================================================================================================
__global__ void fillBSDIndexByParticle(float3* pos_data,
                                       int* num_BSD_data_offset,
                                       int* BSD_iden_idx,
                                       int* BSD_idx,
                                       int* idx_track_data,
                                       int k_n,
                                       float TotLength,
                                       float kernel_h,
                                       float d_domain_x,
                                       float d_domain_y,
                                       float d_domain_z,
                                       int num_domain_x,
                                       int num_domain_y,
                                       int num_domain_z,
                                       float domain_x,
                                       float domain_y,
                                       float domain_z,
                                       float buffer_width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= k_n) {
        return;
    }

    int start_idx = num_BSD_data_offset[idx];

    float dx_2_0 = pos_data[idx].x - (-domain_x / 2);
    float dy_2_0 = pos_data[idx].y - (-domain_y / 2);
    float dz_2_0 = pos_data[idx].z - (-domain_z / 2);

    int x_idx = int(dx_2_0 / d_domain_x);
    int y_idx = int(dy_2_0 / d_domain_y);
    int z_idx = int(dz_2_0 / d_domain_z);

    // the mother BD idx which the current particle belongs to
    int bd_idx = z_idx * num_domain_x * num_domain_y + y_idx * num_domain_x + x_idx;

    // fill in mother BD index
    BSD_idx[start_idx] = bd_idx;
    BSD_iden_idx[start_idx] = 0;
    idx_track_data[start_idx] = idx;

    // we need to check 26 BDs

    int counter = 1;
    for (int i = 0; i < 26; i++) {
        int x_check_idx = x_idx;
        int y_check_idx = y_idx;
        int z_check_idx = z_idx;

        switch (i) {
            case 0:
                x_check_idx = x_idx - 1;
                y_check_idx = y_idx;
                z_check_idx = z_idx;
                break;
            case 1:
                x_check_idx = x_idx - 1;
                y_check_idx = y_idx;
                z_check_idx = z_idx - 1;
                break;
            case 2:
                x_check_idx = x_idx - 1;
                y_check_idx = y_idx;
                z_check_idx = z_idx + 1;
                break;
            case 3:
                x_check_idx = x_idx - 1;
                y_check_idx = y_idx - 1;
                z_check_idx = z_idx;
                break;
            case 4:
                x_check_idx = x_idx - 1;
                y_check_idx = y_idx - 1;
                z_check_idx = z_idx - 1;
                break;
            case 5:
                x_check_idx = x_idx - 1;
                y_check_idx = y_idx - 1;
                z_check_idx = z_idx + 1;
                break;
            case 6:
                x_check_idx = x_idx - 1;
                y_check_idx = y_idx + 1;
                z_check_idx = z_idx;
                break;
            case 7:
                x_check_idx = x_idx - 1;
                y_check_idx = y_idx + 1;
                z_check_idx = z_idx - 1;
                break;
            case 8:
                x_check_idx = x_idx - 1;
                y_check_idx = y_idx + 1;
                z_check_idx = z_idx + 1;
                break;
            case 9:
                x_check_idx = x_idx + 1;
                y_check_idx = y_idx;
                z_check_idx = z_idx;
                break;
            case 10:
                x_check_idx = x_idx + 1;
                y_check_idx = y_idx;
                z_check_idx = z_idx - 1;
                break;
            case 11:
                x_check_idx = x_idx + 1;
                y_check_idx = y_idx;
                z_check_idx = z_idx + 1;
                break;
            case 12:
                x_check_idx = x_idx + 1;
                y_check_idx = y_idx - 1;
                z_check_idx = z_idx;
                break;
            case 13:
                x_check_idx = x_idx + 1;
                y_check_idx = y_idx - 1;
                z_check_idx = z_idx - 1;
                break;
            case 14:
                x_check_idx = x_idx + 1;
                y_check_idx = y_idx - 1;
                z_check_idx = z_idx + 1;
                break;
            case 15:
                x_check_idx = x_idx + 1;
                y_check_idx = y_idx + 1;
                z_check_idx = z_idx;
                break;
            case 16:
                x_check_idx = x_idx + 1;
                y_check_idx = y_idx + 1;
                z_check_idx = z_idx - 1;
                break;
            case 17:
                x_check_idx = x_idx + 1;
                y_check_idx = y_idx + 1;
                z_check_idx = z_idx + 1;
                break;
            case 18:
                x_check_idx = x_idx;
                y_check_idx = y_idx;
                z_check_idx = z_idx - 1;
                break;
            case 19:
                x_check_idx = x_idx;
                y_check_idx = y_idx;
                z_check_idx = z_idx + 1;
                break;
            case 20:
                x_check_idx = x_idx;
                y_check_idx = y_idx - 1;
                z_check_idx = z_idx;
                break;
            case 21:
                x_check_idx = x_idx;
                y_check_idx = y_idx - 1;
                z_check_idx = z_idx - 1;
                break;
            case 22:
                x_check_idx = x_idx;
                y_check_idx = y_idx - 1;
                z_check_idx = z_idx + 1;
                break;
            case 23:
                x_check_idx = x_idx;
                y_check_idx = y_idx + 1;
                z_check_idx = z_idx;
                break;
            case 24:
                x_check_idx = x_idx;
                y_check_idx = y_idx + 1;
                z_check_idx = z_idx - 1;
                break;
            case 25:
                x_check_idx = x_idx;
                y_check_idx = y_idx + 1;
                z_check_idx = z_idx + 1;
                break;
        }

        // check idx validality
        if (x_check_idx < 0 || y_check_idx < 0 || z_check_idx < 0) {
            continue;
        }

        if (x_check_idx >= num_domain_x || y_check_idx >= num_domain_y || z_check_idx >= num_domain_z) {
            continue;
        }

        // expand the SD to BSD, and check whether the particle is in the current BSD
        float BSD_x_start = x_check_idx * d_domain_x - buffer_width * kernel_h;
        float BSD_x_end = (x_check_idx + 1) * d_domain_x + buffer_width * kernel_h;
        float BSD_y_start = y_check_idx * d_domain_y - buffer_width * kernel_h;
        float BSD_y_end = (y_check_idx + 1) * d_domain_y + buffer_width * kernel_h;
        float BSD_z_start = z_check_idx * d_domain_z - buffer_width * kernel_h;
        float BSD_z_end = (z_check_idx + 1) * d_domain_z + buffer_width * kernel_h;

        if (dx_2_0 >= BSD_x_start && dx_2_0 < BSD_x_end) {
            if (dy_2_0 >= BSD_y_start && dy_2_0 < BSD_y_end) {
                if (dz_2_0 >= BSD_z_start && dz_2_0 < BSD_z_end) {
                    BSD_idx[start_idx + counter] =
                        z_check_idx * num_domain_x * num_domain_y + y_check_idx * num_domain_x + x_check_idx;
                    BSD_iden_idx[start_idx + counter] = 1;
                    idx_track_data[start_idx + counter] = idx;
                    counter++;
                }
            }
        }
    }
}

// =================================================================================================================
// Kinematic 5th Step, this is the 1st pass of the kinematic thread
// We will use shared memory to store particle location data
// the 1st pass is going to fill the num_col vector
// Has a constraint that number of particles in a BSD cannot exceed 1024
// This can be guaranteed by allowing at most 8 particles in one dimension
// when particles are perfectly spaced (initial configuration),
// but as they move, this constraint could be broken
// =================================================================================================================

__global__ void findNumCollisions(
    float3* pos_data,            // particle position data vector
    int k_n,                     // total number of particles
    float kernel_h,              // kernel_h of the uni-kernel_h particles
    int* idx_track_data_sorted,  // sorted idx_track data
    int* BSD_iden_idx_sorted,    // vector to indicate whether a particle is in buffer zone or not
    int* num_BSD_data_offset,    // length the same as unique_BSD_idx
    int* length_BSD_data,        // length the same as unique_BSD_idx
    int* unique_BSD_idx,
    int* num_col,
    int* num_col_per_particle,
    int unique_length,
    float buffer_width) {
    __shared__ float3 pos_local[MAX_PARTICLES_PER_BSD];  // request maximum capacity for the shared mem
    __shared__ int idx_local[MAX_PARTICLES_PER_BSD];     // request maximum capacity for track
    __shared__ int iden_local[MAX_PARTICLES_PER_BSD];
    __shared__ int col_local[1];

    int sd_idx = blockIdx.x;
    int idx = threadIdx.x;

    int start_idx = num_BSD_data_offset[sd_idx];
    int global_idx = start_idx + idx;
    int tot_in_bsd = length_BSD_data[sd_idx];
    // if (idx == 0)
    // printf("bsd_idx: %d, tot_in_bsd: %d\n", sd_idx, tot_in_bsd);

    if (idx >= tot_in_bsd) {
        return;
    }

    // there will be idle threads when there are less than 1024 particles
    // in the BSD, but we try to maintain less than 1024 particles per BSD
    // in order to use shared memory
    // load data from global to shared memory
    pos_local[idx] = pos_data[idx_track_data_sorted[global_idx]];
    idx_local[idx] = idx_track_data_sorted[global_idx];
    iden_local[idx] = BSD_iden_idx_sorted[global_idx];
    __syncthreads();

    if (iden_local[idx] == 1) {
        return;
    }
    int count = 0;

    for (int i = 0; i < tot_in_bsd; i++) {
        if (idx_local[idx] < idx_local[i]) {
            float dist2 = (pos_local[i].x - pos_local[idx].x) * (pos_local[i].x - pos_local[idx].x) +
                          (pos_local[i].y - pos_local[idx].y) * (pos_local[i].y - pos_local[idx].y) +
                          (pos_local[i].z - pos_local[idx].z) * (pos_local[i].z - pos_local[idx].z);
            if (dist2 <= kernel_h * buffer_width * kernel_h * buffer_width) {
                count++;
            }
        }
    }
    num_col_per_particle[global_idx] = count;

    atomicAdd(&col_local[0], count);
    // printf("block id: %d, col per block: %d\n", sd_idx, col_local[0]);
    num_col[sd_idx] = col_local[0];
}

// =================================================================================================================
// Kinematic 7th Step, this is the 2nd pass of the kinematic thread
// We will use shared memory to store particle location data
// the 2nd pass is going to fill in particle i-j pairs
// =================================================================================================================

__global__ void fillIJPairs(float3* pos_data,            // particle position data vector
                            int k_n,                     // total number of particles
                            float kernel_h,              // kernel_h of the uni-kernel_h particles
                            int* idx_track_data_sorted,  // sorted idx_track data
                            int* BSD_iden_idx_sorted,  // vector to indicate whether a particle is in buffer zone or not
                            int* num_BSD_data_offset,  // length the same as unique_BSD_idx
                            int* length_BSD_data,      // length the same as unique_BSD_idx
                            int* unique_BSD_idx,
                            int* num_col,
                            int unique_length,
                            int* pair_i_data,
                            int* pair_j_data,
                            int* num_col_offset,
                            int* num_col_per_particle_offset,
                            float buffer_width) {
    __shared__ float3 pos_local[MAX_PARTICLES_PER_BSD];  // request maximum capacity for the shared mem
    __shared__ int idx_local[MAX_PARTICLES_PER_BSD];     // request maximum capacity for track
    __shared__ int iden_local[MAX_PARTICLES_PER_BSD];

    int sd_idx = blockIdx.x;
    int idx = threadIdx.x;

    if (sd_idx >= unique_length) {
        return;
    }

    int start_idx = num_BSD_data_offset[sd_idx];
    int global_idx = start_idx + idx;
    int tot_in_bsd = length_BSD_data[sd_idx];
    // printf("bsd_idx: %d, tot_in_bsd: %d\n", sd_idx, tot_in_bsd);
    if (idx >= tot_in_bsd) {
        return;
    }

    if (iden_local[idx] == 1) {
        return;
    }

    pos_local[idx] = pos_data[idx_track_data_sorted[global_idx]];
    idx_local[idx] = idx_track_data_sorted[global_idx];
    iden_local[idx] = BSD_iden_idx_sorted[global_idx];

    __syncthreads();

    for (int i = 0; i < tot_in_bsd; i++) {
        if (idx_local[idx] < idx_local[i]) {
            float dist2 = (pos_local[i].x - pos_local[idx].x) * (pos_local[i].x - pos_local[idx].x) +
                          (pos_local[i].y - pos_local[idx].y) * (pos_local[i].y - pos_local[idx].y) +
                          (pos_local[i].z - pos_local[idx].z) * (pos_local[i].z - pos_local[idx].z);

            if (dist2 <= kernel_h * buffer_width * kernel_h * buffer_width) {
                pair_i_data[num_col_per_particle_offset[global_idx]] = idx_local[idx];
                pair_j_data[num_col_per_particle_offset[global_idx]] = idx_local[i];
            }
        }
    }
}

__global__ void computeDensityJToI(float3* pos_data,
                                   float* rho_data,
                                   float* pressure_data,
                                   int* i_unique,
                                   int* i_offset,
                                   int* i_length,
                                   int* j_data_sorted,
                                   char* fix_data,
                                   int n_unique,
                                   float h,
                                   float m,
                                   float rho_0) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_unique) {
        return;
    }

    int i_idx = i_unique[idx];
    int start_idx = i_offset[idx];
    int len = i_length[idx];

    if (fix_data[i_idx] == 1) {
        rho_data[i_idx] = rho_0;
        return;
    }

    float rho_sum = 0;

    for (int i = 0; i < len; i++) {
        int j_idx = j_data_sorted[start_idx + i];
        float3 dir = pos_data[i_idx] - pos_data[j_idx];

        float w = W(dir, h);
        rho_sum = rho_sum + m * w;
    }
    rho_data[i_idx] = rho_sum;
}

__global__ void computeDensityIToJ(float3* pos_data,
                                   float* rho_data,
                                   float* pressure_data,
                                   int* j_unique,
                                   int* j_offset,
                                   int* j_length,
                                   int* i_data_sorted,
                                   char* fix_data,
                                   int n_unique,
                                   float h,
                                   float m,
                                   float rho_0) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_unique) {
        return;
    }

    int j_idx = j_unique[idx];
    int start_idx = j_offset[idx];
    int len = j_length[idx];

    if (fix_data[j_idx] == 1) {
        rho_data[j_idx] = rho_0;
        return;
    }

    float rho_sum = 0;

    for (int i = 0; i < len; i++) {
        int i_idx = i_data_sorted[start_idx + i];
        float3 dir = pos_data[j_idx] - pos_data[i_idx];

        float w = W(dir, h);
        rho_sum = rho_sum + m * w;
    }

    rho_data[j_idx] = rho_data[j_idx] + rho_sum;
}

__global__ void computePressure(float* rho_data,
                                float* pressure_data,
                                char* fix_data,
                                int n_sample,
                                float h,
                                float m,
                                float rho_0,
                                float c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_sample || fix_data[idx] == 1) {
        return;
    }

    float w_self = W(make_float3(0.0, 0.0, 0.0), h);
    rho_data[idx] = rho_data[idx] + m * w_self;
    pressure_data[idx] = c * c * (rho_data[idx] - rho_0) + 0.05 * c * c * rho_0;
}
