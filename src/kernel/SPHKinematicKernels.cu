#include <sph/datastruct.h>
#include <kernel/CUDAMathHelpers.cu>
// =================================================================================================================
// ========================================= START of Kinematic kernels ============================================
// =================================================================================================================

// All helper functions
__device__ float W(float3 r, float h) {
    float alpha_d = 3 / (2 * MATH_PI * h * h * h);
    float R = sqrt(r.x * r.x + r.y * r.y + r.z * r.z) / h;
    float res = 0;

    if (R >= 2) {
        res = 0;
    } else if (R < 2 && R >= 1) {
        res = alpha_d * (1.0f / 6.0f) * (2 - R) * (2 - R) * (2 - R);
    } else {
        res = alpha_d * (2 / 3) - R * R + (1 / 2) * R * R * R;
    }

    // printf("%f", res);
    return res;
}

__device__ float3 W_Grad(float3 r, float h) {
    float alpha_d = 3 / (2 * MATH_PI * h * h * h);
    float R = sqrt(r.x * r.x + r.y * r.y + r.z * r.z) / h;

    float coe;
    if (R >= 2) {
        coe = 0;
    } else if (R < 2 && R >= 1) {
        coe = alpha_d * (1 / h) * (-0.5 * (2 - R) * (2 - R));
    } else {
        coe = alpha_d * (1 / h) * (-2 * R + (3 / 2) * R * R);
    }
    float3 r_normalized = r / sqrt(r.x * r.x + r.y * r.y + r.z * r.z);

    return coe * r_normalized;
}

// =================================================================================================================
// Kinematic 1st Step, this pass identifies number of BSDs touched by each particle
// This kernel also fills the idx_track vector
// =================================================================================================================
__global__ void kinematicStep1(float3* pos_data,
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
                               float domain_z) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= k_n) {
        return;
    }

    float dx_2_0 = int(pos_data[idx].x - (-domain_x / 2));
    float dy_2_0 = int(pos_data[idx].y - (-domain_y / 2));
    float dz_2_0 = int(pos_data[idx].z - (-domain_z / 2));

    int x_idx = int(dx_2_0 / d_domain_x);
    int y_idx = int(dy_2_0 / d_domain_x);
    int z_idx = int(dz_2_0 / d_domain_x);

    // the mother BD idx which the current particle belongs to
    // int bd_idx = z_idx * num_domain_x * num_domain_y + y_idx * num_domain_x + x_idx;

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
        float BSD_x_start = x_check_idx * d_domain_x - 4 * kernel_h;
        float BSD_x_end = (x_check_idx + 1) * d_domain_x + 4 * kernel_h;
        float BSD_y_start = y_check_idx * d_domain_y - 4 * kernel_h;
        float BSD_y_end = (y_check_idx + 1) * d_domain_y + 4 * kernel_h;
        float BSD_z_start = z_check_idx * d_domain_z - 4 * kernel_h;
        float BSD_z_end = (z_check_idx + 1) * d_domain_z + 4 * kernel_h;

        if (dx_2_0 >= BSD_x_start && dx_2_0 < BSD_x_end) {
            if (dy_2_0 >= BSD_y_start && dy_2_0 < BSD_y_end) {
                if (dz_2_0 >= BSD_z_start && dz_2_0 < BSD_z_end) {
                    counter++;
                }
            }
        }
    }

    num_BSD_data[idx] = counter;

    __syncthreads();
}

// =================================================================================================================
// Kinematic 2nd Step, this pass identifies the exact index of BSDs touched by each particle
// This kernel also fills the BSD_iden_idx which also identifies whether the particle is in buffer (0 is not in buffer,
// 1 is in buffer)
// =================================================================================================================
__global__ void kinematicStep2(float3* pos_data,
                               int* offset_BSD_data,
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
                               float domain_z) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= k_n) {
        return;
    }

    int start_idx = offset_BSD_data[idx];

    float dx_2_0 = int(pos_data[idx].x - (-domain_x / 2));
    float dy_2_0 = int(pos_data[idx].y - (-domain_y / 2));
    float dz_2_0 = int(pos_data[idx].z - (-domain_z / 2));

    int x_idx = int(dx_2_0 / d_domain_x);
    int y_idx = int(dy_2_0 / d_domain_x);
    int z_idx = int(dz_2_0 / d_domain_x);

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
        float BSD_x_start = x_check_idx * d_domain_x - 4 * kernel_h;
        float BSD_x_end = (x_check_idx + 1) * d_domain_x + 4 * kernel_h;
        float BSD_y_start = y_check_idx * d_domain_y - 4 * kernel_h;
        float BSD_y_end = (y_check_idx + 1) * d_domain_y + 4 * kernel_h;
        float BSD_z_start = z_check_idx * d_domain_z - 4 * kernel_h;
        float BSD_z_end = (z_check_idx + 1) * d_domain_z + 4 * kernel_h;

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
    __syncthreads();
}

// =================================================================================================================
// Kinematic 5th Step, this is the 1st pass of the kinematic thread
// We will use shared memory to store particle location data
// the 1st pass is going to fill the num_col vector
// =================================================================================================================

__global__ void kinematicStep5(
    float3* pos_data,            // particle position data vector
    int k_n,                     // total number of particles
    float tolerance,             // collision detection tolerance
    float kernel_h,              // kernel_h of the uni-kernel_h particles
    int* idx_track_data_sorted,  // sorted idx_track data
    int* BSD_iden_idx_sorted,    // vector to indicate whether a particle is in buffer zone or not
    int* offset_BSD_data,        // length the same as unique_BSD_idx
    int* length_BSD_data,        // length the same as unique_BSD_idx
    int* unique_BSD_idx,
    int* num_col,
    int unique_length) {
    __shared__ float3 pos_local[512];  // request maximum capacity for the shared mem
    __shared__ int idx_local[512];     // request maximum capacity for track
    __shared__ int iden_local[512];
    __shared__ int tot_in_bsd;

    int sd_idx = blockIdx.x;
    int idx = threadIdx.x;

    if (sd_idx >= unique_length) {
        return;
    }

    if (threadIdx.x == 0) {
        tot_in_bsd = length_BSD_data[sd_idx];
        int start_idx = offset_BSD_data[sd_idx];
        int end_idx = offset_BSD_data[sd_idx] + length_BSD_data[sd_idx];
        for (int i = start_idx; i < end_idx; i++) {
            pos_local[i - start_idx] = pos_data[idx_track_data_sorted[i]];
            idx_local[i - start_idx] = idx_track_data_sorted[i];
            iden_local[i - start_idx] = BSD_iden_idx_sorted[i];
        }
    }

    __syncthreads();

    int count = 0;

    if (idx >= tot_in_bsd) {
        return;
    }

    if (iden_local[idx] == 1) {
        return;
    }

    for (int i = 0; i < tot_in_bsd; i++) {
        if ((i != idx) && (idx_local[idx] < idx_local[i])) {
            float dist2 = (pos_local[i].x - pos_local[idx].x) * (pos_local[i].x - pos_local[idx].x) +
                          (pos_local[i].y - pos_local[idx].y) * (pos_local[i].y - pos_local[idx].y) +
                          (pos_local[i].z - pos_local[idx].z) * (pos_local[i].z - pos_local[idx].z);

            if (dist2 <= (kernel_h * 2 + tolerance) * (kernel_h * 2 + tolerance)) {
                count++;
            }
        }
    }

    num_col[sd_idx * 512 + idx] = count;
}

// =================================================================================================================
// Kinematic 5th Step, this is the 1st pass of the kinematic thread
// We will use shared memory to store particle location data
// the 1st pass is going to fill the num_col vector
// =================================================================================================================

__global__ void kinematicStep7(
    float3* pos_data,            // particle position data vector
    int k_n,                     // total number of particles
    float tolerance,             // collision detection tolerance
    float kernel_h,              // kernel_h of the uni-kernel_h particles
    int* idx_track_data_sorted,  // sorted idx_track data
    int* BSD_iden_idx_sorted,    // vector to indicate whether a particle is in buffer zone or not
    int* offset_BSD_data,        // length the same as unique_BSD_idx
    int* length_BSD_data,        // length the same as unique_BSD_idx
    int* unique_BSD_idx,
    int* num_col,
    int unique_length,
    int* pair_i_data,
    int* pair_j_data,
    int* num_col_offset,
    float3* W_grad_data) {
    __shared__ float3 pos_local[512];  // request maximum capacity for the shared mem
    __shared__ int idx_local[512];     // request maximum capacity for track
    __shared__ int iden_local[512];
    __shared__ int tot_in_bsd;

    int sd_idx = blockIdx.x;
    int idx = threadIdx.x;

    if (sd_idx >= unique_length) {
        return;
    }

    if (threadIdx.x == 0) {
        tot_in_bsd = length_BSD_data[sd_idx];
        int start_idx = offset_BSD_data[sd_idx];
        int end_idx = offset_BSD_data[sd_idx] + length_BSD_data[sd_idx];
        for (int i = start_idx; i < end_idx; i++) {
            pos_local[i - start_idx] = pos_data[idx_track_data_sorted[i]];
            idx_local[i - start_idx] = idx_track_data_sorted[i];
            iden_local[i - start_idx] = BSD_iden_idx_sorted[i];
        }
    }

    __syncthreads();

    int count = 0;

    if (idx >= tot_in_bsd) {
        return;
    }

    if (iden_local[idx] == 1) {
        return;
    }

    for (int i = 0; i < tot_in_bsd; i++) {
        if ((i != idx) && (idx_local[idx] < idx_local[i])) {
            float dist2 = (pos_local[i].x - pos_local[idx].x) * (pos_local[i].x - pos_local[idx].x) +
                          (pos_local[i].y - pos_local[idx].y) * (pos_local[i].y - pos_local[idx].y) +
                          (pos_local[i].z - pos_local[idx].z) * (pos_local[i].z - pos_local[idx].z);

            if (dist2 <= (kernel_h * 2 + tolerance) * (kernel_h * 2 + tolerance)) {
                pair_i_data[num_col_offset[sd_idx * 512 + idx] + count] = idx_local[idx];
                pair_j_data[num_col_offset[sd_idx * 512 + idx] + count] = idx_local[i];

                float3 dir = pos_local[idx] - pos_local[i];

                W_grad_data[num_col_offset[sd_idx * 512 + idx] + count] = W_Grad(dir, kernel_h * 2);

                count++;
            }
        }
    }
}

// =================================================================================================================
// Kinematic 8th Step, this is the 1st pass of the kinematic thread
// We will use shared memory to store particle location data
// the 1st pass is going to fill the num_col vector
// =================================================================================================================

__global__ void kinematicStep8(int* pair_i_data, int* pair_j_data, int* inv_pair_i_data, int* inv_pair_j_data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) {
        return;
    }

    inv_pair_i_data[idx] = pair_j_data[idx];
    inv_pair_j_data[idx] = pair_i_data[idx];
}

// pos_data.data(), rho_data.data(), pressure_data.data(), i_unique.data(), i_offset.data(),
// j_data_sorted.data(), i_unique.size()

__global__ void kinematicStep9(float3* pos_data,
                               float* rho_data,
                               float* pressure_data,
                               int* i_unique,
                               int* i_offset,
                               int* i_length,
                               int* j_data_sorted,
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

    float rho_sum = 0;

    for (int i = 0; i < len; i++) {
        int j_idx = j_data_sorted[start_idx + i];
        float3 dir = pos_data[i_idx] - pos_data[j_idx];

        float w = W(dir, 2 * h);
        // printf("%f", w);
        rho_sum = rho_sum + m * w;
    }

    __syncthreads();

    rho_data[i_idx] = rho_sum;
    pressure_data[i_idx] = 100 * (rho_sum - rho_0);
}