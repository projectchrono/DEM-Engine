#include <sph/datastruct.h>
// =================================================================================================================
// ========================================= START of Kinematic kernels ============================================
// =================================================================================================================

// =================================================================================================================
// Kinematic 1st Step, this pass identifies number of BSDs touched by each particle
// This kernel also fills the idx_track vector
// =================================================================================================================
__global__ void kinematicStep1(vector3* pos_data,
                               int* num_BSD_data,
                               int k_n,
                               float radius,
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
    int bd_idx = z_idx * num_domain_x * num_domain_y + y_idx * num_domain_x + x_idx;

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
        float BSD_x_start = x_check_idx * d_domain_x - 4 * radius;
        float BSD_x_end = (x_check_idx + 1) * d_domain_x + 4 * radius;
        float BSD_y_start = y_check_idx * d_domain_y - 4 * radius;
        float BSD_y_end = (y_check_idx + 1) * d_domain_y + 4 * radius;
        float BSD_z_start = z_check_idx * d_domain_z - 4 * radius;
        float BSD_z_end = (z_check_idx + 1) * d_domain_z + 4 * radius;

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
__global__ void kinematicStep2(vector3* pos_data,
                               int* offset_BSD_data,
                               int* BSD_iden_idx,
                               int* BSD_idx,
                               int* idx_track_data,
                               int k_n,
                               float TotLength,
                               float radius,
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
    int length = 0;
    if (idx != TotLength - 1) {
        length = offset_BSD_data[idx + 1] - offset_BSD_data[idx];
    } else {
        length = TotLength - offset_BSD_data[idx - 1];
    }

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
        float BSD_x_start = x_check_idx * d_domain_x - 4 * radius;
        float BSD_x_end = (x_check_idx + 1) * d_domain_x + 4 * radius;
        float BSD_y_start = y_check_idx * d_domain_y - 4 * radius;
        float BSD_y_end = (y_check_idx + 1) * d_domain_y + 4 * radius;
        float BSD_z_start = z_check_idx * d_domain_z - 4 * radius;
        float BSD_z_end = (z_check_idx + 1) * d_domain_z + 4 * radius;

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
    vector3* pos_data,           // particle position data vector
    int k_n,                     // total number of particles
    float tolerance,             // collision detection tolerance
    float radius,                // radius of the uni-radius particles
    int* idx_track_data_sorted,  // sorted idx_track data
    int* BSD_iden_idx_sorted,    // vector to indicate whether a particle is in buffer zone or not
    int* offset_BSD_data,        // length the same as unique_BSD_idx
    int* length_BSD_data,        // length the same as unique_BSD_idx
    int* unique_BSD_idx,
    int* num_col,
    int unique_length) {
    __shared__ vector3 pos_local[512];  // request maximum capacity for the shared mem
    __shared__ int idx_local[512];      // request maximum capacity for track
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

            if (dist2 <= (radius * 2 + tolerance) * (radius * 2 + tolerance)) {
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
    vector3* pos_data,           // particle position data vector
    int k_n,                     // total number of particles
    float tolerance,             // collision detection tolerance
    float radius,                // radius of the uni-radius particles
    int* idx_track_data_sorted,  // sorted idx_track data
    int* BSD_iden_idx_sorted,    // vector to indicate whether a particle is in buffer zone or not
    int* offset_BSD_data,        // length the same as unique_BSD_idx
    int* length_BSD_data,        // length the same as unique_BSD_idx
    int* unique_BSD_idx,
    int* num_col,
    int unique_length,
    contactData* contact_data,
    int contact_n,
    int* num_col_offset) {
    __shared__ vector3 pos_local[512];  // request maximum capacity for the shared mem
    __shared__ int idx_local[512];      // request maximum capacity for track
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

            if (dist2 <= (radius * 2 + tolerance) * (radius * 2 + tolerance)) {
                contact_data[num_col_offset[sd_idx * 512 + idx] + count].contact_pair.x = idx_local[idx];
                contact_data[num_col_offset[sd_idx * 512 + idx] + count].contact_pair.y = idx_local[i];
                count++;
            }
        }
    }
}

// ====================================== The OLD-YOUNG TERMINATOR ===============================

// =================================================================================================================
// Kinematic 1st Pass, this pass identifies each particle into it's corresponding cd
// the cd idx is stored in cd_idx
// =================================================================================================================
__global__ void kinematic1Pass(vector3* pos,
                               int* cd_idx,
                               int* idx_track,
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

    idx_track[idx] = idx;

    float dx_2_0 = int(pos[idx].x - (-domain_x / 2));
    float dy_2_0 = int(pos[idx].y - (-domain_y / 2));
    float dz_2_0 = int(pos[idx].z - (-domain_z / 2));

    int x_idx = int(dx_2_0 / d_domain_x);
    int y_idx = int(dy_2_0 / d_domain_x);
    int z_idx = int(dz_2_0 / d_domain_x);

    cd_idx[idx] = z_idx * num_domain_x * num_domain_y + y_idx * num_domain_x + x_idx;
}

// =================================================================================================================
// Kinematic 2nd Pass, this pass performs a header sweep to find the 'head' and 'tail' of each cd
// if the cd is 'activated', there is(are) particle(s) in the cd - then [cdidx*2] = 'h' and [cdidx*2+1] = 't'
// if the cd is 'inactivated', there is(are) not particle(s) in the cd - then [cdidx*2] = -1 and [cdidx*2+1] = -1
// =================================================================================================================
__global__ void kinematic2Pass(int* idx_sorted, int* idx_ht_data, int idx_size, int hd_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= idx_size) {
        return;
    }

    // marginal case handling
    if (idx == 0) {
        idx_ht_data[idx_sorted[idx] * 2] = idx;
        return;
    }

    if (idx == idx_size - 1) {
        idx_ht_data[idx_sorted[idx] * 2 + 1] = idx;
        return;
    }

    // examine the (current idx) and (current idx - 1)
    // here we already assumed that idx_sorted[idx] >= idx_sorted[idx-1] after radix sort
    if (idx_sorted[idx] != idx_sorted[idx - 1]) {
        idx_ht_data[idx_sorted[idx - 1] * 2 + 1] = idx - 1;
        idx_ht_data[idx_sorted[idx] * 2] = idx;
    }

    // examine the (current idx + 1) and (current idx)
    // here we already assumed that idx_sorted[idx+1] >= idx_sorted[idx] after radix sort
    /*
    if (idx_sorted[idx + 1] != idx_sorted[idx]) {
        idx_ht_data[idx_sorted[idx] * 2 + 1] = idx;
        idx_ht_data[idx_sorted[idx + 1] * 2] = idx + 1;
    }*/
}

// =================================================================================================================
// Kinematic 3rd Pass, this pass compute total number of particles in each sd
// =================================================================================================================

__global__ void kinematic3Pass(int* idx_hd, int* subdomain_decomp, int num_cd_each_sd, int* n_each_sd, int n_sd) {
    int sd_idx = threadIdx.x;
    if (sd_idx >= n_sd) {
        return;
    }

    int total = 0;
    for (int i = 0; i < num_cd_each_sd; i++) {
        int cd_idx = subdomain_decomp[sd_idx * num_cd_each_sd + i];
        if (idx_hd[2 * cd_idx] != -1) {
            total = total + idx_hd[2 * cd_idx + 1] - idx_hd[2 * cd_idx] + 1;
        }
    }

    n_each_sd[sd_idx] = total;
}

// =================================================================================================================
// Kinematic 4th Pass, this pass computes the total number of valid contact for each particle in each subdomain
// This is the 1st pass of the kinematic main loop
// =================================================================================================================

__global__ void kinematic4Pass(vector3* pos,
                               int n,
                               float tolerance,
                               float radius,
                               int* res_arr,
                               int num_cd_each_sd,
                               int* subdomain_decomp,
                               int* idx_track_sorted,
                               int* idx_ht_data,
                               int* n_each_sd,
                               int n_sd) {
    __shared__ vector3 pos_sd[512];  // request maximum capacity for the shared mem
    __shared__ int idx_track[512];   // request maximum capacity for track
    __shared__ int cd_sz_track[64];
    __shared__ int tot_in_sd;
    // It's supposed to be the following line but it's giving me an error
    // extern __shared__ int cd_sz_track[];

    int sd_idx = blockIdx.x;
    int idx = threadIdx.x;

    if (sd_idx >= n_sd) {
        return;
    }

    if (threadIdx.x == 0) {
        tot_in_sd = n_each_sd[sd_idx];
        for (int i = 0; i < num_cd_each_sd; i++) {
            int cd_idx = subdomain_decomp[sd_idx * num_cd_each_sd + i];
            if (idx_ht_data[2 * cd_idx] != -1) {
                cd_sz_track[i] = idx_ht_data[2 * cd_idx + 1] - idx_ht_data[2 * cd_idx] + 1;
            } else {
                cd_sz_track[i] = 0;
            }
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 0; i < num_cd_each_sd; i++) {
            int cd_idx = subdomain_decomp[sd_idx * num_cd_each_sd + i];
            int cur_sz = 0;

            for (int j = 0; j < i; j++) {
                cur_sz = cur_sz + cd_sz_track[j];
            }

            if (idx_ht_data[2 * cd_idx] != -1) {
                int cd_h = idx_ht_data[2 * cd_idx];
                int cd_d = idx_ht_data[2 * cd_idx + 1];

                for (int j = 0; j < (cd_d - cd_h + 1); j++) {
                    pos_sd[cur_sz + j] = pos[idx_track_sorted[cd_h + j]];
                    idx_track[cur_sz + j] = idx_track_sorted[cd_h + j];
                }
            }
        }
    }

    __syncthreads();

    int count = 0;

    if (idx >= tot_in_sd) {
        return;
    }

    for (int i = 0; i < tot_in_sd; i++) {
        if ((i != idx) && (idx_track[idx] < idx_track[i])) {
            float dist2 = (pos_sd[i].x - pos_sd[idx].x) * (pos_sd[i].x - pos_sd[idx].x) +
                          (pos_sd[i].y - pos_sd[idx].y) * (pos_sd[i].y - pos_sd[idx].y) +
                          (pos_sd[i].z - pos_sd[idx].z) * (pos_sd[i].z - pos_sd[idx].z);

            if (dist2 <= (radius * 2 + tolerance) * (radius * 2 + tolerance)) {
                count++;
            }
        }
    }

    res_arr[sd_idx * 512 + idx] = count;
}

// =================================================================================================================
// Kinematic 5th Pass, this pass fills in the contact_data -> contact_pair fields (i and j idx)
// This is the 2nd pass of the kinematic main loop
// =================================================================================================================

__global__ void kinematic5Pass(vector3* pos,
                               int n,
                               int* offset,
                               int* contact_num_arr,
                               float tolerance,
                               float radius,
                               contactData* pair_data,
                               int num_cd_each_sd,
                               int* subdomain_decomp,
                               int* idx_track_sorted,
                               int* idx_ht_data,
                               int* n_each_sd,
                               int n_sd,
                               int contact_sum) {
    __shared__ vector3 pos_sd[512];  // request maximum capacity for the shared mem
    __shared__ int idx_track[512];   // request maximum capacity for track
    __shared__ int cd_sz_track[64];
    __shared__ int tot_in_sd;

    int sd_idx = blockIdx.x;
    int idx = threadIdx.x;

    if (contact_num_arr[sd_idx] == 0) {
        return;
    }

    if (sd_idx >= n_sd) {
        return;
    }

    if (threadIdx.x == 0) {
        tot_in_sd = n_each_sd[sd_idx];
        for (int i = 0; i < num_cd_each_sd; i++) {
            int cd_idx = subdomain_decomp[sd_idx * num_cd_each_sd + i];
            if (idx_ht_data[2 * cd_idx] != -1) {
                cd_sz_track[i] = idx_ht_data[2 * cd_idx + 1] - idx_ht_data[2 * cd_idx] + 1;
            } else {
                cd_sz_track[i] = 0;
            }
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 0; i < num_cd_each_sd; i++) {
            int cd_idx = subdomain_decomp[sd_idx * num_cd_each_sd + i];
            int cur_sz = 0;

            for (int j = 0; j < i; j++) {
                cur_sz = cur_sz + cd_sz_track[j];
            }

            if (idx_ht_data[2 * cd_idx] != -1) {
                int cd_h = idx_ht_data[2 * cd_idx];
                int cd_d = idx_ht_data[2 * cd_idx + 1];

                for (int j = 0; j < (cd_d - cd_h + 1); j++) {
                    pos_sd[cur_sz + j] = pos[idx_track_sorted[cd_h + j]];
                    idx_track[cur_sz + j] = idx_track_sorted[cd_h + j];
                }
            }
        }
    }

    __syncthreads();

    if (idx >= tot_in_sd) {
        return;
    }

    if (contact_num_arr[sd_idx * 512 + idx] == 0)
        return;

    int cur_idx = offset[sd_idx * 512 + idx];

    for (int i = 0; i < tot_in_sd; i++) {
        if ((i != idx) && (idx_track[idx] < idx_track[i])) {
            float dist2 = (pos_sd[i].x - pos_sd[idx].x) * (pos_sd[i].x - pos_sd[idx].x) +
                          (pos_sd[i].y - pos_sd[idx].y) * (pos_sd[i].y - pos_sd[idx].y) +
                          (pos_sd[i].z - pos_sd[idx].z) * (pos_sd[i].z - pos_sd[idx].z);

            if (dist2 <= (radius * 2 + tolerance) * (radius * 2 + tolerance)) {
                pair_data[cur_idx].contact_pair.x = idx_track[idx];
                pair_data[cur_idx].contact_pair.y = idx_track[i];

                cur_idx++;
            }
        }
    }
    __syncthreads();
}

// =================================================================================================================
// ========================================= END of Kinematic kernels ==============================================
// =================================================================================================================