#include <sph/datastruct.h>
// =================================================================================================================
// ========================================= START of Kinematic kernels ============================================
// =================================================================================================================

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
    if (idx_sorted[idx + 1] != idx_sorted[idx]) {
        idx_ht_data[idx_sorted[idx] * 2 + 1] = idx;
        idx_ht_data[idx_sorted[idx + 1] * 2] = idx + 1;
    }
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
// Kinematic 4th Pass, this pass
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
    __shared__ int cd_sz_track[1024];
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
// Kinematic 5th Pass, this pass
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
    __shared__ int cd_sz_track[1024];
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