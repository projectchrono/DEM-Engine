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

// GPU header sweep to find the starting idx of each cell
// if the cell is activated (there are particles in the current cell), fill two elements 'head idx' and 'tail
// idx'
// if the cell is not activated (there is not any particle in the current cell), fill two elements 'head idx' and 'tail
// idx' with -1
__global__ void hdSweep(int* idx_sorted, int* idx_hd_data, int idx_size, int hd_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= idx_size || idx == 0 || idx == idx_size - 1) {
        return;
    }
    __syncthreads();

    // examine the (current idx) and (current idx - 1)
    // here we already assumed that idx_sorted[idx] >= idx_sorted[idx-1] after radix sort
    if (idx_sorted[idx] != idx_sorted[idx - 1]) {
        int diff = idx_sorted[idx] - idx_sorted[idx - 1];
        idx_hd_data[idx_sorted[idx - 1] * 2 + 1] = idx - 1;

        idx_hd_data[idx_sorted[idx] * 2] = idx;
        // for all non - activated cell heading and tail number
        for (int i = 0; i < diff - 1; i++) {
            idx_hd_data[idx_sorted[idx - 1] * 2 + 2 + 2 * i] = -1;
            idx_hd_data[idx_sorted[idx - 1] * 2 + 2 + 2 * i + 1] = -1;
        }
    }

    // examine the (current idx + 1) and (current idx)
    // here we already assumed that idx_sorted[idx+1] >= idx_sorted[idx] after radix sort
    if (idx_sorted[idx + 1] != idx_sorted[idx]) {
        int diff = idx_sorted[idx + 1] - idx_sorted[idx];
        idx_hd_data[idx_sorted[idx] * 2 - 1] = idx;
        idx_hd_data[idx_sorted[idx + 1] * 2 - 2] = idx + 1;
        // for all non-activated cell heading and tail number
        for (int i = 0; i < diff - 1; i++) {
            idx_hd_data[idx_sorted[idx] * 2 + 2 * i] = -1;
            idx_hd_data[idx_sorted[idx] * 2 + 2 * i + 1] = -1;
        }
    }
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