// All Dynamic CUDA kernels for SPH in gpu-physics

#include <sph/datastruct.h>
#include <kernel/CUDAMathHelpers.cu>
// =================================================================================================================
// ========================================= START of Dynamic kernels ==============================================
// =================================================================================================================

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
    // printf("density is: %f\n", rho_data[idx]);
    pressure_data[idx] = c * c * (rho_data[idx] - rho_0) + 0.05 * c * c * rho_0;
}

// =================================================================================================================
// Dynamic 1st Pass, this pass compute the contact force of each contact pair
// the computed force will be filled in contact_force field in each contact pair
// =================================================================================================================
__global__ void getIndividualAcc(
    int* pair_i_data,
    int* pair_j_data,
    float* rho_data,
    float* pressure_data,
    float3* pos_data,
    float3* col_acc_data,
    int n_col,
    float m,
    float kernel_h) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_col) {
        return;
    }

    int i_idx = pair_i_data[idx];
    int j_idx = pair_j_data[idx];

    if (rho_data[i_idx] != 0 && rho_data[j_idx] != 0) {
        float3 rij = pos_data[i_idx] - pos_data[j_idx]; 
        float coe = m * ((pressure_data[i_idx] / (rho_data[i_idx] * rho_data[i_idx])) +
                         (pressure_data[j_idx] / (rho_data[j_idx] * rho_data[j_idx])));
        col_acc_data[idx] = -coe * W_Grad(rij, kernel_h);
    }
}

__global__ void assignAccData(int* pair_i_data_reduced, float3* col_acc_data_reduced, float3* acc_data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) {
        return;
    }

    acc_data[pair_i_data_reduced[idx]] += col_acc_data_reduced[idx];
}

__global__ void negateColAccData(float3* col_acc_data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) {
        return;
    }

    col_acc_data[idx].x = -col_acc_data[idx].x;
    col_acc_data[idx].y = -col_acc_data[idx].y;
    col_acc_data[idx].z = -col_acc_data[idx].z;
}

// ===========================================

__global__ void timeIntegration(float3* pos_data,
                             float3* vel_data,
                             float3* acc_data,
                             char* fix_data,
                             int n,
                             float time_step) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) {
        return;
    }

    if (fix_data[idx] == 0) {
        float grav = -9.8f;
        acc_data[idx].z = grav;

        vel_data[idx].x += acc_data[idx].x * time_step;
        vel_data[idx].y += acc_data[idx].y * time_step;
        vel_data[idx].z += acc_data[idx].z * time_step;

        pos_data[idx].x += vel_data[idx].x * time_step;
        pos_data[idx].y += vel_data[idx].y * time_step;
        pos_data[idx].z += vel_data[idx].z * time_step;
    } else {
        acc_data[idx] = make_float3(0.f, 0.f, 0.f);
    }
}

/*
// =================================================================================================================
// Dynamic 3rd pass, this pass is intended to copy all reduced data into global memory gpu_acc
// TODO: reconsider the necessity of existence of this kernel
// =================================================================================================================
__global__ void dynamicStep3(int* key, float* x_reduced, float* y_reduced, float* z_reduced, int n, vector3* gpu_acc) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) {
        return;
    }
    __syncthreads();

    gpu_acc[key[idx]].x = x_reduced[idx];
    gpu_acc[key[idx]].y = y_reduced[idx];
    gpu_acc[key[idx]].z = z_reduced[idx];
}

// =================================================================================================================
// Dynamic 4th pass, the actual integration pass
// this pass will integrate gpu_acc to gpu_vel to gpu_pos
// and push simulation 1 step forward
// =================================================================================================================
__global__ void dynamicStep4(vector3* gpu_pos,
                             vector3* gpu_vel,
                             vector3* gpu_acc,
                             char* gpu_fix,
                             int gpu_n,
                             float time_step,
                             float kernel_h) {
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
}
*/

// =================================================================================================================
// ========================================= END of Dynamic kernels ================================================
// =================================================================================================================
