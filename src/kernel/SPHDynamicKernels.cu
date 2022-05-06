// All Dynamic CUDA kernels for SPH in gpu-physics

#include <sph/datastruct.h>
#include <kernel/CUDAMathHelpers.cu>
// =================================================================================================================
// ========================================= START of Dynamic kernels ==============================================
// =================================================================================================================

// =================================================================================================================
// Dynamic 1st Pass, this pass compute the contact force of each contact pair
// the computed force will be filled in contact_force field in each contact pair
// =================================================================================================================
__global__ void dynamicStep1(int* pair_i_data,
                             int* pair_j_data,
                             float* rho_data,
                             float* pressure_data,
                             float3* col_acc_data,
                             float3* W_grad_data,
                             int n_col,
                             float kernel_h,
                             float m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_col) {
        return;
    }

    int i_idx = pair_i_data[idx];
    int j_idx = pair_j_data[idx];

    if (rho_data[i_idx] != 0 && rho_data[j_idx] != 0) {
        float coe = m * ((pressure_data[i_idx] / (rho_data[i_idx] * rho_data[i_idx])) +
                         (pressure_data[j_idx] / (rho_data[j_idx] * rho_data[j_idx])));
        col_acc_data[idx] = -coe * W_grad_data[idx];
    }
}

__global__ void dynamicStep2(int* pair_i_data_reduced, float3* col_acc_data_reduced, float3* acc_data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) {
        return;
    }

    acc_data[pair_i_data_reduced[idx]] = col_acc_data_reduced[idx];
}

__global__ void dynamicStep3(float3* col_acc_data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) {
        return;
    }

    col_acc_data[idx].x = -col_acc_data[idx].x;
    col_acc_data[idx].y = -col_acc_data[idx].y;
    col_acc_data[idx].z = -col_acc_data[idx].z;
}

__global__ void dynamicStep4(int* pair_j_data_reduced, float3* col_acc_data_reduced, float3* acc_data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) {
        return;
    }

    acc_data[pair_j_data_reduced[idx]] = col_acc_data_reduced[idx];
}

// ===========================================

__global__ void dynamicStep5(float3* pos_data,
                             float3* vel_data,
                             float3* acc_data,
                             char* fix_data,
                             int n,
                             float time_step,
                             float kernel_h) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) {
        return;
    }

    if (fix_data[idx] == 0) {
        float grav = -9.8f;
        acc_data[idx].z += grav;

        vel_data[idx].x += acc_data[idx].x * time_step;
        vel_data[idx].y += acc_data[idx].y * time_step;
        vel_data[idx].z += acc_data[idx].z * time_step;

        pos_data[idx].x += vel_data[idx].x * time_step;
        pos_data[idx].y += vel_data[idx].y * time_step;
        pos_data[idx].z += vel_data[idx].z * time_step;
    } else {
        acc_data[idx] = make_float3(0.f, 0.f, 0.f);
    }

    __syncthreads();
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