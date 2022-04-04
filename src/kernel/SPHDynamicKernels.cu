// All Dynamic CUDA kernels for SPH in gpu-physics

#include <sph/datastruct.h>

// =================================================================================================================
// ========================================= START of Dynamic kernels ==============================================
// =================================================================================================================

// =================================================================================================================
// Dynamic 1st Pass, this pass compute the contact force of each contact pair
// the computed force will be filled in contact_force field in each contact pair
// =================================================================================================================
__global__ void dynamicStep1(contactData* gpu_pair_data,
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
        float coe = 300.f;
        float dist = sqrt(dist2);

        float dir_x_norm = dir_x / dist;
        float dir_y_norm = dir_y / dist;
        float dir_z_norm = dir_z / dist;

        float penetration = 2 * radius - dist;
        // fill in contact pair data with respect to the first element of the contact pair data
        gpu_pair_data[idx].contact_force.x = dir_x_norm * penetration * coe;
        gpu_pair_data[idx].contact_force.y = dir_y_norm * penetration * coe;
        gpu_pair_data[idx].contact_force.z = dir_z_norm * penetration * coe;
    }
}

// =================================================================================================================
// Dynamic 2nd pass, this pass is intended to flaten the array
// for the original contact_pair array, pair i_a and j_a will only appear once
// this pass makes sure that (i_a,j_a) will be in one contact_pair element and (j_a,i_a) will be in one contact_pair
// =================================================================================================================
__global__ void dynamicStep2(contactData* gpu_pair_data, int gpu_pair_n, contactData* inv_gpu_pair_data) {
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
}

// =================================================================================================================
// ========================================= END of Dynamic kernels ================================================
// =================================================================================================================