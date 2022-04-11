// SPH-DualGPU
// driver code

#include <sph/SPHSystem.h>

using namespace sgps;

int main(int argc, char* argv[]) {
    // initialize particles in a cubic 10x10x10 domain
    float dim_x = 32;
    float dim_y = 32;
    float dim_z = 8;

    // set particle kernel_h
    float kernel_h = 0.2;
    float gap = 0.02;

    // calculate number of particles on each direction
    int num_x = dim_x / (kernel_h * 2 + gap);
    int num_y = dim_y / (kernel_h * 2 + gap);
    int num_z = 2;

    // create position array of all particles
    std::vector<float3> pos_vec;
    std::vector<float3> vel_vec;
    std::vector<float3> acc_vec;
    std::vector<char> fix_vec;

    // sample all particles
    int num_par = 0;
    // sample the bottom layer
    for (int j = 0 / 2; j < num_y / 2; j++) {
        for (int i = 0 / 2; i < num_x / 2; i++) {
            float temp_z = -dim_z / 2 + kernel_h;
            pos_vec.push_back(make_float3(-dim_x / 8 + i * (2 * kernel_h + gap) + kernel_h,
                                          -dim_y / 8 + j * (2 * kernel_h + gap) + kernel_h, temp_z));
            vel_vec.push_back(make_float3(0, 0, 0));
            acc_vec.push_back(make_float3(0, 0, 0));
            fix_vec.push_back(1);
            num_par++;
        }
    }

    // sample a smaller domain for drop test
    for (int k = 1; k < 4; k++) {
        for (int j = 0; j < num_y / 4; j++) {
            for (int i = 0; i < num_x / 4; i++) {
                float temp_z = -dim_z / 2 + 2 * (kernel_h + 2 * gap) * (k) + kernel_h + gap;

                pos_vec.push_back(make_float3(-dim_x / 16 + i * (2 * kernel_h + gap) + kernel_h,
                                              -dim_y / 16 + j * (2 * kernel_h + gap) + kernel_h, temp_z));
                vel_vec.push_back(make_float3(0, 0, 0));
                acc_vec.push_back(make_float3(0, 0, 0));
                fix_vec.push_back(0);

                num_par++;
            }
        }
    }
    std::cout << "Total number of particles is " << num_par << std::endl;

    // create a new GpuManager
    GpuManager gpu_distributor(2);

    // create SPHSystem
    SPHSystem* system = new SPHSystem(gpu_distributor);

    // initialize the SPHSystem
    system->initialize(kernel_h, 330.5103, 1000, pos_vec, vel_vec, acc_vec, fix_vec, dim_x, dim_y, dim_z);
    system->setPrintOut(true, 2);
    system->doStepDynamics(0.0002f, 1.0f);
}
