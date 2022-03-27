// SPH-DualGPU
// driver code

#include <sph/SPHSystem.h>

int main(int argc, char* argv[]) {
    // initialize particles in a cubic 10x10x10 domain
    float dim_x = 32;
    float dim_y = 32;
    float dim_z = 8;

    // set particle radius
    float radius = 0.2;
    float gap = 0.02;

    // calculate number of particles on each direction
    int num_x = dim_x / (radius * 2 + gap);
    int num_y = dim_y / (radius * 2 + gap);
    int num_z = 2;

    // create position array of all particles
    std::vector<vector3> pos_vec;
    std::vector<vector3> vel_vec;
    std::vector<vector3> acc_vec;
    std::vector<char> fix_vec;

    // random z generation
    // float max_z = 2 * -dim_z / 2 + 3 * (2 * radius + gap) + radius;
    // float min_z = 2 * -dim_z / 2 + 1 * (2 * radius + gap) + radius;

    // sample all particles
    int num_par = 0;
    // sample the bottom layer
    for (int j = num_y / 16; j < num_y / 2; j++) {
        for (int i = num_y / 16; i < num_x / 2; i++) {
            float temp_z = -dim_z / 2 + radius;
            pos_vec.push_back(vector3(-dim_x / 8 + i * (2 * radius + gap) + radius,
                                      -dim_y / 8 + j * (2 * radius + gap) + radius, temp_z));
            vel_vec.push_back(vector3(0, 0, 0));
            acc_vec.push_back(vector3(0, 0, 0));
            fix_vec.push_back(1);
            num_par++;
        }
    }

    // sample a smaller domain for drop test
    for (int k = 1; k < num_z; k++) {
        for (int j = num_y / 16; j < num_y / 2; j++) {
            for (int i = num_y / 16; i < num_x / 2; i++) {
                float temp_z = -dim_z / 2 + 2 * (radius + 2 * gap) * (k) + radius + gap;
                /*
                if (k % 2 == 0) {
                    pos_vec.push_back(vector3(-dim_x / 4 + i * (2 * radius + gap) + radius,
                                              -dim_y / 4 + j * (2 * radius + gap) + radius, temp_z));
                } else {
                    pos_vec.push_back(vector3(-dim_x / 4 + 2 * gap + i * (2 * radius + 2 * gap) + radius,
                                              -dim_y / 4 + 2 * gap + j * (2 * radius + 2 * gap) + radius, temp_z));
                }*/
                pos_vec.push_back(vector3(-dim_x / 8 + i * (2 * radius + gap) + radius,
                                          -dim_y / 8 + j * (2 * radius + gap) + radius, temp_z));
                vel_vec.push_back(vector3(0, 0, 0));
                acc_vec.push_back(vector3(0, 0, 0));
                fix_vec.push_back(0);

                num_par++;
            }
        }
    }
    std::cout << "Total number of particles is " << num_par << std::endl;

    /*
        // set particle radius
        float radius = 0.2;
        std::vector<vector3> pos_vec;
        pos_vec.push_back(vector3(0.f, 0.f, 0.21f));
        pos_vec.push_back(vector3(0.f, 0.f, 0.65f));

        std::vector<vector3> vel_vec;
        vel_vec.push_back(vector3(0.f, 0.f, 0.f));
        vel_vec.push_back(vector3(0.f, 0.f, 0.f));

        std::vector<vector3> acc_vec;
        acc_vec.push_back(vector3(0.f, 0.f, 0.f));
        acc_vec.push_back(vector3(0.f, 0.f, 0.f));

        std::vector<bool> fix_vec;
        fix_vec.push_back(true);
        fix_vec.push_back(false);
        */

    // create a new GpuManager
    GpuManager gpu_distributor(2);

    // create SPHSystem
    SPHSystem* system = new SPHSystem(gpu_distributor);

    // initialize the SPHSystem
    system->initialize(radius, pos_vec, vel_vec, acc_vec, fix_vec, dim_x, dim_y, dim_z);
    system->setPrintOut(true, 2);
    system->doStepDynamics(0.0025f, 1.0f);
}
