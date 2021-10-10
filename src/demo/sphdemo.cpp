// SPH-DualGPU
// driver code

#include <sph/SPHSystem.h>

int main(int argc, char* argv[]) {
    // initialize particles in a cubic 10x10x10 domain
    float dim_x = 4;
    float dim_y = 4;
    float dim_z = 4;

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
    float max_z = 2 * -dim_z / 2 + 3 * (2 * radius + gap) + radius;
    float min_z = 2 * -dim_z / 2 + 1 * (2 * radius + gap) + radius;

    // sample all particles
    for (int k = 0; k < num_z; k++) {
        for (int j = 0; j < num_y; j++) {
            for (int i = 0; i < num_x; i++) {
                float temp_z = 0;
                if (k == 1) {
                    temp_z = -dim_z / 2 + radius;
                } else {
                    float r = ((float)rand() / (RAND_MAX));
                    temp_z = r * (max_z - min_z) + min_z;
                }
                pos_vec.push_back(vector3(-dim_x / 2 + i * (2 * radius + gap) + radius,
                                          -dim_y / 2 + j * (2 * radius + gap) + radius, temp_z));
                vel_vec.push_back(vector3(0, 0, 0));
                acc_vec.push_back(vector3(0, 0, 0));
                if (k == 0)
                    fix_vec.push_back(1);
                if (k == 1)
                    fix_vec.push_back(0);
            }
        }
    }

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
    system->initialize(radius, pos_vec, vel_vec, acc_vec, fix_vec);
    system->setPrintOut(true);
    system->doStepDynamics(0.005f, 3.0f);
}
