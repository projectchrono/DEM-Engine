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
    std::vector<bool> fix_vec;

    // sample all particles
    for (int k = 0; k < num_z; k++) {
        for (int j = 0; j < num_y; j++) {
            for (int i = 0; i < num_x; i++) {
                pos_vec.push_back(vector3(-dim_x / 2 + i * (2 * radius + gap) + radius,
                                          -dim_y / 2 + j * (2 * radius + gap) + radius,
                                          -dim_z / 2 + k * (2 * radius + gap) + radius));
                vel_vec.push_back(vector3(0, 0, 0));
                acc_vec.push_back(vector3(0, 0, 0));
                if (k == 0)
                    fix_vec.push_back(true);
                if (k == 1)
                    fix_vec.push_back(false);
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
    system->doStepDynamics(0.01);
    int frame = 0;
    for (int i = 0; i < 1; i++) {
        frame = frame + 1;
        // system->printCSV("test" + std::to_string(frame) + ".csv");
    }

    // print out test
    // print particle data into a csv for paraview visualization
    system->printCSV("test.csv");
}
