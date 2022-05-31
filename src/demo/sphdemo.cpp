// SPH-DualGPU
// driver code

#include <sph/SPHSystem.h>

using namespace sgps;

int main(int argc, char* argv[]) {
    // initialize domain (boundary)
    float dim_x = 16;
    float dim_y = 16;
    float dim_z = 16;

    float fluid_dim_x = 16;
    float fluid_dim_y = 16;
    float fluid_dim_z = 16;

    // set particle kernel_h
    float kernel_h = 1.0;

    // calculate number of particles on each direction
    int num_x = (int)(dim_x / kernel_h);
    int num_y = (int)(dim_y / kernel_h);
    int num_z = (int)(dim_z / kernel_h);
    int num_boundary = 3;

    // create position array of all particles
    std::vector<float3> pos_vec;
    std::vector<float3> vel_vec;
    std::vector<float3> acc_vec;
    std::vector<float> pres_vec;
    std::vector<float> rho_vec;
    std::vector<char> fix_vec;

    // parameter set up, c is sound speed
    int num_par = 0;
    float rho = 1000;
    float gravity = 9.8;
    float c = 2.0 * sqrt(gravity * fluid_dim_z);
    // sample the bottom boundary layers
    for (int j = 0; j < num_y + 2 * num_boundary; j++) {
        for (int i = 0; i < num_x + 2 * num_boundary; i++) {
            for (int k = 0; k < num_boundary; k++) {
                float x_offset = -dim_x / 2 - kernel_h / 2 - 2 * kernel_h;
                float y_offset = -dim_y / 2 - kernel_h / 2 - 2 * kernel_h;
                float z_offset = -dim_z / 2 - kernel_h / 2 - 2 * kernel_h;
                float depth = dim_z / 2 - (z_offset + k * kernel_h);
                pos_vec.push_back(
                    make_float3(x_offset + i * kernel_h, y_offset + j * kernel_h, z_offset + k * kernel_h));
                vel_vec.push_back(make_float3(0, 0, 0));
                acc_vec.push_back(make_float3(0, 0, 0));
                rho_vec.push_back(rho);
                pres_vec.push_back(depth * rho * gravity);
                fix_vec.push_back(1);
                num_par++;

                // the top wall, set pressure to zero
                z_offset = dim_z / 2 + kernel_h / 2;
                pos_vec.push_back(
                    make_float3(x_offset + i * kernel_h, y_offset + j * kernel_h, z_offset + k * kernel_h));
                vel_vec.push_back(make_float3(0, 0, 0));
                acc_vec.push_back(make_float3(0, 0, 0));
                pres_vec.push_back(0.0);
                rho_vec.push_back(rho);
                fix_vec.push_back(1);
                num_par++;
            }
        }
    }
    // side walls in y direction (perpendicular to y axis)
    for (int j = 0; j < num_z; j++) {
        for (int i = 0; i < num_x + 2 * num_boundary; i++) {
            for (int k = 0; k < num_boundary; k++) {
                float x_offset = -dim_x / 2 - kernel_h / 2 - 2 * kernel_h;
                float y_offset = -dim_y / 2 - kernel_h / 2 - 2 * kernel_h;
                float z_offset = -dim_z / 2 + kernel_h / 2;
                pos_vec.push_back(
                    make_float3(x_offset + i * kernel_h, y_offset + k * kernel_h, z_offset + j * kernel_h));
                vel_vec.push_back(make_float3(0, 0, 0));
                acc_vec.push_back(make_float3(0, 0, 0));
                float depth = dim_z / 2 - (z_offset + j * kernel_h);
                pres_vec.push_back(depth * rho * gravity);
                rho_vec.push_back(rho);
                fix_vec.push_back(1);
                num_par++;

                y_offset = dim_y / 2 + kernel_h / 2;
                pos_vec.push_back(
                    make_float3(x_offset + i * kernel_h, y_offset + k * kernel_h, z_offset + j * kernel_h));
                vel_vec.push_back(make_float3(0, 0, 0));
                acc_vec.push_back(make_float3(0, 0, 0));
                pres_vec.push_back(depth * rho * gravity);
                rho_vec.push_back(rho);
                fix_vec.push_back(1);
                num_par++;
            }
        }
    }
    // side walls in x direction (perpendicular to x axis)
    for (int j = 0; j < num_z; j++) {
        for (int i = 0; i < num_y; i++) {
            for (int k = 0; k < num_boundary; k++) {
                float x_offset = -dim_x / 2 - kernel_h / 2 - 2 * kernel_h;
                float y_offset = -dim_y / 2 + kernel_h / 2;
                float z_offset = -dim_z / 2 + kernel_h / 2;
                pos_vec.push_back(
                    make_float3(x_offset + k * kernel_h, y_offset + i * kernel_h, z_offset + j * kernel_h));
                vel_vec.push_back(make_float3(0, 0, 0));
                acc_vec.push_back(make_float3(0, 0, 0));
                float depth = dim_z / 2 - (z_offset + j * kernel_h);
                pres_vec.push_back(depth * rho * gravity);
                rho_vec.push_back(rho);
                fix_vec.push_back(1);
                num_par++;
                x_offset = dim_y / 2 + kernel_h / 2;

                pos_vec.push_back(
                    make_float3(x_offset + k * kernel_h, y_offset + i * kernel_h, z_offset + j * kernel_h));
                vel_vec.push_back(make_float3(0, 0, 0));
                acc_vec.push_back(make_float3(0, 0, 0));
                pres_vec.push_back(depth * rho * gravity);
                rho_vec.push_back(rho);
                fix_vec.push_back(1);
                num_par++;
            }
        }
    }

    // sample a smaller domain for drop test
    for (int j = 0; j < (int)(fluid_dim_z / kernel_h); j++) {
        for (int i = 0; i < (int)(fluid_dim_y / kernel_h); i++) {
            for (int k = 0; k < (int)(fluid_dim_x / kernel_h); k++) {
                float x_offset = -fluid_dim_x / 2 + kernel_h / 2;
                float y_offset = -fluid_dim_y / 2 + kernel_h / 2;
                float z_offset = -dim_z / 2 + kernel_h / 2;

                pos_vec.push_back(
                    make_float3(x_offset + k * kernel_h, y_offset + i * kernel_h, z_offset + j * kernel_h));
                vel_vec.push_back(make_float3(0, 0, 0));
                acc_vec.push_back(make_float3(0, 0, 0));
                float depth = dim_z / 2 - (z_offset + j * kernel_h);
                pres_vec.push_back(depth * rho * gravity);
                rho_vec.push_back(rho);
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
    system->initialize(kernel_h, rho * kernel_h * kernel_h * kernel_h, rho, c, pos_vec, vel_vec, acc_vec, pres_vec,
                       fix_vec, dim_x, dim_y, dim_z);
    // system->printCSV("sph_folder/test" + std::to_string(0) + ".csv", pos_vec.data(), num_par, vel_vec.data(),
    //                  acc_vec.data(), rho_vec.data(), pres_vec.data());
    // system->setPrintOut(true, 200);
    system->doStepDynamics(5e-5, 2.f);
}
