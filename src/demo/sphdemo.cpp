// SPH-DualGPU
// driver code

#include <sph/SPHSystem.cuh>

int main(int argc, char* argv[]) {
    /*
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
    int num_z = dim_z / (radius * 2 + gap);

    // create position array of all particles
    vector3 *pos_arr = new vector3[num_x * num_y * num_z];

    // sample all particles
    for (int k = 0; k < num_z; k++) {
      for (int j = 0; j < num_y; j++) {
        for (int i = 0; i < num_x; i++) {
          int idx = k * num_y * num_x + j * num_x + i;
          pos_arr[idx].x = -dim_x / 2 + i * (2 * radius + gap) + radius;
          pos_arr[idx].y = -dim_y / 2 + j * (2 * radius + gap) + radius;
          pos_arr[idx].z = -dim_z / 2 + k * (2 * radius + gap) + radius;
        }
      }
    }
  */

    // set particle radius
    float radius = 0.2;
    vector3* pos_arr = new vector3[2];
    // first particle at [0,0,0.21]
    pos_arr[0].x = 0.f;
    pos_arr[0].y = 0.f;
    pos_arr[0].z = 0.21f;
    // second particle at [0,0,0.65]
    pos_arr[1].x = 0.f;
    pos_arr[1].y = 0.f;
    pos_arr[1].z = 0.65f;

    // create SPHSystem
    SPHSystem* system = new SPHSystem();

    // initialize the SPHSystem
    system->initialize(radius, pos_arr, 2);
    system->doStepDynamics(0.0001);

    // print out test
    // print particle data into a csv for paraview visualization
    system->printCSV("test.csv");
}
