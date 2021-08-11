// SPH-DualGPU
// SPH system base class header

#include "fstream"
#include "iostream"
#include "string"
#include "interactionManager.h"
#include <cuda_runtime_api.h>


class KinematicTread{
private:
  int nKinematicCycle;
  int k_n;  // total number of particles
  vector3* k_pos;
  vector3* k_vel;
  vector3* k_acc;
  float k_radius;
  InteractionManager* threadManager;

public:
  KinematicTread(InteractionManager* tm){
    cudaSetDevice(device_id);
    cudaStreamCreate(&kStream);
    threadManager = tm;
  }

  void kInitialize(float radius, vector3* pos, vector3* vel, vector3* acc, int n);
  void doKinematicStep();
  int device_id = 0;
  cudaStream_t kStream;
  
};

class DynamicThread{
private:
  
  int nDynamicCycle;
  vector3* d_pos;
  vector3* d_vel;
  vector3* d_acc;
  int d_n;  // total number of particles
  float d_radius;
  InteractionManager* threadManager;
  
public:
  DynamicThread(InteractionManager* tm){
    cudaSetDevice(device_id);
    cudaStreamCreate(&dStream);
    threadManager = tm;
  }
  void dInitialize(float radius, vector3* pos, vector3* vel, vector3* acc, int n);
  void doDynamicStep();
  int device_id = 1;
  cudaStream_t dStream;
};

class SPHSystem {
private:
  KinematicTread* kt;
  DynamicThread* dt;


  vector3 *m_pos; // particle locations, on cpu
  int m_n;      // total number of particles


  contactData *m_contact; // contact pair data

  vector3 *unified_pos; // unified gpu/cpu memory for particle position data


  // main data transfer array

public:
  // initialize the SPHSystem with pos as the particle positions
  // n as the total number of particles initialized in the SPHSystem
  void initialize(float radius,vector3 *pos, int n);

  // start performing simulation dynamics
  void doStepDynamics(float time_step);

  // print particle file to csv for paraview visualization purposes
  void printCSV(std::string filename);
};
