// SPH-DualGPU
// SPH system base class header

#include "fstream"
#include "iostream"
#include "string"
#include <core/utils/GpuManager.h>
#include <cuda_runtime_api.h>
#include <sph/datastruct.cuh>

class KinematicTread{
private:
  int nKinematicCycle;
  int k_n;  // total number of particles
  vector3* k_pos;
  vector3* k_vel;
  vector3* k_acc;
  float k_radius;

  GpuManager* gpuManager;

  ExchangeData* k_shared_data;

  GpuManager::StreamInfo streamInfo;

public:
  KinematicTread(GpuManager* gm,  ExchangeData* shared_data){
    gpuManager = gm;

    k_shared_data = shared_data;

    streamInfo = gpuManager->getAvailableStream();
  }

  void kInitialize(float radius, vector3* pos, vector3* vel, vector3* acc, int n);
  void doKinematicStep(); 
};

class DynamicThread{
private:
  
  int nDynamicCycle;
  vector3* d_pos;
  vector3* d_vel;
  vector3* d_acc;
  int d_n;  // total number of particles
  float d_radius;
  
  GpuManager* gpuManager;

  ExchangeData* d_shared_data;

  GpuManager::StreamInfo streamInfo;
  
public:
  DynamicThread(GpuManager* gm, ExchangeData* shared_data){
    gpuManager = gm;

    d_shared_data = shared_data;

    streamInfo = gpuManager->getAvailableStream();
  }
  void dInitialize(float radius, vector3* pos, vector3* vel, vector3* acc, int n);
  void doDynamicStep(); 
};

class SPHSystem {
private:
  KinematicTread* kt;
  DynamicThread* dt;

  // shared data
  ExchangeData* shared_data;


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
