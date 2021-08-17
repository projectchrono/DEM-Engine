// SPH-DualGPU
// SPH system base class header

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <core/utils/ManagedAllocator.hpp>
#include <core/utils/GpuManager.h>
#include <cuda_runtime_api.h>
#include <sph/datastruct.cuh>

struct DataManager {
    float radius;
    std::vector<vector3, ManagedAllocator<vector3>> m_pos; // particle locations
    std::vector<vector3, ManagedAllocator<vector3>> m_vel; // particle velocities
    std::vector<vector3, ManagedAllocator<vector3>> m_acc; // particle accelerations

    std::vector<contactData, ManagedAllocator<contactData>> m_contact; // contact pair data

    std::vector<int, ManagedAllocator<int>> m_offset; // index offset array for the contact pair data
};

class KinematicThread {
private:
    GpuManager& gpuManager;

    GpuManager::StreamInfo streamInfo;

    DataManager& dataManager;

public:
    KinematicThread(DataManager& dm, GpuManager& gm) : dataManager(dm), gpuManager(gm) {

        streamInfo = gm.getAvailableStream();
    }

    void doKinematicStep();
};

class DynamicThread{
private:
    
    GpuManager& gpuManager;

    GpuManager::StreamInfo streamInfo;

    DataManager& dataManager;
    
public:
    DynamicThread(DataManager& dm, GpuManager& gm) : dataManager(dm), gpuManager(gm) {
        streamInfo = gm.getAvailableStream();
    }

    void doDynamicStep(); 
};

class SPHSystem {
private:

    KinematicThread kt;
    DynamicThread dt;

    DataManager dataManager;

    // main data transfer array

public:

    inline SPHSystem(GpuManager& gm) : kt(dataManager, gm), dt(dataManager, gm) {};

    // initialize the SPHSystem with pos as the particle positions
    // n as the total number of particles initialized in the SPHSystem
    void initialize(float radius,std::vector<vector3> &pos, std::vector<vector3> &vel, std::vector<vector3> &acc);

    // start performing simulation dynamics
    void doStepDynamics(float time_step);

    // print particle file to csv for paraview visualization purposes
    void printCSV(std::string filename);

};
