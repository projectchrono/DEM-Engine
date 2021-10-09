// SPH-DualGPU
// SPH system base class header

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <core/utils/ManagedAllocator.hpp>
#include <core/utils/GpuManager.h>
#include <cuda_runtime_api.h>
#include <sph/datastruct.h>
#include <mutex>
struct DataManager {
    float radius;
    std::vector<vector3, sgps::ManagedAllocator<vector3>> m_pos;              // particle locations
    std::vector<vector3, sgps::ManagedAllocator<vector3>> m_vel;              // particle velocities
    std::vector<vector3, sgps::ManagedAllocator<vector3>> m_acc;              // particle accelerations
    std::vector<bool, sgps::ManagedAllocator<bool>> m_fix;                    // particle fixity
    std::vector<contactData, sgps::ManagedAllocator<contactData>> m_contact;  // contact pair data

    std::vector<int, sgps::ManagedAllocator<int>> m_offset;  // index offset array for the contact pair data
};

class SPHSystem;
class KinematicThread {
  private:
    GpuManager& gpuManager;

    GpuManager::StreamInfo streamInfo;

    DataManager& dataManager;

    SPHSystem& parentSystem;

  public:
    int kinematicCounter;

    KinematicThread(DataManager& dm, GpuManager& gm, SPHSystem& system)
        : dataManager(dm), gpuManager(gm), parentSystem(system) {
        streamInfo = gm.getAvailableStream();
        kinematicCounter = 0;
    }

    void operator()();

    SPHSystem& getParentSystem() { return parentSystem; }
};

class DynamicThread {
  private:
    GpuManager& gpuManager;

    GpuManager::StreamInfo streamInfo;

    DataManager& dataManager;

    SPHSystem& parentSystem;

  public:
    int dynamicCounter;

    DynamicThread(DataManager& dm, GpuManager& gm, SPHSystem& system)
        : dataManager(dm), gpuManager(gm), parentSystem(system) {
        streamInfo = gm.getAvailableStream();
        dynamicCounter = 0;
    }

    void operator()();

    SPHSystem& getParentSystem() { return parentSystem; }
};

class SPHSystem {
  private:
    KinematicThread kt;
    DynamicThread dt;

    DataManager dataManager;

    std::mutex mutex_lock_pos;
    std::mutex mutex_lock_contact;

    // printout indicator
    bool isPrint;

  public:
    inline SPHSystem(GpuManager& gm) : kt(dataManager, gm, *this), dt(dataManager, gm, *this) {
        curr_time = 0.0f;
        pos_data_isFresh = true;
        contact_data_isFresh = false;

        isPrint = false;
    };

    void setPrintOut(bool isPrint) { this->isPrint = isPrint; }

    bool getPrintOut() { return isPrint; }

    // initialize the SPHSystem with pos as the particle positions
    // n as the total number of particles initialized in the SPHSystem
    void initialize(float radius,
                    std::vector<vector3>& pos,
                    std::vector<vector3>& vel,
                    std::vector<vector3>& acc,
                    std::vector<bool>& fix);

    // start performing simulation dynamics
    void doStepDynamics(float time_step, float sim_time);

    // print particle file to csv for paraview visualization purposes
    void printCSV(std::string filename, vector3* pos_arr, int pos_n, vector3* vel_arr, vector3* acc_arr);

    // dual gpu coordinations
    std::mutex& getMutexPos() { return mutex_lock_pos; }
    std::mutex& getMutexContact() { return mutex_lock_contact; }

    int getKiCounter() { return kt.kinematicCounter; }
    int getDyCounter() { return dt.dynamicCounter; }

    float curr_time;
    float sim_time;
    float time_step;

    bool pos_data_isFresh;
    bool contact_data_isFresh;
};
