// SPH-DualGPU
// SPH system base class implementation source code

#include <ostream>
#include <sph/SPHSystem.h>
#include <core/utils/JitHelper.h>
#include <thread>
#include <vector>
#include <core/utils/GpuError.h>
#include "datastruct.h"

void SPHSystem::initialize(float radius,
                           std::vector<vector3>& pos,
                           std::vector<vector3>& vel,
                           std::vector<vector3>& acc,
                           std::vector<bool>& fix) {
    dataManager.radius = radius;
    dataManager.m_pos.assign(pos.begin(), pos.end());
    dataManager.m_vel.assign(vel.begin(), vel.end());
    dataManager.m_acc.assign(acc.begin(), acc.end());
    dataManager.m_fix.assign(fix.begin(), fix.end());
}

void SPHSystem::doStepDynamics(float time_step) {
    std::thread kinematics(kt);
    std::thread dynamics(dt);

    kinematics.join();
    dynamics.join();
}

void SPHSystem::printCSV(std::string filename) {
    // create file
    std::ofstream csvFile(filename);

    csvFile << "x_pos,y_pos,z_pos" << std::endl;

    // write particle data into csv file
    for (int i = 0; i < dataManager.m_pos.size(); i++) {
        csvFile << dataManager.m_pos[i].x << "," << dataManager.m_pos[i].y << "," << dataManager.m_pos[i].z
                << std::endl;
    }

    csvFile.close();
}

void KinematicThread::operator()() {
    cudaSetDevice(streamInfo.device);
    for (int lp = 0; lp < 10; lp++) {
        float tolerance = 0.05;

        // for each step, the kinematic thread needs to do two passes
        // first pass - look for 'number' of potential contacts
        // crate an array to store number of valid potential contacts

        // get total numer of particles
        int k_n;
        {
            getParentSystem().getMutexPos().lock();
            k_n = dataManager.m_pos.size();
            getParentSystem().getMutexPos().unlock();
        }

        // make a copy of the necessary data stored in DataManager
        std::vector<int, sgps::ManagedAllocator<int>> num_arr(k_n, -1);
        std::vector<contactData, sgps::ManagedAllocator<contactData>> contact_data;
        std::vector<vector3, sgps::ManagedAllocator<vector3>> pos_data;
        std::vector<int, sgps::ManagedAllocator<int>> offset_data;

        {
            getParentSystem().getMutexPos().lock();
            pos_data.assign(dataManager.m_pos.begin(), dataManager.m_pos.end());
            getParentSystem().getMutexPos().unlock();
        }

        // initiate JitHelper to perform JITC
        auto kinematic_program =
            JitHelper::buildProgram("sphKernels", JitHelper::KERNEL_DIR / "sphKernels.cu",
                                    std::vector<JitHelper::Header>(), {"-I" + (JitHelper::KERNEL_DIR / "..").string()});

        // kinematic thread first pass
        kinematic_program.kernel("kinematic1stPass")
            .instantiate()
            .configure(dim3(1), dim3(k_n), 0, streamInfo.stream)
            .launch(pos_data.data(), pos_data.size(), tolerance, dataManager.radius, num_arr.data());

        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        // calculate the offset array
        int cur_idx = 0;
        offset_data.clear();
        offset_data.resize(0);
        for (int i = 0; i < k_n; i++) {
            offset_data.push_back(cur_idx);
            cur_idx = cur_idx + num_arr[i];
        }

        // calculate total number of contact
        int contact_sum = 0;

        for (int i = 0; i < k_n; i++) {
            contact_sum = contact_sum + num_arr[i];
        }

        // second kinematic pass to fill the contact pair array
        // dataManager.m_contact.resize(contact_sum);

        contact_data.clear();
        contact_data.resize(contact_sum);

        // if the contact_sum is not 0, we perform the kinematic 2nd pass
        if (contact_sum != 0) {
            kinematic_program.kernel("kinematic2ndPass")
                .instantiate()
                .configure(dim3(1), dim3(k_n), 0, streamInfo.stream)
                .launch(pos_data.data(), pos_data.size(), offset_data.data(), num_arr.data(), tolerance,
                        dataManager.radius, contact_data.data());

            GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
        }

        // copy data back to the dataManager
        {
            getParentSystem().getMutexContact().lock();
            dataManager.m_contact.assign(contact_data.begin(), contact_data.end());
            dataManager.m_offset.assign(offset_data.begin(), offset_data.end());
            getParentSystem().getMutexContact().unlock();
        }

        // display current loop
        std::cout << "ki lp:" << lp << std::endl;
    }

    // std::cout << "contact_arr_length: " << dataManager.m_contact.size() << std::endl;
}

void DynamicThread::operator()() {
    GPU_CALL(cudaSetDevice(streamInfo.device));

    // Touch the CUDA context before the Kernel is accessed

    // Option 1:
    // std::vector<int, sgps::ManagedAllocator<int>> num_arr(dataManager.m_pos.size(), -1);

    // Option 2:
    // cudaEvent_t ev;
    // cudaEventCreate(&ev);

    // create vector to store data from the dataManager
    std::vector<contactData, sgps::ManagedAllocator<contactData>> contact_data;
    std::vector<int, sgps::ManagedAllocator<int>> offset_data;
    std::vector<vector3, sgps::ManagedAllocator<vector3>> pos_data;
    std::vector<vector3, sgps::ManagedAllocator<vector3>> vel_data;
    std::vector<vector3, sgps::ManagedAllocator<vector3>> acc_data;
    std::vector<bool, sgps::ManagedAllocator<bool>> fix_data;
    float radius = dataManager.radius;
    // temp step size explicit definition
    float time_step = 0.01;

    {
        getParentSystem().getMutexContact().lock();
        contact_data.assign(dataManager.m_contact.begin(), dataManager.m_contact.end());
        offset_data.assign(dataManager.m_offset.begin(), dataManager.m_offset.end());
        getParentSystem().getMutexContact().unlock();
    }

    {
        getParentSystem().getMutexPos().lock();
        pos_data.assign(dataManager.m_pos.begin(), dataManager.m_pos.end());
        vel_data.assign(dataManager.m_vel.begin(), dataManager.m_vel.end());
        acc_data.assign(dataManager.m_acc.begin(), dataManager.m_acc.end());
        fix_data.assign(dataManager.m_fix.begin(), dataManager.m_fix.end());
        getParentSystem().getMutexPos().unlock();
    }

    int num_block = 1;
    int num_thread = 1024;

    // display current loop
    for (int lp = 0; lp < 100; lp++) {
        auto dynamic_program =
            JitHelper::buildProgram("sphKernels", JitHelper::KERNEL_DIR / "sphKernels.cu",
                                    std::vector<JitHelper::Header>(), {"-I" + (JitHelper::KERNEL_DIR / "..").string()});

        // handle data fixity
        auto fix_arr = std::make_unique<bool[]>(fix_data.size());
        std::copy(std::begin(fix_data), std::end(fix_data), fix_arr.get());

        // call dynamic first gpu pass
        dynamic_program.kernel("dynamic1stPass")
            .instantiate()
            .configure(dim3(num_block), dim3(num_thread), 0, streamInfo.stream)
            .launch(contact_data.data(), contact_data.size(), pos_data.data(), vel_data.data(), acc_data.data(),
                    fix_arr, radius);

        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        // call dynamic second gpu pass
        dynamic_program.kernel("dynamic2ndPass")
            .instantiate()
            .configure(dim3(num_block), dim3(num_thread), 0, streamInfo.stream)
            .launch(pos_data.data(), dataManager.m_vel.data(), dataManager.m_acc.data(), fix_arr, pos_data.size(),
                    time_step, dataManager.radius);

        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        std::cout << "dy lp:" << lp << std::endl;
    }

    // copy data back to the dataManager
    {
        getParentSystem().getMutexPos().lock();
        dataManager.m_pos.assign(pos_data.begin(), pos_data.end());
        dataManager.m_vel.assign(vel_data.begin(), vel_data.end());
        dataManager.m_acc.assign(acc_data.begin(), acc_data.end());
        getParentSystem().getMutexPos().unlock();
    }
}
