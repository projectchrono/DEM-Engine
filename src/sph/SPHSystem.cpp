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
    for (int lp = 0; lp < 10; lp++) {
        cudaSetDevice(streamInfo.device);

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

        GPU_CALL(cudaDeviceSynchronize());

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

            cudaDeviceSynchronize();
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
    // display current loop
    for (int lp = 0; lp < 10; lp++) {
        std::cout << "dy lp:" << lp << std::endl;
    }
    /*
    std::cout << "dynamic count: " << dynamicCounter++ << std::endl;

    cudaSetDevice(streamInfo.device);
    // calculate number of threads needed and number of block needed
    int num_thread = 64;
    int num_block = dataManager.m_contact.size() / num_thread + 1;
*/
    /*

    // std::cout << "hit_check: 4" << std::endl;
    auto dynamic_program =
        JitHelper::buildProgram("sphKernels", JitHelper::KERNEL_DIR / "sphKernels.cu", std::vector<JitHelper::Header>(),
                                {"-I" + (JitHelper::KERNEL_DIR / "..").string()});


        dynamic_program.kernel("dynamicPass")
            .instantiate()
            .configure(dim3(num_block), dim3(num_thread), 0, streamInfo.stream)
            .launch(dataManager.m_contact.data(), dataManager.m_contact.size(), dataManager.m_pos.data(),
                    dataManager.m_vel.data(), dataManager.m_acc.data(), dataManager.radius);


    dynamic_program.kernel("testKernel").instantiate().configure(dim3(1), dim3(1), 0, streamInfo.stream).launch();

    GPU_CALL(cudaDeviceSynchronize());

    std::cerr << "Kernel should be finished now\n";
    */

    // retreive data from the dataManager
    /*
        std::vector<contactData, sgps::ManagedAllocator<contactData>> contact_data;
        std::vector<vector3, sgps::ManagedAllocator<vector3>> pos_data;
        std::vector<vector3, sgps::ManagedAllocator<vector3>> vel_data;
        std::vector<vector3, sgps::ManagedAllocator<vector3>> acc_data;

        {
            getParentSystem().getMutex().lock();
            contact_data.assign(dataManager.m_contact.begin(), dataManager.m_contact.end());
            pos_data.assign(dataManager.m_pos.begin(), dataManager.m_pos.end());
            vel_data.assign(dataManager.m_vel.begin(), dataManager.m_vel.end());
            acc_data.assign(dataManager.m_acc.begin(), dataManager.m_acc.end());
            getParentSystem().getMutex().unlock();
        }

        for (int i = 0; i < contact_data.size(); i++) {
            float dir_x = pos_data[contact_data[i].contact_pair.x].x - pos_data[contact_data[i].contact_pair.y].x;
            float dir_y = pos_data[contact_data[i].contact_pair.x].y - pos_data[contact_data[i].contact_pair.y].y;
            float dir_z = pos_data[contact_data[i].contact_pair.x].z - pos_data[contact_data[i].contact_pair.y].z;

            float dist2 = dir_x * dir_x + dir_y * dir_y + dir_z * dir_z;

            if (dist2 < (2 * dataManager.radius) * (2 * dataManager.radius)) {
                float coe = 1000.f;

                if (dataManager.m_fix[contact_data[i].contact_pair.x] == false) {
                    acc_data[contact_data[i].contact_pair.x].x = dir_x * coe;
                    acc_data[contact_data[i].contact_pair.x].y = dir_y * coe;
                    acc_data[contact_data[i].contact_pair.x].z = dir_z * coe;
                }

                if (dataManager.m_fix[contact_data[i].contact_pair.y] == false) {
                    acc_data[contact_data[i].contact_pair.y].x = -dir_x * coe;
                    acc_data[contact_data[i].contact_pair.y].y = -dir_y * coe;
                    acc_data[contact_data[i].contact_pair.y].z = -dir_z * coe;
                }

                std::cout << "acc_z:" << acc_data[contact_data[i].contact_pair.y].z << std::endl;
            }
        }

        for (int i = 0; i < pos_data.size(); i++) {
            if (dataManager.m_fix[i] == false) {
                float grav = -9.8f;

                acc_data[i].z = acc_data[i].z + grav;
            }

            vel_data[i].x = vel_data[i].x + acc_data[i].x * 0.01;
            vel_data[i].y = vel_data[i].y + acc_data[i].y * 0.01;
            vel_data[i].z = vel_data[i].z + acc_data[i].z * 0.01;

            pos_data[i].x = pos_data[i].x + vel_data[i].x * 0.01;
            pos_data[i].y = pos_data[i].y + vel_data[i].y * 0.01;
            pos_data[i].z = pos_data[i].z + vel_data[i].z * 0.01;

            acc_data[i].x = 0.f;
            acc_data[i].y = 0.f;
            acc_data[i].z = 0.f;

            std::cout << "part: " << i << " " << pos_data[i].z << " " << std::endl;
        }

        {
            getParentSystem().getMutex().lock();
            dataManager.m_contact.assign(contact_data.begin(), contact_data.end());
            dataManager.m_pos.assign(pos_data.begin(), pos_data.end());
            dataManager.m_vel.assign(vel_data.begin(), vel_data.end());
            dataManager.m_acc.assign(acc_data.begin(), acc_data.end());
            getParentSystem().getMutex().unlock();
        }*/
}
