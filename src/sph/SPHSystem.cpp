// SPH-DualGPU
// SPH system base class implementation source code

#include <ostream>
#include <sph/SPHSystem.h>
#include <core/utils/JitHelper.h>
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
    kt.doKinematicStep();
    dt.doDynamicStep();
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

void KinematicThread::doKinematicStep() {
    cudaSetDevice(streamInfo.device);

    // get total numer of particles
    int k_n = dataManager.m_pos.size();

    // std::cout << "k_n: " << k_n << std::endl;

    float tolerance = 0.05;

    // for each step, the kinematic thread needs to do two passes
    // first pass - look for 'number' of potential contacts
    // crate an array to store number of valid potential contacts

    std::vector<int, sgps::ManagedAllocator<int>> num_arr(k_n, -1);

    // std::cout << "num_arr_size: " << num_arr.size() << std::endl;

    auto kinematic_program =
        JitHelper::buildProgram("sphKernels", JitHelper::KERNEL_DIR / "sphKernels.cu", std::vector<JitHelper::Header>(),
                                {"-I" + (JitHelper::KERNEL_DIR / "..").string()});

    kinematic_program.kernel("kinematic1stPass")
        .instantiate()
        .configure(dim3(1), dim3(k_n), 0, streamInfo.stream)
        .launch(dataManager.m_pos.data(), dataManager.m_pos.size(), tolerance, dataManager.radius, num_arr.data());

    GPU_CALL(cudaDeviceSynchronize());

    // calculate the offset array
    int cur_idx = 0;
    dataManager.m_offset.clear();
    dataManager.m_offset.resize(0);
    for (int i = 0; i < k_n; i++) {
        dataManager.m_offset.push_back(cur_idx);
        cur_idx = cur_idx + num_arr[i];
    }

    // calculate total number of contact
    int contact_sum = 0;

    for (int i = 0; i < k_n; i++) {
        contact_sum = contact_sum + num_arr[i];
    }

    // std::cout << "contact_sum: " << contact_sum << std::endl;

    dataManager.m_contact.clear();

    // second kinematic pass to fill the contact pair array
    dataManager.m_contact.resize(contact_sum);

    if (contact_sum != 0) {
        kinematic_program.kernel("kinematic2ndPass")
            .instantiate()
            .configure(dim3(1), dim3(k_n), 0, streamInfo.stream)
            .launch(dataManager.m_pos.data(), dataManager.m_pos.size(), dataManager.m_offset.data(), num_arr.data(),
                    tolerance, dataManager.radius, dataManager.m_contact.data());

        cudaDeviceSynchronize();
    }

    // std::cout << "contact_arr_length: " << dataManager.m_contact.size() << std::endl;
}

void DynamicThread::doDynamicStep() {
    cudaSetDevice(streamInfo.device);
    // calculate number of threads needed and number of block needed
    int num_thread = 64;
    int num_block = dataManager.m_contact.size() / num_thread + 1;

    // std::cout << "hit_check: 4" << std::endl;

    cudaSetDevice(streamInfo.device);
    // auto dynamic_program =
    //    JitHelper::buildProgram("sphKernels", JitHelper::KERNEL_DIR / "sphKernels.cu",
    //    std::vector<JitHelper::Header>(),
    //{"-I" + (JitHelper::KERNEL_DIR / "..").string()});

    // std::cout << "hit_check: 5" << std::endl;
    /*
        dynamic_program.kernel("dynamicPass")
            .instantiate()
            .configure(dim3(num_block), dim3(num_thread), 0, streamInfo.stream)
            .launch(dataManager.m_contact.data(), dataManager.m_contact.size(), dataManager.m_pos.data(),
                    dataManager.m_vel.data(), dataManager.m_acc.data(), dataManager.radius);
    */
    // dynamic_program.kernel("testKernel").instantiate().configure(dim3(1), dim3(1), 0, streamInfo.stream).launch();

    contactData* gpu_pair_data = dataManager.m_contact.data();
    vector3* gpu_pos = dataManager.m_pos.data();

    for (int i = 0; i < dataManager.m_contact.size(); i++) {
        float dir_x = gpu_pos[gpu_pair_data[i].contact_pair.x].x - gpu_pos[gpu_pair_data[i].contact_pair.y].x;
        float dir_y = gpu_pos[gpu_pair_data[i].contact_pair.x].y - gpu_pos[gpu_pair_data[i].contact_pair.y].y;
        float dir_z = gpu_pos[gpu_pair_data[i].contact_pair.x].z - gpu_pos[gpu_pair_data[i].contact_pair.y].z;

        float dist2 = dir_x * dir_x + dir_y * dir_y + dir_z * dir_z;

        if (dist2 < dataManager.radius * dataManager.radius) {
            float coe = 1000.f;

            if (dataManager.m_fix[dataManager.m_contact[i].contact_pair.x] == false) {
                dataManager.m_acc[dataManager.m_contact[i].contact_pair.x].x = dir_x * coe;
                dataManager.m_acc[dataManager.m_contact[i].contact_pair.x].y = dir_y * coe;
                dataManager.m_acc[dataManager.m_contact[i].contact_pair.x].z = dir_z * coe;
            }

            if (dataManager.m_fix[dataManager.m_contact[i].contact_pair.y] == false) {
                dataManager.m_acc[dataManager.m_contact[i].contact_pair.y].x = -dir_x * coe;
                dataManager.m_acc[dataManager.m_contact[i].contact_pair.y].y = -dir_y * coe;
                dataManager.m_acc[dataManager.m_contact[i].contact_pair.y].z = -dir_z * coe;
            }

            std::cout << "acc_z:" << dataManager.m_acc[dataManager.m_contact[i].contact_pair.y].z << std::endl;
        }
    }

    for (int i = 0; i < dataManager.m_pos.size(); i++) {
        if (dataManager.m_fix[i] == false) {
            float grav = -9.8f;

            dataManager.m_acc[i].z = dataManager.m_acc[i].z + grav;
        }

        dataManager.m_vel[i].x = dataManager.m_vel[i].x + dataManager.m_acc[i].x * 0.01;
        dataManager.m_vel[i].y = dataManager.m_vel[i].y + dataManager.m_acc[i].y * 0.01;
        dataManager.m_vel[i].z = dataManager.m_vel[i].z + dataManager.m_acc[i].z * 0.01;

        dataManager.m_pos[i].x = dataManager.m_pos[i].x + dataManager.m_vel[i].x * 0.01;
        dataManager.m_pos[i].y = dataManager.m_pos[i].y + dataManager.m_vel[i].y * 0.01;
        dataManager.m_pos[i].z = dataManager.m_pos[i].z + dataManager.m_vel[i].z * 0.01;

        dataManager.m_acc[i].x = 0.f;
        dataManager.m_acc[i].y = 0.f;
        dataManager.m_acc[i].z = 0.f;

        std::cout << "part: " << i << " " << dataManager.m_pos[i].z << " " << std::endl;
    }
}
