// SPH-DualGPU
// SPH system base class implementation source code

#include <ostream>
#include <sph/SPHSystem.cuh>
#include <core/utils/JitHelper.h>
#include <vector>
#include <core/utils/GpuError.h>
#include "datastruct.cuh"

void SPHSystem::initialize(float radius,
                           std::vector<vector3>& pos,
                           std::vector<vector3>& vel,
                           std::vector<vector3>& acc) {
    dataManager.radius = radius;
    dataManager.m_pos.assign(pos.begin(), pos.end());
    dataManager.m_vel.assign(vel.begin(), vel.end());
    dataManager.m_acc.assign(acc.begin(), acc.end());
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

    float tolerance = 0.05;
    // kinematicTestKernel<<<1, 1, 0, kStream>>>();

    // for each step, the kinematic thread needs to do two passes
    // first pass - look for 'number' of potential contacts
    // crate an array to store number of valid potential contacts

    std::vector<int, sgps::ManagedAllocator<int>> num_arr(k_n, -1);

    // first kinematic pass to calculate offset array
    // void* args[] = {(void*)(dataManager.m_pos.data()), (void*)(dataManager.m_pos.size()), (void*)(tolerance),
    //                (void*)(dataManager.radius), (void*)(num_arr.data())};
    // cudaLaunchKernel((void*)&kinematic1stPass, dim3(1), dim3(k_n), args, 0, streamInfo.stream);

    auto kinematic_program = JitHelper::buildProgram("cudaKernels", JitHelper::KERNEL_DIR / "cudaKernels.cu");

    kinematic_program.kernel("kinematic1stPass")
        .instantiate()
        .configure(dim3(1), dim3(1), 0, streamInfo.stream)
        .launch(dataManager.m_pos.data(), dataManager.m_pos.size(), tolerance, dataManager.radius, num_arr.data());

    GPU_CALL(cudaDeviceSynchronize());

    /*

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

    // second kinematic pass to fill the contact pair array
    dataManager.m_contact.resize(contact_sum);

    dataManager.m_contact.clear();

    kinematic2ndPass<<<1, k_n, 0, streamInfo.stream>>>(dataManager.m_pos.data(), dataManager.m_pos.size(),
                                                       dataManager.m_offset.data(), num_arr.data(), tolerance,
                                                       dataManager.radius, dataManager.m_contact.data());

    cudaDeviceSynchronize();
    */
}

void DynamicThread::doDynamicStep() {
    cudaSetDevice(streamInfo.device);
    // dynamicTestKernel<<<1, 1, 0, dStream>>>();

    // calculate number of threads needed and number of block needed
    int num_thread = 64;
    int num_block = dataManager.m_contact.size() / num_thread + 1;

    /*
    dynamicPass<<<num_block, num_thread, 0, streamInfo.stream>>>(
        dataManager.m_contact.data(), dataManager.m_contact.size(), dataManager.m_pos.data(), dataManager.m_vel.data(),
        dataManager.m_acc.data(), dataManager.radius);

    cudaDeviceSynchronize();
    */
}
