// SPH-DualGPU
// SPH system base class implementation source code

#include <cstdint>
#include <ostream>
#include <sph/SPHSystem.h>
#include <core/utils/JitHelper.h>
#include <thread>
#include <vector>
#include <algorithm>
#include <chrono>
#include <core/utils/GpuError.h>
#include "SPHSystem.h"
#include "datastruct.h"
#include <algorithms/SPHCubHelperFunctions.h>

int X_SUB_NUM = 0;
int Y_SUB_NUM = 0;
int Z_SUB_NUM = 0;

void SPHSystem::initialize(float kernel_h,
                           float m,
                           float rho_0,
                           float c,
                           std::vector<float3>& pos,
                           std::vector<float3>& vel,
                           std::vector<float3>& acc,
                           std::vector<float>& pres,
                           std::vector<char>& fix,
                           float domain_x,
                           float domain_y,
                           float domain_z) {
    dataManager.kernel_h = kernel_h;
    dataManager.m = m;
    dataManager.rho_0 = rho_0;
    dataManager.c = c;
    dataManager.k_n = pos.size();

    dataManager.m_pos.resize(dataManager.k_n);
    dataManager.m_vel.resize(dataManager.k_n);
    dataManager.m_acc.resize(dataManager.k_n);
    dataManager.m_pressure.resize(dataManager.k_n);
    dataManager.m_fix.resize(dataManager.k_n);
    
    cudaMemcpy(dataManager.m_pos.data(), pos.data(), dataManager.k_n * sizeof(float3), cudaMemcpyDefault);
    cudaMemcpy(dataManager.m_vel.data(), vel.data(), dataManager.k_n * sizeof(float3), cudaMemcpyDefault);
    cudaMemcpy(dataManager.m_acc.data(), acc.data(), dataManager.k_n * sizeof(float3), cudaMemcpyDefault);
    cudaMemcpy(dataManager.m_pressure.data(), pres.data(), dataManager.k_n * sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(dataManager.m_fix.data(), fix.data(), dataManager.k_n * sizeof(char), cudaMemcpyDefault);

    this->domain_x = domain_x + 5 * kernel_h;
    this->domain_y = domain_y + 5 * kernel_h;
    this->domain_z = domain_z + 5 * kernel_h;

    // redeclare the number of subdomain to be size/4 (4 is the BSD side length)
    X_SUB_NUM = (int)(this->domain_x / kernel_h) / 2 + 2;
    Y_SUB_NUM = (int)(this->domain_y / kernel_h) / 2 + 2;
    Z_SUB_NUM = (int)(this->domain_z / kernel_h) / 2 + 2;
    printf("BSD num: %d, %d, %d\n", X_SUB_NUM, Y_SUB_NUM, Z_SUB_NUM);
}

void SPHSystem::doStepDynamics(float time_step, float sim_time) {
    this->time_step = time_step;
    this->sim_time = sim_time;
    std::thread kinematics(kt);
    std::thread dynamics(dt);
    std::thread writeout(wt);

    kinematics.join();
    dynamics.join();

    // join the write-out thread only when write out mode is enabled
    if (this->getPrintOut() == true) {
        writeout.join();
    } else {
        writeout.detach();
    }
}

void SPHSystem::printCSV(std::string filename, float3* pos_arr, int pos_n, float3* vel_arr) {
    // create file
    std::ofstream csvFile(filename);

    csvFile << "x_pos,y_pos,z_pos,x_vel,y_vel,z_vel" << std::endl;

    // write particle data into csv file
    for (int i = 0; i < pos_n; i++) {
        csvFile << pos_arr[i].x << "," << pos_arr[i].y << "," << pos_arr[i].z << "," << vel_arr[i].x << ","
                << vel_arr[i].y << "," << vel_arr[i].z << std::endl;
    }

    csvFile.close();
}

void SPHSystem::printCSV(std::string filename, float3* pos_arr, int pos_n) {
    // create file
    std::ofstream csvFile(filename);

    csvFile << "x_pos,y_pos,z_pos,x_vel,y_vel,z_vel" << std::endl;

    // write particle data into csv file
    for (int i = 0; i < pos_n; i++) {
        csvFile << pos_arr[i].x << "," << pos_arr[i].y << "," << pos_arr[i].z << "," << std::endl;
    }

    csvFile.close();
}

void SPHSystem::printCSV(std::string filename,
                         float3* pos_arr,
                         int pos_n,
                         float3* vel_arr,
                         float3* acc_arr,
                         float* rho_arr,
                         float* pressure_arr) {
    // create file
    std::ofstream csvFile(filename);

    csvFile << "x_pos,y_pos,z_pos,x_vel,y_vel,z_vel,x_acc,y_acc,z_acc,rho,pressure" << std::endl;

    // write particle data into csv file
    for (int i = 0; i < pos_n; i++) {
        csvFile << pos_arr[i].x << "," << pos_arr[i].y << "," << pos_arr[i].z << "," << vel_arr[i].x << ","
                << vel_arr[i].y << "," << vel_arr[i].z << "," << acc_arr[i].x << "," << acc_arr[i].y << ","
                << acc_arr[i].z << "," << rho_arr[i] << "," << pressure_arr[i] << std::endl;
    }

    csvFile.close();
}

void SPHSystem::printCSV(std::string filename, float3* pos_arr, int pos_n, float3* vel_arr, float3* acc_arr) {
    // create file
    std::ofstream csvFile(filename);

    csvFile << "x_pos,y_pos,z_pos,x_vel,y_vel,z_vel,x_acc,y_acc,z_acc" << std::endl;

    // write particle data into csv file
    for (int i = 0; i < pos_n; i++) {
        csvFile << pos_arr[i].x << "," << pos_arr[i].y << "," << pos_arr[i].z << "," << vel_arr[i].x << ","
                << vel_arr[i].y << "," << vel_arr[i].z << "," << acc_arr[i].x << "," << acc_arr[i].y << ","
                << acc_arr[i].z << std::endl;
    }

    csvFile.close();
}

void KinematicThread::operator()() {
    cudaSetDevice(streamInfo.device);
    cudaStreamCreate(&streamInfo.stream);

    // get total numer of particles
    int k_n;
    float kernel_h;
    float m;
    float rho_0;
    float c;
    {
        const std::lock_guard<std::mutex> lock(getParentSystem().getMutexPos());
        k_n = dataManager.k_n;
        kernel_h = dataManager.kernel_h;
        m = dataManager.m;
        rho_0 = dataManager.rho_0;
        c = dataManager.c;
    }

    // set number of blocks and number of threads in each block
    int block_size = 1024;
    int num_thread = (block_size < k_n) ? block_size : k_n;
    int num_block = (k_n % num_thread != 0) ? (k_n / num_thread + 1) : (k_n / num_thread);

    // initial vector declaration
    std::vector<int, sgps::ManagedAllocator<int>>
        num_BSD_data;  // vector stores the number of BSD each particle touches
    std::vector<int, sgps::ManagedAllocator<int>>
        idx_track_data;  // vector to store original particle idx after sorting
    std::vector<int, sgps::ManagedAllocator<int>>
        offset_BSD_data;  // vector to store original particle idx after sorting

    std::vector<int, sgps::ManagedAllocator<int>> pair_i_data;
    std::vector<int, sgps::ManagedAllocator<int>> pair_j_data;

    std::vector<float3, sgps::ManagedAllocator<float3>> pos_data;  // position data of all particles
    std::vector<int, sgps::ManagedAllocator<int>> particle_idx;    // particle index tracking vector
    std::vector<int, sgps::ManagedAllocator<int>> BSD_idx;         // BSD index tracking vector
    std::vector<int, sgps::ManagedAllocator<int>>
        BSD_iden_idx;  // BSD identification tracking vector, this vector identifies whether the current particle is in
                       // the buffer zone or in the actual SD (0 is not in buffer, 1 is in buffer)
    std::vector<int, sgps::ManagedAllocator<int>> num_col;  // number of collision - the output of kinematic 1st pass

    std::vector<float, sgps::ManagedAllocator<float>> rho_data;       // local density data
    std::vector<float, sgps::ManagedAllocator<float>> pressure_data;  // local presusre dataf

    std::vector<float3, sgps::ManagedAllocator<float3>> W_grad_data;  // local W grad data
    std::vector<int, sgps::ManagedAllocator<int>> temp_storage;       // temp storage space for cub functions
    std::vector<char, sgps::ManagedAllocator<char>> fix_data;

    // initiate JitHelper to perform JITC
    auto kinematic_program = JitHelper::buildProgram(
        "SPHKinematicKernels", JitHelper::KERNEL_DIR / "SPHKinematicKernels.cu",
        std::unordered_map<std::string, std::string>(), {"-I" + (JitHelper::KERNEL_DIR / "..").string()});

    // intermediate variables declaration

    while (getParentSystem().curr_time < getParentSystem().sim_time) {
        if (kinematicCounter == 0) {
            const std::lock_guard<std::mutex> lock(getParentSystem().getMutexPos());
            fix_data.resize(k_n);
            pressure_data.resize(k_n);
            pos_data.resize(k_n);
            rho_data.resize(k_n);
            cudaMemcpy(fix_data.data(), dataManager.m_fix.data(), k_n * sizeof(char), cudaMemcpyDefault);
            cudaMemcpy(pressure_data.data(), dataManager.m_pressure.data(), k_n * sizeof(float),
                       cudaMemcpyDefault);
        }

        // for each step, the kinematic thread needs to do two passes
        // first pass - look for 'number' of potential contacts
        // crate an array to store number of valid potential contacts

        if (getParentSystem().pos_data_isFresh == true) {
            const std::lock_guard<std::mutex> lock(getParentSystem().getMutexPos());
            cudaMemcpy(pos_data.data(), dataManager.m_pos.data(), k_n * sizeof(float3), cudaMemcpyDefault);
        }

        // notify the system that the position data has been consumed
        getParentSystem().pos_data_isFresh = false;

        // loop through all vertices to fill in domain vector
        float d_domain_x = getParentSystem().domain_x / X_SUB_NUM;
        float d_domain_y = getParentSystem().domain_y / Y_SUB_NUM;
        float d_domain_z = getParentSystem().domain_z / Z_SUB_NUM;
        float multiplier = 1.33;
        float buffer_width = 2.0;

        // printf("d_domain: %f, %f, %f\n", d_domain_x, d_domain_y, d_domain_z);

        // ==============================================================================================================
        // Kinematic Step 1
        // Identify total number of BSD each particle touches
        // CUB exclusive prefix scan to get totlength
        // ==============================================================================================================
        // resize the num_BSD_data vector
        num_BSD_data.resize(k_n);
        // resize the idx_track_data vector
        idx_track_data.resize(k_n);
        // resize the offset_BSD_data vector
        offset_BSD_data.resize(k_n);

        kinematic_program.kernel("kinematicStep1")
            .instantiate()
            .configure(dim3(num_block), dim3(num_thread), 0, streamInfo.stream)
            .launch(pos_data.data(), num_BSD_data.data(), k_n, multiplier * kernel_h, d_domain_x, d_domain_y,
                    d_domain_z, X_SUB_NUM, Y_SUB_NUM, Z_SUB_NUM, getParentSystem().domain_x, getParentSystem().domain_y,
                    getParentSystem().domain_z, buffer_width);

        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        PrefixScanExclusiveCub(num_BSD_data, offset_BSD_data, num_BSD_data.size(), temp_storage);

        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        int TotLength = offset_BSD_data[k_n - 1] + num_BSD_data[k_n - 1];

        // ==============================================================================================================
        // Kinematic Step 2
        // Fill in all BSD IDs Each Particle Belongs to (GPU)
        // This step also needs to identify whether the particle is in the 'buffer zone' or in the 'actual SD'
        // ==============================================================================================================

        idx_track_data.resize(TotLength);
        BSD_idx.resize(TotLength);
        BSD_iden_idx.resize(TotLength);

        kinematic_program.kernel("kinematicStep2")
            .instantiate()
            .configure(dim3(num_block), dim3(num_thread), 0, streamInfo.stream)
            .launch(pos_data.data(), offset_BSD_data.data(), BSD_iden_idx.data(), BSD_idx.data(), idx_track_data.data(),
                    k_n, TotLength, multiplier * kernel_h, d_domain_x, d_domain_y, d_domain_z, X_SUB_NUM, Y_SUB_NUM,
                    Z_SUB_NUM, getParentSystem().domain_x, getParentSystem().domain_y, getParentSystem().domain_z,
                    buffer_width);

        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        // ==============================================================================================================
        // Kinematic Step 3
        // CUB sort Based on BSD_idx of the particles
        // Sort twice
        // 1st: BSD_idx (key), idx_track_data (value)
        // 2nd: BSD_idx (key), BSD_iden_idx (value)
        // the output of this step will be BSD_idx_sorted, idx_track_data_sorted, and BSD_iden_idx_sorted
        // ==============================================================================================================
        std::vector<int, sgps::ManagedAllocator<int>> BSD_idx_sorted;
        std::vector<int, sgps::ManagedAllocator<int>> idx_track_data_sorted;
        std::vector<int, sgps::ManagedAllocator<int>> BSD_iden_idx_sorted;

        PairRadixSortAscendCub(BSD_idx, BSD_idx_sorted, idx_track_data, idx_track_data_sorted, BSD_idx.size(),
                               temp_storage);

        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        PairRadixSortAscendCub(BSD_idx, BSD_idx_sorted, BSD_iden_idx, BSD_iden_idx_sorted, BSD_idx.size(),
                               temp_storage);

        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        // ==============================================================================================================
        // Kinematic Step 4
        // Compute BSD Offsets and Lengths
        // cub::DeviceRunLengthEncode::Encode​ and cub::ExclusiveScan need to be called
        // ==============================================================================================================
        std::vector<int, sgps::ManagedAllocator<int>> unique_BSD_idx;
        std::vector<int, sgps::ManagedAllocator<int>> length_BSD_idx;
        std::vector<int, sgps::ManagedAllocator<int>> offset_BSD_idx;

        RunLengthEncodeCub(BSD_idx_sorted, unique_BSD_idx, length_BSD_idx, BSD_idx_sorted.size(), temp_storage);

        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        PrefixScanExclusiveCub(length_BSD_idx, offset_BSD_idx, length_BSD_idx.size(), temp_storage);

        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        // ==============================================================================================================
        // Kinematic Step 5
        // Count Collision Events Number​
        // (Kinematic First Pass)
        // ==============================================================================================================

        num_col.resize(unique_BSD_idx.size() * MAX_PARTICLES_PER_BSD);

        // set threads per block to be 512
        // set total number of block to be unique_BSD_idx.size()
        num_block = unique_BSD_idx.size();
        num_thread = MAX_PARTICLES_PER_BSD;

        // launch kinematic first pass kernel
        kinematic_program.kernel("kinematicStep5")
            .instantiate()
            .configure(dim3(num_block), dim3(num_thread), 0, streamInfo.stream)
            .launch(pos_data.data(), k_n, multiplier * kernel_h, idx_track_data_sorted.data(),
                    BSD_iden_idx_sorted.data(), offset_BSD_idx.data(), length_BSD_idx.data(), unique_BSD_idx.data(),
                    num_col.data(), unique_BSD_idx.size(), buffer_width);

        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        // ==============================================================================================================
        // Kinematic Step 6
        // Compute offsets for num_coll_each_bsd
        // This is supposed to be a CUB exclusive scan
        // ==============================================================================================================
        std::vector<int, sgps::ManagedAllocator<int>> num_col_offset;

        PrefixScanExclusiveCub(num_col, num_col_offset, num_col.size(), temp_storage);
        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        int tot_collision = num_col_offset[num_col_offset.size() - 1] + num_col[num_col.size() - 1];

        std::cout << "tot_collision: " << tot_collision << std::endl;

        // ==============================================================================================================
        // Kinematic Step 7
        // Fill in collision pair data
        // ==============================================================================================================

        // resize contact pair data size
        pair_i_data.resize(tot_collision);
        pair_j_data.resize(tot_collision);
        W_grad_data.resize(tot_collision);

        // set threads per block to be 512
        // set total number of block to be unique_BSD_idx.size()
        num_block = unique_BSD_idx.size();
        num_thread = MAX_PARTICLES_PER_BSD;

        // launch kinematic first pass kernel
        kinematic_program.kernel("kinematicStep7")
            .instantiate()
            .configure(dim3(num_block), dim3(num_thread), 0, streamInfo.stream)
            .launch(pos_data.data(), k_n, multiplier * kernel_h, idx_track_data_sorted.data(),
                    BSD_iden_idx_sorted.data(), offset_BSD_idx.data(), length_BSD_idx.data(), unique_BSD_idx.data(),
                    num_col.data(), unique_BSD_idx.size(), pair_i_data.data(), pair_j_data.data(),
                    num_col_offset.data(), W_grad_data.data(), buffer_width);

        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        // ==============================================================================================================
        // Kinematic Step 8
        // This is the 1st step to compute density and pressure
        // Computing particle j's contribution to particle i's density
        // ==============================================================================================================

        // 1st pass to compute particle j's contribution to particle i's density
        std::vector<int, sgps::ManagedAllocator<int>> i_data_sorted_1;
        std::vector<int, sgps::ManagedAllocator<int>> j_data_sorted_1;

        PairRadixSortAscendCub(pair_i_data, i_data_sorted_1, pair_j_data, j_data_sorted_1, tot_collision, temp_storage);
        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        std::vector<int, sgps::ManagedAllocator<int>> i_unique;
        std::vector<int, sgps::ManagedAllocator<int>> i_length;
        std::vector<int, sgps::ManagedAllocator<int>> i_offset;

        RunLengthEncodeCub(i_data_sorted_1, i_unique, i_length, i_data_sorted_1.size(), temp_storage);
        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
        PrefixScanExclusiveCub(i_length, i_offset, i_length.size(), temp_storage);
        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        // initialize density and pressure vectors
        rho_data.resize(pos_data.size());

        num_thread = 512;
        num_block =
            (i_unique.size() % num_thread != 0) ? (i_unique.size() / num_thread + 1) : (i_unique.size() / num_thread);

        kinematic_program.kernel("kinematicStep8")
            .instantiate()
            .configure(dim3(num_block), dim3(num_thread), 0, streamInfo.stream)
            .launch(pos_data.data(), rho_data.data(), pressure_data.data(), i_unique.data(), i_offset.data(),
                    i_length.data(), j_data_sorted_1.data(), fix_data.data(), i_unique.size(), multiplier * kernel_h, m,
                    rho_0);
        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        // ==============================================================================================================
        // Kinematic Step 9
        // This is the 2nd step to compute density and pressure
        // Computing particle i's contribution to particle j's density
        // ==============================================================================================================
        // 1st pass to compute particle j's contribution to particle i's density
        std::vector<int, sgps::ManagedAllocator<int>> i_data_sorted_2;
        std::vector<int, sgps::ManagedAllocator<int>> j_data_sorted_2;

        PairRadixSortAscendCub(pair_j_data, j_data_sorted_2, pair_i_data, i_data_sorted_2, tot_collision, temp_storage);
        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        std::vector<int, sgps::ManagedAllocator<int>> j_unique;
        std::vector<int, sgps::ManagedAllocator<int>> j_length;
        std::vector<int, sgps::ManagedAllocator<int>> j_offset;

        RunLengthEncodeCub(j_data_sorted_2, j_unique, j_length, j_data_sorted_2.size(), temp_storage);
        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
        PrefixScanExclusiveCub(j_length, j_offset, j_length.size(), temp_storage);
        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        num_thread = 512;
        num_block =
            (j_unique.size() % num_thread != 0) ? (j_unique.size() / num_thread + 1) : (j_unique.size() / num_thread);

        kinematic_program.kernel("kinematicStep9")
            .instantiate()
            .configure(dim3(num_block), dim3(num_thread), 0, streamInfo.stream)
            .launch(pos_data.data(), rho_data.data(), pressure_data.data(), j_unique.data(), j_offset.data(),
                    j_length.data(), i_data_sorted_2.data(), fix_data.data(), j_unique.size(), multiplier * kernel_h, m,
                    rho_0);
        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        // ==============================================================================================================
        // Kinematic Step 10
        // Add particle k's contribution to particle k's density (add density of the particle itself)
        // And compute pressure of each particle
        // ==============================================================================================================
        num_thread = 1024;
        num_block =
            (pos_data.size() % num_thread != 0) ? (pos_data.size() / num_thread + 1) : (pos_data.size() / num_thread);

        kinematic_program.kernel("kinematicStep10")
            .instantiate()
            .configure(dim3(num_block), dim3(num_thread), 0, streamInfo.stream)
            .launch(rho_data.data(), pressure_data.data(), fix_data.data(), pos_data.size(), multiplier * kernel_h, m,
                    rho_0, c);
        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        // copy data back to the dataManager
        {
            const std::lock_guard<std::mutex> lock(getParentSystem().getMutexContact());
            dataManager.m_tot_collision = tot_collision;
            if (tot_collision > dataManager.m_pair_i.size()) {
                dataManager.m_pair_i.resize(tot_collision);
                dataManager.m_pair_j.resize(tot_collision);
                dataManager.m_W_grad.resize(tot_collision);
                dataManager.m_rho.resize(tot_collision);
                dataManager.m_pressure.resize(tot_collision);
            }
            cudaMemcpy(dataManager.m_pair_i.data(), pair_i_data.data(), dataManager.m_tot_collision * sizeof(int),
                       cudaMemcpyDeviceToDevice);
            cudaMemcpy(dataManager.m_pair_j.data(), pair_j_data.data(), dataManager.m_tot_collision * sizeof(int),
                       cudaMemcpyDeviceToDevice);
            cudaMemcpy(dataManager.m_rho.data(), rho_data.data(), k_n * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(dataManager.m_pressure.data(), pressure_data.data(), k_n * sizeof(float),
                       cudaMemcpyDeviceToDevice);
            cudaMemcpy(dataManager.m_W_grad.data(), W_grad_data.data(), dataManager.m_tot_collision * sizeof(float3),
                       cudaMemcpyDeviceToDevice);
        }

        // notify the system that the contact data is now fresh
        // increment kinematic counter
        getParentSystem().contact_data_isFresh = true;
        kinematicCounter++;

        // display current loop
        std::cout << "ki ct:" << kinematicCounter << std::endl;
    }

    cudaStreamDestroy(streamInfo.stream);
}

void DynamicThread::operator()() {
    // TOUCH CUDA CONTEXT BEFORE YOU DO ANYTHING
    // Option 1:
    // std::vector<int, sgps::ManagedAllocator<int>> num_arr(dataManager.m_pos.size(), -1);

    // Option 2:
    // cudaEvent_t ev;
    // cudaEventCreate(&ev);

    GPU_CALL(cudaSetDevice(streamInfo.device));
    cudaStreamCreate(&streamInfo.stream);

    // create vector to store data from the dataManager
    std::vector<int, sgps::ManagedAllocator<int>> pair_i_data;
    std::vector<int, sgps::ManagedAllocator<int>> pair_j_data;

    std::vector<float, sgps::ManagedAllocator<float>> rho_data;
    std::vector<float, sgps::ManagedAllocator<float>> pressure_data;
    std::vector<float3, sgps::ManagedAllocator<float3>> W_grad_data;

    std::vector<float3, sgps::ManagedAllocator<float3>> col_acc_data;

    std::vector<int, sgps::ManagedAllocator<int>> offset_data;
    std::vector<float3, sgps::ManagedAllocator<float3>> pos_data;
    std::vector<float3, sgps::ManagedAllocator<float3>> vel_data;
    std::vector<float3, sgps::ManagedAllocator<float3>> acc_data;
    std::vector<char, sgps::ManagedAllocator<char>> fix_data;
    std::vector<float3, sgps::ManagedAllocator<float3>> temp_storage;
    float kernel_h;
    float m;
    float rho_0;
    int output_counter = 0;

    auto dynamic_program = JitHelper::buildProgram("SPHDynamicKernels", JitHelper::KERNEL_DIR / "SPHDynamicKernels.cu",
                                                   std::unordered_map<std::string, std::string>(),
                                                   {"-I" + (JitHelper::KERNEL_DIR / "..").string()});

    // get total number of particles
    int k_n;
    int tot_collision;
    {
        const std::lock_guard<std::mutex> lock(getParentSystem().getMutexPos());
        k_n = dataManager.k_n;
        m = dataManager.m;
        rho_0 = dataManager.rho_0;
    }

    while (getParentSystem().curr_time < getParentSystem().sim_time) {
        // Touch the CUDA context before the Kernel is accessed
        if (dynamicCounter == 0) {
            const std::lock_guard<std::mutex> lock(getParentSystem().getMutexPos());
            pos_data.resize(k_n);
            vel_data.resize(k_n);
            fix_data.resize(k_n);
            cudaMemcpy(pos_data.data(), dataManager.m_pos.data(), k_n * sizeof(float3), cudaMemcpyDefault);
            cudaMemcpy(vel_data.data(), dataManager.m_vel.data(), k_n * sizeof(float3), cudaMemcpyDefault);
            cudaMemcpy(fix_data.data(), dataManager.m_fix.data(), k_n * sizeof(char), cudaMemcpyDefault);
        }

        // temp step size explicit definition
        float time_step = getParentSystem().time_step;

        if (getParentSystem().contact_data_isFresh == true) {
            const std::lock_guard<std::mutex> lock(getParentSystem().getMutexContact());
            tot_collision = dataManager.m_tot_collision;

            if (dynamicCounter == 0) {
                pair_i_data.resize(tot_collision);
                pair_j_data.resize(tot_collision);
                rho_data.resize(k_n);
                pressure_data.resize(k_n);
                W_grad_data.resize(tot_collision);
            } else {
                if (tot_collision > pair_i_data.size()) {
                    pair_i_data.resize(tot_collision);
                    pair_j_data.resize(tot_collision);
                    W_grad_data.resize(tot_collision);
                }
            }
            cudaMemcpy(pair_i_data.data(), dataManager.m_pair_i.data(), tot_collision * sizeof(int),
                       cudaMemcpyDeviceToDevice);
            cudaMemcpy(pair_j_data.data(), dataManager.m_pair_j.data(), tot_collision * sizeof(int),
                       cudaMemcpyDeviceToDevice);
            cudaMemcpy(rho_data.data(), dataManager.m_rho.data(), k_n * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(pressure_data.data(), dataManager.m_pressure.data(), k_n * sizeof(float),
                       cudaMemcpyDeviceToDevice);
            cudaMemcpy(W_grad_data.data(), dataManager.m_W_grad.data(), tot_collision * sizeof(float3),
                       cudaMemcpyDeviceToDevice);
        }

        // notify the system that the contact data is old
        getParentSystem().contact_data_isFresh = false;

        if (tot_collision == 0) {
            continue;
        }

        // clear acc_data each step
        acc_data.resize(pos_data.size());

        // ==============================================================================================================
        // Dynamic Step 1
        // Use GPU to fill in the accelerations in each pair of contact data
        // ==============================================================================================================
        int block_size = 1024;
        int num_thread = (block_size < tot_collision) ? block_size : tot_collision;
        int num_block =
            (tot_collision % num_thread != 0) ? (tot_collision / num_thread + 1) : (tot_collision / num_thread);

        col_acc_data.resize(tot_collision);

        // call dynamic first gpu pass
        // this pass will fill the contact pair data vector
        dynamic_program.kernel("dynamicStep1")
            .instantiate()
            .configure(dim3(num_block), dim3(num_thread), 0, streamInfo.stream)
            .launch(pair_i_data.data(), pair_j_data.data(), rho_data.data(), pressure_data.data(), col_acc_data.data(),
                    W_grad_data.data(), tot_collision, m);
        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        // ==============================================================================================================
        // Dynamic Step 2
        // Reduce acceleration data w.r.t particle i, then add to acc_data for particle i
        // ==============================================================================================================
        std::vector<int, sgps::ManagedAllocator<int>> pair_i_data_sorted_1;
        std::vector<float3, sgps::ManagedAllocator<float3>> col_acc_data_sorted_1;
        std::vector<int, sgps::ManagedAllocator<int>> pair_i_data_reduced_1;
        std::vector<float3, sgps::ManagedAllocator<float3>> col_acc_data_reduced_1;

        // sort
        PairRadixSortAscendCub(pair_i_data, pair_i_data_sorted_1, col_acc_data, col_acc_data_sorted_1, tot_collision,
                               temp_storage);
        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        // reduce
        SumReduceByKeyCub(pair_i_data_sorted_1, pair_i_data_reduced_1, col_acc_data_sorted_1, col_acc_data_reduced_1,
                          pair_i_data_sorted_1.size(), temp_storage);
        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        block_size = 1024;
        num_thread = (block_size < pair_i_data_reduced_1.size()) ? block_size : pair_i_data_reduced_1.size();
        num_block = (pair_i_data_reduced_1.size() % num_thread != 0) ? (pair_i_data_reduced_1.size() / num_thread + 1)
                                                                     : (pair_i_data_reduced_1.size() / num_thread);
        dynamic_program.kernel("dynamicStep2")
            .instantiate()
            .configure(dim3(num_block), dim3(num_thread), 0, streamInfo.stream)
            .launch(pair_i_data_reduced_1.data(), col_acc_data_reduced_1.data(), acc_data.data(),
                    pair_i_data_reduced_1.size());
        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        // ==============================================================================================================
        // Dynamic Step 3
        // Negate col_acc_data to prepare for acceleration sort and reduction w.r.t particle j
        // ==============================================================================================================
        block_size = 512;
        num_thread = (block_size < col_acc_data.size()) ? block_size : col_acc_data.size();
        num_block = (col_acc_data.size() % num_thread != 0) ? (col_acc_data.size() / num_thread + 1)
                                                            : (col_acc_data.size() / num_thread);
        dynamic_program.kernel("dynamicStep3")
            .instantiate()
            .configure(dim3(num_block), dim3(num_thread), 0, streamInfo.stream)
            .launch(col_acc_data.data(), col_acc_data.size());
        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        // ==============================================================================================================
        // Dynamic Step 4
        // Reduce acceleration data w.r.t particle j, then add to acc_data for particle j
        // ==============================================================================================================
        std::vector<int, sgps::ManagedAllocator<int>> pair_j_data_sorted_2;
        std::vector<float3, sgps::ManagedAllocator<float3>> col_acc_data_sorted_2;
        std::vector<int, sgps::ManagedAllocator<int>> pair_j_data_reduced_2;
        std::vector<float3, sgps::ManagedAllocator<float3>> col_acc_data_reduced_2;

        // sort
        PairRadixSortAscendCub(pair_j_data, pair_j_data_sorted_2, col_acc_data, col_acc_data_sorted_2, tot_collision,
                               temp_storage);
        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        // reduce
        SumReduceByKeyCub(pair_j_data_sorted_2, pair_j_data_reduced_2, col_acc_data_sorted_2, col_acc_data_reduced_2,
                          pair_j_data_sorted_2.size(), temp_storage);
        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        block_size = 1024;
        num_thread = (block_size < pair_j_data_reduced_2.size()) ? block_size : pair_j_data_reduced_2.size();
        num_block = (pair_j_data_reduced_2.size() % num_thread != 0) ? (pair_j_data_reduced_2.size() / num_thread + 1)
                                                                     : (pair_j_data_reduced_2.size() / num_thread);
        dynamic_program.kernel("dynamicStep4")
            .instantiate()
            .configure(dim3(num_block), dim3(num_thread), 0, streamInfo.stream)
            .launch(pair_j_data_reduced_2.data(), col_acc_data_reduced_2.data(), acc_data.data(),
                    pair_j_data_reduced_2.size());
        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        // ==============================================================================================================
        // Dynamic Step 5
        // Final integration step to push the simulation one time step forward
        // ==============================================================================================================

        block_size = 1024;
        num_thread = (block_size < pos_data.size()) ? block_size : pos_data.size();
        num_block =
            (pos_data.size() % num_thread != 0) ? (pos_data.size() / num_thread + 1) : (pos_data.size() / num_thread);

        dynamic_program.kernel("dynamicStep5")
            .instantiate()
            .configure(dim3(num_block), dim3(num_thread), 0, streamInfo.stream)
            .launch(pos_data.data(), vel_data.data(), acc_data.data(), fix_data.data(), pos_data.size(), time_step);

        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        // copy data back to the dataManager
        {
            const std::lock_guard<std::mutex> lock(getParentSystem().getMutexPos());
            cudaMemcpy(dataManager.m_pos.data(), pos_data.data(), k_n * sizeof(float3), cudaMemcpyDeviceToDevice);
            cudaMemcpy(dataManager.m_vel.data(), vel_data.data(), k_n * sizeof(float3), cudaMemcpyDeviceToDevice);
            cudaMemcpy(dataManager.m_acc.data(), acc_data.data(), k_n * sizeof(float3), cudaMemcpyDeviceToDevice);
        }

        if (getParentSystem().getPrintOut() == true &&
            getParentSystem().getCurPrint() == getParentSystem().getStepPrint() - 1) {
            while (getParentSystem().wt_thread_busy == true) {
                continue;
            }
            getParentSystem().wt_buffer_fresh = true;
            getParentSystem().setCurPrint(0);
            while (getParentSystem().wt_buffer_fresh == true) {
                continue;
            }

        } else {
            getParentSystem().setCurPrint(getParentSystem().getCurPrint() + 1);
        }

        // notify the system that the position data is fresh
        // increment the dynamic thread
        getParentSystem().pos_data_isFresh = true;
        dynamicCounter++;

        // increment current simulation time
        getParentSystem().curr_time = getParentSystem().curr_time + getParentSystem().time_step;
        std::cout << "dy ct:" << dynamicCounter << std::endl;
    }

    cudaStreamDestroy(streamInfo.stream);
}

void WriteOutThread::operator()() {
    while (getParentSystem().curr_time < getParentSystem().sim_time) {
        if (getParentSystem().wt_buffer_fresh == true && getParentSystem().wt_thread_busy == false) {
            getParentSystem().wt_thread_busy = true;
            std::vector<float3> wt_pos;
            std::vector<float3> wt_vel;
            std::vector<float3> wt_acc;
            std::vector<float> wt_rho;
            std::vector<float> wt_pressure;

            // acquire locks to copy position/velocity/acceleration data from datamanager
            {
                const std::lock_guard<std::mutex> lock(getParentSystem().getMutexPos());
                wt_pos.assign(dataManager.m_pos.begin(), dataManager.m_pos.end());
                wt_vel.assign(dataManager.m_vel.begin(), dataManager.m_vel.end());
                wt_acc.assign(dataManager.m_acc.begin(), dataManager.m_acc.end());
            }

            // acquire locks to copy density/pressure data from datamanager
            {
                const std::lock_guard<std::mutex> lock(getParentSystem().getMutexContact());
                wt_rho.assign(dataManager.m_rho.begin(), dataManager.m_rho.end());
                wt_pressure.assign(dataManager.m_pressure.begin(), dataManager.m_pressure.end());
                getParentSystem().wt_buffer_fresh = false;
            }

            // write into a folder called sph_folder, this folder needs to be created before running demos
            getParentSystem().printCSV("sph_folder/test" + std::to_string(writeOutCounter + 1) + ".csv", wt_pos.data(),
                                       wt_pos.size(), wt_vel.data(), wt_acc.data(), wt_rho.data(), wt_pressure.data());
            getParentSystem().wt_thread_busy = false;
            writeOutCounter++;
            std::cout << "wo ct:" << writeOutCounter << std::endl;
        } else {
            continue;
        }
    }
}
