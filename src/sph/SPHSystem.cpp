// SPH-DualGPU
// SPH system base class implementation source code

#include <cstdint>
#include <ostream>
#include <sph/SPHSystem.h>
#include <core/utils/JitHelper.h>
#include <thread>
#include <vector>
#include <chrono>
#include <core/utils/GpuError.h>
#include "datastruct.h"
#include <algorithms/SPHCubHelperFunctions.h>

int X_SUB_NUM = 0;
int Y_SUB_NUM = 0;
int Z_SUB_NUM = 0;

void SPHSystem::initialize(float kernel_h,
                           float m,
                           float rho_0,
                           std::vector<float3>& pos,
                           std::vector<float3>& vel,
                           std::vector<float3>& acc,
                           std::vector<char>& fix,
                           float domain_x,
                           float domain_y,
                           float domain_z) {
    dataManager.kernel_h = kernel_h;
    dataManager.m = m;
    dataManager.rho_0 = rho_0;
    dataManager.m_pos.assign(pos.begin(), pos.end());
    dataManager.m_vel.assign(vel.begin(), vel.end());
    dataManager.m_acc.assign(acc.begin(), acc.end());
    dataManager.m_fix.assign(fix.begin(), fix.end());
    this->domain_x = domain_x;
    this->domain_y = domain_y;
    this->domain_z = domain_z;

    // redeclare the number of subdomain to be size/4 (4 is the BSD side length)
    X_SUB_NUM = domain_x / 4;
    Y_SUB_NUM = domain_y / 4;
    Z_SUB_NUM = domain_z / 4;
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
                << vel_arr[i].y << "," << vel_arr[i].z << "," << std::endl;
    }

    csvFile.close();
}

void SPHSystem::printCSV(std::string filename, float3* pos_arr, int pos_n, float3* vel_arr, float3* acc_arr) {
    // create file
    std::ofstream csvFile(filename);

    csvFile << "x_pos,y_pos,z_pos,x_vel,y_vel,z_vel" << std::endl;

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
    {
        const std::lock_guard<std::mutex> lock(getParentSystem().getMutexPos());
        k_n = dataManager.m_pos.size();
        kernel_h = dataManager.kernel_h;
        m = dataManager.m;
        rho_0 = dataManager.rho_0;
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

    // intermediate variables declaration

    while (getParentSystem().curr_time < getParentSystem().sim_time) {
        float tolerance = 0.05;

        // for each step, the kinematic thread needs to do two passes
        // first pass - look for 'number' of potential contacts
        // crate an array to store number of valid potential contacts

        if (getParentSystem().pos_data_isFresh == true) {
            const std::lock_guard<std::mutex> lock(getParentSystem().getMutexPos());
            pos_data.assign(dataManager.m_pos.begin(), dataManager.m_pos.end());
        }

        // notify the system that the position data has been consumed
        getParentSystem().pos_data_isFresh = false;

        // loop through all vertices to fill in domain vector
        float d_domain_x = getParentSystem().domain_x / X_SUB_NUM;
        float d_domain_y = getParentSystem().domain_y / Y_SUB_NUM;
        float d_domain_z = getParentSystem().domain_z / Z_SUB_NUM;

        // initiate JitHelper to perform JITC
        auto kinematic_program =
            JitHelper::buildProgram("SPHKinematicKernels", JitHelper::KERNEL_DIR / "SPHKinematicKernels.cu",
                                    std::unordered_map<std::string, std::string>(), {"-I" + (JitHelper::KERNEL_DIR / "..").string()});

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
            .launch(pos_data.data(), num_BSD_data.data(), k_n, kernel_h, d_domain_x, d_domain_y, d_domain_z, X_SUB_NUM,
                    Y_SUB_NUM, Z_SUB_NUM, getParentSystem().domain_x, getParentSystem().domain_y,
                    getParentSystem().domain_z);

        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        PrefixScanExclusiveCub(num_BSD_data, offset_BSD_data);

        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        int TotLength = offset_BSD_data[k_n - 1] + num_BSD_data[k_n - 1];

        // std::cout << "TotLength: " << TotLength << std::endl;

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
                    k_n, TotLength, kernel_h, d_domain_x, d_domain_y, d_domain_z, X_SUB_NUM, Y_SUB_NUM, Z_SUB_NUM,
                    getParentSystem().domain_x, getParentSystem().domain_y, getParentSystem().domain_z);

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

        PairRadixSortAscendCub(BSD_idx, BSD_idx_sorted, idx_track_data, idx_track_data_sorted);

        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        PairRadixSortAscendCub(BSD_idx, BSD_idx_sorted, BSD_iden_idx, BSD_iden_idx_sorted);

        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        // ==============================================================================================================
        // Kinematic Step 4
        // Compute BSD Offsets and Lengths
        // cub::DeviceRunLengthEncode::Encode​ and cub::ExclusiveScan need to be called
        // ==============================================================================================================
        std::vector<int, sgps::ManagedAllocator<int>> unique_BSD_idx;
        std::vector<int, sgps::ManagedAllocator<int>> length_BSD_idx;
        std::vector<int, sgps::ManagedAllocator<int>> offset_BSD_idx;

        RunLengthEncodeCub(BSD_idx_sorted, unique_BSD_idx, length_BSD_idx);

        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        PrefixScanExclusiveCub(length_BSD_idx, offset_BSD_idx);

        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        // ==============================================================================================================
        // Kinematic Step 5
        // Count Collision Events Number​
        // (Kinematic First Pass)
        // ==============================================================================================================

        num_col.clear();
        num_col.resize(unique_BSD_idx.size() * 512);

        // set threads per block to be 512
        // set total number of block to be unique_BSD_idx.size()
        num_block = unique_BSD_idx.size();
        num_thread = 512;

        // launch kinematic first pass kernel
        kinematic_program.kernel("kinematicStep5")
            .instantiate()
            .configure(dim3(num_block), dim3(num_thread), (MAX_NUM_UNIT * UNIT_SHARED_SIZE), streamInfo.stream)
            .launch(pos_data.data(), k_n, tolerance, kernel_h, idx_track_data_sorted.data(), BSD_iden_idx_sorted.data(),
                    offset_BSD_idx.data(), length_BSD_idx.data(), unique_BSD_idx.data(), num_col.data(),
                    unique_BSD_idx.size());

        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        // ==============================================================================================================
        // Kinematic Step 6
        // Compute offsets for num_coll_each_bsd
        // This is supposed to be a CUB exclusive scan
        // ==============================================================================================================
        std::vector<int, sgps::ManagedAllocator<int>> num_col_offset;

        PrefixScanExclusiveCub(num_col, num_col_offset);
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
        num_thread = 512;

        // launch kinematic first pass kernel
        kinematic_program.kernel("kinematicStep7")
            .instantiate()
            .configure(dim3(num_block), dim3(num_thread), (MAX_NUM_UNIT * UNIT_SHARED_SIZE), streamInfo.stream)
            .launch(pos_data.data(), k_n, tolerance, kernel_h, idx_track_data_sorted.data(), BSD_iden_idx_sorted.data(),
                    offset_BSD_idx.data(), length_BSD_idx.data(), unique_BSD_idx.data(), num_col.data(),
                    unique_BSD_idx.size(), pair_i_data.data(), pair_j_data.data(), num_col_offset.data(),
                    W_grad_data.data());

        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        // ================================================================================
        // Kinematic Step 8
        // Flateen the collision pair array to prepare for rho/pressure/W_grad computation
        // Sort all pairs and compute rho/pressure/W_grad
        // ================================================================================
        std::vector<int, sgps::ManagedAllocator<int>> inv_pair_i_data;
        std::vector<int, sgps::ManagedAllocator<int>> inv_pair_j_data;
        inv_pair_i_data.resize(pair_i_data.size());
        inv_pair_j_data.resize(pair_j_data.size());

        num_thread = 1024;
        num_block = (pair_i_data.size() % num_thread != 0) ? (pair_i_data.size() / num_thread + 1)
                                                           : (pair_i_data.size() / num_thread);

        kinematic_program.kernel("kinematicStep8")
            .instantiate()
            .configure(dim3(num_block), dim3(num_thread), 0, streamInfo.stream)
            .launch(pair_i_data.data(), pair_j_data.data(), inv_pair_i_data.data(), inv_pair_j_data.data(),
                    pair_i_data.size());
        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        // reuse two inv vectors
        inv_pair_i_data.insert(inv_pair_i_data.end(), pair_i_data.begin(), pair_i_data.end());
        inv_pair_j_data.insert(inv_pair_j_data.end(), pair_j_data.begin(), pair_j_data.end());

        std::vector<int, sgps::ManagedAllocator<int>> i_data_sorted;
        std::vector<int, sgps::ManagedAllocator<int>> j_data_sorted;

        PairRadixSortAscendCub(inv_pair_i_data, i_data_sorted, inv_pair_j_data, j_data_sorted);

        std::vector<int, sgps::ManagedAllocator<int>> i_unique;
        std::vector<int, sgps::ManagedAllocator<int>> i_length;
        std::vector<int, sgps::ManagedAllocator<int>> i_offset;

        RunLengthEncodeCub(i_data_sorted, i_unique, i_length);
        PrefixScanExclusiveCub(i_length, i_offset);

        rho_data.resize(pos_data.size());
        pressure_data.resize(pos_data.size());

        num_thread = 512;
        num_block =
            (i_unique.size() % num_thread != 0) ? (i_unique.size() / num_thread + 1) : (i_unique.size() / num_thread);

        kinematic_program.kernel("kinematicStep9")
            .instantiate()
            .configure(dim3(num_block), dim3(num_thread), 0, streamInfo.stream)
            .launch(pos_data.data(), rho_data.data(), pressure_data.data(), i_unique.data(), i_offset.data(),
                    i_length.data(), j_data_sorted.data(), i_unique.size(), kernel_h, m, rho_0);
        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        // copy data back to the dataManager
        {
            const std::lock_guard<std::mutex> lock(getParentSystem().getMutexContact());
            dataManager.m_pair_i.assign(pair_i_data.begin(), pair_i_data.end());
            dataManager.m_pair_j.assign(pair_j_data.begin(), pair_j_data.end());
            dataManager.m_rho.assign(rho_data.begin(), rho_data.end());
            dataManager.m_pressure.assign(pressure_data.begin(), pressure_data.end());
            dataManager.m_offset.assign(offset_BSD_data.begin(), offset_BSD_data.end());
            dataManager.m_W_grad.assign(W_grad_data.begin(), W_grad_data.end());
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
    float kernel_h;
    float m;
    float rho_0;

    // get total numer of particles
    int k_n;
    {
        const std::lock_guard<std::mutex> lock(getParentSystem().getMutexPos());
        k_n = dataManager.m_pos.size();
        kernel_h = dataManager.kernel_h;
        m = dataManager.m;
        rho_0 = dataManager.rho_0;
    }

    while (getParentSystem().curr_time < getParentSystem().sim_time) {
        // Touch the CUDA context before the Kernel is accessed

        // Option 1:
        // std::vector<int, sgps::ManagedAllocator<int>> num_arr(dataManager.m_pos.size(), -1);

        // Option 2:
        // cudaEvent_t ev;
        // cudaEventCreate(&ev);

        // temp step size explicit definition
        float time_step = getParentSystem().time_step;

        if (getParentSystem().contact_data_isFresh == true) {
            const std::lock_guard<std::mutex> lock(getParentSystem().getMutexContact());
            pair_i_data.assign(dataManager.m_pair_i.begin(), dataManager.m_pair_i.end());
            pair_j_data.assign(dataManager.m_pair_j.begin(), dataManager.m_pair_j.end());

            rho_data.assign(dataManager.m_rho.begin(), dataManager.m_rho.end());
            pressure_data.assign(dataManager.m_pressure.begin(), dataManager.m_pressure.end());
            W_grad_data.assign(dataManager.m_W_grad.begin(), dataManager.m_W_grad.end());
        }

        // notify the system that the contact data is old
        getParentSystem().contact_data_isFresh = false;

        int contact_size = pair_i_data.size();

        if (contact_size == 0) {
            continue;
        }

        {
            const std::lock_guard<std::mutex> lock(getParentSystem().getMutexPos());
            pos_data.assign(dataManager.m_pos.begin(), dataManager.m_pos.end());
            vel_data.assign(dataManager.m_vel.begin(), dataManager.m_vel.end());
            acc_data.assign(dataManager.m_acc.begin(), dataManager.m_acc.end());
            fix_data.assign(dataManager.m_fix.begin(), dataManager.m_fix.end());
        }

        auto dynamic_program =
            JitHelper::buildProgram("SPHDynamicKernels", JitHelper::KERNEL_DIR / "SPHDynamicKernels.cu",
                                    std::unordered_map<std::string, std::string>(), {"-I" + (JitHelper::KERNEL_DIR / "..").string()});

        // Dynamic Step 1
        // Use GPU to fill in the contact forces in each pair of contactData element
        int block_size = 1024;
        int num_thread = (block_size < pair_i_data.size()) ? block_size : pair_i_data.size();
        int num_block = (pair_i_data.size() % num_thread != 0) ? (pair_i_data.size() / num_thread + 1)
                                                               : (pair_i_data.size() / num_thread);

        col_acc_data.resize(pair_i_data.size());

        // call dynamic first gpu pass
        // this pass will fill the contact pair data vector
        dynamic_program.kernel("dynamicStep1")
            .instantiate()
            .configure(dim3(num_block), dim3(num_thread), 0, streamInfo.stream)
            .launch(pair_i_data.data(), pair_j_data.data(), rho_data.data(), pressure_data.data(), col_acc_data.data(),
                    W_grad_data.data(), pair_i_data.size(), kernel_h, m);
        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        // Dynamic Step 2
        // Flattens the contact data array, this will duplicate each conllision pair into 2 entries
        // A applies force on B + B applies force on A

        std::vector<int, sgps::ManagedAllocator<int>> inv_pair_i_data;
        std::vector<float3, sgps::ManagedAllocator<float3>> inv_col_acc_data;
        inv_pair_i_data.resize(pair_i_data.size());
        inv_col_acc_data.resize(col_acc_data.size());

        dynamic_program.kernel("dynamicStep2")
            .instantiate()
            .configure(dim3(num_block), dim3(num_thread), 0, streamInfo.stream)
            .launch(pair_i_data.data(), pair_j_data.data(), col_acc_data.data(), inv_pair_i_data.data(),
                    inv_col_acc_data.data(), pair_i_data.size());
        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        std::vector<int, sgps::ManagedAllocator<int>> flat_pair_i_data;
        std::vector<float3, sgps::ManagedAllocator<float3>> flat_col_acc_data;

        flat_pair_i_data.insert(flat_pair_i_data.end(), pair_i_data.begin(), pair_i_data.end());
        flat_pair_i_data.insert(flat_pair_i_data.end(), inv_pair_i_data.begin(), inv_pair_i_data.end());

        flat_col_acc_data.insert(flat_col_acc_data.end(), col_acc_data.begin(), col_acc_data.end());
        flat_col_acc_data.insert(flat_col_acc_data.end(), inv_col_acc_data.begin(), inv_col_acc_data.end());

        // Dynamic Step 3
        // Reduce to Get Total Force Applies on Each Particle (CUB)

        std::vector<int, sgps::ManagedAllocator<int>> pair_i_data_sorted;
        std::vector<float3, sgps::ManagedAllocator<float3>> col_acc_data_sorted;
        std::vector<int, sgps::ManagedAllocator<int>> pair_i_data_reduced;
        std::vector<float3, sgps::ManagedAllocator<float3>> col_acc_data_reduced;

        // sort
        PairRadixSortAscendCub(flat_pair_i_data, pair_i_data_sorted, flat_col_acc_data, col_acc_data_sorted);
        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        // reduce
        SumReduceByKeyCub(pair_i_data_sorted, pair_i_data_reduced, col_acc_data_sorted, col_acc_data_reduced);
        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        // Dynamic Step 4
        block_size = 1024;
        num_thread = (block_size < pair_i_data_reduced.size()) ? block_size : pair_i_data_reduced.size();
        num_block = (pair_i_data_reduced.size() % num_thread != 0) ? (pair_i_data_reduced.size() / num_thread + 1)
                                                                   : (pair_i_data_reduced.size() / num_thread);
        dynamic_program.kernel("dynamicStep4")
            .instantiate()
            .configure(dim3(num_block), dim3(num_thread), 0, streamInfo.stream)
            .launch(pair_i_data_reduced.data(), col_acc_data_reduced.data(), acc_data.data(),
                    pair_i_data_reduced.size());
        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        // Dynamic Step 5
        // Final integration step to push the simulation one time step forward

        block_size = 1024;
        num_thread = (block_size < pos_data.size()) ? block_size : pos_data.size();
        num_block =
            (pos_data.size() % num_thread != 0) ? (pos_data.size() / num_thread + 1) : (pos_data.size() / num_thread);

        dynamic_program.kernel("dynamicStep5")
            .instantiate()
            .configure(dim3(num_block), dim3(num_thread), 0, streamInfo.stream)
            .launch(pos_data.data(), vel_data.data(), acc_data.data(), fix_data.data(), pos_data.size(), time_step,
                    kernel_h);

        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        // copy data back to the dataManager
        {
            const std::lock_guard<std::mutex> lock(getParentSystem().getMutexPos());
            dataManager.m_pos.assign(pos_data.begin(), pos_data.end());
            dataManager.m_vel.assign(vel_data.begin(), vel_data.end());
            dataManager.m_acc.assign(acc_data.begin(), acc_data.end());
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

            {
                const std::lock_guard<std::mutex> lock(getParentSystem().getMutexPos());
                wt_pos.assign(dataManager.m_pos.begin(), dataManager.m_pos.end());
                wt_vel.assign(dataManager.m_vel.begin(), dataManager.m_vel.end());
                getParentSystem().wt_buffer_fresh = false;
            }

            getParentSystem().printCSV("sph_folder/test" + std::to_string(writeOutCounter) + ".csv", wt_pos.data(),
                                       wt_pos.size(), wt_vel.data());
            getParentSystem().wt_thread_busy = false;
            writeOutCounter++;
            std::cout << "wo ct:" << writeOutCounter << std::endl;
        } else {
            continue;
        }
    }
}
