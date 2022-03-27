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
#include <core/utils/CpuAlgorithmHelper.h>
#include "datastruct.h"
#include <algorithms/SPHCubHelperFunctions.h>

int X_SUB_NUM = 0;
int Y_SUB_NUM = 0;
int Z_SUB_NUM = 0;

void SPHSystem::initialize(float radius,
                           std::vector<vector3>& pos,
                           std::vector<vector3>& vel,
                           std::vector<vector3>& acc,
                           std::vector<char>& fix,
                           float domain_x,
                           float domain_y,
                           float domain_z) {
    dataManager.radius = radius;
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

void SPHSystem::printCSV(std::string filename, vector3* pos_arr, int pos_n, vector3* vel_arr) {
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

void SPHSystem::printCSV(std::string filename, vector3* pos_arr, int pos_n, vector3* vel_arr, vector3* acc_arr) {
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
    float radius;
    {
        const std::lock_guard<std::mutex> lock(getParentSystem().getMutexPos());
        k_n = dataManager.m_pos.size();
        radius = dataManager.radius;
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
    std::vector<contactData, sgps::ManagedAllocator<contactData>>
        contact_data;  // vector to store contact pair data for the 2nd thread
    std::vector<vector3, sgps::ManagedAllocator<vector3>> pos_data;  // position data of all particles
    std::vector<int, sgps::ManagedAllocator<int>> particle_idx;      // particle index tracking vector
    std::vector<int, sgps::ManagedAllocator<int>> BSD_idx;           // BSD index tracking vector
    std::vector<int, sgps::ManagedAllocator<int>>
        BSD_iden_idx;  // BSD identification tracking vector, this vector identifies whether the current particle is in
                       // the buffer zone or in the actual SD (0 is not in buffer, 1 is in buffer)
    std::vector<int, sgps::ManagedAllocator<int>> num_col;  // number of collision - the output of kinematic 1st pass

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
                                    std::vector<JitHelper::Header>(), {"-I" + (JitHelper::KERNEL_DIR / "..").string()});

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
            .launch(pos_data.data(), num_BSD_data.data(), k_n, radius, d_domain_x, d_domain_y, d_domain_z, X_SUB_NUM,
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
                    k_n, TotLength, radius, d_domain_x, d_domain_y, d_domain_z, X_SUB_NUM, Y_SUB_NUM, Z_SUB_NUM,
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
            .launch(pos_data.data(), k_n, tolerance, radius, idx_track_data_sorted.data(), BSD_iden_idx_sorted.data(),
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
        contact_data.resize(tot_collision);

        // set threads per block to be 512
        // set total number of block to be unique_BSD_idx.size()
        num_block = unique_BSD_idx.size();
        num_thread = 512;

        // launch kinematic first pass kernel
        kinematic_program.kernel("kinematicStep7")
            .instantiate()
            .configure(dim3(num_block), dim3(num_thread), (MAX_NUM_UNIT * UNIT_SHARED_SIZE), streamInfo.stream)
            .launch(pos_data.data(), k_n, tolerance, radius, idx_track_data_sorted.data(), BSD_iden_idx_sorted.data(),
                    offset_BSD_idx.data(), length_BSD_idx.data(), unique_BSD_idx.data(), num_col.data(),
                    unique_BSD_idx.size(), contact_data.data(), contact_data.size(), num_col_offset.data());

        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        // for (int i = 0; i < contact_data.size(); i++) {
        //     std::cout << contact_data[i].contact_pair.x << ", " << contact_data[i].contact_pair.y << std::endl;
        // }

        // copy data back to the dataManager
        {
            const std::lock_guard<std::mutex> lock(getParentSystem().getMutexContact());
            dataManager.m_contact.assign(contact_data.begin(), contact_data.end());
            dataManager.m_offset.assign(offset_BSD_data.begin(), offset_BSD_data.end());
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
    std::vector<contactData, sgps::ManagedAllocator<contactData>> contact_data;
    std::vector<int, sgps::ManagedAllocator<int>> offset_data;
    std::vector<vector3, sgps::ManagedAllocator<vector3>> pos_data;
    std::vector<vector3, sgps::ManagedAllocator<vector3>> vel_data;
    std::vector<vector3, sgps::ManagedAllocator<vector3>> acc_data;
    std::vector<char, sgps::ManagedAllocator<char>> fix_data;
    float radius = dataManager.radius;

    // get total numer of particles
    int k_n;
    {
        const std::lock_guard<std::mutex> lock(getParentSystem().getMutexPos());
        k_n = dataManager.m_pos.size();
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
            contact_data.clear();
            contact_data.assign(dataManager.m_contact.begin(), dataManager.m_contact.end());
        }

        // notify the system that the contact data is old
        getParentSystem().contact_data_isFresh = false;

        int contact_size = contact_data.size();

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
                                    std::vector<JitHelper::Header>(), {"-I" + (JitHelper::KERNEL_DIR / "..").string()});

        // ==============================================================================================================
        // Dynamic Step 1
        // Use GPU to fill in the contact forces in each pair of contactData element
        // ==============================================================================================================
        int block_size = 1024;
        int num_thread = (block_size < contact_data.size()) ? block_size : contact_data.size();
        int num_block = (contact_data.size() % num_thread != 0) ? (contact_data.size() / num_thread + 1)
                                                                : (contact_data.size() / num_thread);

        // call dynamic first gpu pass
        // this pass will fill the contact pair data vector
        dynamic_program.kernel("dynamicStep1")
            .instantiate()
            .configure(dim3(num_block), dim3(num_thread), 0, streamInfo.stream)
            .launch(contact_data.data(), contact_data.size(), pos_data.data(), vel_data.data(), acc_data.data(),
                    fix_data.data(), radius);
        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        // ==============================================================================================================
        // Dynamic Step 2
        // Flattens the contact data array, this will duplicate each conllision pair into 2 entries
        // A applies force on B + B applies force on A
        // ==============================================================================================================
        std::vector<contactData, sgps::ManagedAllocator<contactData>> inv_contact_data;
        inv_contact_data.resize(contact_data.size());

        dynamic_program.kernel("dynamicStep2")
            .instantiate()
            .configure(dim3(num_block), dim3(num_thread), 0, streamInfo.stream)
            .launch(contact_data.data(), contact_data.size(), inv_contact_data.data());
        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        contact_data.insert(contact_data.end(), inv_contact_data.begin(), inv_contact_data.end());

        inv_contact_data.clear();

        // ==============================================================================================================
        // Dynamic Step 3
        // Reduce to Get Total Force Applies on Each Particle (CUB)
        // TODO: This step is supposed to be done on CUB
        // ==============================================================================================================

        // set up CPU data input
        // create a long array to reduce

        std::vector<int, sgps::ManagedAllocator<int>> keys;
        std::vector<float, sgps::ManagedAllocator<float>> x_frcs;
        std::vector<float, sgps::ManagedAllocator<float>> y_frcs;
        std::vector<float, sgps::ManagedAllocator<float>> z_frcs;

        for (int i = 0; i < contact_data.size(); i++) {
            keys.push_back(contact_data[i].contact_pair.x);
            x_frcs.push_back(contact_data[i].contact_force.x);
            y_frcs.push_back(contact_data[i].contact_force.y);
            z_frcs.push_back(contact_data[i].contact_force.z);
        }

        std::vector<int, sgps::ManagedAllocator<int>> keys_sorted;
        std::vector<int, sgps::ManagedAllocator<int>> keys_reduced;

        std::vector<float, sgps::ManagedAllocator<float>> x_frcs_sorted;
        std::vector<float, sgps::ManagedAllocator<float>> x_frcs_reduced;

        std::vector<float, sgps::ManagedAllocator<float>> y_frcs_sorted;
        std::vector<float, sgps::ManagedAllocator<float>> y_frcs_reduced;

        std::vector<float, sgps::ManagedAllocator<float>> z_frcs_sorted;
        std::vector<float, sgps::ManagedAllocator<float>> z_frcs_reduced;

        PairRadixSortAscendCub(keys, keys_sorted, x_frcs, x_frcs_sorted);
        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
        PairRadixSortAscendCub(keys, keys_sorted, y_frcs, y_frcs_sorted);
        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
        PairRadixSortAscendCub(keys, keys_sorted, z_frcs, z_frcs_sorted);
        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
        /*
                std::cout << "====================" << std::endl;
                std::cout << "keys: ";
                for (int i = 0; i < keys.size(); i++) {
                    std::cout << keys[i] << ", ";
                }
                std::cout << std::endl;

                std::cout << "z_frcs: ";
                for (int i = 0; i < z_frcs.size(); i++) {
                    std::cout << z_frcs[i] << ", ";
                }
                std::cout << std::endl;

                std::cout << "keys_sorted: ";
                for (int i = 0; i < keys_sorted.size(); i++) {
                    std::cout << keys_sorted[i] << ", ";
                }
                std::cout << std::endl;

                std::cout << "z_frcs_sorted: ";
                for (int i = 0; i < z_frcs_sorted.size(); i++) {
                    std::cout << z_frcs_sorted[i] << ", ";
                }
                std::cout << std::endl;
        */
        SumReduceByKeyCub(keys_sorted, keys_reduced, x_frcs_sorted, x_frcs_reduced);
        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
        SumReduceByKeyCub(keys_sorted, keys_reduced, y_frcs_sorted, y_frcs_reduced);
        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
        SumReduceByKeyCub(keys_sorted, keys_reduced, z_frcs_sorted, z_frcs_reduced);
        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
        /*
                std::cout << "keys_reduced: ";
                for (int i = 0; i < keys_reduced.size(); i++) {
                    std::cout << keys_reduced[i] << ", ";
                }
                std::cout << std::endl;
                std::cout << "z_frcs_reduced: ";
                for (int i = 0; i < z_frcs_reduced.size(); i++) {
                    std::cout << z_frcs_reduced[i] << ", ";
                }
                std::cout << std::endl;

                std::cout << "====================" << std::endl;
        */
        // ==============================================================================================================
        // Dynamic Step 3
        // Compute accelerations on each particle
        // ==============================================================================================================
        block_size = 1024;
        num_thread = (block_size < keys_reduced.size()) ? block_size : keys_reduced.size();
        num_block = (keys_reduced.size() % num_thread != 0) ? (keys_reduced.size() / num_thread + 1)
                                                            : (keys_reduced.size() / num_thread);

        // call dynamic third gpu pass
        dynamic_program.kernel("dynamicStep3")
            .instantiate()
            .configure(dim3(num_block), dim3(num_thread), 0, streamInfo.stream)
            .launch(keys_reduced.data(), x_frcs_reduced.data(), y_frcs_reduced.data(), z_frcs_reduced.data(),
                    keys_reduced.size(), acc_data.data());

        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        block_size = 1024;
        num_thread = (block_size < k_n) ? block_size : k_n;
        num_block = (k_n % num_thread != 0) ? (k_n / num_thread + 1) : (k_n / num_thread);

        //==============================================================================================================
        // Dynamic Step 4
        // Final integration step to push the simulation one time step forward
        // ==============================================================================================================

        dynamic_program.kernel("dynamicStep4")
            .instantiate()
            .configure(dim3(num_block), dim3(num_thread), 0, streamInfo.stream)
            .launch(pos_data.data(), vel_data.data(), acc_data.data(), fix_data.data(), pos_data.size(), time_step,
                    radius);

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
            std::vector<vector3> wt_pos;
            std::vector<vector3> wt_vel;
            std::vector<vector3> wt_acc;

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
