// SPH-DualGPU
// SPH system base class implementation source code

#include <ostream>
#include <sph/SPHSystem.h>
#include <core/utils/JitHelper.h>
#include <thread>
#include <vector>
#include <chrono>
#include <core/utils/GpuError.h>
#include <core/utils/CpuAlgorithmHelper.h>
#include "datastruct.h"

const int X_SUB_NUM = 10;
const int Y_SUB_NUM = 10;
const int Z_SUB_NUM = 10;

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

void SPHSystem::printCSV(std::string filename, vector3* pos_arr, int pos_n, vector3* vel_arr, vector3* acc_arr) {
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
    {
        const std::lock_guard<std::mutex> lock(getParentSystem().getMutexPos());
        k_n = dataManager.m_pos.size();
    }

    // set number of blocks and number of threads in each block
    int block_size = 1024;
    int num_thread = (block_size < k_n) ? block_size : k_n;
    int num_block = (k_n % num_thread != 0) ? (k_n / num_thread + 1) : (k_n / num_thread);

    // make a copy of the necessary data stored in DataManager
    std::vector<contactData, sgps::ManagedAllocator<contactData>> contact_data;  // contact pair data
    std::vector<vector3, sgps::ManagedAllocator<vector3>> pos_data;              // position data of all particles
    std::vector<int, sgps::ManagedAllocator<int>> offset_data;  // vector to store offset data - helps the 2nd kernel
                                                                // call to identify location to write contact pair data
    std::vector<int, sgps::ManagedAllocator<int>> cd_idx_data;  // vector to store the cd of each particle
    std::vector<int, sgps::ManagedAllocator<int>>
        idx_ht_data;  // vector to store head and tail of each cell after sorting
    std::vector<int, sgps::ManagedAllocator<int>>
        idx_track_data;  // vector to store original particle idx after sorting

    // intermediate variables declaration

    while (getParentSystem().curr_time < getParentSystem().sim_time) {
        float tolerance = 0.05;

        // for each step, the kinematic thread needs to do two passes
        // first pass - look for 'number' of potential contacts
        // crate an array to store number of valid potential contacts

        if (getParentSystem().pos_data_isFresh == true) {
            const std::lock_guard<std::mutex> lock(getParentSystem().getMutexPos());
            pos_data.assign(dataManager.m_pos.begin(), dataManager.m_pos.end());
            cd_idx_data.assign(dataManager.m_idx.begin(), dataManager.m_idx.end());
        }

        // notify the system that the position data has been consumed
        getParentSystem().pos_data_isFresh = false;

        // loop through all vertices to fill in domain vector
        float d_domain_x = getParentSystem().domain_x / X_SUB_NUM;
        float d_domain_y = getParentSystem().domain_y / Y_SUB_NUM;
        float d_domain_z = getParentSystem().domain_z / Z_SUB_NUM;

        // resize cd_idx_data and idx_track_data
        cd_idx_data.resize(k_n);
        idx_track_data.resize(k_n);
        idx_ht_data.resize(2 * X_SUB_NUM * Y_SUB_NUM * Z_SUB_NUM);

        // initialize all idx_ht_data to -1
        for (int i = 0; i < idx_ht_data.size(); i++) {
            idx_ht_data[i] = -1;
        }

        // GPU sweep to put particles into their l1 subdomains
        // initiate JitHelper to perform JITC
        auto kinematic_program =
            JitHelper::buildProgram("SPHKinematicKernels", JitHelper::KERNEL_DIR / "SPHKinematicKernels.cu",
                                    std::vector<JitHelper::Header>(), {"-I" + (JitHelper::KERNEL_DIR / "..").string()});

        // ==============================================================================================================
        // Kinematic Step 1
        // Identify CD each particle belongs to
        // ==============================================================================================================

        // kinematic thread first pass
        kinematic_program.kernel("kinematic1Pass")
            .instantiate()
            .configure(dim3(num_block), dim3(num_thread), (MAX_NUM_UNIT * UNIT_SHARED_SIZE), streamInfo.stream)
            .launch(pos_data.data(), cd_idx_data.data(), idx_track_data.data(), k_n, d_domain_x, d_domain_y, d_domain_z,
                    X_SUB_NUM, Y_SUB_NUM, Z_SUB_NUM, getParentSystem().domain_x, getParentSystem().domain_y,
                    getParentSystem().domain_z);

        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        // ==============================================================================================================
        // Kinematic Step 2
        // Sort all particles based on their CD, store new sequence in idx_track
        // ==============================================================================================================

        // sort to sort the idx data with ascending idx_arr
        std::vector<int> idx_sorted;
        std::vector<int> idx_track_sorted;
        std::vector<int, sgps::ManagedAllocator<int>> idx_sorted_gpu;
        std::vector<int, sgps::ManagedAllocator<int>> idx_track_sorted_gpu;

        // TODO: replace this function call with CUB
        sortOnly(cd_idx_data.data(), idx_track_data.data(), idx_sorted, idx_track_sorted, k_n,
                 count_digit(X_SUB_NUM * Y_SUB_NUM * Z_SUB_NUM));

        // cell domain sorted
        idx_sorted_gpu.assign(idx_sorted.begin(), idx_sorted.end());
        // particle index tracking array sorted
        idx_track_sorted_gpu.assign(idx_track_sorted.begin(), idx_track_sorted.end());

        // ==============================================================================================================
        // Kinematic Step 3
        // Obtain the head and tail of particles starting and ending in each cell - idx in idx_track vector
        // ==============================================================================================================

        // Use a GPU to look up starting idx of each cell
        kinematic_program.kernel("kinematic2Pass")
            .instantiate()
            .configure(dim3(num_block), dim3(num_thread), 0, streamInfo.stream)
            .launch(idx_sorted_gpu.data(), idx_ht_data.data(), idx_sorted_gpu.size(), idx_ht_data.size());

        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        // ==============================================================================================================
        // Kinematic Step 4
        // Slice the entire simulation domain and obtain CDs contained in each SD
        // ==============================================================================================================
        std::vector<int> subdomain_decomp = slice_global_sd(10);
        std::vector<int, sgps::ManagedAllocator<int>> subdomain_decomp_gpu;  // subdomain decomposition data
        subdomain_decomp_gpu.assign(subdomain_decomp.begin(), subdomain_decomp.end());

        int num_cd_each_sd = 64;
        int num_sd = subdomain_decomp_gpu.size() / num_cd_each_sd;

        std::vector<int, sgps::ManagedAllocator<int>> n_each_sd;  // number of particles in each subdomain;
        n_each_sd.resize(subdomain_decomp_gpu.size() / num_cd_each_sd);

        // ==============================================================================================================
        // Kinematic Step 5
        // Compute total number of particles in each SD
        // ==============================================================================================================

        kinematic_program.kernel("kinematic3Pass")
            .instantiate()
            .configure(dim3(1), dim3(1024), 0, streamInfo.stream)
            .launch(idx_ht_data.data(), subdomain_decomp_gpu.data(), num_cd_each_sd, n_each_sd.data(),
                    n_each_sd.size());
        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        std::vector<int, sgps::ManagedAllocator<int>> num_arr(n_each_sd.size() * 1024, -1);

        // ==============================================================================================================
        // Kinematic Step 6
        // Compute total number of contacts for each particle
        // Current we assume each SD has 1024 particles, if particle doesn't exists, give -1, if zero contact, give 0
        // Then compute total number of contacts
        // ==============================================================================================================

        num_block = n_each_sd.size();
        num_thread = 1024;

        // kinematic thread first pass
        kinematic_program.kernel("kinematic4Pass")
            .instantiate()
            .configure(dim3(num_block), dim3(num_thread), 0, streamInfo.stream)
            .launch(pos_data.data(), pos_data.size(), tolerance, dataManager.radius, num_arr.data(), num_cd_each_sd,
                    subdomain_decomp_gpu.data(), idx_track_sorted_gpu.data(), idx_ht_data.data(), n_each_sd.data(),
                    n_each_sd.size());

        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        // calculate total number of contact
        int contact_sum = 0;

        for (int i = 0; i < num_arr.size(); i++) {
            if (num_arr[i] == -1)
                continue;
            contact_sum = contact_sum + num_arr[i];
        }

        // calculate the offset array
        int cur_idx = 0;
        offset_data.clear();
        offset_data.resize(0);
        for (int i = 0; i < num_arr.size(); i++) {
            if (num_arr[i] != -1) {
                offset_data.push_back(cur_idx);
                cur_idx = cur_idx + num_arr[i];
            } else {
                offset_data.push_back(-1);
            }
        }

        std::cout << "total contact: " << contact_sum << std::endl;

        contact_data.clear();
        contact_data.resize(contact_sum);

        // ==============================================================================================================
        // Kinematic Step 7
        // Based on the generated num_arr data, fill in all contact pair data
        // ==============================================================================================================

        // if the contact_sum is not 0, we perform the kinematic 2nd pass
        if (contact_sum != 0) {
            kinematic_program.kernel("kinematic5Pass")
                .instantiate()
                .configure(dim3(num_block), dim3(num_thread), 0, streamInfo.stream)
                .launch(pos_data.data(), pos_data.size(), offset_data.data(), num_arr.data(), tolerance,
                        dataManager.radius, contact_data.data(), num_cd_each_sd, subdomain_decomp_gpu.data(),
                        idx_track_sorted_gpu.data(), idx_ht_data.data(), n_each_sd.data(), n_each_sd.size(),
                        contact_sum);

            GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
        }

        // copy data back to the dataManager
        {
            const std::lock_guard<std::mutex> lock(getParentSystem().getMutexContact());
            dataManager.m_contact.assign(contact_data.begin(), contact_data.end());
            dataManager.m_offset.assign(offset_data.begin(), offset_data.end());
            dataManager.m_idx.assign(cd_idx_data.begin(), cd_idx_data.end());
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
            contact_data.assign(dataManager.m_contact.begin(), dataManager.m_contact.end());
            offset_data.assign(dataManager.m_offset.begin(), dataManager.m_offset.end());
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
        // Fill in all contact force terms in the contact pair data
        // this means compute the contact force for each contact collision pair
        // NOTE: now we have not REDUCE yet
        // ==============================================================================================================

        int block_size = 1024;
        int num_thread = (block_size < contact_data.size()) ? block_size : contact_data.size();
        int num_block = (contact_data.size() % num_thread != 0) ? (contact_data.size() / num_thread + 1)
                                                                : (contact_data.size() / num_thread);

        // call dynamic first gpu pass
        // this pass will fill the contact pair data vector
        dynamic_program.kernel("dynamic1Pass")
            .instantiate()
            .configure(dim3(num_block), dim3(num_thread), 0, streamInfo.stream)
            .launch(contact_data.data(), contact_data.size(), pos_data.data(), vel_data.data(), acc_data.data(),
                    fix_data.data(), radius);
        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        // call dynamic second gpu pass
        // this pass will use gpu to generate another array full of inverse elements of contact_data
        std::vector<contactData, sgps::ManagedAllocator<contactData>> inv_contact_data;
        inv_contact_data.resize(contact_data.size());

        dynamic_program.kernel("dynamic2Pass")
            .instantiate()
            .configure(dim3(num_block), dim3(num_thread), 0, streamInfo.stream)
            .launch(contact_data.data(), contact_data.size(), inv_contact_data.data());
        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        contact_data.insert(contact_data.end(), inv_contact_data.begin(), inv_contact_data.end());

        inv_contact_data.clear();

        // ==============================================================================================================
        // Dynamic Step 2
        // Perform reduction to compute total force applied on each particle
        // ==============================================================================================================

        // TODO: replace this entire step with CUB

        // set up CPU data input
        // create a long array to reduce
        std::vector<int> keys;
        std::vector<float> x_frcs;
        std::vector<float> y_frcs;
        std::vector<float> z_frcs;

        for (int i = 0; i < contact_data.size(); i++) {
            keys.push_back(contact_data[i].contact_pair.x);
            x_frcs.push_back(contact_data[i].contact_force.x);
            y_frcs.push_back(contact_data[i].contact_force.y);
            z_frcs.push_back(contact_data[i].contact_force.z);
        }

        std::vector<int> key_reduced;
        std::vector<float> x_reduced;
        std::vector<float> y_reduced;
        std::vector<float> z_reduced;

        sortReduce(keys.data(), x_frcs.data(), key_reduced, x_reduced, keys.size(), count_digit(k_n));
        key_reduced.clear();
        sortReduce(keys.data(), y_frcs.data(), key_reduced, y_reduced, keys.size(), count_digit(k_n));
        key_reduced.clear();
        sortReduce(keys.data(), z_frcs.data(), key_reduced, z_reduced, keys.size(), count_digit(k_n));

        // transfer data to GPU
        std::vector<int, sgps::ManagedAllocator<int>> gpu_key_reduced;
        std::vector<float, sgps::ManagedAllocator<float>> gpu_x_reduced;
        std::vector<float, sgps::ManagedAllocator<float>> gpu_y_reduced;
        std::vector<float, sgps::ManagedAllocator<float>> gpu_z_reduced;

        gpu_key_reduced.assign(key_reduced.begin(), key_reduced.end());
        gpu_x_reduced.assign(x_reduced.begin(), x_reduced.end());
        gpu_y_reduced.assign(y_reduced.begin(), y_reduced.end());
        gpu_z_reduced.assign(z_reduced.begin(), z_reduced.end());

        // ==============================================================================================================
        // Dynamic Step 3
        // Assign computed acceleration data based on the reduced force computed on each particle
        // ==============================================================================================================

        block_size = 1024;
        num_thread = (block_size < gpu_key_reduced.size()) ? block_size : gpu_key_reduced.size();
        num_block = (gpu_key_reduced.size() % num_thread != 0) ? (gpu_key_reduced.size() / num_thread + 1)
                                                               : (gpu_key_reduced.size() / num_thread);

        // call dynamic third gpu pass
        dynamic_program.kernel("dynamic3Pass")
            .instantiate()
            .configure(dim3(num_block), dim3(num_thread), 0, streamInfo.stream)
            .launch(gpu_key_reduced.data(), gpu_x_reduced.data(), gpu_y_reduced.data(), gpu_z_reduced.data(),
                    gpu_key_reduced.size(), acc_data.data());

        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        block_size = 1024;
        num_thread = (block_size < k_n) ? block_size : k_n;
        num_block = (k_n % num_thread != 0) ? (k_n / num_thread + 1) : (k_n / num_thread);

        // ==============================================================================================================
        // Dynamic Step 4
        // Final integration step to push the simulation one time step forward
        // ==============================================================================================================

        dynamic_program.kernel("dynamic4Pass")
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
                wt_acc.assign(dataManager.m_acc.begin(), dataManager.m_acc.end());
                getParentSystem().wt_buffer_fresh = false;
            }

            getParentSystem().printCSV("sph_folder/test" + std::to_string(writeOutCounter) + ".csv", wt_pos.data(),
                                       wt_pos.size(), wt_vel.data(), wt_acc.data());
            getParentSystem().wt_thread_busy = false;
            writeOutCounter++;
            std::cout << "wo ct:" << writeOutCounter << std::endl;
        } else {
            continue;
        }
    }
}
