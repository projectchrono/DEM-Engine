#include <algorithms/SPHCubHelperFunctions.h>
#include <cub/cub.cuh>
#include <cub/util_allocator.cuh>
#include <cub/util_debug.cuh>

// CustomMin functor
struct CustomCubFloat3Add {
    CUB_RUNTIME_FUNCTION __forceinline__ __device__ __host__ float3 operator()(const float3& a, const float3& b) const {
        return ::make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
    }
};

void PrefixScanExclusiveCub(std::vector<int, sgps::ManagedAllocator<int>>& d_in,
                            std::vector<int, sgps::ManagedAllocator<int>>& d_out,
                            int work_size,
                            std::vector<int, sgps::ManagedAllocator<int>>& temp_storage) {
    d_out.resize(work_size);
    // Determine temporary device storage requirements
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(NULL, temp_storage_bytes, d_in.data(), d_out.data(), work_size);
    // Resize temporary storage
    size_t temp_arr_size = temp_storage_bytes / sizeof(int);
    if (temp_storage.size() < temp_arr_size) {
        temp_storage.resize(temp_arr_size);
    }
    // Run exclusive prefix sum
    cub::DeviceScan::ExclusiveSum(temp_storage.data(), temp_storage_bytes, d_in.data(), d_out.data(), work_size);

    cudaDeviceSynchronize();
}

void PairRadixSortAscendCub(std::vector<int, sgps::ManagedAllocator<int>>& d_keys_in,
                            std::vector<int, sgps::ManagedAllocator<int>>& d_keys_out,
                            std::vector<int, sgps::ManagedAllocator<int>>& d_values_in,
                            std::vector<int, sgps::ManagedAllocator<int>>& d_values_out,
                            int work_size,
                            std::vector<int, sgps::ManagedAllocator<int>>& temp_storage) {
    d_keys_out.resize(work_size);
    d_values_out.resize(work_size);

    // Determine temporary device storage requirements
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(NULL, temp_storage_bytes, d_keys_in.data(), d_keys_out.data(), d_values_in.data(),
                                    d_values_out.data(), work_size);
    // Resize temporary storage
    size_t temp_arr_size = temp_storage_bytes / sizeof(int);
    if (temp_storage.size() < temp_arr_size) {
        temp_storage.resize(temp_arr_size);
    }
    // Run sorting operation
    cub::DeviceRadixSort::SortPairs(temp_storage.data(), temp_storage_bytes, d_keys_in.data(), d_keys_out.data(),
                                    d_values_in.data(), d_values_out.data(), work_size);

    cudaDeviceSynchronize();
}

void PairRadixSortAscendCub(std::vector<int, sgps::ManagedAllocator<int>>& d_keys_in,
                            std::vector<int, sgps::ManagedAllocator<int>>& d_keys_out,
                            std::vector<float, sgps::ManagedAllocator<float>>& d_values_in,
                            std::vector<float, sgps::ManagedAllocator<float>>& d_values_out,
                            int work_size,
                            std::vector<float, sgps::ManagedAllocator<float>>& temp_storage) {
    d_keys_out.resize(work_size);
    d_values_out.resize(work_size);

    // Determine temporary device storage requirements
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(NULL, temp_storage_bytes, d_keys_in.data(), d_keys_out.data(), d_values_in.data(),
                                    d_values_out.data(), work_size);
    // Resize temporary storage
    size_t temp_arr_size = temp_storage_bytes / sizeof(float);
    if (temp_storage.size() < temp_arr_size) {
        temp_storage.resize(temp_arr_size);
    }

    // Run sorting operation
    cub::DeviceRadixSort::SortPairs(temp_storage.data(), temp_storage_bytes, d_keys_in.data(), d_keys_out.data(),
                                    d_values_in.data(), d_values_out.data(), work_size);

    cudaDeviceSynchronize();
}

void PairRadixSortAscendCub(std::vector<int, sgps::ManagedAllocator<int>>& d_keys_in,
                            std::vector<int, sgps::ManagedAllocator<int>>& d_keys_out,
                            std::vector<float3, sgps::ManagedAllocator<float3>>& d_values_in,
                            std::vector<float3, sgps::ManagedAllocator<float3>>& d_values_out,
                            int work_size,
                            std::vector<float3, sgps::ManagedAllocator<float3>>& temp_storage) {
    d_keys_out.resize(work_size);
    d_values_out.resize(work_size);

    // Determine temporary device storage requirements
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(NULL, temp_storage_bytes, d_keys_in.data(), d_keys_out.data(), d_values_in.data(),
                                    d_values_out.data(), work_size);
    // Resize temporary storage
    size_t temp_arr_size = temp_storage_bytes / sizeof(float3);
    if (temp_storage.size() < temp_arr_size) {
        temp_storage.resize(temp_arr_size);
    }

    // Run sorting operation
    cub::DeviceRadixSort::SortPairs(temp_storage.data(), temp_storage_bytes, d_keys_in.data(), d_keys_out.data(),
                                    d_values_in.data(), d_values_out.data(), work_size);

    cudaDeviceSynchronize();
}

void RunLengthEncodeCub(std::vector<int, sgps::ManagedAllocator<int>>& d_in,
                        std::vector<int, sgps::ManagedAllocator<int>>& d_unique_out,
                        std::vector<int, sgps::ManagedAllocator<int>>& d_counts_out,
                        int work_size,
                        std::vector<int, sgps::ManagedAllocator<int>>& temp_storage) {
    std::vector<int, sgps::ManagedAllocator<int>> d_num_runs_out;
    d_unique_out.resize(work_size);
    d_counts_out.resize(work_size);
    d_num_runs_out.resize(1);

    // Determine temporary device storage requirements
    size_t temp_storage_bytes = 0;
    cub::DeviceRunLengthEncode::Encode(NULL, temp_storage_bytes, d_in.data(), d_unique_out.data(), d_counts_out.data(),
                                       d_num_runs_out.data(), work_size);
    // Resize temporary storage
    size_t temp_arr_size = temp_storage_bytes / sizeof(int);
    if (temp_storage.size() < temp_arr_size) {
        temp_storage.resize(temp_arr_size);
    }

    // Run encoding
    cub::DeviceRunLengthEncode::Encode(temp_storage.data(), temp_storage_bytes, d_in.data(), d_unique_out.data(),
                                       d_counts_out.data(), d_num_runs_out.data(), work_size);

    cudaDeviceSynchronize();

    d_unique_out.erase(d_unique_out.begin() + d_num_runs_out[0], d_unique_out.end());
    d_counts_out.erase(d_counts_out.begin() + d_num_runs_out[0], d_counts_out.end());

    cudaDeviceSynchronize();
}

void SumReduceByKeyCub(std::vector<int, sgps::ManagedAllocator<int>>& d_keys_in,
                       std::vector<int, sgps::ManagedAllocator<int>>& d_unique_out,
                       std::vector<float, sgps::ManagedAllocator<float>>& d_values_in,
                       std::vector<float, sgps::ManagedAllocator<float>>& d_aggregates_out,
                       int work_size,
                       std::vector<float, sgps::ManagedAllocator<float>>& temp_storage) {
    d_unique_out.resize(work_size);
    d_aggregates_out.resize(work_size);

    std::vector<int, sgps::ManagedAllocator<int>> d_num_runs_out;
    d_num_runs_out.resize(1);

    // Determine temporary device storage requirements
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::ReduceByKey(NULL, temp_storage_bytes, d_keys_in.data(), d_unique_out.data(), d_values_in.data(),
                                   d_aggregates_out.data(), d_num_runs_out.data(), cub::Sum(), work_size);
    cudaDeviceSynchronize();
    // Resize temporary storage
    size_t temp_arr_size = temp_storage_bytes / sizeof(float);
    if (temp_storage.size() < temp_arr_size) {
        temp_storage.resize(temp_arr_size);
    }

    cub::DeviceReduce::ReduceByKey(temp_storage.data(), temp_storage_bytes, d_keys_in.data(), d_unique_out.data(),
                                   d_values_in.data(), d_aggregates_out.data(), d_num_runs_out.data(), cub::Sum(),
                                   work_size);
    cudaDeviceSynchronize();

    d_unique_out.erase(d_unique_out.begin() + d_num_runs_out[0], d_unique_out.end());
    d_aggregates_out.erase(d_aggregates_out.begin() + d_num_runs_out[0], d_aggregates_out.end());
    cudaDeviceSynchronize();
}

void SumReduceByKeyCub(std::vector<int, sgps::ManagedAllocator<int>>& d_keys_in,
                       std::vector<int, sgps::ManagedAllocator<int>>& d_unique_out,
                       std::vector<float3, sgps::ManagedAllocator<float3>>& d_values_in,
                       std::vector<float3, sgps::ManagedAllocator<float3>>& d_aggregates_out,
                       int work_size,
                       std::vector<float3, sgps::ManagedAllocator<float3>>& temp_storage) {
    d_unique_out.resize(work_size);
    d_aggregates_out.resize(work_size);

    std::vector<int, sgps::ManagedAllocator<int>> d_num_runs_out;
    d_num_runs_out.resize(1);

    // Determine temporary device storage requirements
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::ReduceByKey(NULL, temp_storage_bytes, d_keys_in.data(), d_unique_out.data(), d_values_in.data(),
                                   d_aggregates_out.data(), d_num_runs_out.data(), CustomCubFloat3Add(), work_size);
    cudaDeviceSynchronize();
    // Resize temporary storage
    size_t temp_arr_size = temp_storage_bytes / sizeof(float3);
    if (temp_storage.size() < temp_arr_size) {
        temp_storage.resize(temp_arr_size);
    }

    cub::DeviceReduce::ReduceByKey(temp_storage.data(), temp_storage_bytes, d_keys_in.data(), d_unique_out.data(),
                                   d_values_in.data(), d_aggregates_out.data(), d_num_runs_out.data(),
                                   CustomCubFloat3Add(), work_size);
    cudaDeviceSynchronize();

    d_unique_out.erase(d_unique_out.begin() + d_num_runs_out[0], d_unique_out.end());
    d_aggregates_out.erase(d_aggregates_out.begin() + d_num_runs_out[0], d_aggregates_out.end());
    cudaDeviceSynchronize();
}