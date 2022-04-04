#include <algorithms/SPHCubHelperFunctions.h>
#include <cub/cub.cuh>
#include <cub/util_allocator.cuh>
#include <cub/util_debug.cuh>

// CustomMin functor
struct CustomSum {
    template <typename T>
    CUB_RUNTIME_FUNCTION __forceinline__ T operator()(const T& a, const T& b) const {
        return a + b;
    }
};

void PrefixScanExclusiveCub(std::vector<int, sgps::ManagedAllocator<int>>& d_in,
                            std::vector<int, sgps::ManagedAllocator<int>>& d_out) {
    d_out.resize(d_in.size());
    // Determine temporary device storage requirements
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in.data(), d_out.data(), d_in.size());
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run exclusive prefix sum
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in.data(), d_out.data(), d_in.size());

    cudaDeviceSynchronize();
}

void PairRadixSortAscendCub(std::vector<int, sgps::ManagedAllocator<int>>& d_keys_in,
                            std::vector<int, sgps::ManagedAllocator<int>>& d_keys_out,
                            std::vector<int, sgps::ManagedAllocator<int>>& d_values_in,
                            std::vector<int, sgps::ManagedAllocator<int>>& d_values_out) {
    d_keys_out.resize(d_keys_in.size());
    d_values_out.resize(d_values_in.size());

    // Determine temporary device storage requirements
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in.data(), d_keys_out.data(),
                                    d_values_in.data(), d_values_out.data(), d_keys_in.size());
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run sorting operation
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in.data(), d_keys_out.data(),
                                    d_values_in.data(), d_values_out.data(), d_keys_in.size());

    cudaDeviceSynchronize();
}

void PairRadixSortAscendCub(std::vector<int, sgps::ManagedAllocator<int>>& d_keys_in,
                            std::vector<int, sgps::ManagedAllocator<int>>& d_keys_out,
                            std::vector<float, sgps::ManagedAllocator<float>>& d_values_in,
                            std::vector<float, sgps::ManagedAllocator<float>>& d_values_out) {
    d_keys_out.resize(d_keys_in.size());
    d_values_out.resize(d_values_in.size());

    // Determine temporary device storage requirements
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in.data(), d_keys_out.data(),
                                    d_values_in.data(), d_values_out.data(), d_keys_in.size());
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run sorting operation
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in.data(), d_keys_out.data(),
                                    d_values_in.data(), d_values_out.data(), d_keys_in.size());

    cudaDeviceSynchronize();
}

void RunLengthEncodeCub(std::vector<int, sgps::ManagedAllocator<int>>& d_in,
                        std::vector<int, sgps::ManagedAllocator<int>>& d_unique_out,
                        std::vector<int, sgps::ManagedAllocator<int>>& d_counts_out) {
    std::vector<int, sgps::ManagedAllocator<int>> d_num_runs_out;
    d_unique_out.resize(d_in.size());
    d_counts_out.resize(d_in.size());
    d_num_runs_out.resize(1);

    // Determine temporary device storage requirements
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceRunLengthEncode::Encode(d_temp_storage, temp_storage_bytes, d_in.data(), d_unique_out.data(),
                                       d_counts_out.data(), d_num_runs_out.data(), d_in.size());
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run encoding
    cub::DeviceRunLengthEncode::Encode(d_temp_storage, temp_storage_bytes, d_in.data(), d_unique_out.data(),
                                       d_counts_out.data(), d_num_runs_out.data(), d_in.size());

    cudaDeviceSynchronize();

    d_unique_out.erase(d_unique_out.begin() + d_num_runs_out[0], d_unique_out.end());
    d_counts_out.erase(d_counts_out.begin() + d_num_runs_out[0], d_counts_out.end());

    cudaDeviceSynchronize();
}

void SumReduceByKeyCub(std::vector<int, sgps::ManagedAllocator<int>>& d_keys_in,
                       std::vector<int, sgps::ManagedAllocator<int>>& d_unique_out,
                       std::vector<float, sgps::ManagedAllocator<float>>& d_values_in,
                       std::vector<float, sgps::ManagedAllocator<float>>& d_aggregates_out) {
    d_unique_out.resize(d_keys_in.size());
    d_aggregates_out.resize(d_values_in.size());

    std::vector<int, sgps::ManagedAllocator<int>> d_num_runs_out;
    d_num_runs_out.resize(1);

    // Determine temporary device storage requirements
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes, d_keys_in.data(), d_unique_out.data(),
                                   d_values_in.data(), d_aggregates_out.data(), d_num_runs_out.data(), cub::Sum(),
                                   d_keys_in.size());
    cudaDeviceSynchronize();

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes, d_keys_in.data(), d_unique_out.data(),
                                   d_values_in.data(), d_aggregates_out.data(), d_num_runs_out.data(), cub::Sum(),
                                   d_keys_in.size());
    cudaDeviceSynchronize();

    d_unique_out.erase(d_unique_out.begin() + d_num_runs_out[0], d_unique_out.end());
    d_aggregates_out.erase(d_aggregates_out.begin() + d_num_runs_out[0], d_aggregates_out.end());
    cudaDeviceSynchronize();
}