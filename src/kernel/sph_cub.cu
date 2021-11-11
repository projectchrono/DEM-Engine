// NOTE: This is a backup kernel file awaiting cub to be added to the cmake structure

#define CUB_STDERR  // print CUDA runtime errors to console
#include "cub/cub.cuh"
#include "cub/device/device_reduce.cuh"
#include "cub/util_allocator.cuh"
#include "cub/util_debug.cuh"
#include "stdio.h"
using namespace cub;
CachingDeviceAllocator g_allocator(true);  // Caching allocator for device memory

__global__ void test() {
    int h_in[7] = {5, 6, 1, 4, 2, 7, 1};
    int h_key[7] = {1, 2, 3, 1, 2, 5, 4};
    int* h_kout = new int[7];
    int* h_vout = new int[7];

    // Set up device arrays
    int* d_in;
    g_allocator.DeviceAllocate((void**)&d_in, sizeof(int) * 7);
    cudaMemcpy(d_in, h_in, sizeof(int) * 7, cudaMemcpyHostToDevice);

    // Set up device keys
    int* d_key;
    g_allocator.DeviceAllocate((void**)&d_key, sizeof(int) * 7);
    cudaMemcpy(d_key, h_key, sizeof(int) * 7, cudaMemcpyHostToDevice);

    // Set up device arrays
    int* d_in_sorted;
    g_allocator.DeviceAllocate((void**)&d_in_sorted, sizeof(int) * 7);

    // Set up device keys
    int* d_key_sorted;
    g_allocator.DeviceAllocate((void**)&d_key_sorted, sizeof(int) * 7);

    // set up res key arrays
    int* d_kout;
    g_allocator.DeviceAllocate((void**)&d_kout, sizeof(int) * 7);

    // set up res value arrays
    int* d_vout;
    g_allocator.DeviceAllocate((void**)&d_vout, sizeof(int) * 7);

    int* num_unique;
    int h_unique;
    g_allocator.DeviceAllocate((void**)&num_unique, sizeof(int) * 1);

    void* d_temp_storage_1 = NULL;
    size_t temp_storage_bytes_1 = 0;

    CustomSum sum_op;

    CubDebugExit(
        DeviceRadixSort::SortPairs(d_temp_storage_1, temp_storage_bytes_1, d_key, d_key_sorted, d_in, d_in_sorted, 7));

    g_allocator.DeviceAllocate(&d_temp_storage_1, temp_storage_bytes_1);

    CubDebugExit(
        DeviceRadixSort::SortPairs(d_temp_storage_1, temp_storage_bytes_1, d_key, d_key_sorted, d_in, d_in_sorted, 7));

    void* d_temp_storage_2 = NULL;
    size_t temp_storage_bytes_2 = 0;

    CubDebugExit(DeviceReduce::ReduceByKey(d_temp_storage_2, temp_storage_bytes_2, d_key_sorted, d_kout, d_in_sorted,
                                           d_vout, num_unique, cub::Sum(), 7));

    g_allocator.DeviceAllocate(&d_temp_storage_2, temp_storage_bytes_2);

    CubDebugExit(DeviceReduce::ReduceByKey(d_temp_storage_2, temp_storage_bytes_2, d_key_sorted, d_kout, d_in_sorted,
                                           d_vout, num_unique, cub::Sum(), 7));

    // DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, num_unique, 7);
    // g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes);
    // DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, num_unique, 7);

    cudaMemcpy(h_kout, d_kout, sizeof(int) * 7, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vout, d_vout, sizeof(int) * 7, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_unique, num_unique, sizeof(int), cudaMemcpyDeviceToHost);

    // display result
    std::cout << "num_unique: " << h_unique << std::endl;

    std::cout << "h_kout: " << std::endl;
    for (int i = 0; i < h_unique; i++) {
        std::cout << h_kout[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "h_vout: " << std::endl;
    for (int i = 0; i < h_unique; i++) {
        std::cout << h_vout[i] << " ";
    }
    std::cout << std::endl;

    // Cleanup
    if (d_in)
        CubDebugExit(g_allocator.DeviceFree(d_in));
    if (d_key)
        CubDebugExit(g_allocator.DeviceFree(d_key));
    if (d_kout)
        CubDebugExit(g_allocator.DeviceFree(d_kout));
    if (d_vout)
        CubDebugExit(g_allocator.DeviceFree(d_vout));
    if (num_unique)
        CubDebugExit(g_allocator.DeviceFree(num_unique));
    if (d_temp_storage_1)
        CubDebugExit(g_allocator.DeviceFree(d_temp_storage_1));
    if (d_temp_storage_2)
        CubDebugExit(g_allocator.DeviceFree(d_temp_storage_2));

    return 0;
}