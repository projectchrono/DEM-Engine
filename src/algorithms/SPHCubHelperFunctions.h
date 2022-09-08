#ifndef SMUG_SPH_CUB_HELPERS
#define SMUG_SPH_CUB_HELPERS

#include <core/utils/ManagedAllocator.hpp>
#include <vector>

void PrefixScanExclusiveCub(std::vector<int, smug::ManagedAllocator<int>>& d_in,
                            std::vector<int, smug::ManagedAllocator<int>>& d_out,
                            int work_size,
                            std::vector<int, smug::ManagedAllocator<int>>& temp_storage);

void PairRadixSortAscendCub(std::vector<int, smug::ManagedAllocator<int>>& d_keys_in,
                            std::vector<int, smug::ManagedAllocator<int>>& d_keys_out,
                            std::vector<int, smug::ManagedAllocator<int>>& d_values_in,
                            std::vector<int, smug::ManagedAllocator<int>>& d_values_out,
                            int work_size,
                            std::vector<int, smug::ManagedAllocator<int>>& temp_storage);

void PairRadixSortAscendCub(std::vector<int, smug::ManagedAllocator<int>>& d_keys_in,
                            std::vector<int, smug::ManagedAllocator<int>>& d_keys_out,
                            std::vector<float, smug::ManagedAllocator<float>>& d_values_in,
                            std::vector<float, smug::ManagedAllocator<float>>& d_values_out,
                            int work_size,
                            std::vector<float, smug::ManagedAllocator<float>>& temp_storage);

void PairRadixSortAscendCub(std::vector<int, smug::ManagedAllocator<int>>& d_keys_in,
                            std::vector<int, smug::ManagedAllocator<int>>& d_keys_out,
                            std::vector<float3, smug::ManagedAllocator<float3>>& d_values_in,
                            std::vector<float3, smug::ManagedAllocator<float3>>& d_values_out,
                            int work_size,
                            std::vector<float3, smug::ManagedAllocator<float3>>& temp_storage);

void RunLengthEncodeCub(std::vector<int, smug::ManagedAllocator<int>>& d_in,
                        std::vector<int, smug::ManagedAllocator<int>>& d_unique_out,
                        std::vector<int, smug::ManagedAllocator<int>>& d_counts_out,
                        int work_size,
                        std::vector<int, smug::ManagedAllocator<int>>& temp_storage);

void SumReduceByKeyCub(std::vector<int, smug::ManagedAllocator<int>>& d_keys_in,
                       std::vector<int, smug::ManagedAllocator<int>>& d_uniques_out,
                       std::vector<float, smug::ManagedAllocator<float>>& d_values_in,
                       std::vector<float, smug::ManagedAllocator<float>>& d_aggregates_out,
                       int work_size,
                       std::vector<float, smug::ManagedAllocator<float>>& temp_storage);

void SumReduceByKeyCub(std::vector<int, smug::ManagedAllocator<int>>& d_keys_in,
                       std::vector<int, smug::ManagedAllocator<int>>& d_uniques_out,
                       std::vector<float3, smug::ManagedAllocator<float3>>& d_values_in,
                       std::vector<float3, smug::ManagedAllocator<float3>>& d_aggregates_out,
                       int work_size,
                       std::vector<float3, smug::ManagedAllocator<float3>>& temp_storage);

#endif