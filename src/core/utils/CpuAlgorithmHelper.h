//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#include <vector>
#include <algorithm>
#include <sph/datastruct.h>

#pragma once
#ifndef SGPS_CPU_ALGORITHM
    #define SGPS_CPU_ALGORITHM

// This function is designed to simulate the behavior of cub::DeviceRunLengthEncode
// This function computes the total length of each BSD idx and output array of unique BSD idx.
void deviceRunLength(int* idx_track_data_sorted,
                     int n,
                     std::vector<int>& unique_BSD_idx,
                     std::vector<int>& length_BSD_idx);

void sortReduce(int* key, float* val, std::vector<int>& key_reduced, std::vector<float>& val_reduced, int n, int max);

void sortReduce(int* key, int* val, std::vector<int>& key_reduced, std::vector<int>& val_reduced, int n, int max);

void sortOnly(int* key, int* val, std::vector<int>& key_sorted, std::vector<int>& val_sorted, int n, int max);

int count_digit(int number);

// a test algorithm to slice obtain a std::vector indicating which cell is in that sd
std::vector<int> slice_global_sd(int X_SUB_NUM, int Y_SUB_NUM, int Z_SUB_NUM);

// output the collision data for debugging purposes
void output_collision_data(contactData* contact_data, int contact_size, std::string filename);

// CPU Exclusive Prefix Scan
void PrefixScanExclusive(int* arr, int n, int* prefixSum);

// helper function to look for cell idx from subdomain idx
// std::vector<int>
// SPH_Find_Sub2Cell(int num_c_x, int num_c_y, int num_c_z, int num_s_x, int num_s_y, int num_s_z, int sub_i);

// helper function to look for subdomain idx from cell idx
// int SPH_Find_Cell2Sub(int num_c_x, int num_c_y, int num_c_z, int num_s_x, int num_s_y, int num_s_z, int cell_i);

#endif