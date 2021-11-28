//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.
#pragma once
#ifndef SGPS_CPU_ALGORITHM
    #define SGPS_CPU_ALGORITHM

void sortReduce(int* key, float* val, std::vector<int>& key_reduced, std::vector<float>& val_reduced, int n, int max);

void sortReduce(int* key, int* val, std::vector<int>& key_reduced, std::vector<int>& val_reduced, int n, int max);

void sortOnly(int* key, int* val, std::vector<int>& key_sorted, std::vector<int>& val_sorted, int n, int max);

int count_digit(int number);

#endif