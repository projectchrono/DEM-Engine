//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.
#pragma once
#ifndef SGPS_CPU_ALGORITHM
    #define SGPS_CPU_ALGORITHM

void sortReduce(int* key, float* val, std::vector<int>& key_reduced, std::vector<float>& val_reduced, int n, int max);

#endif