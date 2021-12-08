//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#include <iostream>
#include <list>
#include <cmath>
#include <vector>
#include <algorithm>
#include "CpuAlgorithmHelper.h"
using namespace std;
void display(int* array, int size) {
    for (int i = 0; i < size; i++)
        cout << array[i] << " ";
    cout << endl;
}
void display(float* array, int size) {
    for (int i = 0; i < size; i++)
        cout << array[i] << " ";
    cout << endl;
}

void radixSort(int* key, float* val, int n, int max) {
    int i, j, m, p = 1, index, temp, count = 0;
    list<int> pocket[10];  // radix of decimal number is 10
    list<float> pocket_val[10];
    for (i = 0; i < max; i++) {
        m = pow(10, i + 1);
        p = pow(10, i);
        for (j = 0; j < n; j++) {
            temp = key[j] % m;
            index = temp / p;  // find index for pocket array
            pocket[index].push_back(key[j]);
            pocket_val[index].push_back(val[j]);
        }
        count = 0;
        for (j = 0; j < 10; j++) {
            // delete from linked lists and store to array
            while (!pocket[j].empty()) {
                key[count] = *(pocket[j].begin());
                pocket[j].erase(pocket[j].begin());
                val[count] = *(pocket_val[j].begin());
                pocket_val[j].erase(pocket_val[j].begin());
                count++;
            }
        }
    }
}

void radixSort(int* key, int* val, int n, int max) {
    int i, j, m, p = 1, index, temp, count = 0;
    list<int> pocket[10];  // radix of decimal number is 10
    list<int> pocket_val[10];
    for (i = 0; i < max; i++) {
        m = pow(10, i + 1);
        p = pow(10, i);
        for (j = 0; j < n; j++) {
            temp = key[j] % m;
            index = temp / p;  // find index for pocket array
            pocket[index].push_back(key[j]);
            pocket_val[index].push_back(val[j]);
        }
        count = 0;
        for (j = 0; j < 10; j++) {
            // delete from linked lists and store to array
            while (!pocket[j].empty()) {
                key[count] = *(pocket[j].begin());
                pocket[j].erase(pocket[j].begin());
                val[count] = *(pocket_val[j].begin());
                pocket_val[j].erase(pocket_val[j].begin());
                count++;
            }
        }
    }
}

void reduceByKey(int* key, float* val, int n, std::vector<int>& key_reduced, std::vector<float>& val_reduced) {
    float temp_sum;
    int prev_key = -1;
    for (int i = 0; i < n; i++) {
        if (i == 0) {
            key_reduced.push_back(key[0]);
            temp_sum = val[i];
        } else {
            if (prev_key == key[i]) {
                temp_sum += val[i];
            } else {
                key_reduced.push_back(key[i]);
                val_reduced.push_back(temp_sum);
                temp_sum = val[i];
            }
        }

        if (i == n - 1) {
            val_reduced.push_back(temp_sum);
        }

        prev_key = key[i];
    }
}

void reduceByKey(int* key, int* val, int n, std::vector<int>& key_reduced, std::vector<int>& val_reduced) {
    int temp_sum;
    int prev_key = -1;
    for (int i = 0; i < n; i++) {
        if (i == 0) {
            key_reduced.push_back(key[0]);
            temp_sum = val[i];
        } else {
            if (prev_key == key[i]) {
                temp_sum += val[i];
            } else {
                key_reduced.push_back(key[i]);
                val_reduced.push_back(temp_sum);
                temp_sum = val[i];
            }
        }

        if (i == n - 1) {
            val_reduced.push_back(temp_sum);
        }

        prev_key = key[i];
    }
}

void sortOnly(int* key, int* val, std::vector<int>& key_sorted, std::vector<int>& val_sorted, int n, int max) {
    std::vector<int> key_shallow;
    std::vector<int> val_shallow;
    for (int i = 0; i < n; i++) {
        key_shallow.push_back(key[i]);
        val_shallow.push_back(val[i]);
    }

    radixSort(key_shallow.data(), val_shallow.data(), n, max);
    key_sorted.assign(key_shallow.begin(), key_shallow.end());
    val_sorted.assign(val_shallow.begin(), val_shallow.end());
}

void sortReduce(int* key, float* val, std::vector<int>& key_reduced, std::vector<float>& val_reduced, int n, int max) {
    std::vector<int> key_shallow;
    std::vector<float> val_shallow;
    for (int i = 0; i < n; i++) {
        key_shallow.push_back(key[i]);
        val_shallow.push_back(val[i]);
    }
    radixSort(key_shallow.data(), val_shallow.data(), n, max);
    reduceByKey(key_shallow.data(), val_shallow.data(), n, key_reduced, val_reduced);
}

void sortReduce(int* key, int* val, std::vector<int>& key_reduced, std::vector<int>& val_reduced, int n, int max) {
    std::vector<int> key_shallow;
    std::vector<int> val_shallow;
    for (int i = 0; i < n; i++) {
        key_shallow.push_back(key[i]);
        val_shallow.push_back(val[i]);
    }
    radixSort(key_shallow.data(), val_shallow.data(), n, max);
    reduceByKey(key_shallow.data(), val_shallow.data(), n, key_reduced, val_reduced);
}

int count_digit(int number) {
    int count = 0;
    while (number != 0) {
        number = number / 10;
        count++;
    }
    return count;
}

std::vector<int> slice_global_sd(int num_cd_each_side){
    int num_sd = 3;
    int num_cd_each_sd_side = 4;
    int num_cd_each_sd = num_cd_each_sd_side * num_cd_each_sd_side;
    std::vector<int> res;

    for(int k = 0; k < num_sd; k ++)
    {
        for(int j = 0; j < num_sd; j++)
        {
            for(int i = 0; i < num_sd; i++)
            {
                int start_x = i * (num_cd_each_sd_side-1);   // starting cd idx of the current sd on x
                int start_y = j * (num_cd_each_sd_side-1);   // starting cd idx of the current sd on y  
                int start_z = k * (num_cd_each_sd_side-1);   // starting cd idx of the current sd on z
                for (int a = 0; a < num_cd_each_sd_side; a++)
                {
                    for (int b = 0; b < num_cd_each_sd_side; b++)
                    {
                        for (int c = 0; c < num_cd_each_sd_side; c++)
                        {
                            res.push_back((start_z + a)* num_cd_each_side*num_cd_each_side + (start_y + b)*num_cd_each_side + (start_x+c));
                        }
                    }
                }
            }
        }
    }
    return res;
}

/*
class SPH_Find_Cell2Sub_Exception : public exception {
    virtual const char* what() const throw() { return "SPH_Find_Cell2Sub_Exception happened"; }
} Cell2SubException;

// helper function to look for subdomain idx from cell idx
int SPH_Find_Cell2Sub(int num_c_x, int num_c_y, int num_c_z, int c_x, int c_y, int c_z, int cell_i) {
    if (cell_i >= num_c_x * num_c_y * num_c_z) {
        throw Cell2SubException;
    }

    int idx_c_z = cell_i / (num_c_y * num_c_x);
    int idx_c_y = int((cell_i % (num_c_y * num_c_x)) / num_c_x);
    int idx_c_x = int(cell_i - (idx_c_z * num_c_x * num_c_y) - idx_c_y * num_c_x);

    int idx_s_x = idx_c_x / c_x;
    int idx_s_y = idx_c_y / c_y;
    int idx_s_z = idx_c_z / c_z;

    std::cout << "idx_s_x: " << idx_s_x << std::endl;
    std::cout << "idx_s_y: " << idx_s_y << std::endl;
    std::cout << "idx_s_z: " << idx_s_z << std::endl;
}*/
/*
int main() {
    int n=7;

    int key[n] = {7, 5, 3, 7, 5, 32, 18};
    float val[n] = {0.2, 1.1, 1.7, 3.6, 2.5, 6.1, 7.1};

    int max = *max_element(key , key + n);

    std::vector<int> key_reduced;
    std::vector<float> val_reduced;

    std::cout<<"before: ";
    display(key, n);
    display(val, n);
    sortReduce(key,val,key_reduced, val_reduced,n,max);
    std::cout<<"after: "<<std::endl;
    display(key_reduced.data(), key_reduced.size());
    display(val_reduced.data(), val_reduced.size());
}
*/
