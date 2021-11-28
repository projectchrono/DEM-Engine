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
