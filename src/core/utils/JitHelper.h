//	Copyright (c) 2021, SBEL GPU Development Team
//	Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef DEME_JIT_HELPER_H
#define DEME_JIT_HELPER_H

#include <filesystem>
#include <string>
#include <vector>
#include <unordered_map>
#include <core/utils/GpuError.h>

#include <jitify/jitify.hpp>

#if defined(_WIN32) || defined(_WIN64)
    #undef max
    #undef min
    #undef strtok_r
#endif

// A to-device memcpy wrapper
template <typename T>
void CudaCopyToDevice(T* pD, T* pH) {
    DEME_GPU_CALL(cudaMemcpy(pD, pH, sizeof(T), cudaMemcpyHostToDevice));
}

// A to-host memcpy wrapper
template <typename T>
void CudaCopyToHost(T* pH, T* pD) {
    DEME_GPU_CALL(cudaMemcpy(pH, pD, sizeof(T), cudaMemcpyDeviceToHost));
}

// Used for wrapping data structures so they become usable on GPU
template <typename T>
class DualStruct {
  private:
    T* host_data;           // Pointer to host memory (pinned)
    T* device_data;         // Pointer to device memory
    bool modified_on_host;  // Flag to track if host data has been modified
  public:
    // Constructor: Initialize and allocate memory for both host and device
    DualStruct() : modified_on_host(false) {
        DEME_GPU_CALL(cudaMallocHost((void**)&host_data, sizeof(T)));
        DEME_GPU_CALL(cudaMalloc((void**)&device_data, sizeof(T)));

        syncToDevice();
    }

    // Constructor: Initialize and allocate memory for both host and device with init values
    DualStruct(T init_val) : modified_on_host(false) {
        DEME_GPU_CALL(cudaMallocHost((void**)&host_data, sizeof(T)));
        DEME_GPU_CALL(cudaMalloc((void**)&device_data, sizeof(T)));

        *host_data = init_val;

        syncToDevice();
    }

    // Destructor: Free memory
    ~DualStruct() {
        DEME_GPU_CALL(cudaFreeHost(host_data));  // Free pinned memory
        DEME_GPU_CALL(cudaFree(device_data));    // Free device memory
    }

    // Synchronize changes from host to device
    void syncToDevice() {
        DEME_GPU_CALL(cudaMemcpy(device_data, host_data, sizeof(T), cudaMemcpyHostToDevice));
        modified_on_host = false;
    }

    // Synchronize changes from device to host
    void syncToHost() { DEME_GPU_CALL(cudaMemcpy(host_data, device_data, sizeof(T), cudaMemcpyDeviceToHost)); }

    // Check if host data has been modified and not synced
    bool checkNoPendingModification() { return !modified_on_host; }

    void markModified() { modified_on_host = true; }

    void unmarkModified() { modified_on_host = false; }

    // Accessor for host data (using the arrow operator)
    T* operator->() { return host_data; }

    // Accessor for host data (using the arrow operator)
    T* operator->() const { return host_data; }

    // Dereference operator for simple types (like float) to access the value directly
    T& operator*() {
        return *host_data;  // Return a reference to the value
    }

    // Overloaded operator& for device pointer access
    T* operator&() const {
        return device_data;  // Return device pointer when using &
    }

    // Getter for the device pointer
    T* getDevicePointer() { return device_data; }

    // Getter for the host pointer
    T* getHostPointer() { return host_data; }
};

class JitHelper {
  public:
    class Header {
      public:
        Header(const std::filesystem::path& sourcefile);
        const std::string& getSource();
        void substitute(const std::string& symbol, const std::string& value);

      private:
        std::string _source;
    };

    static jitify::Program buildProgram(
        const std::string& name,
        const std::filesystem::path& source,
        std::unordered_map<std::string, std::string> substitutions = std::unordered_map<std::string, std::string>(),
        std::vector<std::string> flags = std::vector<std::string>());

    //// I'm pretty sure C++17 auto-converts this
    // static jitify::Program buildProgram(
    // 	const std::string& name, const std::string& code,
    // 	std::vector<Header> headers = 0,
    // 	std::vector<std::string> flags = 0
    // );

    static const std::filesystem::path KERNEL_DIR;
    static const std::filesystem::path KERNEL_INCLUDE_DIR;

  private:
    static jitify::JitCache kcache;

    inline static std::string loadSourceFile(const std::filesystem::path& sourcefile) {
        std::string code;
        // If the file exists, read in the entire thing.
        if (std::filesystem::exists(sourcefile)) {
            std::ifstream input(sourcefile);
            std::getline(input, code, std::string::traits_type::to_char_type(std::string::traits_type::eof()));
        }
        return code;
    };
};

#endif
