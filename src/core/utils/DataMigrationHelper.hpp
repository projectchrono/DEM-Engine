//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef DEME_DATA_MIGRATION_HPP
#define DEME_DATA_MIGRATION_HPP

#include <cassert>
#include <core/utils/GpuError.h>

namespace deme {

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

//// TODO: this is currently not tracked...
// ptr being a reference to a pointer is crucial
template <typename T>
inline void DevicePtrAllocAttribBased(T*& ptr, size_t size) {
    cudaPointerAttributes attrib;
    DEME_GPU_CALL(cudaPointerGetAttributes(&attrib, ptr));

    if (attrib.type != cudaMemoryType::cudaMemoryTypeUnregistered)
        DEME_GPU_CALL(cudaFree(ptr));
    DEME_GPU_CALL(cudaMalloc((void**)&ptr, size * sizeof(T)));
}

template <typename T>
inline void DevicePtrDeallocAttribBased(T*& ptr) {
    cudaPointerAttributes attrib;
    DEME_GPU_CALL(cudaPointerGetAttributes(&attrib, ptr));

    if (attrib.type != cudaMemoryType::cudaMemoryTypeUnregistered)
        DEME_GPU_CALL(cudaFree(ptr));
}

// Managed advise doesn't seem to do anything...
#define DEME_ADVISE_DEVICE(vec, device) \
    { advise(vec, ManagedAdvice::PREFERRED_LOC, device); }
#define DEME_MIGRATE_TO_DEVICE(vec, device, stream) \
    { migrate(vec, device, stream); }

// Use (void) to silence unused warnings.
// #define assertm(exp, msg) assert(((void)msg, exp))

// #define OUTPUT_IF_GPU_FAILS(res) \
//     { gpu_assert((res), __FILE__, __LINE__, false); throw std::runtime_error("GPU Assertion Failed!");}

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

// CPU--GPU unified array
template <typename T>
class DualArray {
  public:
    using PinnedVector = std::vector<T, PinnedAllocator<T>>;

    explicit DualArray(bool use_pinned_host = true) : m_use_pinned_host(use_pinned_host) {}

    explicit DualArray(size_t n, bool use_pinned_host = true) : m_use_pinned_host(use_pinned_host) { resize(n); }

    DualArray(const std::vector<T>& vec) : m_use_pinned_host(true) { attachHostVector(&vec, /*deep_copy=*/true); }

    ~DualArray() {
        freeDevice();
        freeHost();
    }

    void resize(size_t n) {
        assert(m_host_vec_ptr == internalHostPtr() && "resize() requires internal host ownership");
        resizeHost(n);
        resizeDevice(n);
    }

    void resizeHost(size_t n) {
        if (!m_host_vec_ptr)
            return;
        m_host_vec_ptr->resize(n);
    }

    // m_device_capacity is allocated memory, not array usable data range
    void resizeDevice(size_t n, bool allow_shrink = false) {
        if (!allow_shrink && m_device_capacity >= n)
            return;
        if (m_device_ptr)
            cudaFree(m_device_ptr);
        DEME_GPU_CALL(cudaMalloc(&m_device_ptr, n * sizeof(T)));
        m_device_capacity = n;
        updateBoundDevicePointer();
    }

    void freeHost() {
        m_std_vec.reset();
        m_pinned_vec.reset();
        m_host_vec_ptr = nullptr;
        m_host_dirty = false;
    }

    void freeDevice() {
        if (m_device_ptr) {
            DEME_GPU_CALL(cudaFree(m_device_ptr));
            m_device_ptr = nullptr;
        }
        m_device_capacity = 0;
        updateBoundDevicePointer();
    }

    void copyToDevice() {
        assert(m_host_vec_ptr);
        size_t count = m_host_vec_ptr->size();
        if (count > m_device_capacity) {
            resizeDevice(count);
        }
        DEME_GPU_CALL(cudaMemcpy(m_device_ptr, m_host_vec_ptr->data(), count * sizeof(T), cudaMemcpyHostToDevice));
        m_host_dirty = false;
    }

    void copyToHost() {
        assert(m_device_ptr && m_host_vec_ptr);
        DEME_GPU_CALL(cudaMemcpy(m_host_vec_ptr->data(), m_device_ptr, size() * sizeof(T), cudaMemcpyDeviceToHost));
        m_host_dirty = false;
    }

    void copyToDeviceAsync(cudaStream_t& stream) {
        assert(m_use_pinned_host && "Async copyToDeviceAsync requires pinned host memory.");
        assert(m_host_vec_ptr && m_device_ptr);
        size_t count = m_host_vec_ptr->size();
        if (count > m_device_capacity) {
            resizeDevice(count);
        }
        cudaMemcpyAsync(m_device_ptr, m_host_vec_ptr->data(), count * sizeof(T), cudaMemcpyHostToDevice, stream);
        m_host_dirty = false;
    }

    void copyToHostAsync(cudaStream_t& stream) {
        assert(m_use_pinned_host && "Async copyToHostAsync requires pinned host memory.");
        assert(m_host_vec_ptr && m_device_ptr);
        size_t count = m_host_vec_ptr->size();
        cudaMemcpyAsync(m_host_vec_ptr->data(), m_device_ptr, count * sizeof(T), cudaMemcpyDeviceToHost, stream);
        m_host_dirty = false;
    }

    void markHostModified() { m_host_dirty = true; }
    void unmarkHostModified() { m_host_dirty = false; }

    // Array's in-use data range is always stored on host by size()
    size_t size() const { return m_host_vec_ptr ? m_host_vec_ptr->size() : 0; }

    T* host() { return m_host_vec_ptr ? m_host_vec_ptr->data() : nullptr; }

    T* device() { return m_device_ptr; }

    std::vector<T>& getHostVector() { return *m_host_vec_ptr; }

    void bindDevicePointer(T** external_ptr_to_ptr) {
        m_bound_device_ptr = external_ptr_to_ptr;
        updateBoundDevicePointer();
    }

    void unbindDevicePointer() { m_bound_device_ptr = nullptr; }

    void attachHostVector(const std::vector<T>* external_vec, bool deep_copy = true) {
        if (deep_copy) {
            m_std_vec = std::make_unique<std::vector<T>>(*external_vec);
            m_host_vec_ptr = m_std_vec.get();
        } else {
            m_std_vec.reset();
            m_host_vec_ptr = const_cast<std::vector<T>*>(external_vec);
        }
        m_host_dirty = true;
    }

    T& operator[](size_t i) { return (*m_host_vec_ptr)[i]; }

    const T& operator[](size_t i) const { return (*m_host_vec_ptr)[i]; }

  private:
    bool m_use_pinned_host = false;

    std::unique_ptr<std::vector<T>> m_std_vec = nullptr;
    std::unique_ptr<PinnedVector> m_pinned_vec = nullptr;
    std::vector<T>* m_host_vec_ptr = nullptr;

    T* m_device_ptr = nullptr;
    size_t m_device_capacity = 0;

    T** m_bound_device_ptr = nullptr;

    bool m_host_dirty = false;

    void updateBoundDevicePointer() {
        if (m_bound_device_ptr) {
            *m_bound_device_ptr = m_device_ptr;
        }
    }

    std::vector<T>* internalHostPtr() {
        return m_use_pinned_host ? static_cast<std::vector<T>*>(m_pinned_vec.get())
                                 : static_cast<std::vector<T>*>(m_std_vec.get());
    }

    // Optional method for cases where syncing to device without resizing (but we don't do that)
    void ensureHostVector(size_t n = 0) {
        if (!m_host_vec_ptr) {
            if (m_use_pinned_host) {
                m_pinned_vec = std::make_unique<PinnedVector>(n);
                m_host_vec_ptr = m_pinned_vec.get();
            } else {
                m_std_vec = std::make_unique<std::vector<T>>(n);
                m_host_vec_ptr = m_std_vec.get();
            }
        }
    }
};

}  // namespace deme

#endif
