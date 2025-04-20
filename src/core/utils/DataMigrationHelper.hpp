//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef DEME_DATA_MIGRATION_HPP
#define DEME_DATA_MIGRATION_HPP

#include <cassert>
#include <optional>
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

// CPU--GPU unified array, leveraging pinned memory
template <typename T>
class DualArray {
  public:
    using PinnedVector = std::vector<T, PinnedAllocator<T>>;

    explicit DualArray(size_t* host_external_counter = nullptr, size_t* device_external_counter = nullptr)
        : m_host_mem_counter(host_external_counter), m_device_mem_counter(device_external_counter) {}

    DualArray(size_t n, size_t* host_external_counter = nullptr, size_t* device_external_counter = nullptr)
        : m_host_mem_counter(host_external_counter), m_device_mem_counter(device_external_counter) {
        resize(n);
    }

    DualArray(const std::vector<T>& vec,
              size_t* host_external_counter = nullptr,
              size_t* device_external_counter = nullptr)
        : m_host_mem_counter(host_external_counter), m_device_mem_counter(device_external_counter) {
        attachHostVector(&vec, /*deep_copy=*/true);
    }

    ~DualArray() {
        freeDevice();
        freeHost();
    }

    void resize(size_t n) {
        assert(m_host_vec_ptr == m_pinned_vec.get() && "resize() requires internal host ownership");
        resizeHost(n);
        resizeDevice(n);
    }

    void resize(size_t n, const T& val) {
        assert(m_host_vec_ptr == m_pinned_vec.get() && "resize() requires internal host ownership");
        resizeHost(n, val);
        resizeDevice(n);
    }

    void resizeHost(size_t n) {
        ensureHostVector();  // allocates pinned vec if null
        size_t old_bytes = m_host_vec_ptr->size() * sizeof(T);
        m_host_vec_ptr->resize(n);
        size_t new_bytes = m_host_vec_ptr->size() * sizeof(T);
        updateHostMemCounter(static_cast<ssize_t>(new_bytes) - static_cast<ssize_t>(old_bytes));
    }

    void resizeHost(size_t n, const T& val) {
        ensureHostVector();  // allocates pinned vec if null
        size_t old_bytes = m_host_vec_ptr->size() * sizeof(T);
        m_host_vec_ptr->resize(n, val);
        size_t new_bytes = m_host_vec_ptr->size() * sizeof(T);
        updateHostMemCounter(static_cast<ssize_t>(new_bytes) - static_cast<ssize_t>(old_bytes));
    }

    // m_device_capacity is allocated memory, not array usable data range
    void resizeDevice(size_t n, bool allow_shrink = false) {
        size_t old_bytes = m_device_capacity * sizeof(T);
        if (!allow_shrink && m_device_capacity >= n)
            return;
        if (m_device_ptr)
            DEME_GPU_CALL(cudaFree(m_device_ptr));
        DEME_GPU_CALL(cudaMalloc((void**)&m_device_ptr, n * sizeof(T)));
        m_device_capacity = n;
        updateBoundDevicePointer();
        updateDeviceMemCounter(static_cast<ssize_t>(n * sizeof(T)) - static_cast<ssize_t>(old_bytes));
    }

    void freeHost() {
        if (m_host_vec_ptr) {
            updateHostMemCounter(-(ssize_t)(m_host_vec_ptr->size() * sizeof(T)));
        }
        m_pinned_vec.reset();
        m_host_vec_ptr = nullptr;
        m_host_dirty = false;
    }

    void freeDevice() {
        if (m_device_ptr) {
            updateDeviceMemCounter(-(ssize_t)(m_device_capacity * sizeof(T)));
            DEME_GPU_CALL(cudaFree(m_device_ptr));
            m_device_ptr = nullptr;
        }
        m_device_capacity = 0;
        updateBoundDevicePointer();
    }

    void copyToDevice() {
        assert(m_host_vec_ptr);
        size_t count = m_host_vec_ptr->size();
        if (count > m_device_capacity)
            resizeDevice(count);
        DEME_GPU_CALL(cudaMemcpy(m_device_ptr, m_host_vec_ptr->data(), count * sizeof(T), cudaMemcpyHostToDevice));
        m_host_dirty = false;
    }

    void copyToHost() {
        assert(m_device_ptr && m_host_vec_ptr);
        DEME_GPU_CALL(cudaMemcpy(m_host_vec_ptr->data(), m_device_ptr, size() * sizeof(T), cudaMemcpyDeviceToHost));
        m_host_dirty = false;
    }

    void copyToDeviceAsync(cudaStream_t& stream) {
        assert(m_host_vec_ptr && m_device_ptr);
        size_t count = m_host_vec_ptr->size();
        if (count > m_device_capacity)
            resizeDevice(count);
        cudaMemcpyAsync(m_device_ptr, m_host_vec_ptr->data(), count * sizeof(T), cudaMemcpyHostToDevice, stream);
        m_host_dirty = false;
    }

    void copyToHostAsync(cudaStream_t& stream) {
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

    PinnedVector& getHostVector() { return *m_host_vec_ptr; }

    void bindDevicePointer(T** external_ptr_to_ptr) {
        m_bound_device_ptr = external_ptr_to_ptr;
        updateBoundDevicePointer();
    }

    void unbindDevicePointer() { m_bound_device_ptr = nullptr; }

    void setHostMemoryCounter(size_t* counter) { m_host_mem_counter = counter; }
    void setDeviceMemoryCounter(size_t* counter) { m_device_mem_counter = counter; }
    // You can use nullptr to unbind

    void attachHostVector(const std::vector<T>* external_vec, bool deep_copy = true) {
        freeHost();  // discard internal memory and update memory tracker
        if (deep_copy) {
            m_pinned_vec = std::make_unique<PinnedVector>(external_vec->begin(), external_vec->end());
            m_host_vec_ptr = m_pinned_vec.get();
            updateHostMemCounter(static_cast<ssize_t>(m_host_vec_ptr->size() * sizeof(T)));
        } else {
            m_host_vec_ptr = const_cast<PinnedVector*>(reinterpret_cast<const PinnedVector*>(external_vec));
        }
        m_host_dirty = true;
    }

    T& operator[](size_t i) { return (*m_host_vec_ptr)[i]; }
    const T& operator[](size_t i) const { return (*m_host_vec_ptr)[i]; }

  private:
    std::unique_ptr<PinnedVector> m_pinned_vec = nullptr;
    PinnedVector* m_host_vec_ptr = nullptr;

    size_t* m_host_mem_counter = nullptr;
    size_t* m_device_mem_counter = nullptr;

    T* m_device_ptr = nullptr;
    size_t m_device_capacity = 0;

    T** m_bound_device_ptr = nullptr;

    bool m_host_dirty = false;

    void ensureHostVector(size_t n = 0) {
        if (!m_host_vec_ptr) {
            m_pinned_vec = std::make_unique<PinnedVector>(n);
            m_host_vec_ptr = m_pinned_vec.get();
        }
    }

    void updateBoundDevicePointer() {
        if (m_bound_device_ptr)
            *m_bound_device_ptr = m_device_ptr;
    }

    void updateHostMemCounter(ssize_t delta) {
        if (m_host_mem_counter)
            *m_host_mem_counter += delta;
    }

    void updateDeviceMemCounter(ssize_t delta) {
        if (m_device_mem_counter)
            *m_device_mem_counter += delta;
    }
};

// Pure device data type, usually used for scratching space
template <typename T>
class DeviceArray {
  public:
    DeviceArray(size_t* external_counter = nullptr) : m_mem_counter(external_counter) {}

    explicit DeviceArray(size_t n, size_t* external_counter = nullptr) : m_mem_counter(external_counter) { resize(n); }

    ~DeviceArray() { free(); }

    void resize(size_t n, bool allow_shrink = false) {
        size_t old_bytes = m_capacity * sizeof(T);
        if (!allow_shrink && m_capacity >= n)
            return;

        if (m_data)
            DEME_GPU_CALL(cudaFree(m_data));

        DEME_GPU_CALL(cudaMalloc((void**)&m_data, n * sizeof(T)));
        m_capacity = n;

        updateMemCounter(static_cast<ssize_t>(m_capacity * sizeof(T)) - static_cast<ssize_t>(old_bytes));
    }

    void free() {
        if (m_data) {
            updateMemCounter(-(ssize_t)(m_capacity * sizeof(T)));
            DEME_GPU_CALL(cudaFree(m_data));
            m_data = nullptr;
            m_capacity = 0;
        }
    }

    size_t size() const { return m_capacity; }

    T* data() { return m_data; }

    const T* data() const { return m_data; }

    void setMemoryCounter(size_t* counter) { m_mem_counter = counter; }

  private:
    T* m_data = nullptr;
    size_t m_capacity = 0;
    size_t* m_mem_counter = nullptr;

    void updateMemCounter(ssize_t delta) {
        if (m_mem_counter)
            *m_mem_counter += delta;
    }
};

template <typename T>
class DeviceVectorPool {
  private:
    std::vector<DeviceArray<T>> vectors;
    std::vector<std::optional<std::string>> in_use;
    std::unordered_map<std::string, size_t> name_to_index;

    size_t* m_mem_counter = nullptr;

  public:
    explicit DeviceVectorPool(size_t* external_counter = nullptr) : m_mem_counter(external_counter) {}

    ~DeviceVectorPool() { releaseAll(); }

    // Claim a vector with at least `size` bytes and associate it with a name
    T* claim(const std::string& name, size_t size) {
        if (name_to_index.count(name)) {
            throw std::runtime_error("Name already claimed: " + name);
        }

        // Look for a free vector
        for (size_t i = 0; i < in_use.size(); ++i) {
            // Found one already allocated but later freed: use it
            if (!in_use[i]) {
                // DeviceArray resize method integrates non-shrink check
                vectors[i].resize(size);
                in_use[i] = name;
                name_to_index[name] = i;
                return vectors[i].data();
            }
        }

        // No free vector: expand pool
        vectors.emplace_back(size, m_mem_counter);
        in_use.emplace_back(name);
        size_t new_index = vectors.size() - 1;
        name_to_index[name] = new_index;
        return vectors[new_index].data();
    }

    T* get(const std::string& name) {
        auto it = name_to_index.find(name);
        if (it == name_to_index.end()) {
            throw std::runtime_error("Name not found: " + name);
        }
        return vectors[it->second].data();
    }

    void resize(const std::string& name, size_t new_size, bool allow_create = false) {
        auto it = name_to_index.find(name);
        if (it == name_to_index.end()) {
            if (allow_create) {
                claim(name, new_size);  // Will allocate if needed
                return;
            } else {
                throw std::runtime_error("Cannot resize: name not found");
            }
        }

        size_t index = it->second;
        vectors[index].resize(new_size);
    }

    // Unclaim a named vector (make it available again)
    void unclaim(const std::string& name) {
        auto it = name_to_index.find(name);
        if (it == name_to_index.end()) {
            return;
        }

        size_t index = it->second;
        in_use[index] = std::nullopt;
        name_to_index.erase(it);
    }

    // This one also deallocate the memory
    void release(const std::string& name) {
        auto it = name_to_index.find(name);
        if (it == name_to_index.end()) {
            throw std::runtime_error("Cannot release: name not found");
        }

        size_t index = it->second;
        vectors[index].free();
        in_use[index] = std::nullopt;
        name_to_index.erase(it);
    }

    // Release all memory (clear the vectors)
    void releaseAll() {
        for (const auto& entry : name_to_index) {
            size_t index = entry.second;
            vectors[index].free();
        }
        in_use.clear();
        name_to_index.clear();
    }

    void setMemoryCounter(size_t* counter) {
        m_mem_counter = counter;
        for (auto& vec : vectors) {
            vec.setMemoryCounter(counter);
        }
    }

    // Debug utility
    void printStatus() const {
        for (size_t i = 0; i < vectors.size(); ++i) {
            if (in_use[i]) {
                std::cout << "Vector[" << i << "] in use as \"" << *in_use[i] << "\"\n";
            } else {
                std::cout << "Vector[" << i << "] is free\n";
            }
        }
    }
};

}  // namespace deme

#endif
