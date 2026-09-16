#ifndef DEME_GPU_MANAGER_H
#define DEME_GPU_MANAGER_H

#include <cuda_runtime_api.h>
#include <vector>
#include <mutex>

namespace deme {

// A device number manager that evenly distributes the streams needed to all the available devices
class GpuManager {
  public:
    GpuManager(unsigned int total_streams = 1);
    /// Create one stream record for each explicitly selected CUDA device.
    /// Repeated device IDs are allowed so multiple workers can share one GPU.
    explicit GpuManager(const std::vector<int>& stream_devices);
    ~GpuManager();

    struct StreamInfo {
      public:
        int device;
        cudaStream_t stream;

        bool _impl_active;  // Reserved for the implementation
    };

    // Returns the LEAST number of streams available on any device.
    unsigned int getStreamsPerDevice();
    // Returns the HIGHEST number of streams per device.
    unsigned int getMaxStreamsPerDevice();

    static int scanNumDevices();

    // DO NOT USE UNLESS YOU INTEND TO MANUALLY HANDLE YOUR STREAMS.
    const std::vector<StreamInfo>& getStreamsFromDevice(int index);

    // DO NOT USE UNLESS YOU INTEND TO MANUALLY HANDLE YOUR STREAMS.
    const std::vector<StreamInfo>& getStreamsFromDevice(const StreamInfo&);

    // Get a stream which hasn't been used yet and mark it as used.
    const StreamInfo& getAvailableStream();
    const StreamInfo& getAvailableStreamFromDevice(int index);

    // Mark a stream as unused.
    void setStreamAvailable(const StreamInfo&);

    // Return the number of distinct devices assigned to workers.
    int getNumDevices() const { return nactive_devices; }
    // Return the number of logical CUDA devices visible to this process.
    int getNumVisibleDevices() const { return nvisible_devices; }

  private:
    int nvisible_devices;
    int nactive_devices;
    std::vector<std::vector<StreamInfo>> streams;
    std::mutex stream_manipulation_mutex;
};

}  // namespace deme

#endif
