#ifndef SGPS_GPU_MANAGER
#define SGPS_GPU_MANAGER

#include <cuda_runtime_api.h>
#include <vector>
#include <core/ApiVersion.h>

class GpuManager {
  public:
    GpuManager(unsigned int total_streams = 1);
    ~GpuManager();

    struct Stream {
      public:
        cudaStream_t stream;
        int device;
    };

    // Returns the LEAST number of streams available on any device
    unsigned int getStreamsPerDevice();
    // Returns the HIGHEST number of streams per device
    unsigned int getMaxStreamsPerDevice();

    unsigned int getNumDevices();

    const std::vector<cudaStream_t>& getStreamsFromDevice(int index);
    const std::vector<cudaStream_t>& getStreamsFromDevice(const struct Stream&);

  private:
    std::vector<std::vector<cudaStream_t>> streams;
};

#endif
