#include <algorithm>
#include "GpuManager.h"

// TODO: add CUDA error checking
GpuManager::GpuManager(unsigned int total_streams) {
    
    int ndevices = 0;
    cudaGetDeviceCount(&ndevices);
    
    this->streams.resize(ndevices);
        
    for (
        unsigned int current_device = 0; 
        total_streams > 0; 
        total_streams--, current_device++
    ) {
    
        cudaStream_t new_stream;
        cudaStreamCreate(&new_stream);
        
        this->streams[current_device].push_back(new_stream);
    
        if (current_device >= ndevices) {
            current_device = 0;
        }
    }
}

// TODO: add CUDA error checking
GpuManager::~GpuManager() {
    for (auto outer = this->streams.begin(); outer != this->streams.end(); outer++) {
        for (auto inner = (*outer).begin(); inner != (*outer).end(); inner++) {
            cudaStreamDestroy(*inner);
        }
    }
}

// TODO: add CUDA error checking
unsigned int GpuManager::getNumDevices() {
    int ndevices = 0;
    cudaGetDeviceCount(&ndevices);
    return ndevices;
}

unsigned int GpuManager::getStreamsPerDevice() {
    auto iter = std::min_element(
        this->streams.begin(), 
        this->streams.end(), 
        [](const auto& a, const auto& b){
            return (a.size() < b.size());
        }
    );
    return (*iter).size();
}

unsigned int GpuManager::getMaxStreamsPerDevice() {
    auto iter = std::max_element(
        this->streams.begin(), 
        this->streams.end(), 
        [](const auto& a, const auto& b){
            return (a.size() < b.size());
        }
    );
    return (*iter).size();
}

const std::vector<cudaStream_t>& GpuManager::getStreamsFromDevice(int index) {
    return this->streams[index];
}

const std::vector<cudaStream_t>& GpuManager::getStreamsFromDevice(const struct Stream& stream) {
    return this->getStreamsFromDevice(stream.device);
}

