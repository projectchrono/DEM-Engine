#include <algorithm>
#include <stdexcept>

#include "GpuManager.h"

// TODO: add CUDA error checking
GpuManager::GpuManager(unsigned int total_streams) {
    int ndevices = 0;
    cudaGetDeviceCount(&ndevices);

    this->streams.resize(ndevices);

    for (unsigned int current_device = 0; total_streams > 0; total_streams--, current_device++) {
        if (current_device >= ndevices) {
            current_device = 0;
        }

        cudaStream_t new_stream;
        // cudaStreamCreate(&new_stream);

        this->streams[current_device].push_back(StreamInfo{(signed)current_device, new_stream, false});
    }
}

// TODO: add CUDA error checking
GpuManager::~GpuManager() {
    for (auto outer = this->streams.begin(); outer != this->streams.end(); outer++) {
        for (auto inner = (*outer).begin(); inner != (*outer).end(); inner++) {
            // cudaStreamDestroy(inner->stream);
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
    auto iter = std::min_element(this->streams.begin(), this->streams.end(),
                                 [](const auto& a, const auto& b) { return (a.size() < b.size()); });
    return (*iter).size();
}

unsigned int GpuManager::getMaxStreamsPerDevice() {
    auto iter = std::max_element(this->streams.begin(), this->streams.end(),
                                 [](const auto& a, const auto& b) { return (a.size() < b.size()); });
    return (*iter).size();
}

const std::vector<GpuManager::StreamInfo>& GpuManager::getStreamsFromDevice(int index) {
    return this->streams[index];
}

const std::vector<GpuManager::StreamInfo>& GpuManager::getStreamsFromDevice(const GpuManager::StreamInfo& info) {
    return this->getStreamsFromDevice(info.device);
}

const GpuManager::StreamInfo& GpuManager::getAvailableStream() {
    // Iterate over stream lists by device
    for (auto by_device = this->streams.begin(); by_device != streams.end(); by_device++) {
        // Iterate over streams in each device
        for (auto stream = by_device->begin(); stream != by_device->end(); stream++) {
            if (!stream->_impl_active) {
                stream->_impl_active = true;
                return *stream;
            }
        }
    }

    // This exception is not meant to be handled, it serves as a notifier that the algorithm is using more streams than
    // it allocated
    throw std::range_error("no available streams!");
}

const GpuManager::StreamInfo& GpuManager::getAvailableStreamFromDevice(int index) {
    for (auto stream = this->streams[index].begin(); stream != this->streams[index].end(); stream++) {
        if (!stream->_impl_active) {
            stream->_impl_active = true;
            return *stream;
        }
    }

    // This exception should rarely be thrown, so it shouldn't have a notable performance impact
    throw std::range_error("no available streams on device [" + std::to_string(index) + "]!");
}

void GpuManager::setStreamAvailable(const StreamInfo& info) {
    for (auto stream = this->streams[info.device].begin(); stream != this->streams[info.device].end(); stream++) {
        if (stream->stream == info.stream) {
            stream->_impl_active = false;
        }
    }
}