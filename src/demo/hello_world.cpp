#include <core/utils/GpuManager.h>
#include <core/utils/JitHelper.h>

int main(int argc, char** argv) {
    GpuManager mgr(2);

    auto stream_info = mgr.getAvailableStream();

    cudaSetDevice(stream_info.device);

    std::cerr << "[NOTICE]: This demo assumes you are running from a typical build directory location inside of the "
                 "source directory!\n";

    std::cerr << "[INFO]: Building kernel from " << JitHelper::KERNEL_DIR << "\n\n";
    auto program = JitHelper::buildProgram("hello", JitHelper::KERNEL_DIR / "hello.cu", {}, {"-I/opt/cuda/include"});

    program.kernel("helloWorldKernel").instantiate().configure(dim3(1), dim3(1), 0, stream_info.stream).launch();

    cudaStreamSynchronize(stream_info.stream);

    exit(0);
}
