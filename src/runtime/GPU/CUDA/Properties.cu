#include "lingodb/runtime/GPU/Properties.h"
#include "lingodb/runtime/GPU/CUDA/CudaUtils.cuh"


std::string lingodb::runtime::gpu::getChipStr(uint32_t deviceId) {
   int major, minor;
   CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, deviceId));
   CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, deviceId));
   return "sm_" + std::to_string(major) + std::to_string(minor);
}