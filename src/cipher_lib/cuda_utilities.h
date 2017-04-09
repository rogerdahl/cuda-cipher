#include "int_types.h"
#include <cuda_runtime.h>

void __cudaCheck(cudaError err, const char* file, const int line);
#define cudaCheck(err) __cudaCheck(err, __FILE__, __LINE__)

void __cudaCheckLastError(
    const char* kernel_name, const char* file, const int line);
#define cudaCheckLastError(msg) __cudaCheckLastError(msg, __FILE__, __LINE__)

u32 DivUp(u32 a, u32 b);

int gpuDeviceInit(int devID);
int _ConvertSMVer2Cores(int major, int minor);
int gpuGetMaxGflopsDeviceId();
void PrintCUDADeviceProperties(u32 cuda_device);