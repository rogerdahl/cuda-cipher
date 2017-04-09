#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utilities.h"
#include <cuda_runtime.h>

using namespace std;

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

void __cudaCheck(cudaError err, const char* file, const int line)
{
  if (cudaSuccess != err) {
    fprintf(
        stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n", file, line,
        (int)err, cudaGetErrorString(err));
    exit(-1);
  }
}

void __cudaCheckLastError(
    const char* kernel_name, const char* file, const int line)
{
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(
        stderr,
        "Kernel failed. kernel=%s, check_loc=%s(%i), error_code=%d, "
        "error_str=%s\n",
        kernel_name, file, line, (int)err, cudaGetErrorString(err));
    exit(-1);
  }
}

u32 DivUp(u32 a, u32 b)
{
  return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

// General GPU Device CUDA Initialization
int gpuDeviceInit(int devID)
{
  int deviceCount;
  cudaCheck(cudaGetDeviceCount(&deviceCount));
  if (deviceCount == 0) {
    fprintf(
        stderr, "gpuDeviceInit() CUDA error: no devices supporting CUDA.\n");
    exit(-1);
  }
  if (devID < 0)
    devID = 0;
  if (devID > deviceCount - 1) {
    fprintf(stderr, "\n");
    fprintf(
        stderr, ">> %d CUDA capable GPU device(s) detected. <<\n", deviceCount);
    fprintf(
        stderr, ">> gpuDeviceInit (-device=%d) is not a valid GPU device. <<\n",
        devID);
    fprintf(stderr, "\n");
    return -devID;
  }

  cudaDeviceProp deviceProp;
  cudaCheck(cudaGetDeviceProperties(&deviceProp, devID));
  if (deviceProp.major < 1) {
    fprintf(stderr, "gpuDeviceInit(): GPU device does not support CUDA.\n");
    exit(-1);
  }

  cudaCheck(cudaSetDevice(devID));
  printf("> gpuDeviceInit() CUDA device [%d]: %s\n", devID, deviceProp.name);
  return devID;
}

// Beginning of GPU Architecture definitions
int _ConvertSMVer2Cores(int major, int minor)
{
  // Defines for GPU Architecture types (using the SM version to determine the #
  // of cores per SM
  typedef struct
  {
    int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM
            // minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = { { 0x10, 8 }, { 0x11, 8 },  { 0x12, 8 },
                                      { 0x13, 8 }, { 0x20, 32 }, { 0x21, 48 },
                                      { -1, -1 } };

  int index = 0;
  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }
    index++;
  }
  printf("MapSMtoCores undefined SMversion %d.%d!\n", major, minor);
  return -1;
}

// This function returns the best GPU (with maximum GFLOPS)
int gpuGetMaxGflopsDeviceId()
{
  int current_device = 0, sm_per_multiproc = 0;
  int max_compute_perf = 0, max_perf_device = 0;
  int device_count = 0, best_SM_arch = 0;
  cudaDeviceProp deviceProp;

  cudaGetDeviceCount(&device_count);
  // Find the best major SM Architecture GPU device
  while (current_device < device_count) {
    cudaGetDeviceProperties(&deviceProp, current_device);
    if (deviceProp.major > 0 && deviceProp.major < 9999) {
      best_SM_arch = MAX(best_SM_arch, deviceProp.major);
    }
    current_device++;
  }

  // Find the best CUDA capable GPU device
  current_device = 0;
  while (current_device < device_count) {
    cudaGetDeviceProperties(&deviceProp, current_device);
    if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
      sm_per_multiproc = 1;
    }
    else {
      sm_per_multiproc =
          _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
    }

    int compute_perf = deviceProp.multiProcessorCount * sm_per_multiproc
                       * deviceProp.clockRate;
    if (compute_perf > max_compute_perf) {
      // If we find GPU with SM major > 2, search only these
      if (best_SM_arch > 2) {
        // If our device==dest_SM_arch, choose this, or else pass
        if (deviceProp.major == best_SM_arch) {
          max_compute_perf = compute_perf;
          max_perf_device = current_device;
        }
      }
      else {
        max_compute_perf = compute_perf;
        max_perf_device = current_device;
      }
    }
    ++current_device;
  }
  return max_perf_device;
}

void PrintCUDADeviceProperties(u32 cuda_device)
{
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, cuda_device);
  wcout << "Name: " << prop.name << endl;
  wcout << "Compute Capability: " << prop.major << L"." << prop.minor << endl;
  wcout << "MultiProcessor Count: " << prop.multiProcessorCount << endl;
  wcout << "Clock Rate: " << prop.clockRate << L" Hz" << endl;
  wcout << "Warp Size: " << prop.warpSize << endl;
  wcout << "Total Constant Memory: " << prop.totalConstMem << L" bytes "
        << endl;
  wcout << "Total Global Memory: " << prop.totalGlobalMem << L" bytes " << endl;
  wcout << "Shared Memory Per Block: " << prop.sharedMemPerBlock << L" bytes "
        << endl;
  wcout << "Max Grid Size: (" << prop.maxGridSize[0] << L", "
        << prop.maxGridSize[1] << L", " << prop.maxGridSize[2] << L")" << endl;
  wcout << "Max Threads Dim: (" << prop.maxThreadsDim[0] << L", "
        << prop.maxThreadsDim[1] << L", " << prop.maxThreadsDim[2] << L")"
        << endl;
  wcout << "Max Threads Per Block: " << prop.maxThreadsPerBlock << endl;
  wcout << "Registers Per Block: " << prop.regsPerBlock << endl;
  wcout << "Memory Pitch: " << prop.memPitch << endl;
  wcout << "Texture Alignment: " << prop.textureAlignment << endl;
  wcout << "Device Overlap: " << prop.deviceOverlap << L"\n" << endl;
}
