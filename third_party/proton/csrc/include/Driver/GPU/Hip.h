#ifndef PROTON_DRIVER_GPU_HIP_H_
#define PROTON_DRIVER_GPU_HIP_H_

#include "Driver/Device.h"
#include <hip/hip_runtime_api.h>

namespace proton {

namespace hip {

template <bool CheckSuccess> hipError_t deviceSynchronize();

template <bool CheckSuccess>
hipError_t deviceGetAttribute(int *value, hipDeviceAttribute_t attribute,
                              int deviceId);

Device getDevice(uint64_t index);

} // namespace hip

} // namespace proton

#endif // PROTON_DRIVER_GPU_HIP_H_
