// cudax experimental APIs require CUDA >= 12.6 and CCCL >= 3.0
#if CUDART_VERSION >= 12060 && CCCL_VERSION_MAJOR >= 3
#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/launch>
#include <cuda/memory_pool>
#include <cuda/stream>
#define HAS_CUDAX 1
#pragma message("Building WITH cudax (CUDART_VERSION=" _CCCL_TO_STRING(CUDART_VERSION) ", CCCL=" _CCCL_TO_STRING( \
    CCCL_VERSION_MAJOR) "." _CCCL_TO_STRING(CCCL_VERSION_MINOR) ")")
#else
#define HAS_CUDAX 0

// Stringify helpers for compile-time messages
#define _CUDAX_STR(x) #x
#define _CUDAX_XSTR(x) _CUDAX_STR(x)
#pragma message("Building WITHOUT cudax (CUDART_VERSION=" _CUDAX_XSTR(CUDART_VERSION) ", CCCL=" _CUDAX_XSTR( \
    CCCL_VERSION_MAJOR) "." _CUDAX_XSTR(CCCL_VERSION_MINOR) ")")
#endif