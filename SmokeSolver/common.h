#pragma once

#ifndef __COMMON_H__
#define __COMMON_H__

#include <cuda.h>
#include <cuda_runtime.h>
#include <google/protobuf/stubs/common.h>


namespace ssv
{
	typedef float T;
	typedef ::google::protobuf::uint8 byte;

	typedef enum
	{
		SSV_SUCCESS = 0,
		SSV_ERROR_INVALID_VALUE = 7001,
		SSV_ERROR_OUT_OF_MEMORY_CPU = 7002,
		SSV_ERROR_OUT_OF_MEMORY_GPU = 7003,
		SSV_ERROR_NOT_INITIALIZED = 7004,
		SSV_ERROR_DEVICE_NOT_READY = 7005,
		SSV_ERROR_UNKNOWN = 7999
	} error_t;
}

#endif // !__COMMON_H__
