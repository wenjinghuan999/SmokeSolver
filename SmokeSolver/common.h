#pragma once

#ifndef __COMMON_H__
#define __COMMON_H__

#include <cuda.h>
#include <cuda_runtime.h>
#include <google/protobuf/stubs/common.h>

#include <type_traits>


namespace ssv
{
	typedef float T;
	typedef float2 T2;
	typedef float3 T3;
	typedef float4 T4;
	typedef ::google::protobuf::uint8 byte;
	typedef ::google::protobuf::uint uint;

	enum class error_t : unsigned int
	{
		SSV_SUCCESS = 0,
		SSV_ERROR_INVALID_VALUE = 7001,
		SSV_ERROR_OUT_OF_MEMORY_CPU = 7002,
		SSV_ERROR_OUT_OF_MEMORY_GPU = 7003,
		SSV_ERROR_NOT_INITIALIZED = 7004,
		SSV_ERROR_TEXTURE_NOT_INITIALIZED = 7005,
		SSV_ERROR_DEVICE_NOT_READY = 7006,
		SSV_ERROR_UNKNOWN = 7999
	};

	template<typename EnumType>
	constexpr typename std::underlying_type<EnumType>::type underlying(EnumType e)
	{
		return static_cast<typename std::underlying_type<EnumType>::type>(e);
	}
}

#endif // !__COMMON_H__
