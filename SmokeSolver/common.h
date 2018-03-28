#pragma once

#ifndef __COMMON_H__
#define __COMMON_H__

#include <cuda.h>
#include <cuda_runtime.h>
#include <google/protobuf/stubs/common.h>
#include <glm/glm.hpp>
#include <exception>

#include <type_traits>


namespace ssv
{
	typedef float real;
	typedef float2 real2;
	typedef float3 real3;
	typedef float4 real4;
	typedef google::protobuf::uint8 byte;
	typedef google::protobuf::uint uint;

	template <typename ...Types>
	__host__ __device__ real2 make_real2(Types ...args)
	{
		return make_float2(std::forward<Types>(args)...);
	}

	template <typename ...Types>
	__host__ __device__ real3 make_real3(Types ...args)
	{
		return make_float3(std::forward<Types>(args)...);
	}

	template <typename ...Types>
	__host__ __device__ real4 make_real4(Types ...args)
	{
		return make_float4(std::forward<Types>(args)...);
	}

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

	class ssv_error
		: public std::exception
	{
		// base of all out-of-range exceptions
	public:
		typedef std::exception mybase_t;

		explicit ssv_error(error_t err)
			: err(err)
		{
		}

	public:
		error_t err;

#if _HAS_EXCEPTIONS

#else /* _HAS_EXCEPTIONS */
	protected:
		virtual void _Doraise() const
		{	// perform class-specific exception handling
			_RAISE(*this);
		}
#endif /* _HAS_EXCEPTIONS */
	};

	template <typename EnumType>
	constexpr typename std::underlying_type<EnumType>::type underlying(EnumType e)
	{
		return static_cast<typename std::underlying_type<EnumType>::type>(e);
	}
}

#endif // !__COMMON_H__
