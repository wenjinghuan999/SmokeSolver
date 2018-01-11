#pragma once

#ifndef __COMMON_CUH__
#define __COMMON_CUH__

#ifdef __INTELLISENSE__
#define __CUDACC__
#endif

#ifdef __RESHARPER__
#define __CUDACC__
#endif

#include "common.h"

#include <device_functions.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>

#ifdef __RESHARPER__
#undef __annotate__
#define __annotate__(a)
#endif

#include <helper_cuda.h>
#include <helper_math.h>

#include <iostream>


namespace ssv
{
	template <typename T>
	void check(T result, char const *const func, const char *const file, int const line, const ssv_error &throwing_error)
	{
		if (result)
		{
			fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
			        file, line, static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
			throw throwing_error;
		}
	}

#define CHECK_CUDA_ERROR_AND_THROW(val, throwing_error) check ((val), #val, __FILE__, __LINE__, throwing_error)
}

#endif // !__COMMON_CUH__
