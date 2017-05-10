#pragma once

#ifndef __COMMON_CUH__
#define __COMMON_CUH__

#include "common.h"

#include <device_functions.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>

#include <helper_cuda.h>
#include <helper_math.h>

#include <iostream>


namespace ssv
{
	template <typename T>
	void check(T result, char const *const func, const char *const file, int const line, ssv::error_t throwing_error)
	{
		if (result)
		{
			fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
				file, line, static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
			throw throwing_error;
		}
	}

#define checkCudaErrorAndThrow(val, throwing_error) check ((val), #val, __FILE__, __LINE__, throwing_error)

}

#endif // !__COMMON_CUH__
