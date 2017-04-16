#pragma once

#ifndef __COMMON_CUH__
#define __COMMON_CUH__

#include "common.h"
#include "Blob.h"

#include <device_functions.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

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

	template <typename _T>
	void PrintRaw(_T *a_dev, size_t length = 1u, std::string tag = "data")
	{
		if (a_dev == nullptr)
		{
			std::cout << tag << " = nullptr" << std::endl;
			return;
		}

		_T *a = new _T[length];
		cudaMemcpy(a, a_dev, length * sizeof(_T), cudaMemcpyDeviceToHost);

		std::cout << tag << ": ";
		for (unsigned int i = 0u; i < length; i++)
		{
			std::cout << a[i] << " ";
		}
		std::cout << std::endl;
		delete[] a;
	}

	template <typename _T>
	void PrintBlobGPU(const Blob<_T> &b, std::string tag = "data")
	{
		for (uint z = 0; z < b.nz(); z++)
		{
			std::cout << tag << "(:,:," << z << "):\n";
			for (uint y = 0; y < b.ny(); y++)
			{
				thrust::copy(b.data_gpu() + z * b.ny() * b.nx() + y * b.nx(),
					b.data_gpu() + z * b.ny() * b.nx() + (y + 1u) * b.nx(), std::ostream_iterator<_T>(std::cout, " "));
				std::cout << std::endl;
			}
			std::cout << std::endl;
		}
	}
}

#endif // !__COMMON_CUH__
