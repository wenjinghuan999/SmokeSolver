#pragma once

#ifndef __DEBUG_OUTPUT_CUH__
#define __DEBUG_OUTPUT_CUH__

#include "common.cuh"
#include "Blob.h"
#include "debug_output.h"
#include <iostream>

namespace ssv
{
	namespace output
	{

		template <typename _T>
		void PrintRawGPU(_T *a_dev, size_t length = 1u, std::string tag = "data", std::ostream &out = std::cout)
		{
			if (a_dev == nullptr)
			{
				out << tag << " = nullptr" << std::endl;
				return;
			}

			_T *a = new _T[length];
			cudaMemcpy(a, a_dev, length * sizeof(_T), cudaMemcpyDeviceToHost);

			out << tag << ": ";
			for (unsigned int i = 0u; i < length; i++)
			{
				out << a[i] << " ";
			}
			out << std::endl;
			delete[] a;
		}

		template <typename _T>
		void PrintBlobGPU(const ssv::Blob<_T> &b, std::string tag = "data", std::ostream &out = std::cout)
		{
			for (uint z = 0; z < b.nz(); z++)
			{
				out << tag << "<GPU>(:,:," << z << "):\n";
				for (uint y = 0; y < b.ny(); y++)
				{
					thrust::copy(b.data_gpu() + z * b.ny() * b.nx() + y * b.nx(),
						b.data_gpu() + z * b.ny() * b.nx() + (y + 1u) * b.nx(), std::ostream_iterator<_T>(out, " "));
					out << std::endl;
				}
				out << std::endl;
			}
		}
	}
}

#endif // !__DEBUG_OUTPUT_CUH__
