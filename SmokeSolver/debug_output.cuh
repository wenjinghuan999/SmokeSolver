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
		/**
		 * \brief Print raw GPU data
		 * \tparam T element data type
		 * \param a_dev raw GPU data pointer to be output
		 * \param length length of data in elements
		 * \param tag output tag string
		 * \param out output ostream, e.g. cout, cerr
		 */
		template <typename T>
		void print_raw_gpu(T *a_dev, size_t length = 1u, std::string tag = "data", std::ostream &out = std::cout)
		{
			if (a_dev == nullptr)
			{
				out << tag << " = nullptr" << std::endl;
				return;
			}

			T *a = new T[length];
			cudaMemcpy(a, a_dev, length * sizeof(T), cudaMemcpyDeviceToHost);

			out << tag << ": ";
			for (unsigned int i = 0u; i < length; i++)
			{
				out << a[i] << " ";
			}
			out << std::endl;
			delete[] a;
		}

		/**
		 * \brief Print GPU Blob data
		 * \tparam T element data type
		 * \param b blob to be output
		 * \param tag output tag string
		 * \param out output ostream, e.g. cout, cerr
		 */
		template <typename T>
		void print_blob_gpu(const ssv::Blob<T> &b, std::string tag = "data", std::ostream &out = std::cout)
		{
			for (uint z = 0; z < b.nz(); z++)
			{
				out << tag << "<GPU>(:,:," << z << "):\n";
				for (uint y = 0; y < b.ny(); y++)
				{
					thrust::copy(b.data_gpu() + z * b.ny() * b.nx() + y * b.nx(),
						b.data_gpu() + z * b.ny() * b.nx() + (y + 1u) * b.nx(), std::ostream_iterator<T>(out, " "));
					out << std::endl;
				}
				out << std::endl;
			}
		}
	}
}

#endif // !__DEBUG_OUTPUT_CUH__
