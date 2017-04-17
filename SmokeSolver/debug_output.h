#pragma once

#ifndef __DEBUG_OUTPUT_H__
#define __DEBUG_OUTPUT_H__


#include "common.h"
#include "Blob.h"

#include <iostream>
#include <algorithm>

namespace ssv
{
	namespace output
	{
		template <typename _T>
		void PrintRawCPU(_T *a, size_t length = 1u, std::string tag = "data")
		{
			if (a == nullptr)
			{
				std::cout << tag << " = nullptr" << std::endl;
				return;
			}

			std::cout << tag << ": ";
			for (unsigned int i = 0u; i < length; i++)
			{
				std::cout << a[i] << " ";
			}
			std::cout << std::endl;
			delete[] a;
		}

		template <typename _T>
		class Blob;

		template <typename _T>
		void PrintBlobCPU(const ssv::Blob<_T> &b, std::string tag = "data")
		{
			for (uint z = 0; z < b.nz(); z++)
			{
				std::cout << tag << "<CPU>(:,:," << z << "):\n";
				for (uint y = 0; y < b.ny(); y++)
				{
					std::copy(b.data_cpu() + z * b.ny() * b.nx() + y * b.nx(),
						b.data_cpu() + z * b.ny() * b.nx() + (y + 1u) * b.nx(), std::ostream_iterator<_T>(std::cout, " "));
					std::cout << std::endl;
				}
				std::cout << std::endl;
			}
		}
	}
}


#endif // !__DEBUG_OUTPUT_H__
