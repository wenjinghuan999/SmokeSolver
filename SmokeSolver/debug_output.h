#pragma once

#ifndef __DEBUG_OUTPUT_H__
#define __DEBUG_OUTPUT_H__


#include "common.h"
#include "Blob.h"

#include <iostream>
#include <fstream>
#include <algorithm>

namespace std
{
	template<class _Elem,
		class _Traits>
	std::basic_ostream<_Elem, _Traits> &
		operator<< (std::basic_ostream<_Elem, _Traits> &out, float2 q)
	{
		out << "(" << q.x << "," << q.y << ")";
		return out;
	}

	template<class _Elem,
		class _Traits>
		std::basic_ostream<_Elem, _Traits> &
		operator<< (std::basic_ostream<_Elem, _Traits> &out, float4 q)
	{
		out << "(" << q.x << "," << q.y << "," << q.z << "," << q.w << ")";
		return out;
	}
}

namespace ssv
{
	namespace output
	{
		template <typename _T>
		void PrintRawCPU(_T *a, size_t length = 1u, const std::string &tag = "data", std::ostream &out = std::cout)
		{
			if (a == nullptr)
			{
				out << tag << " = nullptr" << std::endl;
				return;
			}

			out << tag << ": ";
			for (unsigned int i = 0u; i < length; i++)
			{
				out << a[i] << " ";
			}
			out << std::endl;
			delete[] a;
		}

		template <typename _T>
		class Blob;

		template <typename _T>
		void PrintBlobCPU(const ssv::Blob<_T> &b, const std::string &tag = "data", std::ostream &out = std::cout)
		{
			for (uint z = 0; z < b.nz(); z++)
			{
				out << tag << "<CPU>(:,:," << z << "):\n";
				for (uint y = 0; y < b.ny(); y++)
				{
					std::copy(b.data_cpu() + z * b.ny() * b.nx() + y * b.nx(),
						b.data_cpu() + z * b.ny() * b.nx() + (y + 1u) * b.nx(), std::ostream_iterator<_T>(out, " "));
					out << std::endl;
				}
				out << std::endl;
			}
		}

		inline void SaveBlobCPU(const ssv::BlobBase &b, const std::string &filename = "data.dat")
		{
			std::ofstream fout(filename, std::ios::binary);
			fout.write(reinterpret_cast<const char *>(b.data_cpu()), b.size_cpu_in_bytes());
			fout.close();
		}
	}
}


#endif // !__DEBUG_OUTPUT_H__
