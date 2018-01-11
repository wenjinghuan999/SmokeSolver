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
	template <class Elem, class Traits>
	std::basic_ostream<Elem, Traits> &
	operator<<(std::basic_ostream<Elem, Traits> &out, float2 q)
	{
		out << "(" << q.x << "," << q.y << ")";
		return out;
	}

	template <class Elem, class Traits>
	std::basic_ostream<Elem, Traits> &
	operator<<(std::basic_ostream<Elem, Traits> &out, float4 q)
	{
		out << "(" << q.x << "," << q.y << "," << q.z << "," << q.w << ")";
		return out;
	}
}

namespace ssv
{
	namespace output
	{
		/**
		 * \brief Print raw CPU data
		 * \tparam T element data type
		 * \param a raw CPU data pointer to be output
		 * \param length length of data in elements
		 * \param tag output tag string
		 * \param out output ostream, e.g. cout, cerr
		 */
		template <typename T>
		void print_raw_cpu(T *a, size_t length = 1u, const std::string &tag = "data", std::ostream &out = std::cout)
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

		template <typename T>
		class Blob;

		/**
		 * \brief Print CPU Blob data
		 * \tparam T element data type
		 * \param b blob to be output
		 * \param tag output tag string
		 * \param out output ostream, e.g. cout, cerr
		 */
		template <typename T>
		void print_blob_cpu(const ssv::Blob<T> &b, const std::string &tag = "data", std::ostream &out = std::cout)
		{
			for (uint z = 0; z < b.nz(); z++)
			{
				out << tag << "<CPU>(:,:," << z << "):\n";
				for (uint y = 0; y < b.ny(); y++)
				{
					std::copy(b.data_cpu() + z * b.ny() * b.nx() + y * b.nx(),
					          b.data_cpu() + z * b.ny() * b.nx() + (y + 1u) * b.nx(), std::ostream_iterator<T>(out, " "));
					out << std::endl;
				}
				out << std::endl;
			}
		}

		/**
		 * \brief Save CPU blob to data file (in binary)
		 * \param b blob to be saved
		 * \param filename target filename
		 */
		inline void save_blob_cpu(const BlobBase &b, const std::string &filename = "data.dat")
		{
			std::ofstream fout(filename, std::ios::binary);
			fout.write(reinterpret_cast<const char *>(b.data_cpu_void()), b.size_cpu_in_bytes());
			fout.close();
		}
	}
}


#endif // !__DEBUG_OUTPUT_H__
