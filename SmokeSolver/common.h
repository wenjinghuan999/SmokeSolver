#pragma once

#ifndef __COMMON_H__
#define __COMMON_H__

#include <cuda.h>
#include <cuda_runtime.h>
#include <google/protobuf/stubs/common.h>
#include <tuple>


namespace ssv
{
	typedef float T;
	typedef float2 T2;
	typedef float3 T3;
	typedef float4 T4;
	typedef ::google::protobuf::uint8 byte;

	typedef enum
	{
		SSV_SUCCESS = 0,
		SSV_ERROR_INVALID_VALUE = 7001,
		SSV_ERROR_OUT_OF_MEMORY_CPU = 7002,
		SSV_ERROR_OUT_OF_MEMORY_GPU = 7003,
		SSV_ERROR_NOT_INITIALIZED = 7004,
		SSV_ERROR_TEXTURE_NOT_INITIALIZED = 7005,
		SSV_ERROR_DEVICE_NOT_READY = 7006,
		SSV_ERROR_UNKNOWN = 7999
	} error_t;


	namespace hash_tuple
	{
		template <typename _T>
		struct hash : std::hash<_T> {};

		namespace
		{
			// Code from boost
			// Reciprocal of the golden ratio helps spread entropy
			//     and handles duplicates.
			// See Mike Seymour in magic-numbers-in-boosthash-combine:
			//     http://stackoverflow.com/questions/4948780

			template <class T>
			inline void hash_combine(std::size_t& seed, T const& v)
			{
				seed ^= hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			}

			// Recursive template code derived from Matthieu M.
			template <class Tuple, size_t Index = std::tuple_size<Tuple>::value - 1>
			struct HashValueImpl
			{
				static void apply(size_t& seed, Tuple const& tuple)
				{
					HashValueImpl<Tuple, Index - 1>::apply(seed, tuple);
					hash_combine(seed, std::get<Index>(tuple));
				}
			};

			template <class Tuple>
			struct HashValueImpl<Tuple, 0>
			{
				static void apply(size_t& seed, Tuple const& tuple)
				{
					hash_combine(seed, std::get<0>(tuple));
				}
			};
		}

		template <typename ... TT>
		struct hash<std::tuple<TT...> >
		{
			size_t operator()(std::tuple<TT...> const& tt) const
			{
				size_t seed = 0;
				HashValueImpl<std::tuple<TT...> >::apply(seed, tt);
				return seed;
			}
		};

		template <>
		struct hash<cudaTextureDesc>
		{
			size_t operator()(cudaTextureDesc const& t) const
			{
				size_t seed = 0;
				hash_combine(seed, t.addressMode[0]);
				hash_combine(seed, t.addressMode[1]);
				hash_combine(seed, t.addressMode[2]);
				hash_combine(seed, t.filterMode);
				hash_combine(seed, t.readMode);
				hash_combine(seed, t.sRGB);
				hash_combine(seed, t.borderColor[0]);
				hash_combine(seed, t.borderColor[1]);
				hash_combine(seed, t.borderColor[2]);
				hash_combine(seed, t.borderColor[3]);
				hash_combine(seed, t.normalizedCoords);
				hash_combine(seed, t.maxAnisotropy);
				hash_combine(seed, t.mipmapFilterMode);
				hash_combine(seed, t.mipmapLevelBias);
				hash_combine(seed, t.minMipmapLevelClamp);
				hash_combine(seed, t.maxMipmapLevelClamp);

				return seed;
			}
		};

		template <>
		struct hash<cudaChannelFormatDesc>
		{
			size_t operator()(cudaChannelFormatDesc const& t) const
			{
				size_t seed = 0;
				hash_combine(seed, t.x);
				hash_combine(seed, t.y);
				hash_combine(seed, t.z);
				hash_combine(seed, t.w);
				hash_combine(seed, t.f);

				return seed;
			}
		};
	}
}

namespace
{
	template <typename Type>
	inline bool StructEqual(const Type &_Left,
		const Type &_Right)
	{
		return memcmp(&_Left, &_Right, sizeof(Type)) == 0;
	}
}

inline bool operator == (const cudaTextureDesc &_Left,
	const cudaTextureDesc &_Right)
{
	return StructEqual(_Left, _Right);
}

inline bool operator == (const cudaChannelFormatDesc &_Left,
	const cudaChannelFormatDesc &_Right)
{
	return StructEqual(_Left, _Right);
}

#endif // !__COMMON_H__
