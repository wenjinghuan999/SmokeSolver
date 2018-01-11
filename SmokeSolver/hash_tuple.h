#pragma once

#ifndef __HASH_TUPLE_H__
#define __HASH_TUPLE_H__

#include "common.h"
#include <functional>
#include <tuple>

namespace ssv
{
	namespace hash_tuple
	{
		template <typename T>
		struct hash : std::hash<T>
		{
		};

		// Code from boost
		// Reciprocal of the golden ratio helps spread entropy
		//     and handles duplicates.
		// See Mike Seymour in magic-numbers-in-boosthash-combine:
		//     http://stackoverflow.com/questions/4948780

		template <class T>
		void hash_combine(std::size_t &seed, T const &v)
		{
			seed ^= hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
		}

		// Recursive template code derived from Matthieu M.
		template <class Tuple, size_t Index = std::tuple_size<Tuple>::value - 1>
		struct HashValueImpl
		{
			static void Apply(size_t &seed, Tuple const &tuple)
			{
				HashValueImpl<Tuple, Index - 1>::Apply(seed, tuple);
				hash_combine(seed, std::get<Index>(tuple));
			}
		};

		template <class Tuple>
		struct HashValueImpl<Tuple, 0>
		{
			static void Apply(size_t &seed, Tuple const &tuple)
			{
				hash_combine(seed, std::get<0>(tuple));
			}
		};

		template <typename ... Ts>
		struct hash<std::tuple<Ts...> >
		{
			size_t operator()(std::tuple<Ts...> const &tt) const
			{
				size_t seed = 0;
				HashValueImpl<std::tuple<Ts...> >::Apply(seed, tt);
				return seed;
			}
		};

		template <>
		struct hash<cudaTextureDesc>
		{
			size_t operator()(cudaTextureDesc const &t) const
			{
				size_t seed = 0;
				hash_combine(seed, underlying(t.addressMode[0]));
				hash_combine(seed, underlying(t.addressMode[1]));
				hash_combine(seed, underlying(t.addressMode[2]));
				hash_combine(seed, underlying(t.filterMode));
				hash_combine(seed, underlying(t.readMode));
				hash_combine(seed, t.sRGB);
				hash_combine(seed, t.borderColor[0]);
				hash_combine(seed, t.borderColor[1]);
				hash_combine(seed, t.borderColor[2]);
				hash_combine(seed, t.borderColor[3]);
				hash_combine(seed, t.normalizedCoords);
				hash_combine(seed, t.maxAnisotropy);
				hash_combine(seed, underlying(t.mipmapFilterMode));
				hash_combine(seed, t.mipmapLevelBias);
				hash_combine(seed, t.minMipmapLevelClamp);
				hash_combine(seed, t.maxMipmapLevelClamp);

				return seed;
			}
		};

		template <>
		struct hash<cudaChannelFormatDesc>
		{
			size_t operator()(cudaChannelFormatDesc const &t) const
			{
				size_t seed = 0;
				hash_combine(seed, t.x);
				hash_combine(seed, t.y);
				hash_combine(seed, t.z);
				hash_combine(seed, t.w);
				hash_combine(seed, underlying(t.f));

				return seed;
			}
		};

		template <typename Type>
		bool struct_equal(const Type &left, const Type &right)
		{
			return memcmp(&left, &right, sizeof(Type)) == 0;
		}
	}
}

inline bool operator ==(const cudaTextureDesc &left,
                        const cudaTextureDesc &right)
{
	return ssv::hash_tuple::struct_equal(left, right);
}

inline bool operator ==(const cudaChannelFormatDesc &left,
                        const cudaChannelFormatDesc &right)
{
	return ssv::hash_tuple::struct_equal(left, right);
}

#endif // !__HASH_TUPLE_H__
