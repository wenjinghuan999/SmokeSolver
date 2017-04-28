
#include "common.cuh"
#include "ForceMethod.h"
using namespace ssv;

#include <thrust/transform.h>

namespace
{
	template<typename FType>
	struct simple_buoyancy;

	template<>
	struct simple_buoyancy<T2>
	{
		typedef T argument_type;
		typedef T2 result_type;
		T alpha, beta, tm0;
		__host__ __device__ const T2 operator() (
			const T &rh, const T &tm) const
		{
			return make_float2(0, -alpha * rh + beta * (tm - tm0));
		}
	};
	template<>
	struct simple_buoyancy<T4>
	{
		typedef T argument_type;
		typedef T4 result_type;
		T alpha, beta, tm0;
		__host__ __device__ const T4 operator() (
			const T &rh, const T &tm) const
		{
			return make_float4(0, 0, -alpha * rh + beta * (tm - tm0), 0);
		}
	};
}

template<typename FType>
void ForceMethodSimple::operator()<FType>(
	Blob<FType>& fout, const Blob<T>& rh, const Blob<T>& tm
	) const
{
	thrust::transform(rh.begin_gpu(), rh.end_gpu(),
		tm.begin_gpu(), fout.begin_gpu(),
		simple_buoyancy<FType>{_alpha, _beta, _tm0});
}

template void ForceMethodSimple::operator()<T2>(
	Blob<T2>& fout, const Blob<T>& rh, const Blob<T>& tm) const;
template void ForceMethodSimple::operator()<T4>(
	Blob<T4>& fout, const Blob<T>& rh, const Blob<T>& tm) const;
