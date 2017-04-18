
#include "common.cuh"
#include "ForceMethod.h"
using namespace ssv;

#include <thrust/transform.h>

namespace
{
	template<typename QType, typename FType>
	struct simple_buoyancy;

	template<typename QType>
	struct simple_buoyancy<QType, T2>
	{
		typedef QType argument_type;
		typedef T2 result_type;
		QType alpha, beta, tm0;
		__host__ __device__ const T2 operator() (
			const QType &rh, const QType &tm) const
		{
			return make_float2(0, -alpha * rh + beta * (tm - tm0));
		}
	};
	template<typename QType>
	struct simple_buoyancy<QType, T4>
	{
		typedef QType argument_type;
		typedef T4 result_type;
		QType alpha, beta, tm0;
		__host__ __device__ const T4 operator() (
			const QType &rh, const QType &tm) const
		{
			return make_float4(0, 0, -alpha * rh + beta * (tm - tm0), 0);
		}
	};
}

template<typename QType, typename FType>
void ForceMethodSimple<QType, FType>::operator()(
	Blob<FType>& fout, const Blob<QType>& rh, const Blob<QType>& tm
	) const
{
	T alpha = 0.03f, beta = 2.5f, tm0 = 0.f;

	thrust::transform(rh.data_gpu_begin(), rh.data_gpu_end(),
		tm.data_gpu_begin(), fout.data_gpu_begin(),
		simple_buoyancy<QType, FType>{alpha, beta, tm0});
}

template class ForceMethodSimple<T, T2>;
template class ForceMethodSimple<T, T4>;
