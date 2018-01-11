#include "common.cuh"
#include "ForceMethod.h"
using namespace ssv;

#include <thrust/transform.h>

namespace
{
	template <typename FType>
	struct simple_buoyancy;

	template <>
	struct simple_buoyancy<real2>
	{
		typedef real argument_type;
		typedef real2 result_type;
		real alpha, beta, tm0;
		__host__ __device__ real2 operator()(
			const real &rh, const real &tm) const
		{
			return make_real2(0.f, -alpha * rh + beta * (tm - tm0));
		}
	};

	template <>
	struct simple_buoyancy<real4>
	{
		typedef real argument_type;
		typedef real4 result_type;
		real alpha, beta, tm0;
		__host__ __device__ real4 operator()(
			const real &rh, const real &tm) const
		{
			return make_float4(0, 0, -alpha * rh + beta * (tm - tm0), 0);
		}
	};
}

template <typename FType>
void ForceMethodSimple::operator()(
	Blob<FType> &fout, const Blob<real> &rh, const Blob<real> &tm
) const
{
	thrust::transform(rh.begin_gpu(), rh.end_gpu(),
	                  tm.begin_gpu(), fout.begin_gpu(),
	                  simple_buoyancy<FType>{alpha_, beta_, tm0_});
}

template void ForceMethodSimple::operator()<real2>(
	Blob<real2> &fout, const Blob<real> &rh, const Blob<real> &tm) const;
template void ForceMethodSimple::operator()<real4>(
	Blob<real4> &fout, const Blob<real> &rh, const Blob<real> &tm) const;
