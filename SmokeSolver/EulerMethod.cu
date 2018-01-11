#include "common.cuh"
#include "EulerMethod.h"
using namespace ssv;

#include <thrust/transform.h>
using thrust::placeholders::_1;
using thrust::placeholders::_2;

template <typename QType>
void EulerMethodForward::operator()(
	Blob<QType> &q, const Blob<QType> &d
) const
{
	thrust::transform(
		q.begin_gpu(), q.end_gpu(), d.begin_gpu(), q.begin_gpu(),
		_1 + _2
	);
}

template void EulerMethodForward::operator()<real>(Blob<real> &, const Blob<real> &) const;
template void EulerMethodForward::operator()<real2>(Blob<real2> &, const Blob<real2> &) const;
template void EulerMethodForward::operator()<real4>(Blob<real4> &, const Blob<real4> &) const;
