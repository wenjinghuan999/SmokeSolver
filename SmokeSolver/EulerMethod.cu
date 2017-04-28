
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
		q.data_gpu_begin(), q.data_gpu_end(), d.data_gpu_begin(), q.data_gpu_begin(),
		_1 + _2
	);
}

template void EulerMethodForward::operator()<T>(Blob<T> &, const Blob<T> &) const;
template void EulerMethodForward::operator()<T2>(Blob<T2> &, const Blob<T2> &) const;
template void EulerMethodForward::operator()<T4>(Blob<T4> &, const Blob<T4> &) const;
