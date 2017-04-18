
#include "common.cuh"
#include "EulerMethod.h"
using namespace ssv;

#include <thrust/transform.h>
using thrust::placeholders::_1;
using thrust::placeholders::_2;

template <typename QType>
void EulerMethodForward<QType>::operator() (
	Blob<QType> &q, const Blob<QType> &d
	) const
{
	thrust::transform(
		q.data_gpu_begin(), q.data_gpu_end(), d.data_gpu_begin(), q.data_gpu_begin(),
		_1 + _2
	);
}

template class EulerMethodForward<T>;
template class EulerMethodForward<T2>;
template class EulerMethodForward<T4>;
