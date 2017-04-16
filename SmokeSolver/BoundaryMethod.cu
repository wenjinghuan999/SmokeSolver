
#include "common.cuh"
#include "unary_ops.cuh"
#include "BoundaryMethod.h"
using namespace ssv;

#include <thrust/transform.h>
using thrust::placeholders::_1;


template <typename QType, typename TpType>
void BoundaryMethodClamp<QType, TpType>::operator() (
	Blob<QType> &q, const Blob<TpType> &tp, TpType tp1, QType q1
	) const
{
	thrust::transform_if(
		q.data_gpu_begin(), q.data_gpu_end(), tp.data_gpu_begin(), q.data_gpu_begin(), 
		ops::assign<QType>(q1), _1 == tp1
	);
}

template <typename QType, typename TpType>
void BoundaryMethodClamp<QType, TpType>::operator() (
	Blob<QType> &q, const Blob<TpType> &tp, TpType tp1, QType q1, TpType tp2, QType q2
	) const
{
	thrust::transform_if(
		q.data_gpu_begin(), q.data_gpu_end(), tp.data_gpu_begin(), q.data_gpu_begin(),
		ops::assign<QType>(q1), _1 == tp1
	);
	thrust::transform_if(
		q.data_gpu_begin(), q.data_gpu_end(), tp.data_gpu_begin(), q.data_gpu_begin(),
		ops::assign<QType>(q2), _1 == tp2
	);
}

template class BoundaryMethodClamp<T, byte>;
template class BoundaryMethodClamp<T2, byte>;
template class BoundaryMethodClamp<T4, byte>;
