
#include "common.cuh"
#include "unary_ops.cuh"
#include "BoundaryMethod.h"
using namespace ssv;

#include <thrust/transform.h>


template <typename TpType, typename QType>
void BoundaryMethodClampAll<TpType, QType>::operator() (
	const Blob<TpType> &tp, Blob<QType> &q
	) const
{
	BoundaryOpClamp<TpType, QType> op(tp1, q1);
	thrust::transform(tp.data_gpu_begin(), tp.data_gpu_end(),
		q.data_gpu_begin(), q.data_gpu_begin(), op);
}

template class BoundaryMethodClampAll<byte, T>;
template class BoundaryMethodClampAll<byte, T2>;
template class BoundaryMethodClampAll<byte, T4>;
