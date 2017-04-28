
#include "common.cuh"
#include "unary_ops.cuh"
#include "BoundaryMethod.h"
using namespace ssv;

#include <thrust/transform.h>


template<typename QType, typename TpType, typename OpType>
void BoundaryMethodAll<QType, TpType, OpType>::operator() (
	Blob<QType> &q, const Blob<TpType> &tp
	) const
{
	thrust::transform(q.data_gpu_begin(), q.data_gpu_end(),
		tp.data_gpu_begin(), q.data_gpu_begin(), _op);
}

template class BoundaryMethodAll<T, byte, BoundaryOpClamp<T, byte> >;
template class BoundaryMethodAll<T2, byte, BoundaryOpClamp<T2, byte> >;
template class BoundaryMethodAll<T4, byte, BoundaryOpClamp<T4, byte> >;
