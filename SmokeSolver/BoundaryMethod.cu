
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
	thrust::transform(q.begin_gpu(), q.end_gpu(),
		tp.begin_gpu(), q.begin_gpu(), _op);
}

template class BoundaryMethodAll<T, byte, BoundaryOpClamp<T, byte> >;
template class BoundaryMethodAll<T2, byte, BoundaryOpClamp<T2, byte> >;
template class BoundaryMethodAll<T4, byte, BoundaryOpClamp<T4, byte> >;
template class BoundaryMethodAll<T, byte, BoundaryOpClamp2<T, byte> >;
template class BoundaryMethodAll<T2, byte, BoundaryOpClamp2<T2, byte> >;
template class BoundaryMethodAll<T4, byte, BoundaryOpClamp2<T4, byte> >;
