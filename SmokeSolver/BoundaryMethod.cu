#include "common.cuh"
#include "BoundaryMethod.h"
using namespace ssv;

#include <thrust/transform.h>


template <typename TpType, typename QType, typename OpType>
void BoundaryMethodAll<TpType, QType, OpType>::operator()(
	Blob<QType> &q, const Blob<TpType> &tp
) const
{
	thrust::transform(tp.begin_gpu(), tp.end_gpu(),
	                  q.begin_gpu(), q.begin_gpu(), op_);
}

template class BoundaryMethodAll<byte, real, BoundaryOpClamp<byte, real> >;
template class BoundaryMethodAll<byte, real2, BoundaryOpClamp<byte, real2> >;
template class BoundaryMethodAll<byte, real4, BoundaryOpClamp<byte, real4> >;
template class BoundaryMethodAll<byte, real, BoundaryOpClamp2<byte, real> >;
template class BoundaryMethodAll<byte, real2, BoundaryOpClamp2<byte, real2> >;
template class BoundaryMethodAll<byte, real4, BoundaryOpClamp2<byte, real4> >;
