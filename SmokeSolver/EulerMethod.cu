
#include "common.cuh"
#include "EulerMethod.h"
using namespace ssv;


template <typename QType>
static __global__ void kernelEulerMethodForward(
	BlobWrapper<QType> q, BlobWrapperConst<QType> d
)
{
	size_t y = blockIdx.x;
	size_t x = threadIdx.x;

	q(x, y) += d(x, y);
}

template <typename QType>
void EulerMethodForward<QType>::operator() (
	Blob<QType> &q, const Blob<QType> &d
	) const
{
	kernelEulerMethodForward <<<q.ny(), q.nx()>>>(
		q.wrapper(), d.wrapper_const()
		);
}

template class EulerMethodForward<T>;
template class EulerMethodForward<T2>;
template class EulerMethodForward<T4>;
