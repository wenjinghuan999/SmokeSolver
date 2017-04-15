
#include "common.cuh"
#include "AdvectionMethod.h"
using namespace ssv;


template <typename QType>
static __global__ void kernelAdvectionMethod2dSemiLagrangian(
	BlobWrapper<QType> qout, cudaTextureObject_t q, BlobWrapperConst<T2> u
)
{
	size_t y = blockIdx.x;
	size_t x = threadIdx.x;

	T x0 = (T)(x) + 0.5f - u(x, y).x;
	T y0 = (T)(y) + 0.5f - u(x, y).y;

	qout(x, y) = tex2D<QType>(q, x0, y0);
}

template <typename QType>
void AdvectionMethod2dSemiLagrangian<QType>::operator () (
	Blob<QType> &qout, const Blob<QType> &q, const Blob<T2> &u
	) const
{
	kernelAdvectionMethod2dSemiLagrangian<<<q.ny(), q.nx()>>>(
		qout.helper(), q.data_texture_2d(), u.helper_const()
	);
}

template class AdvectionMethod2dSemiLagrangian<T>;
template class AdvectionMethod2dSemiLagrangian<T2>;
template class AdvectionMethod2dSemiLagrangian<T4>;
