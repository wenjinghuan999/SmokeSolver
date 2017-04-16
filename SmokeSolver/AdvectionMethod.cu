
#include "common.cuh"
#include "AdvectionMethod.h"
using namespace ssv;


template <typename QType>
static __global__ void kernelAdvectionMethodSemiLagrangian(
	BlobWrapper<QType> qout, cudaTextureObject_t q, BlobWrapperConst<T2> u
)
{
	size_t y = blockIdx.x;
	size_t x = threadIdx.x;

	T2 p0 = make_float2(x, y) + 0.5f - u(x, y);

	qout(x, y) = tex2D<QType>(q, p0.x, p0.y);
}

template <typename QType>
static __global__ void kernelAdvectionMethodSemiLagrangian(
	BlobWrapper<QType> qout, cudaTextureObject_t q, BlobWrapperConst<T4> u
)
{
	size_t z = blockIdx.y;
	size_t y = blockIdx.x;
	size_t x = threadIdx.x;

	T4 p0 = make_float4(x, y, z, 0) + 0.5f - u(x, y, z);

	qout(x, y, z) = tex3D<QType>(q, p0.x, p0.y, p0.z);
}


template <typename QType>
void AdvectionMethodSemiLagrangian<QType>::operator () (
	Blob<QType> &qout, const Blob<QType> &q, const Blob<T2> &u
	) const
{
	kernelAdvectionMethodSemiLagrangian<<<q.ny(), q.nx()>>>(
		qout.wrapper(), q.data_texture_2d(), u.wrapper_const()
	);
}

template <typename QType>
void AdvectionMethodSemiLagrangian<QType>::operator () (
	Blob<QType> &qout, const Blob<QType> &q, const Blob<T4> &u
	) const
{
	kernelAdvectionMethodSemiLagrangian<<<q.ny(), q.nx()>>>(
		qout.wrapper(), q.data_texture_2d(), u.wrapper_const()
		);
}

template class AdvectionMethodSemiLagrangian<T>;
template class AdvectionMethodSemiLagrangian<T2>;
template class AdvectionMethodSemiLagrangian<T4>;
