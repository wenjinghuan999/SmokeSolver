
#include "common.cuh"
#include "AdvectMethod.h"
using namespace ssv;


// Semi-Lagrangian Advection
// LAUNCH : block (ny), thread (nx)
// p : nx x ny
// q : nx x ny
// u : nx x ny
template <typename QType>
static __global__ void kernelAdvectMethodSemiLagrangian(
	BlobWrapper<QType> qout, cudaTextureObject_t q, BlobWrapperConst<T2> u
)
{
	size_t y = blockIdx.x;
	size_t x = threadIdx.x;

	T2 p0 = make_float2(x, y) + 0.5f - u(x, y);

	qout(x, y) = tex2D<QType>(q, p0.x, p0.y);
}

// Semi-Lagrangian Advection
// LAUNCH : block (ny, nz), thread (nx)
// p : nx x ny x nz
// q : nx x ny x nz
// u : nx x ny x nz
template <typename QType>
static __global__ void kernelAdvectMethodSemiLagrangian(
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
void AdvectMethodSemiLagrangian<QType>::operator () (
	Blob<QType> &qout, const Blob<QType> &q, const Blob<T2> &u
	) const
{
	kernelAdvectMethodSemiLagrangian<<<q.ny(), q.nx()>>>(
		qout.wrapper(), q.data_texture_2d(), u.wrapper_const()
	);
}

template <typename QType>
void AdvectMethodSemiLagrangian<QType>::operator () (
	Blob<QType> &qout, const Blob<QType> &q, const Blob<T4> &u
	) const
{
	kernelAdvectMethodSemiLagrangian<<<dim3(q.ny(), q.nz()), q.nx()>>>(
		qout.wrapper(), q.data_texture_3d(), u.wrapper_const()
		);
}

template class AdvectMethodSemiLagrangian<T>;
template class AdvectMethodSemiLagrangian<T2>;
template class AdvectMethodSemiLagrangian<T4>;
