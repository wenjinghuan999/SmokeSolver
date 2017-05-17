
#include "common.cuh"
#include "AdvectMethod.h"
using namespace ssv;

namespace
{
	using ssv::uint;

	// Semi-Lagrangian Advection
	// LAUNCH : block (ny), thread (nx)
	// p : nx x ny
	// q : nx x ny
	// u : nx x ny
	template <typename QType>
	__global__ void kernelAdvectMethodSemiLagrangian(
		BlobWrapper<QType> qout, cudaTextureObject_t q, BlobWrapperConst<T2> u
	)
	{
		uint y = blockIdx.x;
		uint x = threadIdx.x;

		T2 p0 = make_T2(x, y) + (T)(0.5) - u(x, y);

		qout(x, y) = tex2D<QType>(q, p0.x, p0.y);
	}

	// Semi-Lagrangian Advection
	// LAUNCH : block (ny, nz), thread (nx)
	// p : nx x ny x nz
	// q : nx x ny x nz
	// u : nx x ny x nz
	template <typename QType>
	__global__ void kernelAdvectMethodSemiLagrangian(
		BlobWrapper<QType> qout, cudaTextureObject_t q, BlobWrapperConst<T4> u
	)
	{
		uint z = blockIdx.y;
		uint y = blockIdx.x;
		uint x = threadIdx.x;

		T4 p0 = make_float4(x, y, z, 0) + (T)(0.5) - u(x, y, z);

		qout(x, y, z) = tex3D<QType>(q, p0.x, p0.y, p0.z);
	}
}

template <typename QType>
void AdvectMethodSemiLagrangian::operator()(
	Blob<QType> &qout, const Blob<QType> &q, const Blob<T2> &u
	) const
{
	kernelAdvectMethodSemiLagrangian<<<q.ny(), q.nx()>>>(
		qout.wrapper(), q.data_texture_2d(), u.wrapper_const()
		);
}

template void AdvectMethodSemiLagrangian::operator()<T>(
	Blob<T> &, const Blob<T> &, const Blob<T2> &) const;
template void AdvectMethodSemiLagrangian::operator()<T2>(
	Blob<T2> &, const Blob<T2> &, const Blob<T2> &) const;
template void AdvectMethodSemiLagrangian::operator()<T4>(
	Blob<T4> &, const Blob<T4> &, const Blob<T2> &) const;

template <typename QType>
void AdvectMethodSemiLagrangian::operator()(
	Blob<QType> &qout, const Blob<QType> &q, const Blob<T4> &u
	) const
{
	kernelAdvectMethodSemiLagrangian<<<dim3(q.ny(), q.nz()), q.nx()>>>(
		qout.wrapper(), q.data_texture_3d(), u.wrapper_const()
		);
}

template void AdvectMethodSemiLagrangian::operator()<T>(
	Blob<T> &, const Blob<T> &, const Blob<T4> &) const;
template void AdvectMethodSemiLagrangian::operator()<T2>(
	Blob<T2> &, const Blob<T2> &, const Blob<T4> &) const;
template void AdvectMethodSemiLagrangian::operator()<T4>(
	Blob<T4> &, const Blob<T4> &, const Blob<T4> &) const;
