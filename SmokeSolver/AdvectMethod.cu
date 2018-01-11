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
		BlobWrapper<QType> qout, cudaTextureObject_t q, BlobWrapperConst<real2> u
	)
	{
		uint y = blockIdx.x;
		uint x = threadIdx.x;

		real2 p0 = make_real2(x, y) + static_cast<real>(0.5) - u(x, y);

		qout(x, y) = tex2D<QType>(q, p0.x, p0.y);
	}

	// Semi-Lagrangian Advection
	// LAUNCH : block (ny, nz), thread (nx)
	// p : nx x ny x nz
	// q : nx x ny x nz
	// u : nx x ny x nz
	template <typename QType>
	__global__ void kernelAdvectMethodSemiLagrangian(
		BlobWrapper<QType> qout, cudaTextureObject_t q, BlobWrapperConst<real4> u
	)
	{
		uint z = blockIdx.y;
		uint y = blockIdx.x;
		uint x = threadIdx.x;

		real4 p0 = make_float4(x, y, z, 0) + static_cast<real>(0.5) - u(x, y, z);

		qout(x, y, z) = tex3D<QType>(q, p0.x, p0.y, p0.z);
	}
}

template <typename QType>
void AdvectMethodSemiLagrangian::operator()(
	Blob<QType> &qout, const Blob<QType> &q, const Blob<real2> &u
) const
{
	kernelAdvectMethodSemiLagrangian<<<q.ny(), q.nx()>>>(
		qout.wrapper(), q.data_texture_2d(), u.wrapper_const()
	);
}

template void AdvectMethodSemiLagrangian::operator()<real>(
	Blob<real> &, const Blob<real> &, const Blob<real2> &) const;
template void AdvectMethodSemiLagrangian::operator()<real2>(
	Blob<real2> &, const Blob<real2> &, const Blob<real2> &) const;
template void AdvectMethodSemiLagrangian::operator()<real4>(
	Blob<real4> &, const Blob<real4> &, const Blob<real2> &) const;

template <typename QType>
void AdvectMethodSemiLagrangian::operator()(
	Blob<QType> &qout, const Blob<QType> &q, const Blob<real4> &u
) const
{
	kernelAdvectMethodSemiLagrangian<<<dim3(q.ny(), q.nz()), q.nx()>>>(
		qout.wrapper(), q.data_texture_3d(), u.wrapper_const()
	);
}

template void AdvectMethodSemiLagrangian::operator()<real>(
	Blob<real> &, const Blob<real> &, const Blob<real4> &) const;
template void AdvectMethodSemiLagrangian::operator()<real2>(
	Blob<real2> &, const Blob<real2> &, const Blob<real4> &) const;
template void AdvectMethodSemiLagrangian::operator()<real4>(
	Blob<real4> &, const Blob<real4> &, const Blob<real4> &) const;
