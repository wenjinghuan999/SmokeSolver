#include "common.cuh"
#include "AdvectMethod.h"
#include "BlobMath.h"
using namespace ssv;

#include <thrust/transform.h>
using thrust::placeholders::_1;
using thrust::placeholders::_2;


// ==================== SemiLagrangian ====================


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
		BlobWrapper<QType> qout, cudaTextureObject_t q, BlobWrapperConst<real2> u, int direction = 1
	)
	{
		uint y = blockIdx.x;
		uint x = threadIdx.x;

		real2 p0 = make_real2(x, y) + static_cast<real>(0.5) - direction * u(x, y);

		qout(x, y) = tex2D<QType>(q, p0.x, p0.y);
	}

	// Semi-Lagrangian Advection
	// LAUNCH : block (ny, nz), thread (nx)
	// p : nx x ny x nz
	// q : nx x ny x nz
	// u : nx x ny x nz
	template <typename QType>
	__global__ void kernelAdvectMethodSemiLagrangian(
		BlobWrapper<QType> qout, cudaTextureObject_t q, BlobWrapperConst<real4> u, int direction = 1
	)
	{
		uint z = blockIdx.y;
		uint y = blockIdx.x;
		uint x = threadIdx.x;

		real4 p0 = make_float4(x, y, z, 0) + static_cast<real>(0.5) - direction * u(x, y, z);

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


// ========================= RK 3 =========================


namespace
{
	using ssv::uint;

	// RK3 Advection
	// LAUNCH : block (ny), thread (nx)
	// p : nx x ny
	// q : nx x ny
	// u : nx x ny
	template <typename QType>
	__global__ void kernelAdvectMethodRK3_2D(
		BlobWrapper<QType> qout, cudaTextureObject_t q, cudaTextureObject_t u, int direction = 1
	)
	{
		uint y = blockIdx.x;
		uint x = threadIdx.x;
		
		real2 p1 = make_real2(x, y) + static_cast<real>(0.5);
		real2 u1 = tex2D<real2>(u, p1.x, p1.y);
		real2 p2 = p1 - static_cast<real>(0.5) * direction * u1;
		real2 u2 = tex2D<real2>(u, p2.x, p2.y);
		real2 p3 = p1 - static_cast<real>(0.75) * direction * u2;
		real2 u3 = tex2D<real2>(u, p3.x, p3.y); 
		real c1 = static_cast<real>(2) / 9, c2 = static_cast<real>(3) / 9, c3 = static_cast<real>(4) / 9;
		p1 -=  direction * (c1 * u1 + c2 * u2 + c3 * u3);

		qout(x, y) = tex2D<QType>(q, p1.x, p1.y);
	}

	// RK3 Advection
	// LAUNCH : block (ny, nz), thread (nx)
	// p : nx x ny x nz
	// q : nx x ny x nz
	// u : nx x ny x nz
	template <typename QType>
	__global__ void kernelAdvectMethodRK3_3D(
		BlobWrapper<QType> qout, cudaTextureObject_t q, cudaTextureObject_t u, int direction = 1
	)
	{
		uint z = blockIdx.y;
		uint y = blockIdx.x;
		uint x = threadIdx.x;
		
		real4 p1 = make_float4(x, y, z, 0) + static_cast<real>(0.5);
		real4 u1 = tex3D<real4>(u, p1.x, p1.y, p1.z);
		real4 p2 = p1 - static_cast<real>(0.5) * direction * u1;
		real4 u2 = tex3D<real4>(u, p2.x, p2.y, p2.z);
		real4 p3 = p1 - static_cast<real>(0.75) * direction * u2;
		real4 u3 = tex3D<real4>(u, p3.x, p3.y, p3.z); 
		real c1 = static_cast<real>(2) / 9, c2 = static_cast<real>(3) / 9, c3 = static_cast<real>(4) / 9;
		p1 -=  direction * (c1 * u1 + c2 * u2 + c3 * u3);

		qout(x, y, z) = tex3D<QType>(q, p1.x, p1.y, p1.z);
	}
}

template <typename QType>
void AdvectMethodRK3::operator()(
	Blob<QType> &qout, const Blob<QType> &q, const Blob<real2> &u
) const
{
	kernelAdvectMethodRK3_2D<<<q.ny(), q.nx()>>>(
		qout.wrapper(), q.data_texture_2d(), u.data_texture_2d()
	);
}

template void AdvectMethodRK3::operator()<real>(
	Blob<real> &, const Blob<real> &, const Blob<real2> &) const;
template void AdvectMethodRK3::operator()<real2>(
	Blob<real2> &, const Blob<real2> &, const Blob<real2> &) const;
template void AdvectMethodRK3::operator()<real4>(
	Blob<real4> &, const Blob<real4> &, const Blob<real2> &) const;

template <typename QType>
void AdvectMethodRK3::operator()(
	Blob<QType> &qout, const Blob<QType> &q, const Blob<real4> &u
) const
{
	kernelAdvectMethodRK3_3D<<<dim3(q.ny(), q.nz()), q.nx()>>>(
		qout.wrapper(), q.data_texture_3d(), u.data_texture_3d()
	);
}

template void AdvectMethodRK3::operator()<real>(
	Blob<real> &, const Blob<real> &, const Blob<real4> &) const;
template void AdvectMethodRK3::operator()<real2>(
	Blob<real2> &, const Blob<real2> &, const Blob<real4> &) const;
template void AdvectMethodRK3::operator()<real4>(
	Blob<real4> &, const Blob<real4> &, const Blob<real4> &) const;


// ========================= RK 4 =========================


namespace
{
	using ssv::uint;

	// RK4 Advection
	// LAUNCH : block (ny), thread (nx)
	// p : nx x ny
	// q : nx x ny
	// u : nx x ny
	template <typename QType>
	__global__ void kernelAdvectMethodRK4_2D(
		BlobWrapper<QType> qout, cudaTextureObject_t q, cudaTextureObject_t u, int direction = 1
	)
	{
		uint y = blockIdx.x;
		uint x = threadIdx.x;

		real2 p1 = make_real2(x, y) + static_cast<real>(0.5);
		real2 u1 = tex2D<real2>(u, p1.x, p1.y);
		real2 p2 = p1 - static_cast<real>(0.5) * direction * u1;
		real2 u2 = tex2D<real2>(u, p2.x, p2.y);
		real2 p3 = p1 - static_cast<real>(0.5) * direction * u2;
		real2 u3 = tex2D<real2>(u, p3.x, p3.y);
		real2 p4 = p1 - direction * u3;
		real2 u4 = tex2D<real2>(u, p4.x, p4.y);

		real c1 = static_cast<real>(1) / 6, c2 = static_cast<real>(2) / 6;
		p1 -=  direction * (c1 * u1 + c2 * u2 + c2 * u3 + c1 * u4);

		qout(x, y) = tex2D<QType>(q, p1.x, p1.y);
	}

	// RK4 Advection
	// LAUNCH : block (ny, nz), thread (nx)
	// p : nx x ny x nz
	// q : nx x ny x nz
	// u : nx x ny x nz
	template <typename QType>
	__global__ void kernelAdvectMethodRK4_3D(
		BlobWrapper<QType> qout, cudaTextureObject_t q, cudaTextureObject_t u, int direction = 1
	)
	{
		uint z = blockIdx.y;
		uint y = blockIdx.x;
		uint x = threadIdx.x;

		real4 p1 = make_float4(x, y, z, 0) + static_cast<real>(0.5);
		real4 u1 = tex3D<real4>(u, p1.x, p1.y, p1.z);
		real4 p2 = p1 - static_cast<real>(0.5) * direction * u1;
		real4 u2 = tex3D<real4>(u, p2.x, p2.y, p2.z);
		real4 p3 = p1 - static_cast<real>(0.5) * direction * u2;
		real4 u3 = tex3D<real4>(u, p3.x, p3.y, p3.z);
		real4 p4 = p1 - direction * u3;
		real4 u4 = tex3D<real4>(u, p4.x, p4.y, p4.z);

		real c1 = static_cast<real>(1) / 6, c2 = static_cast<real>(2) / 6;
		p1 -=  direction * (c1 * u1 + c2 * u2 + c2 * u3 + c1 * u4);

		qout(x, y, z) = tex3D<QType>(q, p1.x, p1.y, p1.z);
	}
}

template <typename QType>
void AdvectMethodRK4::operator()(
	Blob<QType> &qout, const Blob<QType> &q, const Blob<real2> &u
) const
{
	kernelAdvectMethodRK4_2D<<<q.ny(), q.nx()>>>(
		qout.wrapper(), q.data_texture_2d(), u.data_texture_2d()
	);
}

template void AdvectMethodRK4::operator()<real>(
	Blob<real> &, const Blob<real> &, const Blob<real2> &) const;
template void AdvectMethodRK4::operator()<real2>(
	Blob<real2> &, const Blob<real2> &, const Blob<real2> &) const;
template void AdvectMethodRK4::operator()<real4>(
	Blob<real4> &, const Blob<real4> &, const Blob<real2> &) const;

template <typename QType>
void AdvectMethodRK4::operator()(
	Blob<QType> &qout, const Blob<QType> &q, const Blob<real4> &u
) const
{
	kernelAdvectMethodRK4_3D<<<dim3(q.ny(), q.nz()), q.nx()>>>(
		qout.wrapper(), q.data_texture_3d(), u.data_texture_3d()
	);
}

template void AdvectMethodRK4::operator()<real>(
	Blob<real> &, const Blob<real> &, const Blob<real4> &) const;
template void AdvectMethodRK4::operator()<real2>(
	Blob<real2> &, const Blob<real2> &, const Blob<real4> &) const;
template void AdvectMethodRK4::operator()<real4>(
	Blob<real4> &, const Blob<real4> &, const Blob<real4> &) const;


// ========================= BFECC ========================


template <typename QType>
void AdvectMethodBFECC::operator()(
	Blob<QType> &qout, const Blob<QType> &q, const Blob<real2> &u
) const
{
	kernelAdvectMethodRK4_2D<<<q.ny(), q.nx()>>>(
		qout.wrapper(), q.data_texture_2d(), u.data_texture_2d()
	);
	kernelAdvectMethodRK4_2D<<<q.ny(), q.nx()>>>(
		qout.wrapper(), qout.data_texture_2d(), u.data_texture_2d(), -1
	);
	thrust::transform(
		q.begin_gpu(), q.end_gpu(), qout.begin_gpu(), qout.begin_gpu(),
		[] __host__ __device__ (const QType &a, const QType &b)
		{
			return a + (a - b) * static_cast<real>(0.5f);
		});
	kernelAdvectMethodRK4_2D<<<q.ny(), q.nx()>>>(
		qout.wrapper(), qout.data_texture_2d(), u.data_texture_2d()
	);
}

template void AdvectMethodBFECC::operator()<real>(
	Blob<real> &, const Blob<real> &, const Blob<real2> &) const;
template void AdvectMethodBFECC::operator()<real2>(
	Blob<real2> &, const Blob<real2> &, const Blob<real2> &) const;
template void AdvectMethodBFECC::operator()<real4>(
	Blob<real4> &, const Blob<real4> &, const Blob<real2> &) const;

template <typename QType>
void AdvectMethodBFECC::operator()(
	Blob<QType> &qout, const Blob<QType> &q, const Blob<real4> &u
) const
{
	kernelAdvectMethodRK4_3D<<<dim3(q.ny(), q.nz()), q.nx()>>>(
		qout.wrapper(), q.data_texture_3d(), u.data_texture_3d()
	);
	kernelAdvectMethodRK4_3D<<<dim3(q.ny(), q.nz()), q.nx()>>>(
		qout.wrapper(), qout.data_texture_3d(), u.data_texture_3d(), -1
	);
	thrust::transform(
		q.begin_gpu(), q.end_gpu(), qout.begin_gpu(), qout.begin_gpu(),
		[] __host__ __device__ (const QType &a, const QType &b)
		{
			return a + (a - b) * static_cast<real>(0.5f);
		});
	kernelAdvectMethodRK4_3D<<<dim3(q.ny(), q.nz()), q.nx()>>>(
		qout.wrapper(), qout.data_texture_3d(), u.data_texture_3d()
	);
}

template void AdvectMethodBFECC::operator()<real>(
	Blob<real> &, const Blob<real> &, const Blob<real4> &) const;
template void AdvectMethodBFECC::operator()<real2>(
	Blob<real2> &, const Blob<real2> &, const Blob<real4> &) const;
template void AdvectMethodBFECC::operator()<real4>(
	Blob<real4> &, const Blob<real4> &, const Blob<real4> &) const;


// ====================== MacCormack ======================


template <typename QType>
void AdvectMethodMacCormack::operator()(
	Blob<QType> &qout, const Blob<QType> &q, const Blob<real2> &u
) const
{
	kernelAdvectMethodRK4_2D<<<q.ny(), q.nx()>>>(
		qout.wrapper(), q.data_texture_2d(), u.data_texture_2d()
	);
	Blob<QType> q1(q.shape(), q.gpu_device(), Blob<QType>::storage_t::GPU);
	kernelAdvectMethodRK4_2D<<<q.ny(), q.nx()>>>(
		q1.wrapper(), qout.data_texture_2d(), u.data_texture_2d(), -1
	);
	thrust::transform(
		q.begin_gpu(), q.end_gpu(), q1.begin_gpu(), q1.begin_gpu(),
		[] __host__ __device__ (const QType &a, const QType &b)
		{
			return (a - b) * static_cast<real>(0.5f);
		});
	qout += q1;
}

template void AdvectMethodMacCormack::operator()<real>(
	Blob<real> &, const Blob<real> &, const Blob<real2> &) const;
template void AdvectMethodMacCormack::operator()<real2>(
	Blob<real2> &, const Blob<real2> &, const Blob<real2> &) const;
template void AdvectMethodMacCormack::operator()<real4>(
	Blob<real4> &, const Blob<real4> &, const Blob<real2> &) const;

template <typename QType>
void AdvectMethodMacCormack::operator()(
	Blob<QType> &qout, const Blob<QType> &q, const Blob<real4> &u
) const
{
	kernelAdvectMethodRK4_3D<<<dim3(q.ny(), q.nz()), q.nx()>>>(
		qout.wrapper(), q.data_texture_3d(), u.data_texture_3d()
	);
	Blob<QType> q1(q.shape(), q.gpu_device(), Blob<QType>::storage_t::GPU);
	kernelAdvectMethodRK4_3D<<<dim3(q.ny(), q.nz()), q.nx()>>>(
		q1.wrapper(), qout.data_texture_3d(), u.data_texture_3d(), -1
	);
	thrust::transform(
		q.begin_gpu(), q.end_gpu(), q1.begin_gpu(), qout.begin_gpu(),
		[] __host__ __device__ (const QType &a, const QType &b)
		{
			return (a - b) * static_cast<real>(0.5f);
		});
	qout += q1;
}

template void AdvectMethodMacCormack::operator()<real>(
	Blob<real> &, const Blob<real> &, const Blob<real4> &) const;
template void AdvectMethodMacCormack::operator()<real2>(
	Blob<real2> &, const Blob<real2> &, const Blob<real4> &) const;
template void AdvectMethodMacCormack::operator()<real4>(
	Blob<real4> &, const Blob<real4> &, const Blob<real4> &) const;
