#pragma once

#include "common.cuh"
#include "BlobMath.h"
using namespace ssv;

#include <helper_math.h>
#include <glm/gtc/noise.hpp>

#include <thrust/transform.h>
using thrust::placeholders::_1;
using thrust::placeholders::_2;


// Element-wise add
template <typename _T>
void ssv::add(Blob<_T> &qout, const Blob<_T> &q1, const Blob<_T> &q2)
{
	thrust::transform(q1.begin_gpu(), q1.end_gpu(), q2.begin_gpu(), qout.begin_gpu(), _1 + _2);
}

template <typename _T>
void ssv::add(Blob<_T> &qout, const Blob<_T> &q1, _T v)
{
	thrust::transform(q1.begin_gpu(), q1.end_gpu(), qout.begin_gpu(), _1 + v);
}

template<typename _T>
Blob<_T> &ssv::operator+=(Blob<_T>& q1, const Blob<_T>& q2)
{
	thrust::transform(q1.begin_gpu(), q1.end_gpu(), q2.begin_gpu(), q1.begin_gpu(), _1 + _2);
	return q1;
}

template<typename _T>
Blob<_T> &ssv::operator+=(Blob<_T>& q1, _T v)
{
	thrust::transform(q1.begin_gpu(), q1.end_gpu(), q1.begin_gpu(), _1 + v);
	return q1;
}

template void ssv::add<T>(Blob<T> &, const Blob<T> &, const Blob<T> &);
template void ssv::add<T2>(Blob<T2> &, const Blob<T2> &, const Blob<T2> &);
template void ssv::add<T4>(Blob<T4> &, const Blob<T4> &, const Blob<T4> &);

template void ssv::add<T>(Blob<T> &, const Blob<T> &, T);
template void ssv::add<T2>(Blob<T2> &, const Blob<T2> &, T2);
template void ssv::add<T4>(Blob<T4> &, const Blob<T4> &, T4);

template Blob<T> &ssv::operator+=<T>(Blob<T> &, const Blob<T> &);
template Blob<T2> &ssv::operator+=<T2>(Blob<T2> &, const Blob<T2> &);
template Blob<T4> &ssv::operator+=<T4>(Blob<T4> &, const Blob<T4> &);

template Blob<T> &ssv::operator+=<T>(Blob<T> &, T);
template Blob<T2> &ssv::operator+=<T2>(Blob<T2> &, T2);
template Blob<T4> &ssv::operator+=<T4>(Blob<T4> &, T4);

// Element-wise sub
template <typename _T>
void ssv::sub(Blob<_T> &qout, const Blob<_T> &q1, const Blob<_T> &q2)
{
	thrust::transform(q1.begin_gpu(), q1.end_gpu(), q2.begin_gpu(), qout.begin_gpu(), _1 - _2);
}

template <typename _T>
void ssv::sub(Blob<_T> &qout, const Blob<_T> &q1, _T v)
{
	thrust::transform(q1.begin_gpu(), q1.end_gpu(), qout.begin_gpu(), _1 - v);
}

template<typename _T>
Blob<_T> &ssv::operator-=(Blob<_T>& q1, const Blob<_T>& q2)
{
	thrust::transform(q1.begin_gpu(), q1.end_gpu(), q2.begin_gpu(), q1.begin_gpu(), _1 - _2);
	return q1;
}

template<typename _T>
Blob<_T> &ssv::operator-=(Blob<_T>& q1, _T v)
{
	thrust::transform(q1.begin_gpu(), q1.end_gpu(), q1.begin_gpu(), _1 - v);
	return q1;
}

template void ssv::sub<T>(Blob<T> &, const Blob<T> &, const Blob<T> &);
template void ssv::sub<T2>(Blob<T2> &, const Blob<T2> &, const Blob<T2> &);
template void ssv::sub<T4>(Blob<T4> &, const Blob<T4> &, const Blob<T4> &);

template void ssv::sub<T>(Blob<T> &, const Blob<T> &, T);
template void ssv::sub<T2>(Blob<T2> &, const Blob<T2> &, T2);
template void ssv::sub<T4>(Blob<T4> &, const Blob<T4> &, T4);

template Blob<T> &ssv::operator-=<T>(Blob<T> &, const Blob<T> &);
template Blob<T2> &ssv::operator-=<T2>(Blob<T2> &, const Blob<T2> &);
template Blob<T4> &ssv::operator-=<T4>(Blob<T4> &, const Blob<T4> &);

template Blob<T> &ssv::operator-=<T>(Blob<T> &, T);
template Blob<T2> &ssv::operator-=<T2>(Blob<T2> &, T2);
template Blob<T4> &ssv::operator-=<T4>(Blob<T4> &, T4);

template<typename _T>
void ssv::mul(Blob<_T>& qout, const Blob<_T>& q1, const Blob<_T>& q2)
{
	thrust::transform(q1.begin_gpu(), q1.end_gpu(), q2.begin_gpu(), qout.begin_gpu(), _1 * _2);
}

template <typename _T>
void ssv::mul(Blob<_T> &qout, const Blob<_T> &q1, _T v)
{
	thrust::transform(q1.begin_gpu(), q1.end_gpu(), qout.begin_gpu(), _1 * v);
}

template<typename _T>
Blob<_T> &ssv::operator*=(Blob<_T>& q1, const Blob<_T>& q2)
{
	thrust::transform(q1.begin_gpu(), q1.end_gpu(), q2.begin_gpu(), q1.begin_gpu(), _1 * _2);
	return q1;
}

template<typename _T>
Blob<_T> &ssv::operator*=(Blob<_T>& q1, _T v)
{
	thrust::transform(q1.begin_gpu(), q1.end_gpu(), q1.begin_gpu(), _1 * v);
	return q1;
}

template void ssv::mul<T>(Blob<T> &, const Blob<T> &, const Blob<T> &);
template void ssv::mul<T2>(Blob<T2> &, const Blob<T2> &, const Blob<T2> &);
template void ssv::mul<T4>(Blob<T4> &, const Blob<T4> &, const Blob<T4> &);

template void ssv::mul<T>(Blob<T> &, const Blob<T> &, T);
template void ssv::mul<T2>(Blob<T2> &, const Blob<T2> &, T2);
template void ssv::mul<T4>(Blob<T4> &, const Blob<T4> &, T4);

template Blob<T> &ssv::operator*=<T>(Blob<T> &, const Blob<T> &);
template Blob<T2> &ssv::operator*=<T2>(Blob<T2> &, const Blob<T2> &);
template Blob<T4> &ssv::operator*=<T4>(Blob<T4> &, const Blob<T4> &);

template Blob<T> &ssv::operator*=<T>(Blob<T> &, T);
template Blob<T2> &ssv::operator*=<T2>(Blob<T2> &, T2);
template Blob<T4> &ssv::operator*=<T4>(Blob<T4> &, T4);

template<typename _T>
void ssv::div(Blob<_T>& qout, const Blob<_T>& q1, const Blob<_T>& q2)
{
	thrust::transform(q1.begin_gpu(), q1.end_gpu(), q2.begin_gpu(), qout.begin_gpu(), _1 / _2);
}

template <typename _T>
void ssv::div(Blob<_T> &qout, const Blob<_T> &q1, _T v)
{
	thrust::transform(q1.begin_gpu(), q1.end_gpu(), qout.begin_gpu(), _1 / v);
}

template<typename _T>
Blob<_T> &ssv::operator/=(Blob<_T>& q1, const Blob<_T>& q2)
{
	thrust::transform(q1.begin_gpu(), q1.end_gpu(), q2.begin_gpu(), q1.begin_gpu(), _1 / _2);
	return q1;
}

template<typename _T>
Blob<_T> &ssv::operator/=(Blob<_T>& q1, _T v)
{
	thrust::transform(q1.begin_gpu(), q1.end_gpu(), q1.begin_gpu(), _1 / v);
	return q1;
}

template void ssv::div<T>(Blob<T> &, const Blob<T> &, const Blob<T> &);
template void ssv::div<T2>(Blob<T2> &, const Blob<T2> &, const Blob<T2> &);
template void ssv::div<T4>(Blob<T4> &, const Blob<T4> &, const Blob<T4> &);

template void ssv::div<T>(Blob<T> &, const Blob<T> &, T);
template void ssv::div<T2>(Blob<T2> &, const Blob<T2> &, T2);
template void ssv::div<T4>(Blob<T4> &, const Blob<T4> &, T4);

template Blob<T> &ssv::operator/=<T>(Blob<T> &, const Blob<T> &);
template Blob<T2> &ssv::operator/=<T2>(Blob<T2> &, const Blob<T2> &);
template Blob<T4> &ssv::operator/=<T4>(Blob<T4> &, const Blob<T4> &);

template Blob<T> &ssv::operator/=<T>(Blob<T> &, T);
template Blob<T2> &ssv::operator/=<T2>(Blob<T2> &, T2);
template Blob<T4> &ssv::operator/=<T4>(Blob<T4> &, T4);

template <typename _T>
void ssv::neg(Blob<_T> &q)
{
	thrust::transform(q.begin_gpu(), q.end_gpu(), q.begin_gpu(), -_1);
}

template void ssv::neg<T>(Blob<T> &);

namespace
{
	using ssv::uint;

	// Zip
	// LAUNCH : block (ny, nz), thread (nx)
	// params : nx x ny (x nz)
	__global__ void kernelZip(
		BlobWrapper<T2> qout, BlobWrapperConst<T> qx, BlobWrapperConst<T> qy
	)
	{
		uint z = blockIdx.y;
		uint y = blockIdx.x;
		uint x = threadIdx.x;

		qout(x, y, z) = make_T2(qx(x, y, z), qy(x, y, z));
	}

	// Zip
	// LAUNCH : block (ny, nz), thread (nx)
	// params : nx x ny (x nz)
	__global__ void kernelZip(
		BlobWrapper<T4> qout, BlobWrapperConst<T> qx, BlobWrapperConst<T> qy,
		BlobWrapperConst<T> qz
	)
	{
		uint z = blockIdx.y;
		uint y = blockIdx.x;
		uint x = threadIdx.x;

		qout(x, y, z) = make_T4(qx(x, y, z), qy(x, y, z), qz(x, y, z), 0.f);
	}

	// Zip
	// LAUNCH : block (ny, nz), thread (nx)
	// params : nx x ny (x nz)
	__global__ void kernelZip(
		BlobWrapper<T4> qout, BlobWrapperConst<T> qx, BlobWrapperConst<T> qy, 
		BlobWrapperConst<T> qz, BlobWrapperConst<T> qw
	)
	{
		uint z = blockIdx.y;
		uint y = blockIdx.x;
		uint x = threadIdx.x;

		qout(x, y, z) = make_T4(qx(x, y, z), qy(x, y, z), qz(x, y, z), qw(x, y, z));
	}
}

void ssv::zip(Blob<T2>& qout, const Blob<T>& qx, const Blob<T>& qy)
{
	kernelZip<<<dim3(qout.ny(), qout.nz()), qout.nx()>>>(
		qout.wrapper(), qx.wrapper_const(), qy.wrapper_const()
		);
}

void ssv::zip(Blob<T4>& qout, const Blob<T>& qx, const Blob<T>& qy, const Blob<T>& qz)
{
	kernelZip<<<dim3(qout.ny(), qout.nz()), qout.nx()>>>(
		qout.wrapper(), qx.wrapper_const(), qy.wrapper_const(),
		qz.wrapper_const()
		);
}

void ssv::zip(Blob<T4>& qout, const Blob<T>& qx, const Blob<T>& qy, const Blob<T>& qz, const Blob<T>& qw)
{
	kernelZip<<<dim3(qout.ny(), qout.nz()), qout.nx()>>>(
		qout.wrapper(), qx.wrapper_const(), qy.wrapper_const(),
		qz.wrapper_const(), qw.wrapper_const()
		);
}

namespace
{
	using ssv::uint;

	// Unzip
	// LAUNCH : block (ny, nz), thread (nx)
	// params : nx x ny (x nz)
	__global__ void kernelUnzip(
		BlobWrapper<T> qxout, BlobWrapper<T> qyout, BlobWrapperConst<T2> q
	)
	{
		uint z = blockIdx.y;
		uint y = blockIdx.x;
		uint x = threadIdx.x;

		T2 qq = q(x, y, z);
		qxout(x, y, z) = qq.x;
		qyout(x, y, z) = qq.y;
	}

	// Unzip
	// LAUNCH : block (ny, nz), thread (nx)
	// params : nx x ny (x nz)
	__global__ void kernelUnzip(
		BlobWrapper<T> qxout, BlobWrapper<T> qyout,
		BlobWrapper<T> qzout, BlobWrapperConst<T4> q
	)
	{
		uint z = blockIdx.y;
		uint y = blockIdx.x;
		uint x = threadIdx.x;

		T4 qq = q(x, y, z);
		qxout(x, y, z) = qq.x;
		qyout(x, y, z) = qq.y;
		qzout(x, y, z) = qq.z;
	}

	// Unzip
	// LAUNCH : block (ny, nz), thread (nx)
	// params : nx x ny (x nz)
	__global__ void kernelUnzip(
		BlobWrapper<T> qxout, BlobWrapper<T> qyout, 
		BlobWrapper<T> qzout, BlobWrapper<T> qwout, BlobWrapperConst<T4> q
	)
	{
		uint z = blockIdx.y;
		uint y = blockIdx.x;
		uint x = threadIdx.x;

		T4 qq = q(x, y, z);
		qxout(x, y, z) = qq.x;
		qyout(x, y, z) = qq.y;
		qzout(x, y, z) = qq.z;
		qwout(x, y, z) = qq.w;
	}

}

void ssv::unzip(Blob<T>& qxout, Blob<T>& qyout, const Blob<T2>& q)
{
	kernelUnzip<<<dim3(q.ny(), q.nz()), q.nx()>>>(
		qxout.wrapper(), qyout.wrapper(), q.wrapper_const()
		);
}

void ssv::unzip(Blob<T>& qxout, Blob<T>& qyout, Blob<T>& qzout, const Blob<T4>& q)
{
	kernelUnzip<<<dim3(q.ny(), q.nz()), q.nx()>>>(
		qxout.wrapper(), qyout.wrapper(), 
		qzout.wrapper(), q.wrapper_const()
		);
}

void ssv::unzip(Blob<T>& qxout, Blob<T>& qyout, Blob<T>& qzout, Blob<T>& qwout, const Blob<T4>& q)
{
	kernelUnzip<<<dim3(q.ny(), q.nz()), q.nx()>>>(
		qxout.wrapper(), qyout.wrapper(), 
		qzout.wrapper(), qwout.wrapper(), q.wrapper_const()
		);
}

namespace
{
	using ssv::uint;

	// Simple partial difference by x
	// LAUNCH : block (ny, nz), thread (nx)
	// d : nx x ny (x nz)
	// q : nx x ny (x nz)
	template <typename QType>
	__global__ void kernelDiffX(
		BlobWrapper<QType> d, BlobWrapperConst<QType> q
	)
	{
		uint z = blockIdx.y;
		uint y = blockIdx.x;
		uint x = threadIdx.x;

		if (x == 0) d(x, y, z) = q(x + 1u, y, z) - q(x, y, z);
		else if (x == q.nx - 1u) d(x, y, z) = q(x, y, z) - q(x - 1u, y, z);
		else d(x, y, z) = (q(x + 1u, y, z) - q(x - 1u, y, z)) / (T)(2);
	}

	// Simple partial difference by y
	// LAUNCH : block (ny, nz), thread (nx)
	// d : nx x ny (x nz)
	// q : nx x ny (x nz)
	template <typename QType>
	__global__ void kernelDiffY(
		BlobWrapper<QType> d, BlobWrapperConst<QType> q
	)
	{
		uint z = blockIdx.y;
		uint y = blockIdx.x;
		uint x = threadIdx.x;

		if (y == 0) d(x, y, z) = q(x, y + 1u, z) - q(x, y, z);
		else if (y == q.ny - 1u) d(x, y, z) = q(x, y, z) - q(x, y - 1u, z);
		else d(x, y, z) = (q(x, y + 1u, z) - q(x, y - 1u, z)) / (T)(2);
	}

	// Simple partial difference by z
	// LAUNCH : block (ny, nz), thread (nx)
	// d : nx x ny x nz
	// q : nx x ny x nz
	template <typename QType>
	__global__ void kernelDiffZ(
		BlobWrapper<QType> d, BlobWrapperConst<QType> q
	)
	{
		uint z = blockIdx.y;
		uint y = blockIdx.x;
		uint x = threadIdx.x;

		if (z == 0) d(x, y, z) = q(x, y, z + 1u) - q(x, y, z);
		else if (z == q.nz - 1u) d(x, y, z) = q(x, y, z) - q(x, y, z - 1u); 
		else d(x, y, z) = (q(x, y, z + 1u) - q(x, y, z - 1u)) / (T)(2);
	}
}

template <typename _T>
void ssv::diff_x(Blob<_T> &d, const Blob<_T> &q)
{
	kernelDiffX<<<dim3(q.ny(), q.nz()), q.nx()>>>(
		d.wrapper(), q.wrapper_const()
		);
}

template void ssv::diff_x<T>(Blob<T> &, const Blob<T> &);
template void ssv::diff_x<T2>(Blob<T2> &, const Blob<T2> &);
template void ssv::diff_x<T4>(Blob<T4> &, const Blob<T4> &);

template <typename _T>
void ssv::diff_y(Blob<_T> &d, const Blob<_T> &q)
{
	kernelDiffY<<<dim3(q.ny(), q.nz()), q.nx()>>>(
		d.wrapper(), q.wrapper_const()
		);
}

template void ssv::diff_y<T>(Blob<T> &, const Blob<T> &);
template void ssv::diff_y<T2>(Blob<T2> &, const Blob<T2> &);
template void ssv::diff_y<T4>(Blob<T4> &, const Blob<T4> &);

template <typename _T>
void ssv::diff_z(Blob<_T> &d, const Blob<_T> &q)
{
	kernelDiffZ<<<dim3(q.ny(), q.nz()), q.nx()>>>(
		d.wrapper(), q.wrapper_const()
		);
}

template void ssv::diff_z<T>(Blob<T> &, const Blob<T> &);
template void ssv::diff_z<T2>(Blob<T2> &, const Blob<T2> &);
template void ssv::diff_z<T4>(Blob<T4> &, const Blob<T4> &);


namespace
{
	using ssv::uint;

	// Simple divergence 2D
	// LAUNCH : block (ny), thread (nx)
	// d : nx x ny
	// q : nx x ny
	__global__ void kernelDivergence2d(
		BlobWrapper<T> d, BlobWrapperConst<T2> u
	)
	{
		uint y = blockIdx.x;
		uint x = threadIdx.x;

		T2 dd;
		if (x == 0) dd.x = u(x + 1u, y).x - u(x, y).x;
		else if (x >= u.nx - 1u) dd.x = u(x, y).x - u(x - 1u, y).x;
		else dd.x = (u(x + 1u, y).x - u(x - 1u, y).x) / (T)(2);

		if (y == 0) dd.y = u(x, y + 1u).y - u(x, y).y;
		else if (y == u.ny - 1u) dd.y = u(x, y).y - u(x, y - 1u).y;
		else dd.y = (u(x, y + 1u).y - u(x, y - 1u).y) / (T)(2);

		d(x, y) = dd.x + dd.y;
	}

	// Simple divergence 3D
	// LAUNCH : block (ny, nz), thread (nx)
	// d : nx x ny x nz
	// q : nx x ny x nz
	__global__ void kernelDivergence3d(
		BlobWrapper<T> d, BlobWrapperConst<T4> u
	)
	{
		uint z = blockIdx.y;
		uint y = blockIdx.x;
		uint x = threadIdx.x;

		T4 dd;
		if (x == 0) dd.x = u(x + 1u, y, z).x - u(x, y, z).x;
		else if (x == u.nx - 1u) dd.x = u(x, y, z).x - u(x - 1u, y, z).x;
		else dd.x = (u(x + 1u, y, z).x - u(x - 1u, y, z).x) / (T)(2);

		if (y == 0) dd.y = u(x, y + 1u, z).y - u(x, y, z).y;
		else if (y == u.ny - 1u) dd.y = u(x, y, z).y - u(x, y - 1u, z).y;
		else dd.y = (u(x, y + 1u, z).y - u(x, y - 1u, z).y) / (T)(2);

		if (z == 0) dd.z = u(x, y, z + 1u).z - u(x, y, z).z;
		else if (z == u.nz - 1u) dd.z = u(x, y, z).z - u(x, y, z - 1u).z;
		else dd.z = (u(x, y, z + 1u).z - u(x, y, z - 1u).z) / (T)(2);

		d(x, y, z) = dd.x + dd.y + dd.z;
	}
}

// Simple divergence 2D
void ssv::divergence(Blob<T> &d, const Blob<T2> &u)
{
	kernelDivergence2d<<<u.ny(), u.nx()>>>(
		d.wrapper(), u.wrapper_const()
		);
}

// Simple divergence 3D
void ssv::divergence(Blob<T> &d, const Blob<T4> &u)
{
	kernelDivergence3d<<<dim3(u.ny(), u.nz()), u.nx()>>>(
		d.wrapper(), u.wrapper_const()
		);
}

namespace
{
	using ssv::uint;

	// Simple gradient 2D
	// LAUNCH : block (ny), thread (nx)
	// d : nx x ny
	// q : nx x ny
	__global__ void kernelGradient2d(
		BlobWrapper<T2> g, BlobWrapperConst<T> q
	)
	{
		uint y = blockIdx.x;
		uint x = threadIdx.x;

		T2 gg;
		if (x == 0) gg.x = q(x + 1u, y) - q(x, y);
		else if (x == q.nx - 1u) gg.x = q(x, y) - q(x - 1u, y);
		else gg.x = (q(x + 1u, y) - q(x - 1u, y)) / (T)(2);

		if (y == 0) gg.y = q(x, y + 1u) - q(x, y);
		else if (y == q.ny - 1u) gg.y = q(x, y) - q(x, y - 1u);
		else gg.y = (q(x, y + 1u) - q(x, y - 1u)) / (T)(2);

		g(x, y) = gg;
	}

	// Simple gradient 3D
	// LAUNCH : block (ny, nz), thread (nx)
	// d : nx x ny x nz
	// q : nx x ny x nz
	__global__ void kernelGradient3d(
		BlobWrapper<T4> g, BlobWrapperConst<T> q
	)
	{
		uint z = blockIdx.y;
		uint y = blockIdx.x;
		uint x = threadIdx.x;

		T4 gg;
		if (x == 0) gg.x = q(x + 1u, y, z) - q(x, y, z);
		else if (x == q.nx - 1u) gg.x = q(x, y, z) - q(x - 1u, y, z);
		else gg.x = (q(x + 1u, y, z) - q(x - 1u, y, z)) / (T)(2);

		if (y == 0) gg.y = q(x, y + 1u, z) - q(x, y, z);
		else if (y == q.ny - 1u) gg.y = q(x, y, z) - q(x, y - 1u, z);
		else gg.y = (q(x, y + 1u, z) - q(x, y - 1u, z)) / (T)(2);

		if (z == 0) gg.z = q(x, y, z + 1u) - q(x, y, z);
		else if (z == q.nz - 1u) gg.z = q(x, y, z) - q(x, y, z - 1u);
		else gg.z = (q(x, y, z + 1u) - q(x, y, z - 1u)) / (T)(2);

		gg.w = 0;
		g(x, y, z) = gg;
	}
}

// Simple gradient 2D
void ssv::gradient(Blob<T2> &d, const Blob<T> &q)
{
	kernelGradient2d<<<q.ny(), q.nx()>>>(
		d.wrapper(), q.wrapper_const()
		);
}

// Simple gradient 3D
void ssv::gradient(Blob<T4> &d, const Blob<T> &q)
{
	kernelGradient3d<<<dim3(q.ny(), q.nz()), q.nx()>>>(
		d.wrapper(), q.wrapper_const()
		);
}

namespace
{
	using ssv::uint;

	// Simple Laplacian 2D
	// LAUNCH : block (ny - 2), thread (nx - 2)
	// d : nx x ny
	// q : nx x ny
	template<typename _T>
	__global__ void kernelLaplacian2d(
		BlobWrapper<_T> d, BlobWrapperConst<_T> q
	)
	{
		uint y = blockIdx.x + 1u;
		uint x = threadIdx.x + 1u;

		d(x, y) = q(x - 1u, y) + q(x + 1u, y) + q(x, y - 1u) + q(x, y + 1u) - (T)(4) * q(x, y);
	}

	// Simple Laplacian 3D
	// LAUNCH : block (ny - 2, nz - 2), thread (nx - 2)
	// d : nx x ny x nz
	// q : nx x ny x nz
	template<typename _T>
	__global__ void kernelLaplacian3d(
		BlobWrapper<_T> d, BlobWrapperConst<_T> q
	)
	{
		uint z = blockIdx.y + 1u;
		uint y = blockIdx.x + 1u;
		uint x = threadIdx.x + 1u;

		d(x, y, z) = q(x - 1u, y, z) + q(x + 1u, y, z) + q(x, y - 1u, z) 
			+ q(x, y + 1u, z) + q(x, y, z - 1u) + q(x, y, z + 1u)
			- (T)(6) * q(x, y, z);
	}
}

// Laplacian 2D
template <typename _T>
void ssv::laplacian2d(Blob<_T> &d, const Blob<_T> &q)
{
	kernelLaplacian2d<<<q.ny() - 2u, q.nx() - 2u>>>(
		d.wrapper(), q.wrapper_const()
		);
}

template void ssv::laplacian2d<T>(Blob<T> &d, const Blob<T> &q);
template void ssv::laplacian2d<T2>(Blob<T2> &d, const Blob<T2> &q);
template void ssv::laplacian2d<T4>(Blob<T4> &d, const Blob<T4> &q);

// Laplacian 3D
template <typename _T>
void ssv::laplacian3d(Blob<_T> &d, const Blob<_T> &q)
{
	kernelLaplacian3d<<<dim3(q.ny() - 2u, q.nz() - 2u), q.nx() - 2u>>>(
		d.wrapper(), q.wrapper_const()
		);
}

template void ssv::laplacian3d<T>(Blob<T> &d, const Blob<T> &q);
template void ssv::laplacian3d<T2>(Blob<T2> &d, const Blob<T2> &q);
template void ssv::laplacian3d<T4>(Blob<T4> &d, const Blob<T4> &q);

namespace
{
	using ssv::uint;

	// 2D simplex noise
	// LAUNCH : block ny, thread nx
	// q : nx x ny
	__global__ void kernelSimplex2d(
		BlobWrapper<T> q, T2 f, T2 s
	)
	{
		uint y = blockIdx.x;
		uint x = threadIdx.x;
		
		q(x, y) = glm::simplex(glm::vec2(s.x + x / f.x, s.y + y / f.y));
	}
}

void ssv::simplex2d(Blob<T> &q, T2 factor, T2 offset)
{
	kernelSimplex2d<<<q.ny(), q.nx()>>>(
		q.wrapper(), factor, offset
		);
}

void ssv::simplex2d(Blob<T> &q, Blob<T> &dx, Blob<T> &dy, T2 factor, T2 offset)
{
	kernelSimplex2d<<<q.ny(), q.nx()>>>(
		q.wrapper(), factor, offset
		);
	diff_x(dy, q);
	diff_y(dx, q);
	neg(dx);
}
