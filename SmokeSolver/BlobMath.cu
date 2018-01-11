#include "common.cuh"
#include "BlobMath.h"
using namespace ssv;

#include <helper_math.h>
#include <glm/gtc/noise.hpp>

#include <thrust/transform.h>
using thrust::placeholders::_1;
using thrust::placeholders::_2;


// Element-wise add
template <typename T>
void ssv::add(Blob<T> &qout, const Blob<T> &q1, const Blob<T> &q2)
{
	thrust::transform(q1.begin_gpu(), q1.end_gpu(), q2.begin_gpu(), qout.begin_gpu(), _1 + _2);
}

template <typename T>
void ssv::add(Blob<T> &qout, const Blob<T> &q1, T v)
{
	thrust::transform(q1.begin_gpu(), q1.end_gpu(), qout.begin_gpu(), _1 + v);
}

template <typename T>
Blob<T> &ssv::operator+=(Blob<T> &q1, const Blob<T> &q2)
{
	thrust::transform(q1.begin_gpu(), q1.end_gpu(), q2.begin_gpu(), q1.begin_gpu(), _1 + _2);
	return q1;
}

template <typename T>
Blob<T> &ssv::operator+=(Blob<T> &q1, T v)
{
	thrust::transform(q1.begin_gpu(), q1.end_gpu(), q1.begin_gpu(), _1 + v);
	return q1;
}

template void ssv::add<real>(Blob<real> &, const Blob<real> &, const Blob<real> &);
template void ssv::add<real2>(Blob<real2> &, const Blob<real2> &, const Blob<real2> &);
template void ssv::add<real4>(Blob<real4> &, const Blob<real4> &, const Blob<real4> &);

template void ssv::add<real>(Blob<real> &, const Blob<real> &, real);
template void ssv::add<real2>(Blob<real2> &, const Blob<real2> &, real2);
template void ssv::add<real4>(Blob<real4> &, const Blob<real4> &, real4);

template Blob<real> &ssv::operator+=<real>(Blob<real> &, const Blob<real> &);
template Blob<real2> &ssv::operator+=<real2>(Blob<real2> &, const Blob<real2> &);
template Blob<real4> &ssv::operator+=<real4>(Blob<real4> &, const Blob<real4> &);

template Blob<real> &ssv::operator+=<real>(Blob<real> &, real);
template Blob<real2> &ssv::operator+=<real2>(Blob<real2> &, real2);
template Blob<real4> &ssv::operator+=<real4>(Blob<real4> &, real4);

// Element-wise sub
template <typename T>
void ssv::sub(Blob<T> &qout, const Blob<T> &q1, const Blob<T> &q2)
{
	thrust::transform(q1.begin_gpu(), q1.end_gpu(), q2.begin_gpu(), qout.begin_gpu(), _1 - _2);
}

template <typename T>
void ssv::sub(Blob<T> &qout, const Blob<T> &q1, T v)
{
	thrust::transform(q1.begin_gpu(), q1.end_gpu(), qout.begin_gpu(), _1 - v);
}

template <typename T>
Blob<T> &ssv::operator-=(Blob<T> &q1, const Blob<T> &q2)
{
	thrust::transform(q1.begin_gpu(), q1.end_gpu(), q2.begin_gpu(), q1.begin_gpu(), _1 - _2);
	return q1;
}

template <typename T>
Blob<T> &ssv::operator-=(Blob<T> &q1, T v)
{
	thrust::transform(q1.begin_gpu(), q1.end_gpu(), q1.begin_gpu(), _1 - v);
	return q1;
}

template void ssv::sub<real>(Blob<real> &, const Blob<real> &, const Blob<real> &);
template void ssv::sub<real2>(Blob<real2> &, const Blob<real2> &, const Blob<real2> &);
template void ssv::sub<real4>(Blob<real4> &, const Blob<real4> &, const Blob<real4> &);

template void ssv::sub<real>(Blob<real> &, const Blob<real> &, real);
template void ssv::sub<real2>(Blob<real2> &, const Blob<real2> &, real2);
template void ssv::sub<real4>(Blob<real4> &, const Blob<real4> &, real4);

template Blob<real> &ssv::operator-=<real>(Blob<real> &, const Blob<real> &);
template Blob<real2> &ssv::operator-=<real2>(Blob<real2> &, const Blob<real2> &);
template Blob<real4> &ssv::operator-=<real4>(Blob<real4> &, const Blob<real4> &);

template Blob<real> &ssv::operator-=<real>(Blob<real> &, real);
template Blob<real2> &ssv::operator-=<real2>(Blob<real2> &, real2);
template Blob<real4> &ssv::operator-=<real4>(Blob<real4> &, real4);

template <typename T>
void ssv::mul(Blob<T> &qout, const Blob<T> &q1, const Blob<T> &q2)
{
	thrust::transform(q1.begin_gpu(), q1.end_gpu(), q2.begin_gpu(), qout.begin_gpu(), _1 * _2);
}

template <typename T>
void ssv::mul(Blob<T> &qout, const Blob<T> &q1, T v)
{
	thrust::transform(q1.begin_gpu(), q1.end_gpu(), qout.begin_gpu(), _1 * v);
}

template <typename T>
Blob<T> &ssv::operator*=(Blob<T> &q1, const Blob<T> &q2)
{
	thrust::transform(q1.begin_gpu(), q1.end_gpu(), q2.begin_gpu(), q1.begin_gpu(), _1 * _2);
	return q1;
}

template <typename T>
Blob<T> &ssv::operator*=(Blob<T> &q1, T v)
{
	thrust::transform(q1.begin_gpu(), q1.end_gpu(), q1.begin_gpu(), _1 * v);
	return q1;
}

template void ssv::mul<real>(Blob<real> &, const Blob<real> &, const Blob<real> &);
template void ssv::mul<real2>(Blob<real2> &, const Blob<real2> &, const Blob<real2> &);
template void ssv::mul<real4>(Blob<real4> &, const Blob<real4> &, const Blob<real4> &);

template void ssv::mul<real>(Blob<real> &, const Blob<real> &, real);
template void ssv::mul<real2>(Blob<real2> &, const Blob<real2> &, real2);
template void ssv::mul<real4>(Blob<real4> &, const Blob<real4> &, real4);

template Blob<real> &ssv::operator*=<real>(Blob<real> &, const Blob<real> &);
template Blob<real2> &ssv::operator*=<real2>(Blob<real2> &, const Blob<real2> &);
template Blob<real4> &ssv::operator*=<real4>(Blob<real4> &, const Blob<real4> &);

template Blob<real> &ssv::operator*=<real>(Blob<real> &, real);
template Blob<real2> &ssv::operator*=<real2>(Blob<real2> &, real2);
template Blob<real4> &ssv::operator*=<real4>(Blob<real4> &, real4);

template <typename T>
void ssv::div(Blob<T> &qout, const Blob<T> &q1, const Blob<T> &q2)
{
	thrust::transform(q1.begin_gpu(), q1.end_gpu(), q2.begin_gpu(), qout.begin_gpu(), _1 / _2);
}

template <typename T>
void ssv::div(Blob<T> &qout, const Blob<T> &q1, T v)
{
	thrust::transform(q1.begin_gpu(), q1.end_gpu(), qout.begin_gpu(), _1 / v);
}

template <typename T>
Blob<T> &ssv::operator/=(Blob<T> &q1, const Blob<T> &q2)
{
	thrust::transform(q1.begin_gpu(), q1.end_gpu(), q2.begin_gpu(), q1.begin_gpu(), _1 / _2);
	return q1;
}

template <typename T>
Blob<T> &ssv::operator/=(Blob<T> &q1, T v)
{
	thrust::transform(q1.begin_gpu(), q1.end_gpu(), q1.begin_gpu(), _1 / v);
	return q1;
}

template void ssv::div<real>(Blob<real> &, const Blob<real> &, const Blob<real> &);
template void ssv::div<real2>(Blob<real2> &, const Blob<real2> &, const Blob<real2> &);
template void ssv::div<real4>(Blob<real4> &, const Blob<real4> &, const Blob<real4> &);

template void ssv::div<real>(Blob<real> &, const Blob<real> &, real);
template void ssv::div<real2>(Blob<real2> &, const Blob<real2> &, real2);
template void ssv::div<real4>(Blob<real4> &, const Blob<real4> &, real4);

template Blob<real> &ssv::operator/=<real>(Blob<real> &, const Blob<real> &);
template Blob<real2> &ssv::operator/=<real2>(Blob<real2> &, const Blob<real2> &);
template Blob<real4> &ssv::operator/=<real4>(Blob<real4> &, const Blob<real4> &);

template Blob<real> &ssv::operator/=<real>(Blob<real> &, real);
template Blob<real2> &ssv::operator/=<real2>(Blob<real2> &, real2);
template Blob<real4> &ssv::operator/=<real4>(Blob<real4> &, real4);

template <typename T>
void ssv::neg(Blob<T> &q)
{
	thrust::transform(q.begin_gpu(), q.end_gpu(), q.begin_gpu(), thrust::negate<T>());
}

template void ssv::neg<real>(Blob<real> &);

namespace
{
	template <typename T>
	struct is_negtive
	{
		__host__ __device__

		bool operator()(const T &x) const
		{
			return x < 0;
		}
	};
}

template <typename T>
void ssv::abs(Blob<T> &q)
{
	thrust::transform_if(q.begin_gpu(), q.end_gpu(), q.begin_gpu(), thrust::negate<T>(), is_negtive<T>());
}

template void ssv::abs<real>(Blob<real> &);

namespace
{
	struct op_norm2
	{
		__host__ __device__

		real operator()(const real2 &q) const
		{
			return sqrt(q.x * q.x + q.y * q.y);
		}
	};

	struct op_norm3
	{
		__host__ __device__

		real operator()(const real4 &q) const
		{
			return sqrt(q.x * q.x + q.y * q.y + q.z * q.z);
		}
	};
}

void ssv::norm(Blob<real> &n, const Blob<real2> &q)
{
	transform(q.begin_gpu(), q.end_gpu(), n.begin_gpu(), op_norm2());
}

void ssv::norm(Blob<real> &n, const Blob<real4> &q)
{
	transform(q.begin_gpu(), q.end_gpu(), n.begin_gpu(), op_norm3());
}

namespace
{
	struct op_normalize2
	{
		__host__ __device__

		real2 operator()(const real2 &q) const
		{
			real n = sqrt(q.x * q.x + q.y * q.y);
			real2 a = q;
			a.x /= n;
			a.y /= n;
			return a;
		}
	};

	struct op_normalize3
	{
		__host__ __device__

		real4 operator()(const real4 &q) const
		{
			real n = sqrt(q.x * q.x + q.y * q.y + q.z * q.z);
			real4 a = q;
			a.x /= n;
			a.y /= n;
			a.z /= n;
			a.w = 1.0f;
			return a;
		}
	};
}

void ssv::normalize(Blob<real2> &q)
{
	transform(q.begin_gpu(), q.end_gpu(), q.begin_gpu(), op_normalize2());
}

void ssv::normalize(Blob<real4> &q)
{
	transform(q.begin_gpu(), q.end_gpu(), q.begin_gpu(), op_normalize3());
}

namespace
{
	using ssv::uint;

	// Zip
	// LAUNCH : block (ny, nz), thread (nx)
	// params : nx x ny (x nz)
	__global__ void kernelZip(
		BlobWrapper<real2> qout, BlobWrapperConst<real> qx, BlobWrapperConst<real> qy
	)
	{
		uint z = blockIdx.y;
		uint y = blockIdx.x;
		uint x = threadIdx.x;

		qout(x, y, z) = make_real2(qx(x, y, z), qy(x, y, z));
	}

	// Zip
	// LAUNCH : block (ny, nz), thread (nx)
	// params : nx x ny (x nz)
	__global__ void kernelZip(
		BlobWrapper<real4> qout, BlobWrapperConst<real> qx, BlobWrapperConst<real> qy,
		BlobWrapperConst<real> qz
	)
	{
		uint z = blockIdx.y;
		uint y = blockIdx.x;
		uint x = threadIdx.x;

		qout(x, y, z) = make_real4(qx(x, y, z), qy(x, y, z), qz(x, y, z), 0.f);
	}

	// Zip
	// LAUNCH : block (ny, nz), thread (nx)
	// params : nx x ny (x nz)
	__global__ void kernelZip(
		BlobWrapper<real4> qout, BlobWrapperConst<real> qx, BlobWrapperConst<real> qy,
		BlobWrapperConst<real> qz, BlobWrapperConst<real> qw
	)
	{
		uint z = blockIdx.y;
		uint y = blockIdx.x;
		uint x = threadIdx.x;

		qout(x, y, z) = make_real4(qx(x, y, z), qy(x, y, z), qz(x, y, z), qw(x, y, z));
	}
}

void ssv::zip(Blob<real2> &qout, const Blob<real> &qx, const Blob<real> &qy)
{
	kernelZip<<<dim3(qout.ny(), qout.nz()), qout.nx()>>>(
		qout.wrapper(), qx.wrapper_const(), qy.wrapper_const()
	);
}

void ssv::zip(Blob<real4> &qout, const Blob<real> &qx, const Blob<real> &qy, const Blob<real> &qz)
{
	kernelZip<<<dim3(qout.ny(), qout.nz()), qout.nx()>>>(
		qout.wrapper(), qx.wrapper_const(), qy.wrapper_const(),
		qz.wrapper_const()
	);
}

void ssv::zip(Blob<real4> &qout, const Blob<real> &qx, const Blob<real> &qy, const Blob<real> &qz, const Blob<real> &qw)
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
		BlobWrapper<real> qxout, BlobWrapper<real> qyout, BlobWrapperConst<real2> q
	)
	{
		uint z = blockIdx.y;
		uint y = blockIdx.x;
		uint x = threadIdx.x;

		real2 qq = q(x, y, z);
		qxout(x, y, z) = qq.x;
		qyout(x, y, z) = qq.y;
	}

	// Unzip
	// LAUNCH : block (ny, nz), thread (nx)
	// params : nx x ny (x nz)
	__global__ void kernelUnzip(
		BlobWrapper<real> qxout, BlobWrapper<real> qyout,
		BlobWrapper<real> qzout, BlobWrapperConst<real4> q
	)
	{
		uint z = blockIdx.y;
		uint y = blockIdx.x;
		uint x = threadIdx.x;

		real4 qq = q(x, y, z);
		qxout(x, y, z) = qq.x;
		qyout(x, y, z) = qq.y;
		qzout(x, y, z) = qq.z;
	}

	// Unzip
	// LAUNCH : block (ny, nz), thread (nx)
	// params : nx x ny (x nz)
	__global__ void kernelUnzip(
		BlobWrapper<real> qxout, BlobWrapper<real> qyout,
		BlobWrapper<real> qzout, BlobWrapper<real> qwout, BlobWrapperConst<real4> q
	)
	{
		uint z = blockIdx.y;
		uint y = blockIdx.x;
		uint x = threadIdx.x;

		real4 qq = q(x, y, z);
		qxout(x, y, z) = qq.x;
		qyout(x, y, z) = qq.y;
		qzout(x, y, z) = qq.z;
		qwout(x, y, z) = qq.w;
	}
}

void ssv::unzip(Blob<real> &qxout, Blob<real> &qyout, const Blob<real2> &q)
{
	kernelUnzip<<<dim3(q.ny(), q.nz()), q.nx()>>>(
		qxout.wrapper(), qyout.wrapper(), q.wrapper_const()
	);
}

void ssv::unzip(Blob<real> &qxout, Blob<real> &qyout, Blob<real> &qzout, const Blob<real4> &q)
{
	kernelUnzip<<<dim3(q.ny(), q.nz()), q.nx()>>>(
		qxout.wrapper(), qyout.wrapper(),
		qzout.wrapper(), q.wrapper_const()
	);
}

void ssv::unzip(Blob<real> &qxout, Blob<real> &qyout, Blob<real> &qzout, Blob<real> &qwout, const Blob<real4> &q)
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
		else d(x, y, z) = (q(x + 1u, y, z) - q(x - 1u, y, z)) / static_cast<real>(2);
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
		else d(x, y, z) = (q(x, y + 1u, z) - q(x, y - 1u, z)) / static_cast<real>(2);
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
		else d(x, y, z) = (q(x, y, z + 1u) - q(x, y, z - 1u)) / static_cast<real>(2);
	}
}

template <typename T>
void ssv::diff_x(Blob<T> &d, const Blob<T> &q)
{
	kernelDiffX<<<dim3(q.ny(), q.nz()), q.nx()>>>(
		d.wrapper(), q.wrapper_const()
	);
}

template void ssv::diff_x<real>(Blob<real> &, const Blob<real> &);
template void ssv::diff_x<real2>(Blob<real2> &, const Blob<real2> &);
template void ssv::diff_x<real4>(Blob<real4> &, const Blob<real4> &);

template <typename T>
void ssv::diff_y(Blob<T> &d, const Blob<T> &q)
{
	kernelDiffY<<<dim3(q.ny(), q.nz()), q.nx()>>>(
		d.wrapper(), q.wrapper_const()
	);
}

template void ssv::diff_y<real>(Blob<real> &, const Blob<real> &);
template void ssv::diff_y<real2>(Blob<real2> &, const Blob<real2> &);
template void ssv::diff_y<real4>(Blob<real4> &, const Blob<real4> &);

template <typename T>
void ssv::diff_z(Blob<T> &d, const Blob<T> &q)
{
	kernelDiffZ<<<dim3(q.ny(), q.nz()), q.nx()>>>(
		d.wrapper(), q.wrapper_const()
	);
}

template void ssv::diff_z<real>(Blob<real> &, const Blob<real> &);
template void ssv::diff_z<real2>(Blob<real2> &, const Blob<real2> &);
template void ssv::diff_z<real4>(Blob<real4> &, const Blob<real4> &);


namespace
{
	using ssv::uint;

	// Simple divergence 2D
	// LAUNCH : block (ny), thread (nx)
	// d : nx x ny
	// q : nx x ny
	__global__ void kernelDivergence2D(
		BlobWrapper<real> d, BlobWrapperConst<real2> u
	)
	{
		uint y = blockIdx.x;
		uint x = threadIdx.x;

		real2 dd;
		if (x == 0) dd.x = u(x + 1u, y).x - u(x, y).x;
		else if (x >= u.nx - 1u) dd.x = u(x, y).x - u(x - 1u, y).x;
		else dd.x = (u(x + 1u, y).x - u(x - 1u, y).x) / static_cast<real>(2);

		if (y == 0) dd.y = u(x, y + 1u).y - u(x, y).y;
		else if (y == u.ny - 1u) dd.y = u(x, y).y - u(x, y - 1u).y;
		else dd.y = (u(x, y + 1u).y - u(x, y - 1u).y) / static_cast<real>(2);

		d(x, y) = dd.x + dd.y;
	}

	// Simple divergence 3D
	// LAUNCH : block (ny, nz), thread (nx)
	// d : nx x ny x nz
	// q : nx x ny x nz
	__global__ void kernelDivergence3D(
		BlobWrapper<real> d, BlobWrapperConst<real4> u
	)
	{
		uint z = blockIdx.y;
		uint y = blockIdx.x;
		uint x = threadIdx.x;

		real4 dd;
		if (x == 0) dd.x = u(x + 1u, y, z).x - u(x, y, z).x;
		else if (x == u.nx - 1u) dd.x = u(x, y, z).x - u(x - 1u, y, z).x;
		else dd.x = (u(x + 1u, y, z).x - u(x - 1u, y, z).x) / static_cast<real>(2);

		if (y == 0) dd.y = u(x, y + 1u, z).y - u(x, y, z).y;
		else if (y == u.ny - 1u) dd.y = u(x, y, z).y - u(x, y - 1u, z).y;
		else dd.y = (u(x, y + 1u, z).y - u(x, y - 1u, z).y) / static_cast<real>(2);

		if (z == 0) dd.z = u(x, y, z + 1u).z - u(x, y, z).z;
		else if (z == u.nz - 1u) dd.z = u(x, y, z).z - u(x, y, z - 1u).z;
		else dd.z = (u(x, y, z + 1u).z - u(x, y, z - 1u).z) / static_cast<real>(2);

		d(x, y, z) = dd.x + dd.y + dd.z;
	}
}

// Simple divergence 2D
void ssv::divergence(Blob<real> &d, const Blob<real2> &u)
{
	kernelDivergence2D<<<u.ny(), u.nx()>>>(
		d.wrapper(), u.wrapper_const()
	);
}

// Simple divergence 3D
void ssv::divergence(Blob<real> &d, const Blob<real4> &u)
{
	kernelDivergence3D<<<dim3(u.ny(), u.nz()), u.nx()>>>(
		d.wrapper(), u.wrapper_const()
	);
}

namespace
{
	using ssv::uint;

	// Simple curl 2D
	// LAUNCH : block (ny), thread (nx)
	// d : nx x ny
	// q : nx x ny
	__global__ void kernelCurl2D(
		BlobWrapper<real> c, BlobWrapperConst<real2> q
	)
	{
		uint y = blockIdx.x;
		uint x = threadIdx.x;

		real qypx;
		if (x == 0) qypx = q(x + 1u, y).y - q(x, y).y;
		else if (x == q.nx - 1u) qypx = q(x, y).y - q(x - 1u, y).y;
		else qypx = (q(x + 1u, y).y - q(x - 1u, y).y) / static_cast<real>(2);

		real qxpy;
		if (y == 0) qxpy = q(x, y + 1u).x - q(x, y).x;
		else if (y == q.ny - 1u) qxpy = q(x, y).x - q(x, y - 1u).x;
		else qxpy = (q(x, y + 1u).x - q(x, y - 1u).x) / static_cast<real>(2);

		c(x, y) = qypx - qxpy;
	}

	// Simple curl 3D
	// LAUNCH : block (ny, nz), thread (nx)
	// d : nx x ny x nz
	// q : nx x ny x nz
	__global__ void kernelCurl3D(
		BlobWrapper<real4> c, BlobWrapperConst<real4> q
	)
	{
		uint z = blockIdx.y;
		uint y = blockIdx.x;
		uint x = threadIdx.x;

		real4 qpx;
		if (x == 0) qpx = q(x + 1u, y, z) - q(x, y, z);
		else if (x == q.nx - 1u) qpx = q(x, y, z) - q(x - 1u, y, z);
		else qpx = (q(x + 1u, y, z) - q(x - 1u, y, z)) / static_cast<real>(2);

		real4 qpy;
		if (y == 0) qpy = q(x, y + 1u, z) - q(x, y, z);
		else if (y == q.ny - 1u) qpy = q(x, y, z) - q(x, y - 1u, z);
		else qpy = (q(x, y + 1u, z) - q(x, y - 1u, z)) / static_cast<real>(2);

		real4 qpz;
		if (z == 0) qpz = q(x, y, z + 1u) - q(x, y, z);
		else if (z == q.nz - 1u) qpz = q(x, y, z) - q(x, y, z - 1u);
		else qpz = (q(x, y, z + 1u) - q(x, y, z - 1u)) / static_cast<real>(2);

		real4 cc;
		cc.x = qpy.z - qpz.y;
		cc.y = qpz.x - qpx.z;
		cc.z = qpx.y - qpy.x;
		cc.w = 0;
		c(x, y, z) = cc;
	}
}

// Simple curl 2D
void ssv::curl(Blob<real> &d, const Blob<real2> &q)
{
	kernelCurl2D<<<q.ny(), q.nx()>>>(
		d.wrapper(), q.wrapper_const()
	);
}

// Simple curl 3D
void ssv::curl(Blob<real4> &d, const Blob<real4> &q)
{
	kernelCurl3D<<<dim3(q.ny(), q.nz()), q.nx()>>>(
		d.wrapper(), q.wrapper_const()
	);
}

namespace
{
	using ssv::uint;

	// Simple gradient 2D
	// LAUNCH : block (ny), thread (nx)
	// d : nx x ny
	// q : nx x ny
	__global__ void kernelGradient2D(
		BlobWrapper<real2> g, BlobWrapperConst<real> q
	)
	{
		uint y = blockIdx.x;
		uint x = threadIdx.x;

		real2 gg;
		if (x == 0) gg.x = q(x + 1u, y) - q(x, y);
		else if (x == q.nx - 1u) gg.x = q(x, y) - q(x - 1u, y);
		else gg.x = (q(x + 1u, y) - q(x - 1u, y)) / static_cast<real>(2);

		if (y == 0) gg.y = q(x, y + 1u) - q(x, y);
		else if (y == q.ny - 1u) gg.y = q(x, y) - q(x, y - 1u);
		else gg.y = (q(x, y + 1u) - q(x, y - 1u)) / static_cast<real>(2);

		g(x, y) = gg;
	}

	// Simple gradient 3D
	// LAUNCH : block (ny, nz), thread (nx)
	// d : nx x ny x nz
	// q : nx x ny x nz
	__global__ void kernelGradient3D(
		BlobWrapper<real4> g, BlobWrapperConst<real> q
	)
	{
		uint z = blockIdx.y;
		uint y = blockIdx.x;
		uint x = threadIdx.x;

		real4 gg;
		if (x == 0) gg.x = q(x + 1u, y, z) - q(x, y, z);
		else if (x == q.nx - 1u) gg.x = q(x, y, z) - q(x - 1u, y, z);
		else gg.x = (q(x + 1u, y, z) - q(x - 1u, y, z)) / static_cast<real>(2);

		if (y == 0) gg.y = q(x, y + 1u, z) - q(x, y, z);
		else if (y == q.ny - 1u) gg.y = q(x, y, z) - q(x, y - 1u, z);
		else gg.y = (q(x, y + 1u, z) - q(x, y - 1u, z)) / static_cast<real>(2);

		if (z == 0) gg.z = q(x, y, z + 1u) - q(x, y, z);
		else if (z == q.nz - 1u) gg.z = q(x, y, z) - q(x, y, z - 1u);
		else gg.z = (q(x, y, z + 1u) - q(x, y, z - 1u)) / static_cast<real>(2);

		gg.w = 0;
		g(x, y, z) = gg;
	}
}

// Simple gradient 2D
void ssv::gradient(Blob<real2> &d, const Blob<real> &q)
{
	kernelGradient2D<<<q.ny(), q.nx()>>>(
		d.wrapper(), q.wrapper_const()
	);
}

// Simple gradient 3D
void ssv::gradient(Blob<real4> &d, const Blob<real> &q)
{
	kernelGradient3D<<<dim3(q.ny(), q.nz()), q.nx()>>>(
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
	template <typename T>
	__global__ void kernelLaplacian2D(
		BlobWrapper<T> d, BlobWrapperConst<T> q
	)
	{
		uint y = blockIdx.x + 1u;
		uint x = threadIdx.x + 1u;

		d(x, y) = q(x - 1u, y) + q(x + 1u, y) + q(x, y - 1u) + q(x, y + 1u) - static_cast<real>(4) * q(x, y);
	}

	// Simple Laplacian 3D
	// LAUNCH : block (ny - 2, nz - 2), thread (nx - 2)
	// d : nx x ny x nz
	// q : nx x ny x nz
	template <typename T>
	__global__ void kernelLaplacian3D(
		BlobWrapper<T> d, BlobWrapperConst<T> q
	)
	{
		uint z = blockIdx.y + 1u;
		uint y = blockIdx.x + 1u;
		uint x = threadIdx.x + 1u;

		d(x, y, z) = q(x - 1u, y, z) + q(x + 1u, y, z) + q(x, y - 1u, z)
			+ q(x, y + 1u, z) + q(x, y, z - 1u) + q(x, y, z + 1u)
			- static_cast<real>(6) * q(x, y, z);
	}
}

// Laplacian 2D
template <typename T>
void ssv::laplacian_2d(Blob<T> &d, const Blob<T> &q)
{
	kernelLaplacian2D<<<q.ny() - 2u, q.nx() - 2u>>>(
		d.wrapper(), q.wrapper_const()
	);
}

template void ssv::laplacian_2d<real>(Blob<real> &d, const Blob<real> &q);
template void ssv::laplacian_2d<real2>(Blob<real2> &d, const Blob<real2> &q);
template void ssv::laplacian_2d<real4>(Blob<real4> &d, const Blob<real4> &q);

// Laplacian 3D
template <typename T>
void ssv::laplacian_3d(Blob<T> &d, const Blob<T> &q)
{
	kernelLaplacian3D<<<dim3(q.ny() - 2u, q.nz() - 2u), q.nx() - 2u>>>(
		d.wrapper(), q.wrapper_const()
	);
}

template void ssv::laplacian_3d<real>(Blob<real> &d, const Blob<real> &q);
template void ssv::laplacian_3d<real2>(Blob<real2> &d, const Blob<real2> &q);
template void ssv::laplacian_3d<real4>(Blob<real4> &d, const Blob<real4> &q);

namespace
{
	using ssv::uint;

	// 2D simplex noise
	// LAUNCH : block ny, thread nx
	// q : nx x ny
	__global__ void kernelSimplex2D(
		BlobWrapper<real> q, real2 f, real2 s
	)
	{
		uint y = blockIdx.x;
		uint x = threadIdx.x;

		q(x, y) = simplex(glm::vec2(s.x + x / f.x, s.y + y / f.y));
	}
}

void ssv::simplex_2d(Blob<real> &q, real2 factor, real2 offset)
{
	kernelSimplex2D<<<q.ny(), q.nx()>>>(
		q.wrapper(), factor, offset
	);
}

void ssv::simplex_2d(Blob<real> &q, Blob<real> &dx, Blob<real> &dy, real2 factor, real2 offset)
{
	kernelSimplex2D<<<q.ny(), q.nx()>>>(
		q.wrapper(), factor, offset
	);
	diff_x(dy, q);
	diff_y(dx, q);
	neg(dx);
}
