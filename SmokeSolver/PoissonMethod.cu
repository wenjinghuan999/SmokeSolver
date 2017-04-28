
#include "common.cuh"
#include "PoissonMethod.h"
using namespace ssv;

namespace
{
	using ssv::uint;

	// Red-black Gauss-Seidel
	// LAUNCH : block (ny - 2), thread (nx - 2)
	// p : nx x ny
	// g : nx x ny
	// omega : SOR coefficient
	// redblack : 0 or 1 indicating red or black
	template <typename QType>
	__global__ void kernelGS2d(
		BlobWrapper<QType> q, BlobWrapperConst<QType> g,
		T omega, uint redblack
	)
	{
		uint y = blockIdx.x * 2u + 1u;
		uint x = threadIdx.x + 1u;

		// Red - all cells with (x + y) % 2 == 0
		y += (x & 1) ^ redblack;

		QType v = q(x - 1u, y);
		v += q(x + 1u, y);
		v += q(x, y - 1u);
		v += q(x, y + 1u);

		q(x, y) = omega * (v - g(x, y)) / (T)(4) + ((T)(1) - omega) * q(x, y);
	}

	// Red-black Gauss-Seidel
	// LAUNCH : block (ny - 2, (nz - 2) / 2), thread (nx - 2)
	// p : nx x ny x nz
	// g : nx x ny x nz
	// omega : SOR coefficient
	// redblack : 0 or 1 indicating red or black
	template <typename QType>
	__global__ void kernelGS3d(
		BlobWrapper<QType> q, BlobWrapperConst<QType> g,
		T omega, uint redblack
	)
	{
		uint z = blockIdx.y * 2u + 1u;
		uint y = blockIdx.x + 1u;
		uint x = threadIdx.x + 1u;

		// Red - all cells with (x + y + z) % 2 == 0
		z += ((x + y) & 1) ^ redblack;

		QType v = q(x - 1u, y, z);
		v += q(x + 1u, y, z);
		v += q(x, y - 1u, z);
		v += q(x, y + 1u, z);
		v += q(x, y, z - 1u);
		v += q(x, y, z + 1u);

		q(x, y, z) = omega * (v - g(x, y, z)) / (T)(6) 
			+ ((T)(1) - omega) * q(x, y, z);
	}
}

template<typename QType>
void PoissonMethodGS::operator()(
	Blob<QType>& q, const Blob<QType>& g
	) const
{
	if (q.nz() < 3u)
	{
		for (uint i = 0; i < _iterations; i++)
		{
			kernelGS2d<<<(q.ny() - 2u) / 2u, q.nx() - 2u>>>(
				q.wrapper(), g.wrapper_const(), (T)(1), 0
			);
			kernelGS2d<<<(q.ny() - 2u) / 2u, q.nx() - 2u>>>(
				q.wrapper(), g.wrapper_const(), (T)(1), 1u
			);
		}
	}
	else
	{
		for (uint i = 0; i < _iterations; i++)
		{
			kernelGS3d<<<dim3(q.ny() - 2u, (q.nz() - 2u) / 2u), q.nx() - 2u>>>(
				q.wrapper(), g.wrapper_const(), (T)(1), 0
			);
			kernelGS3d<<<dim3(q.ny() - 2u, (q.nz() - 2u) / 2u), q.nx() - 2u>>>(
				q.wrapper(), g.wrapper_const(), (T)(1), 1u
			);
		}
	}
}

template void PoissonMethodGS::operator()(Blob<T>&, const Blob<T>&) const;
template void PoissonMethodGS::operator()(Blob<T2>&, const Blob<T2>&) const;
template void PoissonMethodGS::operator()(Blob<T4>&, const Blob<T4>&) const;

namespace
{
	using ssv::uint;

	// Down sampling
	// LAUNCH : block (ny), thread (nx)
	// qout :  nx x  ny
	// qin  : 2nx x 2ny
	template <typename QType>
	__global__ void kernelDownSample2d(
		BlobWrapper<QType> qout, BlobWrapperConst<QType> qin
	)
	{
		uint y = blockIdx.x;
		uint x = threadIdx.x;

		qout(x, y) = qin(2u * x, 2u * y) + qin(2u * x + 1u, 2u * y)
			+ qin(2u * x, 2u * y + 1u) + qin(2u * x + 1u, 2u * y + 1u);
	}
	// Down sampling
	// LAUNCH : block (ny, nz), thread (nx)
	// qout :  nx x  ny x  nz
	// qin  : 2nx x 2ny x 2nz
	template <typename QType>
	__global__ void kernelDownSample3d(
		BlobWrapper<QType> qout, BlobWrapperConst<QType> qin
	)
	{
		uint z = blockIdx.y;
		uint y = blockIdx.x;
		uint x = threadIdx.x;

		QType q = qin(2u * x,	2u * y,			2u * z);
		q += qin(2u * x + 1u,	2u * y,			2u * z);
		q += qin(2u * x + 1u,	2u * y + 1u,	2u * z);
		q += qin(2u * x,		2u * y + 1u,	2u * z);
		q += qin(2u * x,		2u * y + 1u,	2u * z + 1u);
		q += qin(2u * x + 1u,	2u * y + 1u,	2u * z + 1u);
		q += qin(2u * x + 1u,	2u * y,			2u * z + 1u);
		q += qin(2u * x,		2u * y,			2u * z + 1u);
		qout(x, y, z) = q;
	}

	// Up sampling
	// LAUNCH : block (ny), thread (nx)
	// qout  : 2nx x 2ny
	// qin   :  nx x  ny
	template <typename QType>
	__global__ void kernelUpSample2d(
		BlobWrapper<QType> qout, BlobWrapperConst<QType> qin
	)
	{
		uint y = blockIdx.x;
		uint x = threadIdx.x;

		QType q = qin(x, y);
		qout(2u * x,		2u * y) = q;
		qout(2u * x + 1u,	2u * y) = q;
		qout(2u * x + 1u,	2u * y + 1u) = q;
		qout(2u * x,		2u * y + 1u) = q;
	}

	// Up sampling
	// LAUNCH : block (ny, nz), thread (nx)
	// qout  : 2nx x 2ny x 2nz
	// qin   :  nx x  ny x  nz
	template <typename QType>
	__global__ void kernelUpSample3d(
		BlobWrapper<QType> qout, BlobWrapperConst<QType> qin
	)
	{
		uint z = blockIdx.y;
		uint y = blockIdx.x;
		uint x = threadIdx.x;

		QType q = qin(x, y);
		qout(2u * x,		2u * y,			2u * z) = q;
		qout(2u * x + 1u,	2u * y,			2u * z) = q;
		qout(2u * x + 1u,	2u * y + 1u,	2u * z) = q;
		qout(2u * x,		2u * y + 1u,	2u * z) = q;
		qout(2u * x,		2u * y + 1u,	2u * z + 1u) = q;
		qout(2u * x + 1u,	2u * y + 1u,	2u * z + 1u) = q;
		qout(2u * x + 1u,	2u * y,			2u * z + 1u) = q;
		qout(2u * x,		2u * y,			2u * z + 1u) = q;
	}

	typename BlobShape _NextShape(
		const BlobShape &shape
	)
	{
		uint nx, ny, nz;
		std::tie(nx, ny, nz) = shape;
		nx /= 2u; if (nx < 3u) nx = 3u;
		ny /= 2u; if (ny < 3u) ny = 3u;
		if (nz >= 3u)
		{
			nz = (nz + 1u) / 2u + 1u; if (nz < 3u) nz = 3u;
		}
		return std::make_tuple(nx, ny, nz);
	}

	template <typename QType>
	void _DownSample(
		Blob<QType> &qout, const Blob<QType> &qin
	)
	{
		assert(qout.shape() == _NextShape(qin.shape()));
		if (qout.nz() == qin.nz())
		{
			kernelDownSample2d<<<qout.ny(), qout.nx()>>>(
				qout.wrapper(), qin.wrapper_const()
			);
		}
		else
		{
			kernelDownSample3d<<<dim3(qout.ny(), qout.nz()), qout.nx()>>>(
				qout.wrapper(), qin.wrapper_const()
			);
		}
	}

	template <typename QType>
	void _UpSample(
		Blob<QType> &qout, const Blob<QType> &qin
	)
	{
		assert(qin.shape() == _NextShape(qout.shape()));
		if (qout.nz() == qin.nz())
		{
			kernelUpSample2d<<<qin.ny(), qin.nx()>>>(
				qout.wrapper(), qin.wrapper_const()
			);
		}
		else
		{
			kernelUpSample3d<<<dim3(qin.ny(), qin.nz()), qin.nx()>>>(
				qout.wrapper(), qin.wrapper_const()
			);
		}
	}
	
	template <typename QType>
	void _VCycle(
		std::vector<Blob<QType> > &buffers_q, std::vector<Blob<QType> > &buffers_g, 
		const PoissonMethodGS &gs, uint level, uint levels
	)
	{
		if (level + 1u < levels)
		{
			_DownSample(buffers_q[level + 1u], buffers_q[level]);
			_DownSample(buffers_g[level + 1u], buffers_g[level]);

			_VCycle(buffers_q, buffers_g, gs, level + 1u, levels);

			_UpSample(buffers_q[level], buffers_q[level + 1]);
		}
		gs(buffers_q[level], buffers_g[level]);
	}
}

template<typename QType>
void PoissonMethodVCycle::operator()(Blob<QType>& q, const Blob<QType>& g) const
{
	static std::vector<Blob<QType> > buffers_q;
	static std::vector<Blob<QType> > buffers_g;

	typename Blob<QType>::shape_t shape = q.shape();
	shape = _NextShape(shape);

	if (buffers_q.empty() || buffers_q.front().shape() != shape)
	{
		buffers_q.clear();
		buffers_g.clear();
		for (uint i = 0; i < _levels; i++)
		{
			buffers_q.emplace_back(shape, q.gpu_device(), false);
			buffers_g.emplace_back(shape, q.gpu_device(), false);
			shape = _NextShape(shape);
		}
	}

	if (_levels > 0)
	{
		_DownSample(buffers_q[0], q);
		_DownSample(buffers_g[0], g);

		_VCycle(buffers_q, buffers_g, _gs, 0, _levels);

		_UpSample(q, buffers_q[0]);
	}
	_gs(q, g);
}

template void PoissonMethodVCycle::operator()<T>(Blob<T>& q, const Blob<T>& g) const;
template void PoissonMethodVCycle::operator()<T2>(Blob<T2>& q, const Blob<T2>& g) const;
template void PoissonMethodVCycle::operator()<T4>(Blob<T4>& q, const Blob<T4>& g) const;
