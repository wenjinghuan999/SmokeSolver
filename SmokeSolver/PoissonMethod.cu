
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
		Blob<QType>::wrapper_t q, Blob<QType>::wrapper_const_t g, 
		T omega, uint redblack
	)
	{
		uint y = blockIdx.x * 2u + 1u;
		uint x = threadIdx.x + 1u;

		// Red - all cells with (x + y) % 2 == 0
		y += (x & 1) ^ redblack;

		T v = 0;
		v += q(x - 1u, y);
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
		Blob<QType>::wrapper_t q, Blob<QType>::wrapper_const_t g, 
		T omega, uint redblack
	)
	{
		uint z = blockIdx.y * 2u + 1u;
		uint y = blockIdx.x + 1u;
		uint x = threadIdx.x + 1u;

		// Red - all cells with (x + y + z) % 2 == 0
		z += ((x + y) & 1) ^ redblack;

		T v = 0;
		v += q(x - 1u, y, z);
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
void PoissonMethodGS<QType>::operator()(
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

template class PoissonMethodGS<T>;



template<typename QType>
void PoissonMethodVCycle<QType>::operator()(Blob<QType>& q, const Blob<QType>& g) const
{
	if (_buffers.empty() || _buffers.front().shape() != q.shape())
	{
		_buffers.clear();

		for (uint i = 0; i < _levels; i++)
		{
			_buffers.emplace_back(q.shape(), q.gpu_device(), false);
		}
	}
}
