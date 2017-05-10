
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
				q.wrapper(), g.wrapper_const(), _omega, 0
			);
			kernelGS2d<<<(q.ny() - 2u) / 2u, q.nx() - 2u>>>(
				q.wrapper(), g.wrapper_const(), _omega, 1u
			);
		}
	}
	else
	{
		for (uint i = 0; i < _iterations; i++)
		{
			kernelGS3d<<<dim3(q.ny() - 2u, (q.nz() - 2u) / 2u), q.nx() - 2u>>>(
				q.wrapper(), g.wrapper_const(), _omega, 0
			);
			kernelGS3d<<<dim3(q.ny() - 2u, (q.nz() - 2u) / 2u), q.nx() - 2u>>>(
				q.wrapper(), g.wrapper_const(), _omega, 1u
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
			buffers_q.emplace_back(shape, q.gpu_device(), Blob<QType>::storage_t::GPU);
			buffers_g.emplace_back(shape, q.gpu_device(), Blob<QType>::storage_t::GPU);
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

namespace
{
	using ssv::uint;

	void PCGLaplacianInit(int *row_ptr, int *col_ind, T *val, int Nx, int Ny, int Nz)
	{
		int N = Nx * Ny * Nz;
		int Nxy = Nx * Ny;
		int idx = 0;

		// loop over degrees of freedom
		for (int i = 0; i < N; i++)
		{
			int ix = i % Nx;
			int iy = (i % Nxy) / Nx;
			int iz = i / Nxy;

			row_ptr[i] = idx;

			// down
			if (iz > 0)
			{
				val[idx] = 1.0;
				col_ind[idx] = i - Nxy;
				idx++;
			}

			// front
			if (iy > 0)
			{
				val[idx] = 1.0;
				col_ind[idx] = i - Nx;
				idx++;
			}

			// left
			if (ix > 0)
			{
				val[idx] = 1.0;
				col_ind[idx] = i - 1;
				idx++;
			}

			// center
			val[idx] = -6.0;
			col_ind[idx] = i;
			idx++;

			//right
			if (ix < Nx - 1)
			{
				val[idx] = 1.0;
				col_ind[idx] = i + 1;
				idx++;
			}

			// back
			if (iy < Ny - 1)
			{
				val[idx] = 1.0;
				col_ind[idx] = i + Nx;
				idx++;
			}

			// up
			if (iz < Nz - 1)
			{
				val[idx] = 1.0;
				col_ind[idx] = i + Nxy;
				idx++;
			}

		}

		row_ptr[N] = idx;
	}
}

void PoissonMethodCG::_Init()
{
	/* Create CUBLAS & CUSPARSE context */
	_cublasHandle = 0;
	checkCudaErrorAndThrow(cublasCreate(&_cublasHandle), error_t::SSV_ERROR_UNKNOWN);
	_cusparseHandle = 0;
	checkCudaErrorAndThrow(cusparseCreate(&_cusparseHandle), error_t::SSV_ERROR_UNKNOWN);
}

void PoissonMethodCG::_PCGInit(const BlobShape &shape) const
{
	uint nx, ny, nz;
	std::tie(nx, ny, nz) = shape;
	_buffer_shape = shape;
	_nnz = 7u * nx * ny * nz - 2u * (nx * ny + ny * nz + nz * nx);
	_nall = nx * ny * nz;
	int *h_row = new int[_nall + 1u];
	int *h_col = new int[_nnz];
	float *h_val = new float[_nnz];

	PCGLaplacianInit(h_row, h_col, h_val, nx, ny, nz);

	/* Allocate required memory */
	checkCudaErrorAndThrow(cudaMalloc((void **)&_d_col, _nnz * sizeof(int)),
		error_t::SSV_ERROR_OUT_OF_MEMORY_GPU);
	checkCudaErrorAndThrow(cudaMalloc((void **)&_d_row, (_nall + 1) * sizeof(int)),
		error_t::SSV_ERROR_OUT_OF_MEMORY_GPU);
	checkCudaErrorAndThrow(cudaMalloc((void **)&_d_val, _nnz * sizeof(float)),
		error_t::SSV_ERROR_OUT_OF_MEMORY_GPU);
	checkCudaErrorAndThrow(cudaMalloc((void **)&_d_y, _nall * sizeof(float)),
		error_t::SSV_ERROR_OUT_OF_MEMORY_GPU);
	checkCudaErrorAndThrow(cudaMalloc((void **)&_d_p, _nall * sizeof(float)),
		error_t::SSV_ERROR_OUT_OF_MEMORY_GPU);
	checkCudaErrorAndThrow(cudaMalloc((void **)&_d_omega, _nall * sizeof(float)),
		error_t::SSV_ERROR_OUT_OF_MEMORY_GPU);
	checkCudaErrorAndThrow(cudaMalloc((void **)&_d_x, _nall * sizeof(float)),
		error_t::SSV_ERROR_OUT_OF_MEMORY_GPU);
	checkCudaErrorAndThrow(cudaMalloc((void **)&_d_r, _nall * sizeof(float)),
		error_t::SSV_ERROR_OUT_OF_MEMORY_GPU);

	checkCudaErrorAndThrow(cudaMemcpy(_d_col, h_col, _nnz * sizeof(int), cudaMemcpyHostToDevice),
		error_t::SSV_ERROR_UNKNOWN);
	checkCudaErrorAndThrow(cudaMemcpy(_d_row, h_row, (_nall + 1) * sizeof(int), cudaMemcpyHostToDevice),
		error_t::SSV_ERROR_UNKNOWN);
	checkCudaErrorAndThrow(cudaMemcpy(_d_val, h_val, _nnz * sizeof(float), cudaMemcpyHostToDevice),
		error_t::SSV_ERROR_UNKNOWN);

	delete[] h_col;
	delete[] h_row;
	delete[] h_val;

	checkCudaErrorAndThrow(cudaMalloc((void **)&_d_valsILU0, _nnz * sizeof(float)),
		error_t::SSV_ERROR_OUT_OF_MEMORY_GPU);
	checkCudaErrorAndThrow(cudaMalloc((void **)&_d_zm1, _nall * sizeof(float)),
		error_t::SSV_ERROR_OUT_OF_MEMORY_GPU);
	checkCudaErrorAndThrow(cudaMalloc((void **)&_d_zm2, _nall * sizeof(float)),
		error_t::SSV_ERROR_OUT_OF_MEMORY_GPU);
	checkCudaErrorAndThrow(cudaMalloc((void **)&_d_rm2, _nall * sizeof(float)),
		error_t::SSV_ERROR_OUT_OF_MEMORY_GPU);

	/* Description of the matrices*/
	_descrA = 0;
	checkCudaErrorAndThrow(cusparseCreateMatDescr(&_descrA),
		error_t::SSV_ERROR_UNKNOWN);
	checkCudaErrorAndThrow(cusparseSetMatType(_descrA, CUSPARSE_MATRIX_TYPE_GENERAL),
		error_t::SSV_ERROR_UNKNOWN);
	checkCudaErrorAndThrow(cusparseSetMatIndexBase(_descrA, CUSPARSE_INDEX_BASE_ZERO),
		error_t::SSV_ERROR_UNKNOWN);

	_descrL = 0;
	checkCudaErrorAndThrow(cusparseCreateMatDescr(&_descrL),
		error_t::SSV_ERROR_UNKNOWN);
	checkCudaErrorAndThrow(cusparseSetMatType(_descrL, CUSPARSE_MATRIX_TYPE_GENERAL),
		error_t::SSV_ERROR_UNKNOWN);
	checkCudaErrorAndThrow(cusparseSetMatIndexBase(_descrL, CUSPARSE_INDEX_BASE_ZERO),
		error_t::SSV_ERROR_UNKNOWN);
	checkCudaErrorAndThrow(cusparseSetMatFillMode(_descrL, CUSPARSE_FILL_MODE_LOWER),
		error_t::SSV_ERROR_UNKNOWN);
	checkCudaErrorAndThrow(cusparseSetMatDiagType(_descrL, CUSPARSE_DIAG_TYPE_UNIT),
		error_t::SSV_ERROR_UNKNOWN);

	_descrU = 0;
	checkCudaErrorAndThrow(cusparseCreateMatDescr(&_descrU),
		error_t::SSV_ERROR_UNKNOWN);
	checkCudaErrorAndThrow(cusparseSetMatType(_descrU, CUSPARSE_MATRIX_TYPE_GENERAL),
		error_t::SSV_ERROR_UNKNOWN);
	checkCudaErrorAndThrow(cusparseSetMatIndexBase(_descrU, CUSPARSE_INDEX_BASE_ZERO),
		error_t::SSV_ERROR_UNKNOWN);
	checkCudaErrorAndThrow(cusparseSetMatFillMode(_descrU, CUSPARSE_FILL_MODE_UPPER),
		error_t::SSV_ERROR_UNKNOWN);
	checkCudaErrorAndThrow(cusparseSetMatDiagType(_descrU, CUSPARSE_DIAG_TYPE_NON_UNIT),
		error_t::SSV_ERROR_UNKNOWN);

	/* create the analysis info object for the A matrix */
	_infoA = 0;
	checkCudaErrorAndThrow(cusparseCreateSolveAnalysisInfo(&_infoA),
		error_t::SSV_ERROR_UNKNOWN);

	/* Create info objects for the ILU0 preconditioner */
	_infoU = 0;
	checkCudaErrorAndThrow(cusparseCreateSolveAnalysisInfo(&_infoU),
		error_t::SSV_ERROR_UNKNOWN);

	/* Perform the analysis for the Non-Transpose case */
	checkCudaErrorAndThrow(cusparseScsrsv_analysis(_cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
		(int)(_nall), (int)(_nnz), _descrA, _d_val, _d_row, _d_col, _infoA),
		error_t::SSV_ERROR_UNKNOWN);

	/* Copy A data to ILU0 vals as input*/
	checkCudaErrorAndThrow(cudaMemcpy(_d_valsILU0, _d_val, _nnz * sizeof(float), cudaMemcpyDeviceToDevice),
		error_t::SSV_ERROR_UNKNOWN);

	/* generate the Incomplete LU factor H for the matrix A using cudsparseScsrilu0 */
	checkCudaErrorAndThrow(cusparseScsrilu0(_cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
		(int)(_nall), _descrA, _d_valsILU0, _d_row, _d_col, _infoA),
		error_t::SSV_ERROR_UNKNOWN);

	checkCudaErrorAndThrow(cusparseScsrsv_analysis(_cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
		(int)(_nall), (int)(_nnz), _descrU, _d_val, _d_row, _d_col, _infoU),
		error_t::SSV_ERROR_UNKNOWN);
}

void PoissonMethodCG::_PCGExecute() const
{
	if (std::get<0>(_buffer_shape) == 0)
	{
		throw error_t::SSV_ERROR_NOT_INITIALIZED;
	}

	const T tol = 1e-10f;
	const T floatone = 1.f;
	const T floatzero = 0.f;

	checkCudaErrorAndThrow(cudaMemset(_d_x, 0, _nall * sizeof(float)),
		error_t::SSV_ERROR_UNKNOWN);

	uint k = 0;
	float r1, alpha, beta;
	float numerator, denominator, nalpha;
	checkCudaErrorAndThrow(cublasSdot(_cublasHandle, (int)(_nall), _d_r, 1, _d_r, 1, &r1),
		error_t::SSV_ERROR_UNKNOWN);

	while (r1 > tol * tol && k <= _iterations)
	{
		// Forward Solve, we can re-use infoA since the sparsity pattern of A matches that of L
		checkCudaErrorAndThrow(cusparseScsrsv_solve(_cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
			(int)(_nall), &floatone, _descrL,
			_d_valsILU0, _d_row, _d_col, _infoA, _d_r, _d_y),
			error_t::SSV_ERROR_UNKNOWN);

		// Back Substitution
		checkCudaErrorAndThrow(cusparseScsrsv_solve(_cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
			(int)(_nall), &floatone, _descrU,
			_d_valsILU0, _d_row, _d_col, _infoU, _d_y, _d_zm1),
			error_t::SSV_ERROR_UNKNOWN);

		k++;

		if (k == 1u)
		{
			checkCudaErrorAndThrow(cublasScopy(_cublasHandle, (int)(_nall), _d_zm1, 1, _d_p, 1),
				error_t::SSV_ERROR_UNKNOWN);
		}
		else
		{
			checkCudaErrorAndThrow(cublasSdot(_cublasHandle, (int)(_nall), _d_r, 1, _d_zm1, 1, &numerator),
				error_t::SSV_ERROR_UNKNOWN);
			checkCudaErrorAndThrow(cublasSdot(_cublasHandle, (int)(_nall), _d_rm2, 1, _d_zm2, 1, &denominator),
				error_t::SSV_ERROR_UNKNOWN);
			beta = numerator / denominator;
			checkCudaErrorAndThrow(cublasSscal(_cublasHandle, (int)(_nall), &beta, _d_p, 1),
				error_t::SSV_ERROR_UNKNOWN);
			checkCudaErrorAndThrow(cublasSaxpy(_cublasHandle, (int)(_nall), &floatone, _d_zm1, 1, _d_p, 1),
				error_t::SSV_ERROR_UNKNOWN);
		}

		int nzILU0 = 2 * (int)(_nall) - 1;
		checkCudaErrorAndThrow(cusparseScsrmv(_cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
			(int)(_nall), (int)(_nall), nzILU0, &floatone, _descrU, _d_val, _d_row, _d_col, _d_p, &floatzero, _d_omega),
			error_t::SSV_ERROR_UNKNOWN);
		checkCudaErrorAndThrow(cublasSdot(_cublasHandle, (int)(_nall), _d_r, 1, _d_zm1, 1, &numerator),
			error_t::SSV_ERROR_UNKNOWN);
		checkCudaErrorAndThrow(cublasSdot(_cublasHandle, (int)(_nall), _d_p, 1, _d_omega, 1, &denominator),
			error_t::SSV_ERROR_UNKNOWN);
		alpha = numerator / denominator;
		checkCudaErrorAndThrow(cublasSaxpy(_cublasHandle, (int)(_nall), &alpha, _d_p, 1, _d_x, 1),
			error_t::SSV_ERROR_UNKNOWN);
		checkCudaErrorAndThrow(cublasScopy(_cublasHandle, (int)(_nall), _d_r, 1, _d_rm2, 1),
			error_t::SSV_ERROR_UNKNOWN);
		checkCudaErrorAndThrow(cublasScopy(_cublasHandle, (int)(_nall), _d_zm1, 1, _d_zm2, 1),
			error_t::SSV_ERROR_UNKNOWN);
		nalpha = -alpha;
		checkCudaErrorAndThrow(cublasSaxpy(_cublasHandle, (int)(_nall), &nalpha, _d_omega, 1, _d_r, 1),
			error_t::SSV_ERROR_UNKNOWN);
		checkCudaErrorAndThrow(cublasSdot(_cublasHandle, (int)(_nall), _d_r, 1, _d_r, 1, &r1),
			error_t::SSV_ERROR_UNKNOWN);
	}
}

void PoissonMethodCG::_PCGDestroy() const
{
	if (std::get<0>(_buffer_shape) == 0)
	{
		return;
	}
	_buffer_shape = std::make_tuple(0, 0, 0);

	checkCudaErrorAndThrow(cudaFree(_d_col), error_t::SSV_ERROR_UNKNOWN);
	checkCudaErrorAndThrow(cudaFree(_d_row), error_t::SSV_ERROR_UNKNOWN);
	checkCudaErrorAndThrow(cudaFree(_d_val), error_t::SSV_ERROR_UNKNOWN);
	checkCudaErrorAndThrow(cudaFree(_d_y), error_t::SSV_ERROR_UNKNOWN);
	checkCudaErrorAndThrow(cudaFree(_d_p), error_t::SSV_ERROR_UNKNOWN);
	checkCudaErrorAndThrow(cudaFree(_d_omega), error_t::SSV_ERROR_UNKNOWN);
	checkCudaErrorAndThrow(cudaFree(_d_valsILU0), error_t::SSV_ERROR_UNKNOWN);
	checkCudaErrorAndThrow(cudaFree(_d_zm1), error_t::SSV_ERROR_UNKNOWN);
	checkCudaErrorAndThrow(cudaFree(_d_zm2), error_t::SSV_ERROR_UNKNOWN);
	checkCudaErrorAndThrow(cudaFree(_d_rm2), error_t::SSV_ERROR_UNKNOWN);
	checkCudaErrorAndThrow(cudaFree(_d_x), error_t::SSV_ERROR_UNKNOWN);
	checkCudaErrorAndThrow(cudaFree(_d_r), error_t::SSV_ERROR_UNKNOWN);

	checkCudaErrorAndThrow(cusparseDestroyMatDescr(_descrA),
		error_t::SSV_ERROR_UNKNOWN); _descrA = 0;
	checkCudaErrorAndThrow(cusparseDestroyMatDescr(_descrL),
		error_t::SSV_ERROR_UNKNOWN); _descrL = 0;
	checkCudaErrorAndThrow(cusparseDestroyMatDescr(_descrU),
		error_t::SSV_ERROR_UNKNOWN); _descrU = 0;

	checkCudaErrorAndThrow(cusparseDestroySolveAnalysisInfo(_infoA),
		error_t::SSV_ERROR_UNKNOWN); _infoA = 0;
	checkCudaErrorAndThrow(cusparseDestroySolveAnalysisInfo(_infoU),
		error_t::SSV_ERROR_UNKNOWN); _infoU = 0;
}

template<>
void PoissonMethodCG::operator()(
	Blob<T>& q, const Blob<T>& g
	) const
{
	if (q.shape() != _buffer_shape)
	{
		_PCGDestroy();
		_PCGInit(q.shape());
	}
	g.copyTo(_d_r, Blob<T>::storage_t::GPU, Blob<T>::storage_t::GPU);
	_PCGExecute();
	q.copyFrom(_d_x, Blob<T>::storage_t::GPU, Blob<T>::storage_t::GPU);
}

template void PoissonMethodCG::operator()(Blob<T>&, const Blob<T>&) const;
