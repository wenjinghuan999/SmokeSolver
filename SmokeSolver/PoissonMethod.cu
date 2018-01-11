#include "common.cuh"
#include "PoissonMethod.h"
#include <vector>
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
	__global__ void kernelGS2D(
		BlobWrapper<QType> q, BlobWrapperConst<QType> g,
		real omega, uint redblack
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

		q(x, y) = omega * (v - g(x, y)) / static_cast<real>(4) + (static_cast<real>(1) - omega) * q(x, y);
	}

	// Red-black Gauss-Seidel
	// LAUNCH : block (ny - 2, (nz - 2) / 2), thread (nx - 2)
	// p : nx x ny x nz
	// g : nx x ny x nz
	// omega : SOR coefficient
	// redblack : 0 or 1 indicating red or black
	template <typename QType>
	__global__ void kernelGS3D(
		BlobWrapper<QType> q, BlobWrapperConst<QType> g,
		real omega, uint redblack
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

		q(x, y, z) = omega * (v - g(x, y, z)) / static_cast<real>(6)
			+ (static_cast<real>(1) - omega) * q(x, y, z);
	}
}

template <typename QType>
void PoissonMethodGS::operator()(
	Blob<QType> &q, const Blob<QType> &g
) const
{
	if (q.nz() < 3u)
	{
		for (uint i = 0; i < iterations_; i++)
		{
			kernelGS2D<<<(q.ny() - 2u) / 2u, q.nx() - 2u>>>(
				q.wrapper(), g.wrapper_const(), omega_, 0
			);
			kernelGS2D<<<(q.ny() - 2u) / 2u, q.nx() - 2u>>>(
				q.wrapper(), g.wrapper_const(), omega_, 1u
			);
		}
	}
	else
	{
		for (uint i = 0; i < iterations_; i++)
		{
			kernelGS3D<<<dim3(q.ny() - 2u, (q.nz() - 2u) / 2u), q.nx() - 2u>>>(
				q.wrapper(), g.wrapper_const(), omega_, 0
			);
			kernelGS3D<<<dim3(q.ny() - 2u, (q.nz() - 2u) / 2u), q.nx() - 2u>>>(
				q.wrapper(), g.wrapper_const(), omega_, 1u
			);
		}
	}
}

template void PoissonMethodGS::operator()(Blob<real> &, const Blob<real> &) const;
template void PoissonMethodGS::operator()(Blob<real2> &, const Blob<real2> &) const;
template void PoissonMethodGS::operator()(Blob<real4> &, const Blob<real4> &) const;

namespace
{
	using ssv::uint;

	// Down sampling
	// LAUNCH : block (ny), thread (nx)
	// qout :  nx x  ny
	// qin  : 2nx x 2ny
	template <typename QType>
	__global__ void kernelDownSample2D(
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
	__global__ void kernelDownSample3D(
		BlobWrapper<QType> qout, BlobWrapperConst<QType> qin
	)
	{
		uint z = blockIdx.y;
		uint y = blockIdx.x;
		uint x = threadIdx.x;

		QType q = qin(2u * x, 2u * y, 2u * z);
		q += qin(2u * x + 1u, 2u * y, 2u * z);
		q += qin(2u * x + 1u, 2u * y + 1u, 2u * z);
		q += qin(2u * x, 2u * y + 1u, 2u * z);
		q += qin(2u * x, 2u * y + 1u, 2u * z + 1u);
		q += qin(2u * x + 1u, 2u * y + 1u, 2u * z + 1u);
		q += qin(2u * x + 1u, 2u * y, 2u * z + 1u);
		q += qin(2u * x, 2u * y, 2u * z + 1u);
		qout(x, y, z) = q;
	}

	// Up sampling
	// LAUNCH : block (ny), thread (nx)
	// qout  : 2nx x 2ny
	// qin   :  nx x  ny
	template <typename QType>
	__global__ void kernelUpSample2D(
		BlobWrapper<QType> qout, BlobWrapperConst<QType> qin
	)
	{
		uint y = blockIdx.x;
		uint x = threadIdx.x;

		QType q = qin(x, y);
		qout(2u * x, 2u * y) = q;
		qout(2u * x + 1u, 2u * y) = q;
		qout(2u * x + 1u, 2u * y + 1u) = q;
		qout(2u * x, 2u * y + 1u) = q;
	}

	// Up sampling
	// LAUNCH : block (ny, nz), thread (nx)
	// qout  : 2nx x 2ny x 2nz
	// qin   :  nx x  ny x  nz
	template <typename QType>
	__global__ void kernelUpSample3D(
		BlobWrapper<QType> qout, BlobWrapperConst<QType> qin
	)
	{
		uint z = blockIdx.y;
		uint y = blockIdx.x;
		uint x = threadIdx.x;

		QType q = qin(x, y);
		qout(2u * x, 2u * y, 2u * z) = q;
		qout(2u * x + 1u, 2u * y, 2u * z) = q;
		qout(2u * x + 1u, 2u * y + 1u, 2u * z) = q;
		qout(2u * x, 2u * y + 1u, 2u * z) = q;
		qout(2u * x, 2u * y + 1u, 2u * z + 1u) = q;
		qout(2u * x + 1u, 2u * y + 1u, 2u * z + 1u) = q;
		qout(2u * x + 1u, 2u * y, 2u * z + 1u) = q;
		qout(2u * x, 2u * y, 2u * z + 1u) = q;
	}

	blob_shape_t _NextShape(
		const blob_shape_t &shape
	)
	{
		uint nx, ny, nz;
		std::tie(nx, ny, nz) = shape;
		nx /= 2u;
		if (nx < 3u) nx = 3u;
		ny /= 2u;
		if (ny < 3u) ny = 3u;
		if (nz >= 3u)
		{
			nz = (nz + 1u) / 2u + 1u;
			if (nz < 3u) nz = 3u;
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
			kernelDownSample2D<<<qout.ny(), qout.nx()>>>(
				qout.wrapper(), qin.wrapper_const()
			);
		}
		else
		{
			kernelDownSample3D<<<dim3(qout.ny(), qout.nz()), qout.nx()>>>(
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
			kernelUpSample2D<<<qin.ny(), qin.nx()>>>(
				qout.wrapper(), qin.wrapper_const()
			);
		}
		else
		{
			kernelUpSample3D<<<dim3(qin.ny(), qin.nz()), qin.nx()>>>(
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

template <typename QType>
void PoissonMethodVCycle::operator()(Blob<QType> &q, const Blob<QType> &g) const
{
	static std::vector<Blob<QType> > buffers_q;
	static std::vector<Blob<QType> > buffers_g;

	typename Blob<QType>::shape_t shape = q.shape();
	shape = _NextShape(shape);

	if (buffers_q.empty() || buffers_q.front().shape() != shape)
	{
		buffers_q.clear();
		buffers_g.clear();
		for (uint i = 0; i < levels_; i++)
		{
			buffers_q.emplace_back(shape, q.gpu_device(), Blob<QType>::storage_t::GPU);
			buffers_g.emplace_back(shape, q.gpu_device(), Blob<QType>::storage_t::GPU);
			shape = _NextShape(shape);
		}
	}

	if (levels_ > 0)
	{
		_DownSample(buffers_q[0], q);
		_DownSample(buffers_g[0], g);

		_VCycle(buffers_q, buffers_g, gs_, 0, levels_);

		_UpSample(q, buffers_q[0]);
	}
	gs_(q, g);
}

template void PoissonMethodVCycle::operator()<real>(Blob<real> &q, const Blob<real> &g) const;
template void PoissonMethodVCycle::operator()<real2>(Blob<real2> &q, const Blob<real2> &g) const;
template void PoissonMethodVCycle::operator()<real4>(Blob<real4> &q, const Blob<real4> &g) const;

namespace
{
	using ssv::uint;

	void _PCGLaplacianInit(int *row_ptr, int *col_ind, real *val, int nx, int ny, int nz)
	{
		int n = nx * ny * nz;
		int nxy = nx * ny;
		int idx = 0;

		// loop over degrees of freedom
		for (int i = 0; i < n; i++)
		{
			int ix = i % nx;
			int iy = (i % nxy) / nx;
			int iz = i / nxy;

			row_ptr[i] = idx;

			// down
			if (iz > 0)
			{
				val[idx] = 1.0;
				col_ind[idx] = i - nxy;
				idx++;
			}

			// front
			if (iy > 0)
			{
				val[idx] = 1.0;
				col_ind[idx] = i - nx;
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
			if (ix < nx - 1)
			{
				val[idx] = 1.0;
				col_ind[idx] = i + 1;
				idx++;
			}

			// back
			if (iy < ny - 1)
			{
				val[idx] = 1.0;
				col_ind[idx] = i + nx;
				idx++;
			}

			// up
			if (iz < nz - 1)
			{
				val[idx] = 1.0;
				col_ind[idx] = i + nxy;
				idx++;
			}
		}

		row_ptr[n] = idx;
	}
}

void PoissonMethodCG::_Init()
{
	/* Create CUBLAS & CUSPARSE context */
	cublas_handle_ = nullptr;
	CHECK_CUDA_ERROR_AND_THROW(cublasCreate(&cublas_handle_), ssv_error(error_t::SSV_ERROR_UNKNOWN));
	cusparse_handle_ = nullptr;
	CHECK_CUDA_ERROR_AND_THROW(cusparseCreate(&cusparse_handle_), ssv_error(error_t::SSV_ERROR_UNKNOWN));
}

void PoissonMethodCG::_PCGInit(const blob_shape_t &shape) const
{
	uint nx, ny, nz;
	std::tie(nx, ny, nz) = shape;
	buffer_shape_ = shape;
	nnz_ = 7u * nx * ny * nz - 2u * (nx * ny + ny * nz + nz * nx);
	nall_ = nx * ny * nz;
	int *h_row = new int[nall_ + 1u];
	int *h_col = new int[nnz_];
	float *h_val = new float[nnz_];

	_PCGLaplacianInit(h_row, h_col, h_val, nx, ny, nz);

	/* Allocate required memory */
	CHECK_CUDA_ERROR_AND_THROW(cudaMalloc(reinterpret_cast<void **>(&d_col_), nnz_ * sizeof(int)),
		ssv_error(error_t::SSV_ERROR_OUT_OF_MEMORY_GPU));
	CHECK_CUDA_ERROR_AND_THROW(cudaMalloc(reinterpret_cast<void **>(&d_row_), (nall_ + 1) * sizeof(int)),
		ssv_error(error_t::SSV_ERROR_OUT_OF_MEMORY_GPU));
	CHECK_CUDA_ERROR_AND_THROW(cudaMalloc(reinterpret_cast<void **>(&d_val_), nnz_ * sizeof(float)),
		ssv_error(error_t::SSV_ERROR_OUT_OF_MEMORY_GPU));
	CHECK_CUDA_ERROR_AND_THROW(cudaMalloc(reinterpret_cast<void **>(&d_y_), nall_ * sizeof(float)),
		ssv_error(error_t::SSV_ERROR_OUT_OF_MEMORY_GPU));
	CHECK_CUDA_ERROR_AND_THROW(cudaMalloc(reinterpret_cast<void **>(&d_p_), nall_ * sizeof(float)),
		ssv_error(error_t::SSV_ERROR_OUT_OF_MEMORY_GPU));
	CHECK_CUDA_ERROR_AND_THROW(cudaMalloc(reinterpret_cast<void **>(&d_omega_), nall_ * sizeof(float)),
		ssv_error(error_t::SSV_ERROR_OUT_OF_MEMORY_GPU));
	CHECK_CUDA_ERROR_AND_THROW(cudaMalloc(reinterpret_cast<void **>(&d_x_), nall_ * sizeof(float)),
		ssv_error(error_t::SSV_ERROR_OUT_OF_MEMORY_GPU));
	CHECK_CUDA_ERROR_AND_THROW(cudaMalloc(reinterpret_cast<void **>(&d_r_), nall_ * sizeof(float)),
		ssv_error(error_t::SSV_ERROR_OUT_OF_MEMORY_GPU));

	CHECK_CUDA_ERROR_AND_THROW(cudaMemcpy(d_col_, h_col, nnz_ * sizeof(int), cudaMemcpyHostToDevice),
		ssv_error(error_t::SSV_ERROR_UNKNOWN));
	CHECK_CUDA_ERROR_AND_THROW(cudaMemcpy(d_row_, h_row, (nall_ + 1) * sizeof(int), cudaMemcpyHostToDevice),
		ssv_error(error_t::SSV_ERROR_UNKNOWN));
	CHECK_CUDA_ERROR_AND_THROW(cudaMemcpy(d_val_, h_val, nnz_ * sizeof(float), cudaMemcpyHostToDevice),
		ssv_error(error_t::SSV_ERROR_UNKNOWN));

	delete[] h_col;
	delete[] h_row;
	delete[] h_val;

	CHECK_CUDA_ERROR_AND_THROW(cudaMalloc(reinterpret_cast<void **>(&d_vals_ilu0_), nnz_ * sizeof(float)),
		ssv_error(error_t::SSV_ERROR_OUT_OF_MEMORY_GPU));
	CHECK_CUDA_ERROR_AND_THROW(cudaMalloc(reinterpret_cast<void **>(&d_zm1_), nall_ * sizeof(float)),
		ssv_error(error_t::SSV_ERROR_OUT_OF_MEMORY_GPU));
	CHECK_CUDA_ERROR_AND_THROW(cudaMalloc(reinterpret_cast<void **>(&d_zm2_), nall_ * sizeof(float)),
		ssv_error(error_t::SSV_ERROR_OUT_OF_MEMORY_GPU));
	CHECK_CUDA_ERROR_AND_THROW(cudaMalloc(reinterpret_cast<void **>(&d_rm2_), nall_ * sizeof(float)),
		ssv_error(error_t::SSV_ERROR_OUT_OF_MEMORY_GPU));

	/* Description of the matrices*/
	descr_a_ = nullptr;
	CHECK_CUDA_ERROR_AND_THROW(cusparseCreateMatDescr(&descr_a_),
		ssv_error(error_t::SSV_ERROR_UNKNOWN));
	CHECK_CUDA_ERROR_AND_THROW(cusparseSetMatType(descr_a_, CUSPARSE_MATRIX_TYPE_GENERAL),
		ssv_error(error_t::SSV_ERROR_UNKNOWN));
	CHECK_CUDA_ERROR_AND_THROW(cusparseSetMatIndexBase(descr_a_, CUSPARSE_INDEX_BASE_ZERO),
		ssv_error(error_t::SSV_ERROR_UNKNOWN));

	descr_l_ = nullptr;
	CHECK_CUDA_ERROR_AND_THROW(cusparseCreateMatDescr(&descr_l_),
		ssv_error(error_t::SSV_ERROR_UNKNOWN));
	CHECK_CUDA_ERROR_AND_THROW(cusparseSetMatType(descr_l_, CUSPARSE_MATRIX_TYPE_GENERAL),
		ssv_error(error_t::SSV_ERROR_UNKNOWN));
	CHECK_CUDA_ERROR_AND_THROW(cusparseSetMatIndexBase(descr_l_, CUSPARSE_INDEX_BASE_ZERO),
		ssv_error(error_t::SSV_ERROR_UNKNOWN));
	CHECK_CUDA_ERROR_AND_THROW(cusparseSetMatFillMode(descr_l_, CUSPARSE_FILL_MODE_LOWER),
		ssv_error(error_t::SSV_ERROR_UNKNOWN));
	CHECK_CUDA_ERROR_AND_THROW(cusparseSetMatDiagType(descr_l_, CUSPARSE_DIAG_TYPE_UNIT),
		ssv_error(error_t::SSV_ERROR_UNKNOWN));

	descr_u_ = nullptr;
	CHECK_CUDA_ERROR_AND_THROW(cusparseCreateMatDescr(&descr_u_),
		ssv_error(error_t::SSV_ERROR_UNKNOWN));
	CHECK_CUDA_ERROR_AND_THROW(cusparseSetMatType(descr_u_, CUSPARSE_MATRIX_TYPE_GENERAL),
		ssv_error(error_t::SSV_ERROR_UNKNOWN));
	CHECK_CUDA_ERROR_AND_THROW(cusparseSetMatIndexBase(descr_u_, CUSPARSE_INDEX_BASE_ZERO),
		ssv_error(error_t::SSV_ERROR_UNKNOWN));
	CHECK_CUDA_ERROR_AND_THROW(cusparseSetMatFillMode(descr_u_, CUSPARSE_FILL_MODE_UPPER),
		ssv_error(error_t::SSV_ERROR_UNKNOWN));
	CHECK_CUDA_ERROR_AND_THROW(cusparseSetMatDiagType(descr_u_, CUSPARSE_DIAG_TYPE_NON_UNIT),
		ssv_error(error_t::SSV_ERROR_UNKNOWN));

	/* create the analysis info object for the A matrix */
	info_a_ = nullptr;
	CHECK_CUDA_ERROR_AND_THROW(cusparseCreateSolveAnalysisInfo(&info_a_),
		ssv_error(error_t::SSV_ERROR_UNKNOWN));

	/* Create info objects for the ILU0 preconditioner */
	info_u_ = nullptr;
	CHECK_CUDA_ERROR_AND_THROW(cusparseCreateSolveAnalysisInfo(&info_u_),
		ssv_error(error_t::SSV_ERROR_UNKNOWN));

	/* Perform the analysis for the Non-Transpose case */
	CHECK_CUDA_ERROR_AND_THROW(cusparseScsrsv_analysis(cusparse_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
			static_cast<int>(nall_), static_cast<int>(nnz_), descr_a_, d_val_, d_row_, d_col_, info_a_),
		ssv_error(error_t::SSV_ERROR_UNKNOWN));

	/* Copy A data to ILU0 vals as input*/
	CHECK_CUDA_ERROR_AND_THROW(cudaMemcpy(d_vals_ilu0_, d_val_, nnz_ * sizeof(float), cudaMemcpyDeviceToDevice),
		ssv_error(error_t::SSV_ERROR_UNKNOWN));

	/* generate the Incomplete LU factor H for the matrix A using cudsparseScsrilu0 */
	CHECK_CUDA_ERROR_AND_THROW(cusparseScsrilu0(cusparse_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
			static_cast<int>(nall_), descr_a_, d_vals_ilu0_, d_row_, d_col_, info_a_),
		ssv_error(error_t::SSV_ERROR_UNKNOWN));

	CHECK_CUDA_ERROR_AND_THROW(cusparseScsrsv_analysis(cusparse_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
			static_cast<int>(nall_), static_cast<int>(nnz_), descr_u_, d_val_, d_row_, d_col_, info_u_),
		ssv_error(error_t::SSV_ERROR_UNKNOWN));
}

void PoissonMethodCG::_PCGExecute() const
{
	if (std::get<0>(buffer_shape_) == 0)
	{
		throw ssv_error(error_t::SSV_ERROR_NOT_INITIALIZED);
	}

	const real tol = 1e-10f;
	const real floatone = 1.f;
	const real floatzero = 0.f;

	CHECK_CUDA_ERROR_AND_THROW(cudaMemset(d_x_, 0, nall_ * sizeof(float)),
		ssv_error(error_t::SSV_ERROR_UNKNOWN));

	uint k = 0;
	float r1;
	float numerator, denominator;
	CHECK_CUDA_ERROR_AND_THROW(cublasSdot(cublas_handle_, static_cast<int>(nall_), d_r_, 1, d_r_, 1, &r1),
		ssv_error(error_t::SSV_ERROR_UNKNOWN));

	while (r1 > tol * tol && k <= iterations_)
	{
		// Forward Solve, we can re-use infoA since the sparsity pattern of A matches that of L
		CHECK_CUDA_ERROR_AND_THROW(cusparseScsrsv_solve(cusparse_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
				static_cast<int>(nall_), &floatone, descr_l_,
				d_vals_ilu0_, d_row_, d_col_, info_a_, d_r_, d_y_),
			ssv_error(error_t::SSV_ERROR_UNKNOWN));

		// Back Substitution
		CHECK_CUDA_ERROR_AND_THROW(cusparseScsrsv_solve(cusparse_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
				static_cast<int>(nall_), &floatone, descr_u_,
				d_vals_ilu0_, d_row_, d_col_, info_u_, d_y_, d_zm1_),
			ssv_error(error_t::SSV_ERROR_UNKNOWN));

		k++;

		if (k == 1u)
		{
			CHECK_CUDA_ERROR_AND_THROW(cublasScopy(cublas_handle_, static_cast<int>(nall_), d_zm1_, 1, d_p_, 1),
				ssv_error(error_t::SSV_ERROR_UNKNOWN));
		}
		else
		{
			CHECK_CUDA_ERROR_AND_THROW(cublasSdot(cublas_handle_, static_cast<int>(nall_), d_r_, 1, d_zm1_, 1, &numerator),
				ssv_error(error_t::SSV_ERROR_UNKNOWN));
			CHECK_CUDA_ERROR_AND_THROW(cublasSdot(cublas_handle_, static_cast<int>(nall_), d_rm2_, 1, d_zm2_, 1, &denominator),
				ssv_error(error_t::SSV_ERROR_UNKNOWN));
			float beta = numerator / denominator;
			CHECK_CUDA_ERROR_AND_THROW(cublasSscal(cublas_handle_, static_cast<int>(nall_), &beta, d_p_, 1),
				ssv_error(error_t::SSV_ERROR_UNKNOWN));
			CHECK_CUDA_ERROR_AND_THROW(cublasSaxpy(cublas_handle_, static_cast<int>(nall_), &floatone, d_zm1_, 1, d_p_, 1),
				ssv_error(error_t::SSV_ERROR_UNKNOWN));
		}

		int nz_ilu0 = 2 * static_cast<int>(nall_) - 1;
		CHECK_CUDA_ERROR_AND_THROW(cusparseScsrmv(cusparse_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
				static_cast<int>(nall_), static_cast<int>(nall_), nz_ilu0, &floatone, descr_u_, d_val_, d_row_, d_col_, d_p_, &
				floatzero, d_omega_),
			ssv_error(error_t::SSV_ERROR_UNKNOWN));
		CHECK_CUDA_ERROR_AND_THROW(cublasSdot(cublas_handle_, static_cast<int>(nall_), d_r_, 1, d_zm1_, 1, &numerator),
			ssv_error(error_t::SSV_ERROR_UNKNOWN));
		CHECK_CUDA_ERROR_AND_THROW(cublasSdot(cublas_handle_, static_cast<int>(nall_), d_p_, 1, d_omega_, 1, &denominator),
			ssv_error(error_t::SSV_ERROR_UNKNOWN));
		float alpha = numerator / denominator;
		CHECK_CUDA_ERROR_AND_THROW(cublasSaxpy(cublas_handle_, static_cast<int>(nall_), &alpha, d_p_, 1, d_x_, 1),
			ssv_error(error_t::SSV_ERROR_UNKNOWN));
		CHECK_CUDA_ERROR_AND_THROW(cublasScopy(cublas_handle_, static_cast<int>(nall_), d_r_, 1, d_rm2_, 1),
			ssv_error(error_t::SSV_ERROR_UNKNOWN));
		CHECK_CUDA_ERROR_AND_THROW(cublasScopy(cublas_handle_, static_cast<int>(nall_), d_zm1_, 1, d_zm2_, 1),
			ssv_error(error_t::SSV_ERROR_UNKNOWN));
		float nalpha = -alpha;
		CHECK_CUDA_ERROR_AND_THROW(cublasSaxpy(cublas_handle_, static_cast<int>(nall_), &nalpha, d_omega_, 1, d_r_, 1),
			ssv_error(error_t::SSV_ERROR_UNKNOWN));
		CHECK_CUDA_ERROR_AND_THROW(cublasSdot(cublas_handle_, static_cast<int>(nall_), d_r_, 1, d_r_, 1, &r1),
			ssv_error(error_t::SSV_ERROR_UNKNOWN));
	}
}

void PoissonMethodCG::_PCGDestroy() const
{
	if (std::get<0>(buffer_shape_) == 0)
	{
		return;
	}
	buffer_shape_ = std::make_tuple(0, 0, 0);

	CHECK_CUDA_ERROR_AND_THROW(cudaFree(d_col_), ssv_error(error_t::SSV_ERROR_UNKNOWN));
	CHECK_CUDA_ERROR_AND_THROW(cudaFree(d_row_), ssv_error(error_t::SSV_ERROR_UNKNOWN));
	CHECK_CUDA_ERROR_AND_THROW(cudaFree(d_val_), ssv_error(error_t::SSV_ERROR_UNKNOWN));
	CHECK_CUDA_ERROR_AND_THROW(cudaFree(d_y_), ssv_error(error_t::SSV_ERROR_UNKNOWN));
	CHECK_CUDA_ERROR_AND_THROW(cudaFree(d_p_), ssv_error(error_t::SSV_ERROR_UNKNOWN));
	CHECK_CUDA_ERROR_AND_THROW(cudaFree(d_omega_), ssv_error(error_t::SSV_ERROR_UNKNOWN));
	CHECK_CUDA_ERROR_AND_THROW(cudaFree(d_vals_ilu0_), ssv_error(error_t::SSV_ERROR_UNKNOWN));
	CHECK_CUDA_ERROR_AND_THROW(cudaFree(d_zm1_), ssv_error(error_t::SSV_ERROR_UNKNOWN));
	CHECK_CUDA_ERROR_AND_THROW(cudaFree(d_zm2_), ssv_error(error_t::SSV_ERROR_UNKNOWN));
	CHECK_CUDA_ERROR_AND_THROW(cudaFree(d_rm2_), ssv_error(error_t::SSV_ERROR_UNKNOWN));
	CHECK_CUDA_ERROR_AND_THROW(cudaFree(d_x_), ssv_error(error_t::SSV_ERROR_UNKNOWN));
	CHECK_CUDA_ERROR_AND_THROW(cudaFree(d_r_), ssv_error(error_t::SSV_ERROR_UNKNOWN));

	CHECK_CUDA_ERROR_AND_THROW(cusparseDestroyMatDescr(descr_a_),
		ssv_error(error_t::SSV_ERROR_UNKNOWN));
	descr_a_ = nullptr;
	CHECK_CUDA_ERROR_AND_THROW(cusparseDestroyMatDescr(descr_l_),
		ssv_error(error_t::SSV_ERROR_UNKNOWN));
	descr_l_ = nullptr;
	CHECK_CUDA_ERROR_AND_THROW(cusparseDestroyMatDescr(descr_u_),
		ssv_error(error_t::SSV_ERROR_UNKNOWN));
	descr_u_ = nullptr;

	CHECK_CUDA_ERROR_AND_THROW(cusparseDestroySolveAnalysisInfo(info_a_),
		ssv_error(error_t::SSV_ERROR_UNKNOWN));
	info_a_ = nullptr;
	CHECK_CUDA_ERROR_AND_THROW(cusparseDestroySolveAnalysisInfo(info_u_),
		ssv_error(error_t::SSV_ERROR_UNKNOWN));
	info_u_ = nullptr;
}

template <>
void PoissonMethodCG::operator()(
	Blob<real> &q, const Blob<real> &g
) const
{
	if (q.shape() != buffer_shape_)
	{
		_PCGDestroy();
		_PCGInit(q.shape());
	}
	g.copy_to(d_r_, Blob<real>::storage_t::GPU, Blob<real>::storage_t::GPU);
	_PCGExecute();
	q.copy_from(d_x_, Blob<real>::storage_t::GPU, Blob<real>::storage_t::GPU);
}

template void PoissonMethodCG::operator()(Blob<real> &, const Blob<real> &) const;
