#pragma once

#ifndef __POISSON_METHOD_H__
#define __POISSON_METHOD_H__

#include "common.h"
#include "Blob.h"

#include <cusparse_v2.h>
#include <cublas_v2.h>


namespace ssv
{
	// Solve Poisson equation
	//  __2
	//  \/ q = g
	//
	class PoissonMethod
	{
	public:
		template <typename QType>
		using type = std::function<void(Blob<QType> &, const Blob<QType> &)>;
	};

	class PoissonMethodGS : public PoissonMethod
	{
	public:
		PoissonMethodGS(uint iterations, real omega)
			: iterations_(iterations), omega_(omega)
		{
		}

		template <typename QType>
		void operator()(
			Blob<QType> &q, const Blob<QType> &g
		) const;
	private:
		uint iterations_;
		real omega_;
	};

	class PoissonMethodVCycle : public PoissonMethod
	{
	public:
		PoissonMethodVCycle(uint levels, uint iterations, real omega)
			: levels_(levels), gs_(iterations, omega)
		{
		}

		template <typename QType>
		void operator()(
			Blob<QType> &q, const Blob<QType> &g
		) const;
	private:
		uint levels_;
		PoissonMethodGS gs_;
	};

	class PoissonMethodCG : public PoissonMethod
	{
	public:
		explicit PoissonMethodCG(uint iterations)
			: iterations_(iterations)
		{
			_Init();
		}

		template <typename QType>
		void operator()(
			Blob<QType> &q, const Blob<QType> &g
		) const;
	private:
		void _Init();
		void _PCGInit(const blob_shape_t &shape) const;
		void _PCGExecute() const;
		void _PCGDestroy() const;
	private:
		uint iterations_;
		// Handle
		cublasHandle_t cublas_handle_;
		cusparseHandle_t cusparse_handle_;
		// Buffer
		mutable blob_shape_t buffer_shape_;
		mutable cusparseMatDescr_t descr_a_, descr_l_, descr_u_;
		mutable cusparseSolveAnalysisInfo_t info_a_, info_u_;
		mutable size_t nnz_, nall_;
		mutable real *d_val_;
		mutable int *d_col_, *d_row_;
		mutable real *d_x_, *d_r_;
		mutable real *d_vals_ilu0_, *d_zm1_, *d_zm2_, *d_rm2_, *d_y_, *d_p_, *d_omega_;
	};
}

#endif // !__POISSON_METHOD_H__
