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
		PoissonMethodGS(ssv::uint iterations, ssv::T omega)
			: _iterations(iterations), _omega(omega) {}
		template <typename QType>
		void operator() (
			Blob<QType> &q, const Blob<QType> &g
			) const;
	private:
		ssv::uint _iterations;
		ssv::T _omega;
	};

	class PoissonMethodVCycle : public PoissonMethod
	{
	public:
		PoissonMethodVCycle(ssv::uint levels, ssv::uint iterations, ssv::T omega)
			: _levels(levels), _gs(iterations, omega) {}
		template <typename QType>
		void operator() (
			Blob<QType> &q, const Blob<QType> &g
			) const;
	private:
		ssv::uint _levels;
		PoissonMethodGS _gs;
	};

	class PoissonMethodCG : public PoissonMethod
	{
	public:
		PoissonMethodCG(ssv::uint iterations)
			: _iterations(iterations)
		{
			_Init(); 
		}
		template <typename QType>
		void operator() (
			Blob<QType> &q, const Blob<QType> &g
			) const;
	private:
		void _Init();
		void _PCGInit(const BlobShape &shape) const;
		void _PCGExecute() const;
		void _PCGDestroy() const;
	private:
		ssv::uint _iterations;
		// Handle
		cublasHandle_t _cublasHandle;
		cusparseHandle_t _cusparseHandle;
		// Buffer
		mutable ssv::BlobShape _buffer_shape;
		mutable cusparseMatDescr_t _descrA, _descrL, _descrU;
		mutable cusparseSolveAnalysisInfo_t _infoA, _infoU;
		mutable size_t _nnz, _nall;
		mutable T *_d_val;
		mutable int *_d_col, *_d_row;
		mutable T *_d_x, *_d_r;
		mutable T *_d_valsILU0, *_d_zm1, *_d_zm2, *_d_rm2, *_d_y, *_d_p, *_d_omega;
	};
}

#endif // !__POISSON_METHOD_H__
