#pragma once

#ifndef __BOUNDARY_METHOD_H__
#define __BOUNDARY_METHOD_H__

#include "common.h"
#include "Blob.h"


namespace ssv
{
	// Operator for boundary conditions
	template<typename TpType, typename QType>
	class BoundaryOp
	{
	public:
		typedef TpType first_argument_type;
		typedef QType second_argument_type;
		typedef QType result_type;
		// Apply boundary condition for a cell
		// tp : cell type
		// q  : current cell value
		__host__ __device__ virtual QType operator() (
			const TpType &tp, const QType &q
			) const = 0;
	};

	// Clamp q to q1 for tp = tp1, ...
	// q remains the same for other tp
	template <typename TpType, typename QType>
	class BoundaryOpClamp : public BoundaryOp<TpType, QType>
	{
	public:
		BoundaryOpClamp(TpType tp1, QType q1)
			: tp1(tp1), q1(q1) {}
	public:
		__host__ __device__ virtual QType operator() (
			const TpType &tp, const QType &q
			) const
		{
			return tp == tp1 ? q1 : q; 
		}
	private:
		TpType tp1;
		QType q1;
	};

	// Deal with boundary conditions
	template <typename TpType, typename QType>
	class BoundaryMethod
	{
	public:
		virtual void operator () (
			const Blob<TpType> &tp, Blob<QType> &q
			) const = 0;
	};

	// Apply the same boundary operator for all cells
	template <typename TpType, typename QType>
	class BoundaryMethodClampAll : public BoundaryMethod<TpType, QType>
	{
	public:
		BoundaryMethodClampAll(TpType tp1, QType q1)
			: tp1(tp1), q1(q1) {}
	public:
		virtual void operator () (
			const Blob<TpType> &tp, Blob<QType> &q
			) const;
	private:
		TpType tp1;
		QType q1;
	};
}

#endif // !__BOUNDARY_METHOD_H__
