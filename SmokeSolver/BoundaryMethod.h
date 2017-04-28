#pragma once

#ifndef __BOUNDARY_METHOD_H__
#define __BOUNDARY_METHOD_H__

#include "common.h"
#include "Blob.h"


namespace ssv
{
	// Boundary operators
	template <typename QType, typename TpType>
	struct BoundaryOpClamp
	{
		typedef QType first_argument_type;
		typedef TpType second_argument_type;
		typedef QType result_type;

		QType q1;
		TpType tp1;
		__host__ __device__ QType operator() (
			const QType &q, const TpType &tp
			) const
		{
			return tp == tp1 ? q1 : q;
		}
	};

	// Boundary operators
	template <typename QType, typename TpType>
	struct BoundaryOpClamp2
	{
		typedef QType first_argument_type;
		typedef TpType second_argument_type;
		typedef QType result_type;

		QType q1;
		TpType tp1;
		QType q2;
		TpType tp2;
		__host__ __device__ QType operator() (
			const QType &q, const TpType &tp
			) const
		{
			return tp == tp1 ? q1 : tp == tp2 ? q2: q;
		}
	};

	// Deal with boundary conditions
	class BoundaryMethod
	{
	public:
		template<typename QType, typename TpType>
		using type = std::function<void(Blob<QType> &, const Blob<TpType> &)>;
	};

	// Apply the same boundary operator for all cells
	// op : boundary operator
	template<typename QType, typename TpType, typename OpType>
	class BoundaryMethodAll : public BoundaryMethod
	{
	public:
		BoundaryMethodAll(OpType &&op)
			: _op(std::forward<OpType>(op)) {}
	public:
		void operator()(
			Blob<QType> &q, const Blob<TpType> &tp
			) const;
	private:
		OpType _op;
	};

	template<typename OpType, 
		typename QType = OpType::first_argument_type, 
		typename TpType = OpType::second_argument_type>
	inline BoundaryMethodAll<QType, TpType, OpType> make_boundary_method_all(OpType &&op)
	{
		return BoundaryMethodAll<QType, TpType, OpType>(std::forward<OpType>(op));
	}
}

#endif // !__BOUNDARY_METHOD_H__
