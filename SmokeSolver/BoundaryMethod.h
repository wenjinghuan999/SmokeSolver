#pragma once

#ifndef __BOUNDARY_METHOD_H__
#define __BOUNDARY_METHOD_H__

#include "common.h"
#include "Blob.h"


namespace ssv
{
	/** \brief Apply clamp operator on boundary */
	template <typename TpType, typename QType>
	struct BoundaryOpClamp
	{
		typedef TpType first_argument_type;
		typedef QType second_argument_type;
		typedef QType result_type;

		TpType tp1;
		QType q1;
		__host__ __device__ QType operator()(
			const TpType &tp, const QType &q
		) const
		{
			return tp == tp1 ? q1 : q;
		}
	};

	/** \brief Make boundary operator that clamps q[i] = \p q1 for tp[i] = \p tp1 */
	template <typename EnumType, typename QType,
	          typename TpType = typename std::underlying_type<EnumType>::type>
	BoundaryOpClamp<TpType, QType> make_boundary_op_clamp(const EnumType &tp1, const QType &q1)
	{
		return BoundaryOpClamp<TpType, QType>{underlying(tp1), q1};
	}

	/** \brief Apply clamp operator on boundary */
	template <typename TpType, typename QType>
	struct BoundaryOpClamp2
	{
		typedef TpType first_argument_type;
		typedef QType second_argument_type;
		typedef QType result_type;

		TpType tp1;
		QType q1;
		TpType tp2;
		QType q2;
		__host__ __device__ QType operator()(
			const TpType &tp, const QType &q
		) const
		{
			return tp == tp1 ? q1 : tp == tp2 ? q2 : q;
		}
	};

	/** \brief Make boundary operator that clamps q[i] = \p q1 for tp[i] = \p tp1 and q[i] = \p q2 for tp[i] = \p tp2 */
	template <typename EnumType, typename QType,
	          typename TpType = typename std::underlying_type<EnumType>::type>
	BoundaryOpClamp2<TpType, QType> make_boundary_op_clamp2(
		const EnumType &tp1, const QType &q1, const EnumType &tp2, const QType &q2)
	{
		return BoundaryOpClamp2<TpType, QType>{underlying(tp1), q1, underlying(tp2), q2};
	}

	/** \brief Apply boundary conditions */
	class BoundaryMethod
	{
	public:
		template <typename QType, typename TpType>
		using type = std::function<void(Blob<QType> &, const Blob<TpType> &)>;
	};

	/** \brief Apply the same boundary operator for all cells */
	template <typename TpType, typename QType, typename OpType>
	class BoundaryMethodAll : public BoundaryMethod
	{
	public:
		explicit BoundaryMethodAll(OpType &&op)
			: op_(std::forward<OpType>(op))
		{
		}

	public:
		void operator()(
			Blob<QType> &q, const Blob<TpType> &tp
		) const;
	private:
		/** \brief boundary operator, \code q[i] = op(tp[i], q[i]) \endcode */
		OpType op_;
	};

	/**
	 * \brief Make boundary method that applies the same boundary operator for all cells
	 * \tparam OpType boundary operator type
	 * \tparam TpType the first argument type of OpType
	 * \tparam QType the second argument type (as well as the result type) of OpType
	 * \param op boundary operator, \code q[i] = op(tp[i], q[i]) \endcode, try using make_boundary_op_... 
	 */
	template <typename OpType,
	          typename TpType = typename OpType::first_argument_type,
	          typename QType = typename OpType::second_argument_type>
	BoundaryMethodAll<TpType, QType, OpType> make_boundary_all(OpType &&op)
	{
		return BoundaryMethodAll<TpType, QType, OpType>(std::forward<OpType>(op));
	}
}

#endif // !__BOUNDARY_METHOD_H__
