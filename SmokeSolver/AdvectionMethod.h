#pragma once

#ifndef __ADVECTION_METHOD_H__
#define __ADVECTION_METHOD_H__

#include "common.h"
#include "Blob.h"

namespace ssv
{
	// Solve 2D advection problem
	//  dq          __
	// ---- = -(u . \/) q
	//  dt
	template <typename QType>
	class AdvectionMethod2d
	{
	public:
		virtual void operator () (
			Blob<QType> &qout, const Blob<QType> &q, const Blob<T2> &u
			) const = 0;
	};

	template <typename QType>
	class AdvectionMethod2dSemiLagrangian : public AdvectionMethod2d<QType>
	{
	public:
		virtual void operator () (
			Blob<QType> &qout, const Blob<QType> &q, const Blob<T2> &u
			) const override;
	};
}

#endif // !__ADVECTION_METHOD_H__
