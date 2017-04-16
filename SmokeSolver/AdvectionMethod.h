#pragma once

#ifndef __ADVECTION_METHOD_H__
#define __ADVECTION_METHOD_H__

#include "common.h"
#include "Blob.h"


namespace ssv
{
	// Solve advection problem
	//  dq          __
	// ---- = -(u . \/) q
	//  dt
	template <typename QType>
	class AdvectionMethod
	{
	public:
		// 2D
		virtual void operator() (
			Blob<QType> &qout, const Blob<QType> &q, const Blob<T2> &u
			) const = 0;
		// 3D
		virtual void operator() (
			Blob<QType> &qout, const Blob<QType> &q, const Blob<T4> &u
			) const = 0;
	};

	template <typename QType>
	class AdvectionMethodSemiLagrangian : public AdvectionMethod<QType>
	{
	public:
		virtual void operator() (
			Blob<QType> &qout, const Blob<QType> &q, const Blob<T2> &u
			) const override;
		virtual void operator() (
			Blob<QType> &qout, const Blob<QType> &q, const Blob<T4> &u
			) const override;
	};
}

#endif // !__ADVECTION_METHOD_H__
