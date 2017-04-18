#pragma once

#ifndef __POISSON_METHOD_H__
#define __POISSON_METHOD_H__

#include "common.h"
#include "Blob.h"


namespace ssv
{
	// Solve Poisson equation
	//  __2
	//  \/ q = g
	//
	template <typename QType>
	class PoissonMethod
	{
	public:
		virtual void operator() (
			Blob<QType> &q, const Blob<QType> &g
			) const = 0;
	};

	template <typename QType>
	class PoissonMethodGS : public PoissonMethod<QType>
	{
	public:
		virtual void operator() (
			Blob<QType> &q, const Blob<QType> &g
			) const override;
	};
}

#endif // !__POISSON_METHOD_H__
