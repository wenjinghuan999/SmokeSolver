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
		PoissonMethodGS(ssv::uint iterations)
			: _iterations(iterations) {}
		virtual void operator() (
			Blob<QType> &q, const Blob<QType> &g
			) const override;
	private:
		ssv::uint _iterations;
	};

	template <typename QType>
	class PoissonMethodVCycle : public PoissonMethod<QType>
	{
	public:
		PoissonMethodVCycle(ssv::uint levels, ssv::uint iterations)
			: _levels(levels), _gs(iterations) {}
		virtual void operator() (
			Blob<QType> &q, const Blob<QType> &g
			) const override;
	private:
		ssv::uint _levels;
		PoissonMethodGS _gs;
		std::vector<Blob<QType> > _buffers;
	};
}

#endif // !__POISSON_METHOD_H__
