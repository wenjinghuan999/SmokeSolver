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
	class PoissonMethod
	{
	public:
		template <typename QType>
		using type = std::function<void(Blob<QType> &, const Blob<QType> &)>;
	};

	class PoissonMethodGS : public PoissonMethod
	{
	public:
		PoissonMethodGS(ssv::uint iterations)
			: _iterations(iterations) {}
		template <typename QType>
		void operator() (
			Blob<QType> &q, const Blob<QType> &g
			) const;
	private:
		ssv::uint _iterations;
	};

	class PoissonMethodVCycle : public PoissonMethod
	{
	public:
		PoissonMethodVCycle(ssv::uint levels, ssv::uint iterations)
			: _levels(levels), _gs(iterations) {}
		template <typename QType>
		void operator() (
			Blob<QType> &q, const Blob<QType> &g
			) const;
	private:
		ssv::uint _levels;
		PoissonMethodGS _gs;
	};
}

#endif // !__POISSON_METHOD_H__
