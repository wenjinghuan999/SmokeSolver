#pragma once

#ifndef __ADVECT_METHOD_H__
#define __ADVECT_METHOD_H__

#include "common.h"
#include "Blob.h"

#include <functional>


namespace ssv
{
	// Solve advection problem
	//  dq          __
	// ---- = -(u . \/) q
	//  dt
	class AdvectMethod
	{
	public:
		template <typename QType, typename UType>
		using type = std::function<void(Blob<QType> &, const Blob<QType> &, const Blob<UType> &)>;
	};

	class AdvectMethodSemiLagrangian : public AdvectMethod
	{
	public:
		// 2D
		template <typename QType>
		void operator()(
			Blob<QType> &qout, const Blob<QType> &q, const Blob<real2> &u
		) const;
		// 3D
		template <typename QType>
		void operator()(
			Blob<QType> &qout, const Blob<QType> &q, const Blob<real4> &u
		) const;
	};

	class AdvectMethodRK3 : public AdvectMethod
	{
	public:
		// 2D
		template <typename QType>
		void operator()(
			Blob<QType> &qout, const Blob<QType> &q, const Blob<real2> &u
		) const;
		// 3D
		template <typename QType>
		void operator()(
			Blob<QType> &qout, const Blob<QType> &q, const Blob<real4> &u
		) const;
	};

	class AdvectMethodRK4 : public AdvectMethod
	{
	public:
		// 2D
		template <typename QType>
		void operator()(
			Blob<QType> &qout, const Blob<QType> &q, const Blob<real2> &u
		) const;
		// 3D
		template <typename QType>
		void operator()(
			Blob<QType> &qout, const Blob<QType> &q, const Blob<real4> &u
		) const;
	};

	class AdvectMethodBFECC : public AdvectMethod
	{
	public:
		// 2D
		template <typename QType>
		void operator()(
			Blob<QType> &qout, const Blob<QType> &q, const Blob<real2> &u
		) const;
		// 3D
		template <typename QType>
		void operator()(
			Blob<QType> &qout, const Blob<QType> &q, const Blob<real4> &u
		) const;
	};

	class AdvectMethodMacCormack : public AdvectMethod
	{
	public:
		// 2D
		template <typename QType>
		void operator()(
			Blob<QType> &qout, const Blob<QType> &q, const Blob<real2> &u
		) const;
		// 3D
		template <typename QType>
		void operator()(
			Blob<QType> &qout, const Blob<QType> &q, const Blob<real4> &u
		) const;
	};
}

#endif // !__ADVECT_METHOD_H__
