#pragma once

#ifndef __FORCE_METHOD_H__
#define __FORCE_METHOD_H__

#include "common.h"
#include "Blob.h"


namespace ssv
{
	// Calculate force
	class ForceMethod
	{
	public:
		template <typename FType>
		using type = std::function<void(Blob<FType> &, const Blob<real> &, const Blob<real> &)>;
	};

	// Calculate force according to density(rh) and temperature(tm)
	class ForceMethodSimple : public ForceMethod
	{
	public:
		ForceMethodSimple(real alpha, real beta, real tm0)
			: alpha_(alpha), beta_(beta), tm0_(tm0)
		{
		}

	public:
		template <typename FType>
		void operator()(
			Blob<FType> &fout, const Blob<real> &rh, const Blob<real> &tm
		) const;
	private:
		real alpha_, beta_, tm0_;
	};
}

#endif // !__FORCE_METHOD_H__
