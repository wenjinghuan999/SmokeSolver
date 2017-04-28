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
		using type = std::function<void(Blob<FType> &, const Blob<T> &, const Blob<T> &)>;
	};

	// Calculate force according to density(rh) and temperature(tm)
	class ForceMethodSimple : public ForceMethod
	{
	public:
		ForceMethodSimple(T alpha, T beta, T tm0)
			: _alpha(alpha), _beta(beta), _tm0(tm0) {}
	public:
		template <typename FType>
		void operator() (
			Blob<FType> &fout, const Blob<T> &rh, const Blob<T> &tm
			) const;
	private:
		T _alpha, _beta, _tm0;
	};
}

#endif // !__FORCE_METHOD_H__
