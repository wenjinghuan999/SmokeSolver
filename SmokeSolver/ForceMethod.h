#pragma once

#ifndef __FORCE_METHOD_H__
#define __FORCE_METHOD_H__

#include "common.h"
#include "Blob.h"


namespace ssv
{
	// Calculate force
	template <typename QType, typename FType>
	class ForceMethod
	{
	public:
		// Calculate force according to density(rh) and temperature(tm)
		virtual void operator() (
			Blob<FType> &fout, const Blob<QType> &rh, const Blob<QType> &tm
			) const = 0;
	};

	template <typename QType, typename FType>
	class ForceMethodSimple : public ForceMethod<QType, FType>
	{
	public:
		virtual void operator() (
			Blob<FType> &fout, const Blob<QType> &rh, const Blob<QType> &tm
			) const override;
	};
}

#endif // !__FORCE_METHOD_H__
