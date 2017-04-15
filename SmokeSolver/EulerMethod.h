#pragma once

#ifndef __EULER_METHOD_H__
#define __EULER_METHOD_H__

#include "common.h"
#include "Blob.h"

namespace ssv
{
	// Solve euler problem
	//  dq
	// ---- = d
	//  dt
	template <typename QType>
	class EulerMethod
	{
	public:
		virtual void operator () (
			Blob<QType> &q, const Blob<QType> &d
			) const = 0;
	};

	template <typename QType>
	class EulerMethodForward : public EulerMethod<QType>
	{
	public:
		virtual void operator () (
			Blob<QType> &q, const Blob<QType> &d
			) const override
		{
			EulerCuda(q, d);
		}

	private:
		void EulerCuda(
			Blob<QType> &q, const Blob<QType> &d
		) const;
	};
}

#endif // !__EULER_METHOD_H__
