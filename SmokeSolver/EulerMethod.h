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
	class EulerMethod
	{
	public:
		template <typename QType>
		using type = std::function<void(Blob<QType> &, const Blob<QType> &)>;
	};

	class EulerMethodForward : public EulerMethod
	{
	public:
		template <typename QType>
		void operator()(
			Blob<QType> &q, const Blob<QType> &d
		) const;
	};
}

#endif // !__EULER_METHOD_H__
