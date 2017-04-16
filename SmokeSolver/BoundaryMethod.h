#pragma once

#ifndef __BOUNDARY_METHOD_H__
#define __BOUNDARY_METHOD_H__

#include "common.h"
#include "Blob.h"


namespace ssv
{
	// Deal with boundary conditions
	template <typename QType, typename TpType>
	class BoundaryMethod
	{
	public:
		virtual void operator () (
			Blob<QType> &q, const Blob<TpType> &tp
			) const = 0;
	};


	// Clamp q to q1 for tp = tp1, ...
	// q remains the same for other tp
	template <typename QType, typename TpType>
	class BoundaryMethodClamp
	{
	public:
		virtual void operator () (
			Blob<QType> &q, const Blob<TpType> &tp, TpType tp1, QType q1
			) const;
		virtual void operator () (
			Blob<QType> &q, const Blob<TpType> &tp, TpType tp1, QType q1, TpType tp2, QType q2
			) const;
	};
}

#endif // !__BOUNDARY_METHOD_H__
