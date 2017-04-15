#pragma once

#ifndef __ADVECTION_METHOD_H__
#define __ADVECTION_METHOD_H__

#include "common.h"
#include "Blob.h"

namespace ssv
{
	class AdvectionMethod2d
	{
	public:
		virtual void operator () (Blob<T> &q, Blob<T2> &u) = 0;
		virtual void operator () (Blob<T2> &q, Blob<T2> &u) = 0;
	};

	class AcvectionMethod2dSemiLagrangian
	{
	public:
		virtual void operator () (Blob<T> &q, Blob<T2> &u)
		{

		}

		virtual void operator () (Blob<T2> &q, Blob<T2> &u)
		{

		}
	private:
		void AdvectCuda(Blob<T> &q, Blob<T2> &u);
	};
}


#endif // !__ADVECTION_METHOD_H__
