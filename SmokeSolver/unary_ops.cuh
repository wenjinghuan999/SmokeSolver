#pragma once

#ifndef __UNARY_OPS_CUH__
#define __UNARY_OPS_CUH__

#include "common.cuh"

namespace ssv
{
	namespace ops
	{
		template<typename _T>
		struct assign
		{
			typedef _T argument_type;
			typedef _T result_type;
			_T val;
			assign(_T val) : val(val) {}
			__host__ __device__ const _T &operator() (const _T &) const { return val; }
		};
	}
}

#endif // !__UNARY_OPS_CUH__
