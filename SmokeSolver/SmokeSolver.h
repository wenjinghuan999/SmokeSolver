#pragma once

#ifndef __SMOKER_SOLVER_H__
#define __SMOKER_SOLVER_H__

#include "common.h"


namespace ssv
{
	class SmokeSolver
	{
	public:
		virtual void init() = 0;
		virtual void step() = 0;
		virtual void destory() = 0;
	};
}


#endif // !__SMOKER_SOLVER_H__
