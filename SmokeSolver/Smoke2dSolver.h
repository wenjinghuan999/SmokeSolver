#pragma once

#ifndef __SMOKE2D_SOLVER_H__
#define __SMOKE2D_SOLVER_H__

#include "common.h"
#include "SmokeSolver.h"
#include "Blob.h"

namespace ssv
{
	class Smoke2dSolver: public SmokeSolver
	{
	public:
		Smoke2dSolver();
		~Smoke2dSolver();
	
	public:
		void setSize(uint nx, uint ny);
		//void setEulerMethod();
		//void setAdvectionMethod();
		//void setProjectionMethod();

	public:
		virtual void init();
		virtual void step();
		virtual void destory();

	private:
		void _InitCuda();
		void _StepCuda();
		void _DestroyCuda();
	
	private:
		uint _nx, _ny;
		Blob<T> _data;
	};
}

#endif // !__SMOKE2D_SOLVER_H__
