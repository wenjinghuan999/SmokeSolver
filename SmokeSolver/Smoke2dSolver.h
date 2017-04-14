#pragma once

#ifndef __SMOKE2D_SOLVER_H__
#define __SMOKE2D_SOLVER_H__

#include "common.h"
#include "Blob.h"

namespace ssv
{
	class Smoke2dSolver
	{
	public:
		Smoke2dSolver();
		~Smoke2dSolver();
	public:
		void Init(size_t nx, size_t ny);
		void Step();
		void Destory();
	private:
		void _InitCuda();
		void _StepCuda();
	private:
		size_t _nx, _ny;
		Blob<T> _data;
	};
}

#endif // !__SMOKE2D_SOLVER_H__
