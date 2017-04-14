
#include "Smoke2dSolver.h"
using namespace ssv;

Smoke2dSolver::Smoke2dSolver()
{
}


Smoke2dSolver::~Smoke2dSolver()
{

}

void ssv::Smoke2dSolver::Init(size_t nx, size_t ny)
{
	if (nx == 0 || ny == 0)
	{
		throw SSV_ERROR_INVALID_VALUE;
	}

	_nx = nx;
	_ny = ny;

	_InitCuda();
}

void ssv::Smoke2dSolver::Step()
{
	_StepCuda();
}

void ssv::Smoke2dSolver::Destory()
{

}
