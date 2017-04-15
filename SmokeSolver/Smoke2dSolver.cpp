
#include "Smoke2dSolver.h"
using namespace ssv;

Smoke2dSolver::Smoke2dSolver()
{
}


Smoke2dSolver::~Smoke2dSolver()
{

}

void Smoke2dSolver::setSize(size_t nx, size_t ny)
{
	if (nx == 0 || ny == 0)
	{
		throw SSV_ERROR_INVALID_VALUE;
	}

	_nx = nx;
	_ny = ny;
}

void Smoke2dSolver::init()
{
	if (_nx == 0 || _ny == 0)
	{
		throw SSV_ERROR_NOT_INITIALIZED;
	}
	_InitCuda();
}

void Smoke2dSolver::step()
{
	_StepCuda();
}

void Smoke2dSolver::destory()
{
	_DestroyCuda();
}
