
#include "Smoke2dSolver.h"
using namespace ssv;


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
