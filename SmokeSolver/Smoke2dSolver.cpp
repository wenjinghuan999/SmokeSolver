
#include "Smoke2dSolver.h"
#include "debug_output.h"
using namespace ssv;


void Smoke2dSolver::init()
{
	if (_nx == 0 || _ny == 0)
	{
		throw SSV_ERROR_NOT_INITIALIZED;
	}
	//_InitCuda();
}

void Smoke2dSolver::step()
{
	Blob<byte> tp(5, 5);
	Blob<T> rh(5, 5);
	Blob<T> tm(5, 5);
	Blob<T2> u(5, 5);
	Blob<T2> f(5, 5);

	rh.copyToCpu(); output::PrintBlobCPU(rh, "rh");
	tm.copyToCpu(); output::PrintBlobCPU(tm, "tm");

	_boundary(rh, tp);
	_boundary2(u, tp);

	rh.copyToCpu(); output::PrintBlobCPU(rh, "rh");
	u.copyToCpu(); output::PrintBlobCPU(u, "u");

	_force(f, rh, tm);

	f.copyToCpu(); output::PrintBlobCPU(f, "f");

	_euler2(u, f);

	u.copyToCpu(); output::PrintBlobCPU(u, "u");

	_advect(rh, rh, u);
	_advect(tm, tm, u);
	_advect2(u, u, u);

	rh.copyToCpu(); output::PrintBlobCPU(rh, "rh");
	tm.copyToCpu(); output::PrintBlobCPU(tm, "tm");
	u.copyToCpu(); output::PrintBlobCPU(u, "u");

	_poisson(rh, rh);
	_poisson2(u, u);

	rh.copyToCpu(); output::PrintBlobCPU(rh, "rh");
	u.copyToCpu(); output::PrintBlobCPU(u, "u");


	//_StepCuda();
}

void Smoke2dSolver::destory()
{
	//_DestroyCuda();
}
