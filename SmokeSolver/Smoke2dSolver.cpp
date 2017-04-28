
#include "Smoke2dSolver.h"
#include "BlobMath.h"
#include "debug_output.h"
using namespace ssv;


void Smoke2dSolver::init()
{
	if (_nx == 0 || _ny == 0)
	{
		throw error_t::SSV_ERROR_NOT_INITIALIZED;
	}

	_tp = Blob<byte>(_nx, _ny);
	_rh[0] = Blob<T>(_nx, _ny);
	_rh[1] = Blob<T>(_nx, _ny);
	_tm[0] = Blob<T>(_nx, _ny);
	_tm[1] = Blob<T>(_nx, _ny);
	_u = Blob<T2>(_nx, _ny);
	_f = Blob<T2>(_nx, _ny);
	_temp = Blob<T>(_nx, _ny);
	_temp2a = Blob<T2>(_nx, _ny);
	_temp2b = Blob<T2>(_nx, _ny);

	_tp.setDataCubeCpu(underlying(CellType::CellTypeWall), 0, 0, 0, _ny - 1u);
	_tp.setDataCubeCpu(underlying(CellType::CellTypeWall), _nx - 1u, _nx - 1u, 0, _ny - 1u);
	_tp.setDataCubeCpu(underlying(CellType::CellTypeWall), 0, _nx - 1u, 0, 0);
	_tp.setDataCubeCpu(underlying(CellType::CellTypeWall), 0, _nx - 1u, _ny - 1u, _ny - 1u);
	_tp.copyToGpu();

	ping = 0;
}

void ssv::Smoke2dSolver::addSource(uint x0, uint x1, uint y0, uint y1)
{
	_tp.setDataCubeCpu(underlying(CellType::CellTypeSource), x0, x1, y0, y1);
	_tp.copyToGpu();
}

void Smoke2dSolver::step()
{
	Blob<byte> &tp = _tp;
	Blob<T> &rh = _rh[ping], &rh2 = _rh[ping ^ 1];
	Blob<T> &tm = _tm[ping], &tm2 = _tm[ping ^ 1];
	Blob<T2> &u = _u;
	Blob<T2> &f= _f;
	Blob<T> &temp = _temp;
	Blob<T2> &u1 = _temp2a, &u2 = _temp2b;

	//tp.copyToCpu(); output::PrintBlobCPU(tp, "tp");

	_boundary(rh, tp);
	_boundary(tm, tp);
	_boundary2(u, tp);

	rh.copyToCpu(); output::PrintBlobCPU(rh, "rh");
	//u.copyToCpu(); output::PrintBlobCPU(u, "u");

	_force(f, rh, tm);

	//f.copyToCpu(); output::PrintBlobCPU(f, "f");

	_euler2(u, f);

	//u.copyToCpu(); output::PrintBlobCPU(u, "u");

	laplacian2d(u1, u);
	_euler2(u, u1);

	//u.copyToCpu(); output::PrintBlobCPU(u, "u");

	_advect(rh2, rh, u);
	_advect(tm2, tm, u);
	_advect2(u1, u, u);

	//rh2.copyToCpu(); output::PrintBlobCPU(rh2, "rh");
	//tm2.copyToCpu(); output::PrintBlobCPU(tm2, "tm");
	//u1.copyToCpu(); output::PrintBlobCPU(u1, "u1");

	divergence(temp, u1);
	_poisson(temp, temp);
	gradient(u2, temp);

	sub(u, u1, u2);

	//u.copyToCpu(); output::PrintBlobCPU(u, "u");

	ping ^= 1;
}

void Smoke2dSolver::destory()
{

}
