
#include "Smoke2dSolver.h"
#include "BlobMath.h"
#include "debug_output.h"
using namespace ssv;

#include <fstream>
#include <sstream>
#include <random>


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
	_temp1a = Blob<T>(_nx, _ny);
	_temp1b = Blob<T>(_nx, _ny);
	_temp2a = Blob<T2>(_nx, _ny);
	_temp2b = Blob<T2>(_nx, _ny);

	_tp.setDataCubeCpu(underlying(CellType::CellTypeWall), 0, 0, 0, _ny - 1u);
	_tp.setDataCubeCpu(underlying(CellType::CellTypeWall), _nx - 1u, _nx - 1u, 0, _ny - 1u);
	_tp.setDataCubeCpu(underlying(CellType::CellTypeWall), 0, _nx - 1u, 0, 0);
	_tp.setDataCubeCpu(underlying(CellType::CellTypeWall), 0, _nx - 1u, _ny - 1u, _ny - 1u);
	_tp.syncCpu2Gpu();

	ping = 0;
	get_data_ping = 0;
}

void Smoke2dSolver::addSource(uint x0, uint x1, uint y0, uint y1)
{
	_tp.setDataCubeCpu(underlying(CellType::CellTypeSource), x0, x1, y0, y1);
	_tp.syncCpu2Gpu();
}

void Smoke2dSolver::genNoise()
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dis(0.f, 32768.f);
	T2 offset = make_T2(dis(gen), dis(gen));
	simplex2d(_rh[ping], _temp1a, _temp1b, make_T2(16.f, 16.f), offset);
	zip(_u, _temp1a, _temp1b);

	offset = make_T2(dis(gen), dis(gen));
	simplex2d(_rh[ping], make_T2(16.f, 16.f), offset);
	simplex2d(_tm[ping], make_T2(16.f, 16.f), offset);

	get_data_ping = 0;
}

void *Smoke2dSolver::getData(size_t *size)
{
	if (get_data_ping == 0)
	{
		get_data_ping ^= 1;
		if (size != nullptr)
		{
			*size = _rh[ping].size_cpu_in_bytes();
		}
		_rh[ping].syncGpu2Cpu();
		return _rh[ping].data_cpu();
	}
	else
	{
		get_data_ping ^= 1;
		if (size != nullptr)
		{
			*size = _u.size_cpu_in_bytes();
		}
		_u.syncGpu2Cpu();
		return _u.data_cpu();
	}
}

void Smoke2dSolver::saveData(const std::string &filename)
{
	_rh[ping].syncGpu2Cpu();
	output::SaveBlobCPU(_rh[ping], filename + "_rh");
	_u.syncGpu2Cpu();
	output::SaveBlobCPU(_u, filename + "_u");
}

void Smoke2dSolver::step()
{
	Blob<byte> &tp = _tp;
	Blob<T> &rh = _rh[ping], &rh2 = _rh[ping ^ 1];
	Blob<T> &tm = _tm[ping], &tm2 = _tm[ping ^ 1];
	Blob<T2> &u = _u;
	Blob<T2> &f= _f;
	Blob<T> &temp1 = _temp1a, &temp2 = _temp1b;
	Blob<T2> &u1 = _temp2a, &u2 = _temp2b;

	//tp.syncGpu2Cpu(); output::PrintBlobCPU(tp, "tp");

	_boundary(rh, tp);
	_boundary(tm, tp);
	_boundary2(u, tp);

	//static int frame_no = 0;
	//std::stringstream ss;
	//ss << "data/" << frame_no << ".txt";
	//std::ofstream fout(ss.str());
	//frame_no++;
	//u.syncGpu2Cpu(); output::PrintBlobCPU(u, "u");

	_force(f, rh, tm);

	//f.syncGpu2Cpu(); output::PrintBlobCPU(f, "f");

	_euler2(u, f);

	//u.syncGpu2Cpu(); output::PrintBlobCPU(u, "u");

	laplacian2d(u1, u);
	u1 *= make_T2(0.2f, 0.2f);
	_euler2(u, u1);

	laplacian2d(temp1, rh);
	temp1 *= 0.01f;
	_euler(rh, temp1);

	laplacian2d(temp1, tm);
	temp1 *= 0.01f;
	_euler(tm, temp1);

	//u.syncGpu2Cpu(); output::PrintBlobCPU(u, "u");

	_advect(rh2, rh, u);
	_advect(tm2, tm, u);
	_advect2(u1, u, u);

	//rh2.syncGpu2Cpu(); output::PrintBlobCPU(rh2, "rh");
	//tm2.syncGpu2Cpu(); output::PrintBlobCPU(tm2, "tm");
	//u1.syncGpu2Cpu(); output::PrintBlobCPU(u1, "u1");

	divergence(temp1, u1);
	_poisson(temp2, temp1);
	gradient(u2, temp2);

	sub(u, u1, u2);

	//u.syncGpu2Cpu(); output::PrintBlobCPU(u, "u");

	//std::cout << "rh size = " << rh2.size_cpu_in_bytes() << std::endl;
	//rh2.syncGpu2Cpu(); output::PrintBlobCPU(rh2, "rh");

	ping ^= 1;
}

void Smoke2dSolver::destory()
{

}
