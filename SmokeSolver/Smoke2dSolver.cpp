#include "Smoke2dSolver.h"
#include "BlobMath.h"
#include "debug_output.h"
using namespace ssv;

#include <random>


void Smoke2DSolver::init()
{
	if (nx_ == 0 || ny_ == 0)
	{
		throw ssv_error(error_t::SSV_ERROR_NOT_INITIALIZED);
	}

	tp_ = Blob<byte>(nx_, ny_);
	rh_[0] = Blob<real>(nx_, ny_);
	rh_[1] = Blob<real>(nx_, ny_);
	tm_[0] = Blob<real>(nx_, ny_);
	tm_[1] = Blob<real>(nx_, ny_);
	u_ = Blob<real2>(nx_, ny_);
	w_ = Blob<real>(nx_, ny_);
	f_ = Blob<real2>(nx_, ny_);
	temp1_a_ = Blob<real>(nx_, ny_);
	temp1_b_ = Blob<real>(nx_, ny_);
	temp2_c_ = Blob<real2>(nx_, ny_);
	temp2_d_ = Blob<real2>(nx_, ny_);

	tp_.set_data_cube_cpu(underlying(CellType::CELL_TYPE_WALL), 0, 0, 0, ny_ - 1u);
	tp_.set_data_cube_cpu(underlying(CellType::CELL_TYPE_WALL), nx_ - 1u, nx_ - 1u, 0, ny_ - 1u);
	tp_.set_data_cube_cpu(underlying(CellType::CELL_TYPE_WALL), 0, nx_ - 1u, 0, 0);
	tp_.set_data_cube_cpu(underlying(CellType::CELL_TYPE_WALL), 0, nx_ - 1u, ny_ - 1u, ny_ - 1u);
	tp_.sync_cpu_to_gpu();

	ping_ = 0;
}

void Smoke2DSolver::add_source(uint x0, uint x1, uint y0, uint y1)
{
	tp_.set_data_cube_cpu(underlying(CellType::CELL_TYPE_SOURCE), x0, x1, y0, y1);
	tp_.sync_cpu_to_gpu();
}

void Smoke2DSolver::gen_noise()
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dis(0.f, 32768.f);
	real2 offset = make_real2(dis(gen), dis(gen));
	//simplex_2d(_rh[ping], _temp1a, _temp1b, make_real2(4.f, 4.f), offset);
	simplex_2d(temp1_a_, make_real2(16.f, 16.f), offset);
	offset = make_real2(dis(gen), dis(gen));
	simplex_2d(temp1_b_, make_real2(16.f, 16.f), offset);
	zip(u_, temp1_a_, temp1_b_);

	offset = make_real2(dis(gen), dis(gen));
	simplex_2d(rh_[ping_], make_real2(16.f, 16.f), offset);
	simplex_2d(tm_[ping_], make_real2(16.f, 16.f), offset);
}

void *Smoke2DSolver::get_data(Property property, size_t *size)
{
	switch (property)
	{
	case Property::PROPERTY_DENSITY:
		if (size != nullptr) *size = rh_[ping_].size_cpu_in_bytes();
		rh_[ping_].sync_gpu_to_cpu();
		return rh_[ping_].data_cpu();
	case Property::PROPERTY_TEMPERATURE:
		if (size != nullptr) *size = tm_[ping_].size_cpu_in_bytes();
		tm_[ping_].sync_gpu_to_cpu();
		return tm_[ping_].data_cpu();
	case Property::PROPERTY_VELOCITY:
		if (size != nullptr) *size = u_.size_cpu_in_bytes();
		u_.sync_gpu_to_cpu();
		return u_.data_cpu();
	default:
		if (size != nullptr) *size = 0;
		return nullptr;
	}
}

void Smoke2DSolver::save_data(const std::string &filename)
{
	rh_[ping_].sync_gpu_to_cpu();
	output::save_blob_cpu(rh_[ping_], filename + "_rh");
	u_.sync_gpu_to_cpu();
	output::save_blob_cpu(u_, filename + "_u");
}

void Smoke2DSolver::step()
{
	Blob<byte> &tp = tp_;
	Blob<real> &rh = rh_[ping_], &rh2 = rh_[ping_ ^ 1];
	Blob<real> &tm = tm_[ping_], &tm2 = tm_[ping_ ^ 1];
	Blob<real2> &u = u_;
	Blob<real> &w = w_;
	Blob<real2> &f = f_;
	Blob<real> &temp1_a = temp1_a_, &temp1_b = temp1_b_;
	Blob<real2> &u1 = temp2_c_, &u2 = temp2_d_;
	Blob<real2> &eta = temp2_c_;

	boundary_den_(rh, tp);
	boundary_den_(tm, tp);
	boundary_vel_(u, tp);

	force_(f, rh, tm);
	euler_vel_(u, f);

	curl(w, u);
	temp1_a = w;
	abs(temp1_a);
	gradient(eta, temp1_a);
	normalize(eta);
	unzip(temp1_a, temp1_b, eta);
	temp1_a *= w;
	neg(temp1_a);
	temp1_b *= w;
	zip(f, temp1_b, temp1_a);
	f *= make_real2(0.5f, 0.5f);

	euler_vel_(u, f);

	laplacian_2d(u1, u);
	u1 *= make_real2(0.2f, 0.2f);
	euler_vel_(u, u1);

	laplacian_2d(temp1_a, rh);
	temp1_a *= 0.1f;
	euler_den_(rh, temp1_a);

	laplacian_2d(temp1_a, tm);
	temp1_a *= 0.1f;
	euler_den_(tm, temp1_a);

	advect_den_(rh2, rh, u);
	advect_den_(tm2, tm, u);
	advect_vel_(u1, u, u);

	divergence(temp1_a, u1);
	poisson_(temp1_b, temp1_a);
	gradient(u2, temp1_b);

	sub(u, u1, u2);
	ping_ ^= 1;
}

void Smoke2DSolver::destory()
{
}
