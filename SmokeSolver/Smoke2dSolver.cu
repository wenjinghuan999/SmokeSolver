
#if 0
#include "common.cuh"
#include "Smoke2dSolver.h"
using namespace ssv;

#include "pitched_ptr.h"
#include "debug_output.h"
#include "debug_output.cuh"
using namespace ssv::output;

#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
using thrust::placeholders::_1;

void ssv::Smoke2dSolver::_InitCuda()
{
	cudaSetDevice(0);

	_data = Blob<T>(_nx, _ny);
	Blob<T> a(5, 2, 3);
	_data = a;
}


static __global__ void kernelWWW(
	cudaTextureObject_t texObj, cudaPitchedPtr anotherptr
	)
{
	size_t j = blockIdx.x;
	size_t i = threadIdx.x;

	float u = i / (float)1; 
	float v = j / (float)1;

	float *p = (float *)anotherptr.ptr;
	p[j * anotherptr.pitch / sizeof(T) + i] = tex3D<float>(texObj, u, v, 2);
}


std::ostream &operator<< (std::ostream &out, float2 q)
{
	out << q.x << "," << q.y;
	return out;
}


void Smoke2dSolver::_StepCuda()
{
	cudaSetDevice(0);

	T *p = _data.data_cpu();
	for (size_t k = 0; k < 3; k++)
	{
		for (size_t j = 0; j < _ny; j++)
		{
			for (size_t i = 0; i < _nx; i++)
			{
				p[k * _ny * _nx + j * _nx + i] = i * 100 + j * 10 + k;
			}
		}
	}

	for (size_t k = 0; k < 3; k++)
	{
		for (size_t j = 0; j < _ny; j++)
		{
			for (size_t i = 0; i < _nx; i++)
			{
				std::cout << p[k * _ny * _nx + j * _nx + i]  << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
	for (size_t k = 0; k < 3 * _nx * _ny; k++)
	{
		std::cout << p[k] << " ";
	}
	std::cout << std::endl;
	_data.copyToGpu();

	const cudaPitchedPtr *ppp = _data.data_gpu_cuda_pitched_ptr();
	T *pd = _data.data_gpu_raw();
	PrintRawGPU(pd, _data.pitch_in_elements() * _data.ny() * _data.nz(), "data:\n");

	Blob<T> another = Blob<T>(5, 2, 3);
	const cudaPitchedPtr *ppap = another.data_gpu_cuda_pitched_ptr();

	cudaTextureObject_t texObj = _data.data_texture_3d();
	
	kernelWWW<<<2, 5>>>(texObj, *ppap);

	cudaDeviceSynchronize();

	PrintRawGPU(another.data_gpu_raw(), ppap->pitch / sizeof(T) * ppap->ysize * 3, "data:\n");
	another.copyToCpu();
	p = another.data_cpu();

	std::cout << "another" << std::endl;
	for (size_t k = 0; k < 3; k++)
	{
		for (size_t j = 0; j < _ny; j++)
		{
			for (size_t i = 0; i < _nx; i++)
			{
				std::cout << p[k * _ny * _nx + j * _nx + i] << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}

	Blob<T2> u = Blob<T2>(5, 2, 3);

	Blob<byte> tp(5, 2, 3);

	BoundaryMethodClampAll<byte, T2> bnd(0, make_float2(1.f, 0.5));
	bnd(tp, u);
	PrintBlobGPU(u, "uu");

	thrust::transform(_data.data_gpu(), _data.data_gpu() + _nx*_ny*3, _data.data_gpu(), 1.f + _1 * _1);
	PrintRawGPU(pd, _data.pitch_in_elements() * _data.ny() * _data.nz(), "data:\n");
	_data.copyToCpu();
	p = _data.data_cpu();


	AdvectMethodSemiLagrangian<T> adv_lag;
	AdvectMethod<T> &adv = adv_lag;
	adv(another, _data, u);

	EulerMethodForward<T> euler_forward;
	EulerMethod<T> &euler = euler_forward;
	euler(another, another);


	texObj = _data.data_texture_2d();
	std::cout << "TEX: " << texObj << std::endl;
	texObj = _data.data_texture_2d();
	std::cout << "TEX: " << texObj << std::endl;
	texObj = _data.data_texture_2d();
	std::cout << "TEX: " << texObj << std::endl;
	texObj = _data.data_texture_3d();
	std::cout << "TEX: " << texObj << std::endl;
	texObj = _data.data_texture_3d();
	std::cout << "TEX: " << texObj << std::endl;
	texObj = _data.data_texture_3d();
	std::cout << "TEX: " << texObj << std::endl;


	std::cout << "before" << std::endl;

	PrintBlobGPU(another, "another");
	PrintBlobCPU(another, "another");
	PrintBlobGPU(_data, "_data");
	PrintBlobCPU(_data, "_data");

	std::swap(_data, another);

	std::cout << "after" << std::endl;

	PrintBlobGPU(another, "another");
	PrintBlobCPU(another, "another");
	PrintBlobGPU(_data, "_data");
	PrintBlobCPU(_data, "_data");

}

void Smoke2dSolver::_DestroyCuda()
{

}
#endif