
#include "common.cuh"
#include "Smoke2dSolver.h"
using namespace ssv;

#include "pitched_ptr.h"

#include "AdvectionMethod.h"
#include "EulerMethod.h"

#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
using thrust::placeholders::_1;

void ssv::Smoke2dSolver::_InitCuda()
{
	cudaSetDevice(0);

	_data.setSize(_nx, _ny);
	_data.setSize(5, 2, 3);
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
	Print(pd, _data.pitch_in_elements() * _data.ny() * _data.nz(), "data:\n");

	Blob<T> another;
	another.setSize(5, 2, 3);
	const cudaPitchedPtr *ppap = another.data_gpu_cuda_pitched_ptr();

	cudaTextureObject_t texObj = _data.data_texture_3d();
	
	kernelWWW<<<2, 5>>>(texObj, *ppap);

	cudaDeviceSynchronize();

	Print(another.data_gpu_raw(), ppap->pitch / sizeof(T) * ppap->ysize * 3, "data:\n");
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


	thrust::transform(_data.data_gpu(), _data.data_gpu() + _nx*_ny*3, _data.data_gpu(), 1.f + _1 * _1);
	Print(pd, _data.pitch_in_elements() * _data.ny() * _data.nz(), "data:\n");
	_data.copyToCpu();
	p = _data.data_cpu();

	Blob<T2> u;
	u.setSize(5, 2, 3);

	AdvectionMethodSemiLagrangian<T> adv_lag;
	AdvectionMethod<T> &adv = adv_lag;
	adv(another, _data, u);

	EulerMethodForward<T> euler_forward;
	EulerMethod<T> &euler = euler_forward;
	euler(another, another);


	std::cout << std::endl;
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

	for (size_t k = 0; k < _nx * _ny * 3; k++)
	{
		std::cout << p[k] << " ";
	}
	std::cout << std::endl;

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


	Print(another.data_gpu_raw(), another.pitch_in_elements() * another.ny() * another.nz(), "data:\n");
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

	texObj = _data.data_texture_3d();
	kernelWWW << <2, 5 >> >(texObj, *ppap);

	cudaDeviceSynchronize();

	Print(another.data_gpu_raw(), ppap->pitch / sizeof(T) * ppap->ysize * 3, "data:\n");
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

}

void Smoke2dSolver::_DestroyCuda()
{

}
