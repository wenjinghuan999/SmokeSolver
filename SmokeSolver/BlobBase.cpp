
#include "BlobBase.h"
using namespace ssv;

BlobBase::BlobBase()
	: _nx_in_bytes(0), _ny(0), _nz(0), _size_in_bytes(0), _data_cpu(nullptr)
{
	memset(&_data_gpu, 0, sizeof(cudaPitchedPtr));
	memset(&_data_gpu_extent, 0, sizeof(cudaExtent));
	
	_storage_gpu_device = -1;
	_data_texture_default_2d = 0;
	_data_texture_default_3d = 0;
	_data_cuda_array = nullptr;
}

BlobBase::~BlobBase()
{
	reset();
}

void BlobBase::setSize(size_t nx_in_bytes, uint ny, uint nz,
	int gpu_device, bool cpu_copy)
{
	if (nx_in_bytes == 0 || ny == 0)
	{
		throw SSV_ERROR_INVALID_VALUE;
	}
	
	reset();

	_nx_in_bytes = nx_in_bytes;
	_ny = ny; 
	_nz = nz;
	_size_in_bytes = _nx_in_bytes * _ny * _nz;
	_storage_gpu_device = gpu_device;

	if (cpu_copy)
	{
		_data_cpu = new byte[_size_in_bytes];
		if (!_data_cpu)
		{
			throw SSV_ERROR_OUT_OF_MEMORY_CPU;
		}
	}

	_InitCuda(gpu_device);
}

void BlobBase::reset()
{
	_DestroyCuda();

	if (_data_cpu)
	{
		delete[] _data_cpu;
		_data_cpu = nullptr;
	}
	_nx_in_bytes = 0;
	_ny = 0;
	_nz = 0;
	_size_in_bytes = 0;
}
