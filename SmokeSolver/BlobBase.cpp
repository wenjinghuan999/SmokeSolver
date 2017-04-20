
#include "BlobBase.h"
using namespace ssv;

BlobBase::BlobBase()
	: _nx_in_bytes(0), _ny(0), _nz(0), _data_cpu(nullptr)
{
	_InitCuda();
}

BlobBase::BlobBase(size_t nx_in_bytes, uint ny, uint nz,
	int gpu_device, bool cpu_copy)
{
	if (nx_in_bytes == 0 || ny == 0 || nz == 0)
	{
		throw SSV_ERROR_INVALID_VALUE;
	}

	_nx_in_bytes = nx_in_bytes;
	_ny = ny; 
	_nz = nz;

	if (cpu_copy)
	{
		_data_cpu = new byte[_nx_in_bytes * _ny * _nz];
		if (!_data_cpu)
		{
			throw SSV_ERROR_OUT_OF_MEMORY_CPU;
		}
	}

	_InitCuda(gpu_device);
}

BlobBase::BlobBase(const BlobBase &other)
{
	_nx_in_bytes = other._nx_in_bytes;
	_ny = other._ny;
	_nz = other._nz;
	if (other._data_cpu)
	{
		_data_cpu = new byte[_nx_in_bytes * _ny * _nz];
		if (!_data_cpu)
		{
			throw SSV_ERROR_OUT_OF_MEMORY_CPU;
		}
		memcpy(_data_cpu, other._data_cpu, _nx_in_bytes * _ny * _nz);
	}

	_CopyCuda(other, other._storage_gpu_device);
}

BlobBase &BlobBase::operator= (const BlobBase &other)
{
	if (this != &other)
	{
		_DestroyCuda();
		if (_data_cpu)
		{
			delete[] _data_cpu;
			_data_cpu = nullptr;
		}

		_nx_in_bytes = other._nx_in_bytes;
		_ny = other._ny;
		_nz = other._nz;
		if (other._data_cpu)
		{
			_data_cpu = new byte[_nx_in_bytes * _ny * _nz];
			if (!_data_cpu)
			{
				throw SSV_ERROR_OUT_OF_MEMORY_CPU;
			}
			memcpy(_data_cpu, other._data_cpu, _nx_in_bytes * _ny * _nz);
		}

		_CopyCuda(other, other._storage_gpu_device);
	}
	return *this;
}

BlobBase::BlobBase(BlobBase &&other)
	: _nx_in_bytes(other._nx_in_bytes), _ny(other._ny), _nz(other._nz),
	_data_cpu(other._data_cpu)
{
	_MoveCuda(std::forward<BlobBase>(other));

	other._nx_in_bytes = 0;
	other._ny = 0;
	other._nz = 0;
	other._data_cpu = nullptr;
}

BlobBase &BlobBase::operator= (BlobBase &&other)
{
	if (this != &other)
	{
		_DestroyCuda();
		if (_data_cpu)
		{
			delete[] _data_cpu;
			_data_cpu = nullptr;
		}

		_nx_in_bytes = other._nx_in_bytes;
		_ny = other._ny;
		_nz = other._nz;
		_data_cpu = other._data_cpu;

		_MoveCuda(std::forward<BlobBase>(other));

		other._nx_in_bytes = 0;
		other._ny = 0;
		other._nz = 0;
		other._data_cpu = nullptr;
	}
	return *this;
}

BlobBase::~BlobBase()
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
}
