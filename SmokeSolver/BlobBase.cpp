#include "BlobBase.h"
using namespace ssv;

BlobBase::BlobBase()
	: nx_in_bytes_(0), ny_(0), nz_(0), data_cpu_(nullptr)
{
	_InitCuda();
}

BlobBase::BlobBase(size_t nx_in_bytes, uint ny, uint nz,
                   int gpu_device, storage_t storage)
	: nx_in_bytes_(0), ny_(0), nz_(0), data_cpu_(nullptr)
{
	if (nx_in_bytes == 0 || ny == 0 || nz == 0)
	{
		throw ssv_error(error_t::SSV_ERROR_INVALID_VALUE);
	}

	nx_in_bytes_ = nx_in_bytes;
	ny_ = ny;
	nz_ = nz;

	if (underlying(storage) & underlying(storage_t::CPU))
	{
		data_cpu_ = new byte[nx_in_bytes_ * ny_ * nz_];
		if (!data_cpu_)
		{
			throw ssv_error(error_t::SSV_ERROR_OUT_OF_MEMORY_CPU);
		}
		memset(data_cpu_, 0, nx_in_bytes_ * ny_ * nz_);
	}
	
	_InitCuda(gpu_device);
}

BlobBase::BlobBase(const BlobBase &other)
	: nx_in_bytes_(0), ny_(0), nz_(0), data_cpu_(nullptr)
{
	nx_in_bytes_ = other.nx_in_bytes_;
	ny_ = other.ny_;
	nz_ = other.nz_;
	if (other.data_cpu_)
	{
		data_cpu_ = new byte[nx_in_bytes_ * ny_ * nz_];
		if (!data_cpu_)
		{
			throw ssv_error(error_t::SSV_ERROR_OUT_OF_MEMORY_CPU);
		}
		memcpy(data_cpu_, other.data_cpu_, nx_in_bytes_ * ny_ * nz_);
	}

	_CopyCuda(other, other.storage_gpu_device_);
}

BlobBase &BlobBase::operator=(const BlobBase &other)
{
	if (this != &other)
	{
		_DestroyCuda();
		if (data_cpu_)
		{
			delete[] static_cast<byte *>(data_cpu_);
			data_cpu_ = nullptr;
		}

		nx_in_bytes_ = other.nx_in_bytes_;
		ny_ = other.ny_;
		nz_ = other.nz_;
		if (other.data_cpu_)
		{
			data_cpu_ = new byte[nx_in_bytes_ * ny_ * nz_];
			if (!data_cpu_)
			{
				throw ssv_error(error_t::SSV_ERROR_OUT_OF_MEMORY_CPU);
			}
			memcpy(data_cpu_, other.data_cpu_, nx_in_bytes_ * ny_ * nz_);
		}

		_CopyCuda(other, other.storage_gpu_device_);
	}
	return *this;
}

BlobBase::BlobBase(BlobBase &&other) noexcept
	: nx_in_bytes_(other.nx_in_bytes_), ny_(other.ny_), nz_(other.nz_),
	  data_cpu_(other.data_cpu_)
{
	_MoveCuda(std::forward<BlobBase>(other));

	other.nx_in_bytes_ = 0;
	other.ny_ = 0;
	other.nz_ = 0;
	other.data_cpu_ = nullptr;
}

BlobBase &BlobBase::operator=(BlobBase &&other) noexcept
{
	if (this != &other)
	{
		_DestroyCuda();
		if (data_cpu_)
		{
			delete[] static_cast<byte *>(data_cpu_);
			data_cpu_ = nullptr;
		}

		nx_in_bytes_ = other.nx_in_bytes_;
		ny_ = other.ny_;
		nz_ = other.nz_;
		data_cpu_ = other.data_cpu_;

		_MoveCuda(std::forward<BlobBase>(other));

		other.nx_in_bytes_ = 0;
		other.ny_ = 0;
		other.nz_ = 0;
		other.data_cpu_ = nullptr;
	}
	return *this;
}

BlobBase::~BlobBase()
{
	_DestroyCuda();

	if (data_cpu_)
	{
		delete[] static_cast<byte *>(data_cpu_);
		data_cpu_ = nullptr;
	}
	nx_in_bytes_ = 0;
	ny_ = 0;
	nz_ = 0;
}
