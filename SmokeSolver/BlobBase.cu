
#include "common.cuh"
#include "pitched_ptr.h"
#include "BlobBase.h"
using namespace ssv;

#include "thrust/copy.h"


void BlobBase::syncGpu2Cpu()
{
	if (_storage_gpu_device < 0) throw error_t::SSV_ERROR_NOT_INITIALIZED;
	checkCudaErrorAndThrow(cudaSetDevice(_storage_gpu_device),
		error_t::SSV_ERROR_DEVICE_NOT_READY);
	if (_data_cpu == nullptr) return;

	copyTo(_data_cpu, storage_t::GPU, storage_t::CPU);
}

void BlobBase::syncCpu2Gpu()
{
	if (_storage_gpu_device < 0) throw error_t::SSV_ERROR_NOT_INITIALIZED;
	checkCudaErrorAndThrow(cudaSetDevice(_storage_gpu_device),
		error_t::SSV_ERROR_DEVICE_NOT_READY);
	if (_data_cpu == nullptr) return;

	copyFrom(_data_cpu, storage_t::CPU, storage_t::GPU);
}

namespace
{
	inline enum cudaMemcpyKind make_cudaMemcpyKind(
		BlobBase::storage_t from, BlobBase::storage_t to
	)
	{
		if (from == BlobBase::storage_t::CPU)
		{
			if (to == BlobBase::storage_t::CPU)
				return cudaMemcpyHostToHost;
			else
				return cudaMemcpyHostToDevice;
		}
		else
		{
			if (to == BlobBase::storage_t::CPU)
				return cudaMemcpyDeviceToHost;
			else
				return cudaMemcpyDeviceToDevice;
		}
	}
}

void BlobBase::copyTo(void *dst, storage_t from, storage_t to) const
{
	if ((from != storage_t::CPU && from != storage_t::GPU)
		|| (to != storage_t::CPU && to != storage_t::GPU)
		|| dst == nullptr)
	{
		throw error_t::SSV_ERROR_INVALID_VALUE;
	}

	cudaPitchedPtr dst_pitched_ptr =
		make_cudaPitchedPtr(dst, _nx_in_bytes, _nx_in_bytes, _ny);
	cudaPitchedPtr data_cpu_pitched_ptr =
		make_cudaPitchedPtr(_data_cpu, _nx_in_bytes, _nx_in_bytes, _ny);
	cudaMemcpy3DParms params = { 0 };
	if (from == storage_t::CPU)
		params.srcPtr = data_cpu_pitched_ptr;
	else
		params.srcPtr = _data_gpu;
	params.dstPtr = dst_pitched_ptr;
	params.kind = make_cudaMemcpyKind(from, to);
	params.extent = _data_gpu_extent;
	checkCudaErrorAndThrow(cudaMemcpy3D(&params),
		error_t::SSV_ERROR_INVALID_VALUE);
}

void BlobBase::copyTo(cudaPitchedPtr *dst, storage_t from, storage_t to) const
{
	if ((from != storage_t::CPU && from != storage_t::GPU)
		|| (to != storage_t::CPU && to != storage_t::GPU)
		|| dst == nullptr || dst->ptr == nullptr)
	{
		throw error_t::SSV_ERROR_INVALID_VALUE;
	}

	cudaPitchedPtr data_cpu_pitched_ptr =
		make_cudaPitchedPtr(_data_cpu, _nx_in_bytes, _nx_in_bytes, _ny);
	cudaMemcpy3DParms params = { 0 };
	if (from == storage_t::CPU)
		params.srcPtr = data_cpu_pitched_ptr;
	else
		params.srcPtr = _data_gpu;
	params.dstPtr = *dst;
	params.kind = make_cudaMemcpyKind(from, to);
	params.extent = _data_gpu_extent;
	checkCudaErrorAndThrow(cudaMemcpy3D(&params),
		error_t::SSV_ERROR_INVALID_VALUE);
}

void BlobBase::copyFrom(void *src, storage_t from, storage_t to)
{
	if ((from != storage_t::CPU && from != storage_t::GPU)
		|| src == nullptr)
	{
		throw error_t::SSV_ERROR_INVALID_VALUE;
	}

	cudaPitchedPtr src_pitched_ptr =
		make_cudaPitchedPtr(src, _nx_in_bytes, _nx_in_bytes, _ny);
	cudaPitchedPtr data_cpu_pitched_ptr =
		make_cudaPitchedPtr(_data_cpu, _nx_in_bytes, _nx_in_bytes, _ny);
	cudaMemcpy3DParms params = { 0 };
	if (to == storage_t::CPU)
		params.dstPtr = data_cpu_pitched_ptr;
	else
		params.dstPtr = _data_gpu;
	params.srcPtr = src_pitched_ptr;
	params.kind = make_cudaMemcpyKind(from, to);
	params.extent = _data_gpu_extent;
	checkCudaErrorAndThrow(cudaMemcpy3D(&params),
		error_t::SSV_ERROR_INVALID_VALUE);
}

void BlobBase::copyFrom(cudaPitchedPtr *src, storage_t from, storage_t to)
{
	if ((from != storage_t::CPU && from != storage_t::GPU)
		|| src == nullptr || src->ptr == nullptr)
	{
		throw error_t::SSV_ERROR_INVALID_VALUE;
	}

	cudaPitchedPtr data_cpu_pitched_ptr =
		make_cudaPitchedPtr(_data_cpu, _nx_in_bytes, _nx_in_bytes, _ny);
	cudaMemcpy3DParms params = { 0 };
	if (from == storage_t::CPU)
		params.dstPtr = data_cpu_pitched_ptr;
	else
		params.dstPtr = _data_gpu;
	params.srcPtr = *src;
	params.kind = make_cudaMemcpyKind(from, to);
	params.extent = _data_gpu_extent;
	checkCudaErrorAndThrow(cudaMemcpy3D(&params),
		error_t::SSV_ERROR_INVALID_VALUE);
}

void BlobBase::destroyTexture(cudaTextureObject_t texture_object)
{
	if (_storage_gpu_device < 0) throw error_t::SSV_ERROR_NOT_INITIALIZED;
	checkCudaErrorAndThrow(cudaSetDevice(_storage_gpu_device),
		error_t::SSV_ERROR_DEVICE_NOT_READY);

	if (!texture_object)
	{
		if (_data_texture_default_2d)
		{
			checkCudaErrorAndThrow(cudaDestroyTextureObject(_data_texture_default_2d),
				error_t::SSV_ERROR_UNKNOWN);
			return;
		}
		if (_data_texture_default_3d)
		{
			checkCudaErrorAndThrow(cudaDestroyTextureObject(_data_texture_default_3d),
				error_t::SSV_ERROR_UNKNOWN);
			return;
		}
		for (auto kv : _data_textures)
		{
			checkCudaErrorAndThrow(cudaDestroyTextureObject(kv.second),
				error_t::SSV_ERROR_UNKNOWN);
		}
		if (_data_cuda_array)
		{
			checkCudaErrorAndThrow(cudaFreeArray(_data_cuda_array),
				error_t::SSV_ERROR_UNKNOWN);
		}
	}
	else
	{
		if (_data_texture_default_2d == texture_object)
		{
			checkCudaErrorAndThrow(cudaDestroyTextureObject(_data_texture_default_2d),
				error_t::SSV_ERROR_UNKNOWN);
			return;
		}
		if (_data_texture_default_3d == texture_object)
		{
			checkCudaErrorAndThrow(cudaDestroyTextureObject(_data_texture_default_3d),
				error_t::SSV_ERROR_UNKNOWN);
			return;
		}
		for (auto kv : _data_textures)
		{
			if (kv.second == texture_object)
			{
				_data_textures.erase(kv.first);
				checkCudaErrorAndThrow(cudaDestroyTextureObject(kv.second),
					error_t::SSV_ERROR_UNKNOWN);
				return;
			}
		}
		throw error_t::SSV_ERROR_INVALID_VALUE;
	}
}

void BlobBase::clear()
{
	if (_data_cpu)
	{
		memset(_data_cpu, 0, size_cpu_in_bytes());
	}
	checkCudaErrorAndThrow(cudaSetDevice(_storage_gpu_device),
		error_t::SSV_ERROR_DEVICE_NOT_READY);
	checkCudaErrorAndThrow(cudaMemset(_data_gpu.ptr, 0, size_gpu_in_bytes()),
		error_t::SSV_ERROR_UNKNOWN);
}

cudaTextureObject_t BlobBase::_CreateTexture2d(
	const texture_param_t &params
) const
{
	unsigned char dimension;
	struct cudaTextureDesc sTexDesc;
	cudaChannelFormatDesc sChannelDesc; 
	size_t layer_id;

	std::tie(sTexDesc, sChannelDesc, dimension, layer_id) = params;

	if (dimension != 2u || layer_id >= _nz)
	{
		throw error_t::SSV_ERROR_INVALID_VALUE;
	}

	if (_storage_gpu_device < 0) throw error_t::SSV_ERROR_NOT_INITIALIZED;
	checkCudaErrorAndThrow(cudaSetDevice(_storage_gpu_device),
		error_t::SSV_ERROR_DEVICE_NOT_READY);

	cudaResourceDesc sResDesc;
	memset(&sResDesc, 0, sizeof(sResDesc));
	sResDesc.resType = cudaResourceTypePitch2D;
	sResDesc.res.pitch2D.desc = sChannelDesc;
	sResDesc.res.pitch2D.devPtr = 
		static_cast<byte *>(_data_gpu.ptr)
		+ layer_id * _data_gpu.pitch * _data_gpu.ysize;
	sResDesc.res.pitch2D.width = _data_gpu.xsize;
	sResDesc.res.pitch2D.height = _data_gpu.ysize;
	sResDesc.res.pitch2D.pitchInBytes = _data_gpu.pitch;

	cudaTextureObject_t texture_object = 0;
	checkCudaErrorAndThrow(cudaCreateTextureObject(&texture_object, &sResDesc, &sTexDesc, NULL),
		error_t::SSV_ERROR_INVALID_VALUE);

	return texture_object;
}

cudaTextureObject_t BlobBase::_CreateTexture3d(
	const texture_param_t &params
) const
{
	unsigned char dimension;
	struct cudaTextureDesc sTexDesc;
	cudaChannelFormatDesc sChannelDesc;
	size_t layer_id;

	std::tie(sTexDesc, sChannelDesc, dimension, layer_id) = params;

	if (dimension != 3u || layer_id != 0)
	{
		throw error_t::SSV_ERROR_INVALID_VALUE;
	}

	if (_storage_gpu_device < 0) throw error_t::SSV_ERROR_NOT_INITIALIZED;
	checkCudaErrorAndThrow(cudaSetDevice(_storage_gpu_device),
		error_t::SSV_ERROR_DEVICE_NOT_READY);

	if (!_data_cuda_array)
	{
		size_t element_size_in_bytes =
			(sChannelDesc.x + sChannelDesc.y + sChannelDesc.z + sChannelDesc.w) / 8u;
		cudaExtent extent_in_elements = make_cudaExtent(
			_nx_in_bytes / element_size_in_bytes, _ny, _nz
		);

		checkCudaErrorAndThrow(cudaMalloc3DArray(&_data_cuda_array, &sChannelDesc, extent_in_elements),
			error_t::SSV_ERROR_OUT_OF_MEMORY_GPU);
	}

	cudaResourceDesc sResDesc;
	memset(&sResDesc, 0, sizeof(sResDesc));
	sResDesc.resType = cudaResourceTypeArray;
	sResDesc.res.array.array = _data_cuda_array;

	cudaTextureObject_t texture_object = 0;
	checkCudaErrorAndThrow(cudaCreateTextureObject(&texture_object, &sResDesc, &sTexDesc, NULL),
		error_t::SSV_ERROR_INVALID_VALUE);

	return texture_object;
}

void BlobBase::_CopyToCudaArray() const
{
	if (_storage_gpu_device < 0) throw error_t::SSV_ERROR_NOT_INITIALIZED;
	checkCudaErrorAndThrow(cudaSetDevice(_storage_gpu_device),
		error_t::SSV_ERROR_DEVICE_NOT_READY);

	cudaExtent extent_in_elements;
	cudaArrayGetInfo(nullptr, &extent_in_elements, nullptr, _data_cuda_array);

	cudaMemcpy3DParms params = { 0 };
	params.srcPtr = _data_gpu;
	params.dstArray = _data_cuda_array;
	params.kind = cudaMemcpyDeviceToDevice;
	params.extent = extent_in_elements;
	checkCudaErrorAndThrow(cudaMemcpy3D(&params),
		error_t::SSV_ERROR_INVALID_VALUE);
}

BlobBase::texture_param_t BlobBase::_MakeTextureParam(
	const cudaTextureDesc * texDesc, const cudaChannelFormatDesc * channelDesc, 
	unsigned char dimension, uint layer_id
)
{
	struct cudaTextureDesc sTexDesc;
	if (texDesc == nullptr)
	{
		memset(&sTexDesc, 0, sizeof(sTexDesc));
		sTexDesc.addressMode[0] = cudaAddressModeClamp;
		sTexDesc.addressMode[1] = cudaAddressModeClamp;
		sTexDesc.addressMode[2] = cudaAddressModeClamp;
		sTexDesc.filterMode = cudaFilterModeLinear;
		sTexDesc.readMode = cudaReadModeElementType;
		sTexDesc.normalizedCoords = 0;
		texDesc = &sTexDesc;
	}

	cudaChannelFormatDesc sChannelDesc;
	if (channelDesc == nullptr)
	{
		sChannelDesc = cudaCreateChannelDesc<byte>();
		channelDesc = &sChannelDesc;
	}

	return std::make_tuple(*texDesc, *channelDesc, dimension, layer_id);
}

void BlobBase::_InitCuda(int gpu_device)
{
	if (gpu_device < 0)
	{
		_storage_gpu_device = -1;
		memset(&_data_gpu_extent, 0, sizeof(cudaExtent));
		memset(&_data_gpu, 0, sizeof(cudaPitchedPtr));
		_data_texture_default_2d = 0;
		_data_texture_default_3d = 0;
		_data_textures.clear();
		_data_cuda_array = nullptr;
	}
	else
	{
		_storage_gpu_device = gpu_device;
		_data_gpu_extent = make_cudaExtent(_nx_in_bytes,_ny, _nz);

		checkCudaErrorAndThrow(cudaSetDevice(_storage_gpu_device),
			error_t::SSV_ERROR_DEVICE_NOT_READY);
		checkCudaErrorAndThrow(cudaMalloc3D(&_data_gpu, _data_gpu_extent),
			error_t::SSV_ERROR_OUT_OF_MEMORY_GPU);
		checkCudaErrorAndThrow(cudaMemset(_data_gpu.ptr, 0, size_gpu_in_bytes()),
			error_t::SSV_ERROR_UNKNOWN);

		_data_texture_default_2d = 0;
		_data_texture_default_3d = 0;
		_data_textures.clear();
		_data_cuda_array = nullptr;
	}
}

void BlobBase::_CopyCuda(const BlobBase &other, int gpu_device)
{
	_storage_gpu_device = gpu_device;
	_data_gpu_extent = other._data_gpu_extent;

	checkCudaErrorAndThrow(cudaSetDevice(_storage_gpu_device),
		error_t::SSV_ERROR_DEVICE_NOT_READY);
	checkCudaErrorAndThrow(cudaMalloc3D(&_data_gpu, _data_gpu_extent),
		error_t::SSV_ERROR_OUT_OF_MEMORY_GPU);

	cudaMemcpy3DParms params = { 0 };
	params.srcPtr = other._data_gpu;
	params.dstPtr = _data_gpu;
	params.kind = cudaMemcpyDeviceToDevice;
	params.extent = _data_gpu_extent;
	checkCudaErrorAndThrow(cudaMemcpy3D(&params),
		error_t::SSV_ERROR_INVALID_VALUE);

	_data_texture_default_2d = 0;
	_data_texture_default_3d = 0;
	_data_textures.clear();
	_data_cuda_array = nullptr;
}

void BlobBase::_MoveCuda(BlobBase &&other)
{
	_storage_gpu_device = other._storage_gpu_device;
	_data_gpu_extent = other._data_gpu_extent;
	_data_gpu = other._data_gpu;
	_data_texture_default_2d = other._data_texture_default_2d;
	_data_texture_default_3d = other._data_texture_default_3d;
	_data_textures = std::move(other._data_textures);
	_data_cuda_array = other._data_cuda_array;

	memset(&other._data_gpu, 0, sizeof(cudaPitchedPtr));
	memset(&other._data_gpu_extent, 0, sizeof(cudaExtent));
	other._storage_gpu_device = -1;
	other._data_texture_default_2d = 0;
	other._data_texture_default_3d = 0;
	other._data_textures.clear();
	other._data_cuda_array = nullptr;
}

void BlobBase::_DestroyCuda()
{
	if (_storage_gpu_device >= 0)
	{
		checkCudaErrorAndThrow(cudaSetDevice(_storage_gpu_device),
			error_t::SSV_ERROR_DEVICE_NOT_READY);
		if (_data_gpu.ptr)
		{
			checkCudaErrorAndThrow(cudaFree(_data_gpu.ptr),
				error_t::SSV_ERROR_INVALID_VALUE);
		}
		destroyTexture();
	}

	memset(&_data_gpu, 0, sizeof(cudaPitchedPtr));
	memset(&_data_gpu_extent, 0, sizeof(cudaExtent));
	_storage_gpu_device = -1;
	_data_texture_default_2d = 0;
	_data_texture_default_3d = 0;
	_data_textures.clear();
	_data_cuda_array = nullptr;
}
