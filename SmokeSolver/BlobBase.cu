
#include "common.cuh"
#include "BlobBase.h"
using namespace ssv;


void BlobBase::copyToCpu(cudaPitchedPtr *from_gpu_data)
{
	checkCudaErrorAndThrow(cudaSetDevice(_storage_gpu_device),
		SSV_ERROR_DEVICE_NOT_READY);

	if (from_gpu_data == nullptr)
	{
		from_gpu_data = &_data_gpu;
	}
	cudaPitchedPtr data_cpu_pitched_ptr =
		make_cudaPitchedPtr(_data_cpu, _nx_in_bytes, _nx_in_bytes, _ny);
	cudaMemcpy3DParms params = { 0 };
	params.srcPtr = *from_gpu_data;
	params.dstPtr = data_cpu_pitched_ptr;
	params.kind = cudaMemcpyDeviceToHost;
	params.extent = _data_gpu_extent;
	checkCudaErrorAndThrow(cudaMemcpy3D(&params),
		SSV_ERROR_INVALID_VALUE);
}

void BlobBase::copyToGpu(void *from_cpu_data)
{
	checkCudaErrorAndThrow(cudaSetDevice(_storage_gpu_device),
		SSV_ERROR_DEVICE_NOT_READY);

	if (from_cpu_data == nullptr)
	{
		from_cpu_data = _data_cpu;
	}
	cudaPitchedPtr data_cpu_pitched_ptr =
		make_cudaPitchedPtr(from_cpu_data, _nx_in_bytes, _nx_in_bytes, _ny);
	cudaMemcpy3DParms params = { 0 };
	params.srcPtr = data_cpu_pitched_ptr;
	params.dstPtr = _data_gpu;
	params.kind = cudaMemcpyHostToDevice;
	params.extent = _data_gpu_extent;
	checkCudaErrorAndThrow(cudaMemcpy3D(&params), 
		SSV_ERROR_INVALID_VALUE);
}

void BlobBase::destroyTexture(cudaTextureObject_t texture_object)
{
	checkCudaErrorAndThrow(cudaSetDevice(_storage_gpu_device),
		SSV_ERROR_DEVICE_NOT_READY);

	if (!texture_object)
	{
		if (_data_texture_default_2d)
		{
			checkCudaErrorAndThrow(cudaDestroyTextureObject(_data_texture_default_2d),
				SSV_ERROR_UNKNOWN);
			return;
		}
		if (_data_texture_default_3d)
		{
			checkCudaErrorAndThrow(cudaDestroyTextureObject(_data_texture_default_3d),
				SSV_ERROR_UNKNOWN);
			return;
		}
		for (auto kv : _data_textures)
		{
			checkCudaErrorAndThrow(cudaDestroyTextureObject(kv.second),
				SSV_ERROR_UNKNOWN);
		}
		if (_data_cuda_array)
		{
			checkCudaErrorAndThrow(cudaFreeArray(_data_cuda_array),
				SSV_ERROR_UNKNOWN);
		}
	}
	else
	{
		if (_data_texture_default_2d == texture_object)
		{
			checkCudaErrorAndThrow(cudaDestroyTextureObject(_data_texture_default_2d),
				SSV_ERROR_UNKNOWN);
			return;
		}
		if (_data_texture_default_3d == texture_object)
		{
			checkCudaErrorAndThrow(cudaDestroyTextureObject(_data_texture_default_3d),
				SSV_ERROR_UNKNOWN);
			return;
		}
		for (auto kv : _data_textures)
		{
			if (kv.second == texture_object)
			{
				_data_textures.erase(kv.first);
				checkCudaErrorAndThrow(cudaDestroyTextureObject(kv.second),
					SSV_ERROR_UNKNOWN);
				return;
			}
		}
		throw SSV_ERROR_INVALID_VALUE;
	}
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
		throw SSV_ERROR_INVALID_VALUE;
	}

	checkCudaErrorAndThrow(cudaSetDevice(_storage_gpu_device),
		SSV_ERROR_DEVICE_NOT_READY);

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
		SSV_ERROR_INVALID_VALUE);

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
		throw SSV_ERROR_INVALID_VALUE;
	}

	checkCudaErrorAndThrow(cudaSetDevice(_storage_gpu_device),
		SSV_ERROR_DEVICE_NOT_READY);

	if (!_data_cuda_array)
	{
		size_t element_size_in_bytes =
			(sChannelDesc.x + sChannelDesc.y + sChannelDesc.z + sChannelDesc.w) / 8u;
		cudaExtent extent_in_elements = make_cudaExtent(
			_nx_in_bytes / element_size_in_bytes, _ny, _nz
		);

		checkCudaErrorAndThrow(cudaMalloc3DArray(&_data_cuda_array, &sChannelDesc, extent_in_elements),
			SSV_ERROR_OUT_OF_MEMORY_GPU);
	}

	cudaResourceDesc sResDesc;
	memset(&sResDesc, 0, sizeof(sResDesc));
	sResDesc.resType = cudaResourceTypeArray;
	sResDesc.res.array.array = _data_cuda_array;

	cudaTextureObject_t texture_object = 0;
	checkCudaErrorAndThrow(cudaCreateTextureObject(&texture_object, &sResDesc, &sTexDesc, NULL),
		SSV_ERROR_INVALID_VALUE);

	return texture_object;
}

void BlobBase::_CopyToCudaArray() const
{
	checkCudaErrorAndThrow(cudaSetDevice(_storage_gpu_device),
		SSV_ERROR_DEVICE_NOT_READY);

	cudaExtent extent_in_elements;
	cudaArrayGetInfo(nullptr, &extent_in_elements, nullptr, _data_cuda_array);

	cudaMemcpy3DParms params = { 0 };
	params.srcPtr = _data_gpu;
	params.dstArray = _data_cuda_array;
	params.kind = cudaMemcpyDeviceToDevice;
	params.extent = extent_in_elements;
	checkCudaErrorAndThrow(cudaMemcpy3D(&params),
		SSV_ERROR_INVALID_VALUE);
}

void BlobBase::_InitCuda(int gpu_device)
{
	_storage_gpu_device = gpu_device;
	_data_gpu_extent.width = _nx_in_bytes;
	_data_gpu_extent.height = _ny;
	_data_gpu_extent.depth = _nz;

	if (gpu_device >= 0)
	{
		checkCudaErrorAndThrow(cudaSetDevice(_storage_gpu_device),
			SSV_ERROR_DEVICE_NOT_READY);
		checkCudaErrorAndThrow(cudaMalloc3D(&_data_gpu, _data_gpu_extent),
			SSV_ERROR_OUT_OF_MEMORY_GPU);
	}
}

void BlobBase::_DestroyCuda()
{
	if (_storage_gpu_device >= 0)
	{
		checkCudaErrorAndThrow(cudaSetDevice(_storage_gpu_device),
			SSV_ERROR_DEVICE_NOT_READY);
		if (_data_gpu.ptr)
		{
			checkCudaErrorAndThrow(cudaFree(_data_gpu.ptr),
				SSV_ERROR_INVALID_VALUE);
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
