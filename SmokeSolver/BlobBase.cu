
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
		make_cudaPitchedPtr(_data_cpu, _nx, _nx, _ny);
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
		make_cudaPitchedPtr(from_cpu_data, _nx, _nx, _ny);
	cudaMemcpy3DParms params = { 0 };
	params.srcPtr = data_cpu_pitched_ptr;
	params.dstPtr = _data_gpu;
	params.kind = cudaMemcpyHostToDevice;
	params.extent = _data_gpu_extent;
	checkCudaErrorAndThrow(cudaMemcpy3D(&params), 
		SSV_ERROR_INVALID_VALUE);
}

cudaTextureObject_t BlobBase::createTexture2d(
	const cudaTextureDesc *texDesc,
	const cudaChannelFormatDesc *channelDesc,
	size_t layer_id)
{
	if (layer_id >= _nz)
	{
		throw SSV_ERROR_INVALID_VALUE;
	}

	checkCudaErrorAndThrow(cudaSetDevice(_storage_gpu_device),
		SSV_ERROR_DEVICE_NOT_READY);

	if (_data_texture_2d)
	{
		checkCudaErrorAndThrow(cudaDestroyTextureObject(_data_texture_2d),
			SSV_ERROR_UNKNOWN);
		_data_texture_2d = 0;
	}

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

	cudaResourceDesc sResDesc;
	memset(&sResDesc, 0, sizeof(sResDesc));
	sResDesc.resType = cudaResourceTypePitch2D;
	sResDesc.res.pitch2D.desc = *channelDesc;
	sResDesc.res.pitch2D.devPtr = 
		static_cast<byte *>(_data_gpu.ptr)
		+ layer_id * _data_gpu.pitch * _data_gpu.ysize;
	sResDesc.res.pitch2D.width = _data_gpu.xsize;
	sResDesc.res.pitch2D.height = _data_gpu.ysize;
	sResDesc.res.pitch2D.pitchInBytes = _data_gpu.pitch;

	checkCudaErrorAndThrow(cudaCreateTextureObject(&_data_texture_2d, &sResDesc, texDesc, NULL),
		SSV_ERROR_INVALID_VALUE);

	return _data_texture_2d;
}

cudaTextureObject_t BlobBase::createTexture3d(
	const cudaTextureDesc *texDesc,
	const cudaChannelFormatDesc *channelDesc
)
{
	checkCudaErrorAndThrow(cudaSetDevice(_storage_gpu_device),
		SSV_ERROR_DEVICE_NOT_READY);

	if (_data_texture_3d)
	{
		checkCudaErrorAndThrow(cudaDestroyTextureObject(_data_texture_3d),
			SSV_ERROR_UNKNOWN);
		_data_texture_3d = 0;
	}

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

	size_t element_size_in_bytes = 
		(channelDesc->x + channelDesc->y + channelDesc->z + channelDesc->w) / 8u;
	cudaExtent extent_in_elements = make_cudaExtent(
		_nx / element_size_in_bytes, _ny, _nz
	);

	checkCudaErrorAndThrow(cudaMalloc3DArray(&_data_cuda_array, channelDesc, extent_in_elements),
		SSV_ERROR_OUT_OF_MEMORY_GPU);

	cudaResourceDesc sResDesc;
	memset(&sResDesc, 0, sizeof(sResDesc));
	sResDesc.resType = cudaResourceTypeArray;
	sResDesc.res.array.array = _data_cuda_array;

	checkCudaErrorAndThrow(cudaCreateTextureObject(&_data_texture_3d, &sResDesc, &sTexDesc, NULL),
		SSV_ERROR_INVALID_VALUE);

	return data_texture_3d();
}

cudaTextureObject_t BlobBase::data_texture_3d() const
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

	return _data_texture_3d;
}

void BlobBase::_InitCuda(int gpu_device)
{
	_storage_gpu_device = gpu_device;
	_data_gpu_extent.width = _nx;
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
		if (_data_texture_2d)
		{
			checkCudaErrorAndThrow(cudaDestroyTextureObject(_data_texture_2d),
				SSV_ERROR_UNKNOWN);
		}
		if (_data_texture_3d)
		{
			checkCudaErrorAndThrow(cudaDestroyTextureObject(_data_texture_3d),
				SSV_ERROR_UNKNOWN);
		}
	}

	memset(&_data_gpu, 0, sizeof(cudaPitchedPtr));
	memset(&_data_gpu_extent, 0, sizeof(cudaExtent));
	_storage_gpu_device = -1;
	_data_texture_2d = 0;
	_data_texture_3d = 0;
	_data_cuda_array = nullptr;
}
