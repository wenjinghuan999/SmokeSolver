#include "common.cuh"
#include "pitched_ptr.h"
#include "BlobBase.h"
using namespace ssv;


void BlobBase::sync_gpu_to_cpu()
{
	if (storage_gpu_device_ < 0) throw ssv_error(error_t::SSV_ERROR_NOT_INITIALIZED);
	CHECK_CUDA_ERROR_AND_THROW(cudaSetDevice(storage_gpu_device_),
		ssv_error(error_t::SSV_ERROR_DEVICE_NOT_READY));
	if (data_cpu_ == nullptr) return;

	_CopyTo(data_cpu_, storage_t::GPU, storage_t::CPU);
}

void BlobBase::sync_cpu_to_gpu()
{
	if (storage_gpu_device_ < 0) throw ssv_error(error_t::SSV_ERROR_NOT_INITIALIZED);
	CHECK_CUDA_ERROR_AND_THROW(cudaSetDevice(storage_gpu_device_),
		ssv_error(error_t::SSV_ERROR_DEVICE_NOT_READY));
	if (data_cpu_ == nullptr) return;

	_CopyFrom(data_cpu_, storage_t::CPU, storage_t::GPU);
}

namespace
{
	enum cudaMemcpyKind make_cudaMemcpyKind(
		BlobBase::storage_t from, BlobBase::storage_t to
	)
	{
		if (from == BlobBase::storage_t::CPU)
		{
			if (to == BlobBase::storage_t::CPU)
				return cudaMemcpyHostToHost;
			return cudaMemcpyHostToDevice;
		}
		if (to == BlobBase::storage_t::CPU)
			return cudaMemcpyDeviceToHost;
		return cudaMemcpyDeviceToDevice;
	}
}

void BlobBase::_CopyTo(void *dst, storage_t from, storage_t to) const
{
	if ((from != storage_t::CPU && from != storage_t::GPU)
		|| (to != storage_t::CPU && to != storage_t::GPU)
		|| dst == nullptr)
	{
		throw ssv_error(error_t::SSV_ERROR_INVALID_VALUE);
	}

	cudaPitchedPtr dst_pitched_ptr =
		make_cudaPitchedPtr(dst, nx_in_bytes_, nx_in_bytes_, ny_);
	cudaPitchedPtr data_cpu_pitched_ptr =
		make_cudaPitchedPtr(data_cpu_, nx_in_bytes_, nx_in_bytes_, ny_);
	cudaMemcpy3DParms params = {nullptr};
	if (from == storage_t::CPU)
		params.srcPtr = data_cpu_pitched_ptr;
	else
		params.srcPtr = data_gpu_;
	params.dstPtr = dst_pitched_ptr;
	params.kind = make_cudaMemcpyKind(from, to);
	params.extent = data_gpu_extent_;
	CHECK_CUDA_ERROR_AND_THROW(cudaMemcpy3D(&params),
		ssv_error(error_t::SSV_ERROR_INVALID_VALUE));
}

void BlobBase::_CopyTo(cudaPitchedPtr *dst, storage_t from, storage_t to) const
{
	if ((from != storage_t::CPU && from != storage_t::GPU)
		|| (to != storage_t::CPU && to != storage_t::GPU)
		|| dst == nullptr || dst->ptr == nullptr)
	{
		throw ssv_error(error_t::SSV_ERROR_INVALID_VALUE);
	}

	cudaPitchedPtr data_cpu_pitched_ptr =
		make_cudaPitchedPtr(data_cpu_, nx_in_bytes_, nx_in_bytes_, ny_);
	cudaMemcpy3DParms params = {nullptr};
	if (from == storage_t::CPU)
		params.srcPtr = data_cpu_pitched_ptr;
	else
		params.srcPtr = data_gpu_;
	params.dstPtr = *dst;
	params.kind = make_cudaMemcpyKind(from, to);
	params.extent = data_gpu_extent_;
	CHECK_CUDA_ERROR_AND_THROW(cudaMemcpy3D(&params),
		ssv_error(error_t::SSV_ERROR_INVALID_VALUE));
}

void BlobBase::_CopyFrom(void *src, storage_t from, storage_t to)
{
	if ((from != storage_t::CPU && from != storage_t::GPU)
		|| src == nullptr)
	{
		throw ssv_error(error_t::SSV_ERROR_INVALID_VALUE);
	}

	cudaPitchedPtr src_pitched_ptr =
		make_cudaPitchedPtr(src, nx_in_bytes_, nx_in_bytes_, ny_);
	cudaPitchedPtr data_cpu_pitched_ptr =
		make_cudaPitchedPtr(data_cpu_, nx_in_bytes_, nx_in_bytes_, ny_);
	cudaMemcpy3DParms params = {nullptr};
	if (to == storage_t::CPU)
		params.dstPtr = data_cpu_pitched_ptr;
	else
		params.dstPtr = data_gpu_;
	params.srcPtr = src_pitched_ptr;
	params.kind = make_cudaMemcpyKind(from, to);
	params.extent = data_gpu_extent_;
	CHECK_CUDA_ERROR_AND_THROW(cudaMemcpy3D(&params),
		ssv_error(error_t::SSV_ERROR_INVALID_VALUE));
}

void BlobBase::_CopyFrom(cudaPitchedPtr *src, storage_t from, storage_t to)
{
	if ((from != storage_t::CPU && from != storage_t::GPU)
		|| src == nullptr || src->ptr == nullptr)
	{
		throw ssv_error(error_t::SSV_ERROR_INVALID_VALUE);
	}

	cudaPitchedPtr data_cpu_pitched_ptr =
		make_cudaPitchedPtr(data_cpu_, nx_in_bytes_, nx_in_bytes_, ny_);
	cudaMemcpy3DParms params = {nullptr};
	if (from == storage_t::CPU)
		params.dstPtr = data_cpu_pitched_ptr;
	else
		params.dstPtr = data_gpu_;
	params.srcPtr = *src;
	params.kind = make_cudaMemcpyKind(from, to);
	params.extent = data_gpu_extent_;
	CHECK_CUDA_ERROR_AND_THROW(cudaMemcpy3D(&params),
		ssv_error(error_t::SSV_ERROR_INVALID_VALUE));
}

void BlobBase::destroy_texture(cudaTextureObject_t texture_object)
{
	if (storage_gpu_device_ < 0) throw ssv_error(error_t::SSV_ERROR_NOT_INITIALIZED);
	CHECK_CUDA_ERROR_AND_THROW(cudaSetDevice(storage_gpu_device_),
		ssv_error(error_t::SSV_ERROR_DEVICE_NOT_READY));

	if (!texture_object)
	{
		if (data_texture_default_2d_)
		{
			CHECK_CUDA_ERROR_AND_THROW(cudaDestroyTextureObject(data_texture_default_2d_),
				ssv_error(error_t::SSV_ERROR_UNKNOWN));
			return;
		}
		if (data_texture_default_3d_)
		{
			CHECK_CUDA_ERROR_AND_THROW(cudaDestroyTextureObject(data_texture_default_3d_),
				ssv_error(error_t::SSV_ERROR_UNKNOWN));
			return;
		}
		for (auto kv : data_textures_)
		{
			CHECK_CUDA_ERROR_AND_THROW(cudaDestroyTextureObject(kv.second),
				ssv_error(error_t::SSV_ERROR_UNKNOWN));
		}
		if (data_cuda_array_)
		{
			CHECK_CUDA_ERROR_AND_THROW(cudaFreeArray(data_cuda_array_),
				ssv_error(error_t::SSV_ERROR_UNKNOWN));
		}
	}
	else
	{
		if (data_texture_default_2d_ == texture_object)
		{
			CHECK_CUDA_ERROR_AND_THROW(cudaDestroyTextureObject(data_texture_default_2d_),
				ssv_error(error_t::SSV_ERROR_UNKNOWN));
			return;
		}
		if (data_texture_default_3d_ == texture_object)
		{
			CHECK_CUDA_ERROR_AND_THROW(cudaDestroyTextureObject(data_texture_default_3d_),
				ssv_error(error_t::SSV_ERROR_UNKNOWN));
			return;
		}
		for (auto kv : data_textures_)
		{
			if (kv.second == texture_object)
			{
				data_textures_.erase(kv.first);
				CHECK_CUDA_ERROR_AND_THROW(cudaDestroyTextureObject(kv.second),
					ssv_error(error_t::SSV_ERROR_UNKNOWN));
				return;
			}
		}
		throw ssv_error(error_t::SSV_ERROR_INVALID_VALUE);
	}
}

void BlobBase::clear()
{
	if (data_cpu_)
	{
		memset(data_cpu_, 0, size_cpu_in_bytes());
	}
	CHECK_CUDA_ERROR_AND_THROW(cudaSetDevice(storage_gpu_device_),
		ssv_error(ssv_error(error_t::SSV_ERROR_DEVICE_NOT_READY)));
	CHECK_CUDA_ERROR_AND_THROW(cudaMemset(data_gpu_.ptr, 0, size_gpu_in_bytes()),
		ssv_error(ssv_error(error_t::SSV_ERROR_UNKNOWN)));
}

cudaTextureObject_t BlobBase::_CreateTexture2D(
	const texture_param_t &params
) const
{
	unsigned char dimension;
	struct cudaTextureDesc tex_desc;
	cudaChannelFormatDesc channel_desc;
	size_t layer_id;

	std::tie(tex_desc, channel_desc, dimension, layer_id) = params;

	if (dimension != 2u || layer_id >= nz_)
	{
		throw ssv_error(ssv_error(error_t::SSV_ERROR_INVALID_VALUE));
	}

	if (storage_gpu_device_ < 0) throw ssv_error(ssv_error(error_t::SSV_ERROR_NOT_INITIALIZED));
	CHECK_CUDA_ERROR_AND_THROW(cudaSetDevice(storage_gpu_device_),
		ssv_error(error_t::SSV_ERROR_DEVICE_NOT_READY));

	cudaResourceDesc res_desc;
	memset(&res_desc, 0, sizeof(res_desc));
	res_desc.resType = cudaResourceTypePitch2D;
	res_desc.res.pitch2D.desc = channel_desc;
	res_desc.res.pitch2D.devPtr =
		static_cast<byte *>(data_gpu_.ptr)
		+ layer_id * data_gpu_.pitch * data_gpu_.ysize;
	res_desc.res.pitch2D.width = data_gpu_.xsize;
	res_desc.res.pitch2D.height = data_gpu_.ysize;
	res_desc.res.pitch2D.pitchInBytes = data_gpu_.pitch;

	cudaTextureObject_t texture_object = 0;
	CHECK_CUDA_ERROR_AND_THROW(cudaCreateTextureObject(&texture_object, &res_desc, &tex_desc, NULL),
		ssv_error(error_t::SSV_ERROR_INVALID_VALUE));

	return texture_object;
}

cudaTextureObject_t BlobBase::_CreateTexture3D(
	const texture_param_t &params
) const
{
	unsigned char dimension;
	struct cudaTextureDesc s_tex_desc;
	cudaChannelFormatDesc s_channel_desc;
	size_t layer_id;

	std::tie(s_tex_desc, s_channel_desc, dimension, layer_id) = params;

	if (dimension != 3u || layer_id != 0)
	{
		throw ssv_error(error_t::SSV_ERROR_INVALID_VALUE);
	}

	if (storage_gpu_device_ < 0) throw ssv_error(error_t::SSV_ERROR_NOT_INITIALIZED);
	CHECK_CUDA_ERROR_AND_THROW(cudaSetDevice(storage_gpu_device_),
		ssv_error(error_t::SSV_ERROR_DEVICE_NOT_READY));

	if (!data_cuda_array_)
	{
		size_t element_size_in_bytes =
			(s_channel_desc.x + s_channel_desc.y + s_channel_desc.z + s_channel_desc.w) / 8u;
		cudaExtent extent_in_elements = make_cudaExtent(
			nx_in_bytes_ / element_size_in_bytes, ny_, nz_
		);

		CHECK_CUDA_ERROR_AND_THROW(cudaMalloc3DArray(&data_cuda_array_, &s_channel_desc, extent_in_elements),
			ssv_error(error_t::SSV_ERROR_OUT_OF_MEMORY_GPU));
	}

	cudaResourceDesc s_res_desc;
	memset(&s_res_desc, 0, sizeof(s_res_desc));
	s_res_desc.resType = cudaResourceTypeArray;
	s_res_desc.res.array.array = data_cuda_array_;

	cudaTextureObject_t texture_object = 0;
	CHECK_CUDA_ERROR_AND_THROW(cudaCreateTextureObject(&texture_object, &s_res_desc, &s_tex_desc, NULL),
		ssv_error(error_t::SSV_ERROR_INVALID_VALUE));

	return texture_object;
}

void BlobBase::_CopyToCudaArray() const
{
	if (storage_gpu_device_ < 0) throw ssv_error(error_t::SSV_ERROR_NOT_INITIALIZED);
	CHECK_CUDA_ERROR_AND_THROW(cudaSetDevice(storage_gpu_device_),
		ssv_error(error_t::SSV_ERROR_DEVICE_NOT_READY));

	cudaExtent extent_in_elements;
	cudaArrayGetInfo(nullptr, &extent_in_elements, nullptr, data_cuda_array_);

	cudaMemcpy3DParms params = {nullptr};
	params.srcPtr = data_gpu_;
	params.dstArray = data_cuda_array_;
	params.kind = cudaMemcpyDeviceToDevice;
	params.extent = extent_in_elements;
	CHECK_CUDA_ERROR_AND_THROW(cudaMemcpy3D(&params),
		ssv_error(error_t::SSV_ERROR_INVALID_VALUE));
}

BlobBase::texture_param_t BlobBase::_MakeTextureParam(
	const cudaTextureDesc *tex_desc, const cudaChannelFormatDesc *channel_desc,
	unsigned char dimension, uint layer_id
)
{
	struct cudaTextureDesc s_tex_desc;
	if (tex_desc == nullptr)
	{
		memset(&s_tex_desc, 0, sizeof(s_tex_desc));
		s_tex_desc.addressMode[0] = cudaAddressModeClamp;
		s_tex_desc.addressMode[1] = cudaAddressModeClamp;
		s_tex_desc.addressMode[2] = cudaAddressModeClamp;
		s_tex_desc.filterMode = cudaFilterModeLinear;
		s_tex_desc.readMode = cudaReadModeElementType;
		s_tex_desc.normalizedCoords = 0;
		tex_desc = &s_tex_desc;
	}

	if (channel_desc == nullptr)
	{
		cudaChannelFormatDesc s_channel_desc = cudaCreateChannelDesc<byte>();
		channel_desc = &s_channel_desc;
	}

	return std::make_tuple(*tex_desc, *channel_desc, dimension, layer_id);
}

void BlobBase::_InitCuda(int gpu_device)
{
	if (gpu_device < 0)
	{
		storage_gpu_device_ = -1;
		memset(&data_gpu_extent_, 0, sizeof(cudaExtent));
		memset(&data_gpu_, 0, sizeof(cudaPitchedPtr));
		data_texture_default_2d_ = 0;
		data_texture_default_3d_ = 0;
		data_textures_.clear();
		data_cuda_array_ = nullptr;
	}
	else
	{
		storage_gpu_device_ = gpu_device;
		data_gpu_extent_ = make_cudaExtent(nx_in_bytes_, ny_, nz_);

		CHECK_CUDA_ERROR_AND_THROW(cudaSetDevice(storage_gpu_device_),
			ssv_error(error_t::SSV_ERROR_DEVICE_NOT_READY));
		CHECK_CUDA_ERROR_AND_THROW(cudaMalloc3D(&data_gpu_, data_gpu_extent_),
			ssv_error(error_t::SSV_ERROR_OUT_OF_MEMORY_GPU));
		CHECK_CUDA_ERROR_AND_THROW(cudaMemset(data_gpu_.ptr, 0, size_gpu_in_bytes()),
			ssv_error(error_t::SSV_ERROR_UNKNOWN));

		data_texture_default_2d_ = 0;
		data_texture_default_3d_ = 0;
		data_textures_.clear();
		data_cuda_array_ = nullptr;
	}
}

void BlobBase::_CopyCuda(const BlobBase &other, int gpu_device)
{
	storage_gpu_device_ = gpu_device;
	data_gpu_extent_ = other.data_gpu_extent_;

	CHECK_CUDA_ERROR_AND_THROW(cudaSetDevice(storage_gpu_device_),
		ssv_error(error_t::SSV_ERROR_DEVICE_NOT_READY));
	CHECK_CUDA_ERROR_AND_THROW(cudaMalloc3D(&data_gpu_, data_gpu_extent_),
		ssv_error(error_t::SSV_ERROR_OUT_OF_MEMORY_GPU));

	cudaMemcpy3DParms params = {nullptr};
	params.srcPtr = other.data_gpu_;
	params.dstPtr = data_gpu_;
	params.kind = cudaMemcpyDeviceToDevice;
	params.extent = data_gpu_extent_;
	CHECK_CUDA_ERROR_AND_THROW(cudaMemcpy3D(&params),
		ssv_error(error_t::SSV_ERROR_INVALID_VALUE));

	data_texture_default_2d_ = 0;
	data_texture_default_3d_ = 0;
	data_textures_.clear();
	data_cuda_array_ = nullptr;
}

void BlobBase::_MoveCuda(BlobBase &&other)
{
	storage_gpu_device_ = other.storage_gpu_device_;
	data_gpu_extent_ = other.data_gpu_extent_;
	data_gpu_ = other.data_gpu_;
	data_texture_default_2d_ = other.data_texture_default_2d_;
	data_texture_default_3d_ = other.data_texture_default_3d_;
	data_textures_ = std::move(other.data_textures_);
	data_cuda_array_ = other.data_cuda_array_;

	memset(&other.data_gpu_, 0, sizeof(cudaPitchedPtr));
	memset(&other.data_gpu_extent_, 0, sizeof(cudaExtent));
	other.storage_gpu_device_ = -1;
	other.data_texture_default_2d_ = 0;
	other.data_texture_default_3d_ = 0;
	other.data_textures_.clear();
	other.data_cuda_array_ = nullptr;
}

void BlobBase::_DestroyCuda()
{
	if (storage_gpu_device_ >= 0)
	{
		CHECK_CUDA_ERROR_AND_THROW(cudaSetDevice(storage_gpu_device_),
			ssv_error(error_t::SSV_ERROR_DEVICE_NOT_READY));
		if (data_gpu_.ptr)
		{
			CHECK_CUDA_ERROR_AND_THROW(cudaFree(data_gpu_.ptr),
				ssv_error(error_t::SSV_ERROR_INVALID_VALUE));
		}
		destroy_texture();
	}

	memset(&data_gpu_, 0, sizeof(cudaPitchedPtr));
	memset(&data_gpu_extent_, 0, sizeof(cudaExtent));
	storage_gpu_device_ = -1;
	data_texture_default_2d_ = 0;
	data_texture_default_3d_ = 0;
	data_textures_.clear();
	data_cuda_array_ = nullptr;
}
