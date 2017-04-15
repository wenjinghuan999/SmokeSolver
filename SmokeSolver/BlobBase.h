#pragma once

#ifndef __BLOB_BASE_H__
#define __BLOB_BASE_H__

#include "common.h"

#include <unordered_map>

namespace ssv
{
	class BlobBase
	{
	public:
		BlobBase();
		~BlobBase();

	public:
		// nx: size in bytes
		// ny, nz: size in elements
		// gpu_device: cuda device id
		// cpu_copy: if true, copying from and to CPU is enabled
		virtual void setSize(size_t nx, size_t ny, size_t nz = 1u, 
			int gpu_device = 0, bool cpu_copy = true);

		// Reset Blob parameters (as if just initialized)
		void reset();

		// Copy Data to CPU
		// from_gpu_data: cudaPitchedPtr from where data should be copied
		//                [default] = nullptr: from GPU data of this Blob 
		void copyToCpu(cudaPitchedPtr *from_gpu_data = nullptr);

		// Copy Data to GPU
		// from_cpu_data: pointer from where data should be copied
		//                [default] = nullptr: from CPU data of this Blob 
		void copyToGpu(void *from_cpu_data = nullptr);

		// Destroy the specific texture
		// if texture_object == 0, destroy all textures
		void destroyTexture(cudaTextureObject_t texture_object = 0);

	public:
		// Return nx (in elements)
		virtual size_t nx() const = 0;

		// Return nx (in bytes)
		size_t nx_in_bytes() const { return _nx_in_bytes; }

		// Return ny (in elements)
		size_t ny() const { return _ny; }

		// Return nz (in elements)
		size_t nz() const { return _nz; }

		// Return total number of elements 
		// ( = nx() * ny() * nz())
		virtual size_t numel() const = 0;

		// Return total number of bytes 
		// ( = nx_in_bytes() * ny() * nz())
		size_t size_in_bytes() const { return _size_in_bytes; }

		// Return pitch in bytes
		size_t pitch_in_bytes() const
		{
			return _data_gpu.pitch;
		}

		// Return raw pointer of CPU data
		void *data_cpu()
		{
			return _data_cpu;
		}

		// Return cudaPitchedPtr of GPU data
		cudaPitchedPtr *data_gpu_cuda_pitched_ptr()
		{
			return &_data_gpu;
		}

		// Return cudaTextureObject of GPU data in 2D
		// If no texture of specific parameters exists, a new texture object will be created.
		// If the Blob is 3D, use layer_id to specify which layer should be sampled.
		// Each layer has a unique texture. Consider using 3D texture to avoid creating too many textures.
		// texDesc is optional
		// default texDesc: clamp addr mode, linear filter, not normalized
		virtual cudaTextureObject_t data_texture_2d(
			const cudaTextureDesc *texDesc = nullptr,
			size_t layer_id = 0) = 0;

		// Return cudaTextureObject of GPU data in 3D
		// If no texture of specific parameters exists, a new texture object will be created.
		// Call this method to re-obtain the texture after GPU data are modified
		// A memory copy is needed. Consider using linear memory for better performance.
		// texDesc is optional
		// default texDesc: clamp addr mode, linear filter, not normalized
		virtual cudaTextureObject_t data_texture_3d(
			const cudaTextureDesc *texDesc = nullptr) = 0;

	public:
		typedef std::tuple<int, cudaTextureDesc, cudaChannelFormatDesc, size_t> texture_param_t;

	protected:
		texture_param_t _MakeTextureParam(
			unsigned char dimension, const cudaTextureDesc * texDesc,
			const cudaChannelFormatDesc * channelDesc, size_t layer_id);

		// Create cudaTextureObject of GPU data in 2D
		cudaTextureObject_t _CreateTexture2d(const texture_param_t &params);

		// Create cudaTextureObject of GPU data in 3D
		cudaTextureObject_t _CreateTexture3d(const texture_param_t &params);

		// Copy data from pitched pointer to 3d CUDA array
		void _CopyToCudaArray();

	protected:
		void _InitCuda(int gpu_device);
		void _DestroyCuda();

	protected:
		size_t _nx_in_bytes, _ny, _nz, _size_in_bytes;
		void *_data_cpu;

	protected:
		int _storage_gpu_device;
		cudaExtent _data_gpu_extent;
		cudaPitchedPtr _data_gpu;
		cudaTextureObject_t _data_texture_default_2d, _data_texture_default_3d;
		std::unordered_map<texture_param_t, cudaTextureObject_t, hash_tuple::hash<texture_param_t> > _data_textures;
		cudaArray_t _data_cuda_array;
	};
}

#endif // !__BLOB_BASE_H__
