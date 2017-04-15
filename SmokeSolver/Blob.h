#pragma once

#ifndef __BLOB_H__
#define __BLOB_H__

#include "common.h"
#include "BlobBase.h"
#include "pitched_ptr.h"

#include <cuda_texture_types.h>

namespace ssv
{
	template <typename>
	class Blob;
	template <typename>
	struct BlobWrapper;
	template <typename>
	struct BlobWrapperConst;

	// Basic data structure
	// Contains 2D or 3D data on CPU/GPU and provides managed texture objects
	template <typename _T>
	class Blob
		: public BlobBase
	{
	public:
		// Set or reset Blob parameters
		// nx, ny, nz: size in elements
		// gpu_device: cuda device id
		// cpu_copy: if true, copying from and to CPU is enabled
		void setSize(uint nx, uint ny, uint nz = 1u, int gpu_device = 0, bool cpu_copy = true)
		{
			BlobBase::setSize(nx * sizeof(_T), ny, nz, gpu_device, cpu_copy);
		}

		// Return cudaTextureObject of GPU data in 2D
		// If no texture of specific parameters exists, a new texture object will be created.
		// If the Blob is 3D, use layer_id to specify which layer should be sampled.
		// Each layer has a unique texture. Consider using 3D texture to avoid creating too many textures.
		// texDesc is optional
		// default texDesc: clamp addr mode, linear filter, not normalized
		virtual cudaTextureObject_t data_texture_2d(
			const cudaTextureDesc *texDesc = nullptr,
			uint layer_id = 0
		) const override
		{
			if (texDesc == nullptr && layer_id == 0)
			{
				if (_data_texture_default_2d)
				{
					return _data_texture_default_2d;
				}
			}

			cudaChannelFormatDesc sChannelDesc = cudaCreateChannelDesc<_T>();
			texture_param_t params = _MakeTextureParam(
				texDesc, &sChannelDesc, 2u, layer_id
			);
			auto iter = _data_textures.find(params);
			if (iter != _data_textures.end())
			{
				return iter->second;
			}

			cudaTextureObject_t texture_object = _CreateTexture2d(params);

			if (texDesc == nullptr && layer_id == 0)
			{
				_data_texture_default_2d = texture_object;
			}
			else
			{
				_data_textures[params] = texture_object;
			}

			return texture_object;
		}

		// Return cudaTextureObject of GPU data in 3D
		// If no texture of specific parameters exists, a new texture object will be created.
		// Call this method to re-obtain the texture after GPU data are modified
		// A memory copy is needed. Consider using linear memory for better performance.
		// texDesc is optional
		// default texDesc: clamp addr mode, linear filter, not normalized
		virtual cudaTextureObject_t data_texture_3d(
			const cudaTextureDesc *texDesc = nullptr
		) const override
		{
			if (texDesc == nullptr)
			{
				if (_data_texture_default_3d)
				{
					_CopyToCudaArray();
					return _data_texture_default_3d;
				}
			}
			cudaChannelFormatDesc sChannelDesc = cudaCreateChannelDesc<_T>();
			texture_param_t params = _MakeTextureParam(
				texDesc, &sChannelDesc, 3u, 0
			);
			auto iter = _data_textures.find(params);
			if (iter != _data_textures.end())
			{
				_CopyToCudaArray();
				return iter->second;
			}

			cudaTextureObject_t texture_object = _CreateTexture3d(params);

			if (texDesc == nullptr)
			{
				_data_texture_default_3d = texture_object;
			}
			else
			{
				_data_textures[params] = texture_object;
			}

			_CopyToCudaArray();
			return texture_object;
		}

	public:
		// Return nx (in elements)
		virtual uint nx() const override
		{
			return (uint)(_nx_in_bytes / sizeof(_T));
		}

		// Return total number of elements 
		// ( = nx() * ny() * nz())
		virtual uint numel() const override
		{
			return (uint)(_size_in_bytes / sizeof(_T));
		}

		// Return pitch in elements
		// Total memory allocated = pitch_in_elements * ny * nz * sizeof(_T)
		virtual uint pitch_in_elements() const override
		{
			return (uint)(_data_gpu.pitch / sizeof(_T));
		}

		// Raw pointer of CPU data
		_T *data_cpu() const
		{
			return static_cast<_T *>(_data_cpu);
		}

		// pitched_ptr of GPU data
		pitched_ptr<_T> data_gpu() const
		{
			return pitched_ptr<_T>(&_data_gpu);
		}

		// Raw pointer of GPU data
		_T *data_gpu_raw() const
		{
			return static_cast<_T *>(_data_gpu.ptr);
		}

		// Return BlobWrapper of this Blob
		BlobWrapper<_T> helper() 
		{
			return BlobWrapper<_T> {
				static_cast<_T *>(_data_gpu.ptr), (uint)(_data_gpu.pitch / sizeof(_T)),
					(uint)(_nx_in_bytes / sizeof(_T)), _ny, _nz };
		}

		// Return BlobWrapperConst of this Blob
		BlobWrapperConst<_T> helper_const() const
		{
			return BlobWrapperConst<_T> {
				static_cast<const _T *>(_data_gpu.ptr), (uint)(_data_gpu.pitch / sizeof(_T)),
					(uint)(_nx_in_bytes / sizeof(_T)), _ny, _nz };
		}
	};


	// A light-weighted helper class for easier kernel implementation using Blob
	template <typename _T>
	struct BlobWrapper
	{
		_T *ptr;
		uint pitch;
		uint nx;
		uint ny;
		uint nz;

		operator BlobWrapperConst<_T>() const
		{
			return BlobWrapperConst<_T> { ptr, pitch, nx, ny, nz };
		}

		__device__ _T &operator()(uint x, uint y, uint z)
		{
			return ptr[z * pitch * ny + y * pitch + x];
		}

		__device__ _T &operator()(uint x, uint y)
		{
			return ptr[y * pitch + x];
		}
	};

	// A light-weighted helper class for easier kernel implementation using Blob
	template <typename _T>
	struct BlobWrapperConst
	{
		const _T *ptr;
		uint pitch;
		uint nx;
		uint ny;
		uint nz;

		__device__ const _T &operator () (uint x, uint y, uint z)
		{
			return ptr[z * pitch * ny + y * pitch + x];
		}

		__device__ const _T &operator () (uint x, uint y)
		{
			return ptr[y * pitch + x];
		}
	};
}


#endif // !__BLOB_H__