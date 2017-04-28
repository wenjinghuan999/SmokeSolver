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
		typedef BlobBase::shape_t shape_t;
		typedef BlobWrapper<_T> wrapper_t;
		typedef BlobWrapperConst<_T> wrapper_const_t;

	public:
		Blob() : BlobBase() {}

		// nx, ny, nz: size in elements
		// gpu_device: cuda device id
		// cpu_copy: if true, copying from and to CPU is enabled
		Blob(uint nx, uint ny, uint nz = 1u, int gpu_device = 0, bool cpu_copy = true)
			: BlobBase(nx * sizeof(_T), ny, nz, gpu_device, cpu_copy) {}

		// shape: size in elements
		// gpu_device: cuda device id
		// cpu_copy: if true, copying from and to CPU is enabled
		Blob(std::tuple<uint, uint, uint> shape, int gpu_device = 0, bool cpu_copy = true)
			: BlobBase(std::get<0>(shape) * sizeof(_T), std::get<1>(shape), std::get<2>(shape),
				gpu_device, cpu_copy) {}

	public:
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
		// Set data in cube
		void setDataCubeCpu(_T value, uint x0, uint x1, uint y0, uint y1, uint z0 = 0, uint z1 = 0)
		{
			uint _nx = nx();
			if (x0 >= _nx || x1 >= _nx || x0 > x1
				|| y0 >= _nx || y0 >= _ny || y0 > y1
				|| z0 >= _nx || z0 >= _nz || z0 > z1)
			{
				throw error_t::SSV_ERROR_INVALID_VALUE;
			}
			_T *pa = data_cpu();
			for (uint z = z0; z <= z1; z++)
			{
				_T *pz = pa + z * _ny * _nx;
				for (uint y = y0; y <= y1; y++)
				{
					_T *p = pz + y * _nx + x0;
					for (uint x = x0; x <= x1; x++)
					{
						*(p++) = value;
					}
				}
			}
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
			return (uint)(_nx_in_bytes * _ny * _nz / sizeof(_T));
		}

		// Return shape 
		// ( = make_tuple(nx(), ny(), nz()))
		virtual shape_t shape() const override
		{
			return std::make_tuple((uint)(_nx_in_bytes / sizeof(_T)), _ny, _nz);
		}

		// Return pitch in elements
		// Total memory allocated = pitch_in_elements * ny * nz * sizeof(_T)
		virtual uint pitch_in_elements() const override
		{
			return (uint)(_data_gpu.pitch / sizeof(_T));
		}

		// Raw pointer of CPU data
		const _T *data_cpu() const
		{
			return static_cast<const _T *>(_data_cpu);
		}

		// Raw pointer of CPU data
		_T *data_cpu()
		{
			return static_cast<_T *>(_data_cpu);
		}

		// Begining pointer of CPU data
		// = data_cpu()
		const _T *begin_cpu() const
		{
			return static_cast<const _T *>(_data_cpu);
		}

		// Begining pointer of CPU data
		// = data_cpu()
		_T *begin_cpu()
		{
			return static_cast<_T *>(_data_cpu);
		}

		// Ending pointer of CPU data
		// = data_cpu() + numel()
		const _T *end_cpu() const
		{
			return static_cast<const _T *>(_data_cpu) + numel();
		}

		// Ending pointer of CPU data
		// = data_cpu() + numel()
		_T *end_cpu()
		{
			return static_cast<_T *>(_data_cpu) + numel();
		}

		// pitched_ptr of GPU data
		pitched_ptr<const _T> data_gpu() const
		{
			return pitched_ptr<const _T>(&_data_gpu);
		}

		// pitched_ptr of GPU data
		pitched_ptr<_T> data_gpu()
		{
			return pitched_ptr<_T>(&_data_gpu);
		}

		// Begining pitched_ptr of GPU data
		// = data_gpu()
		pitched_ptr<const _T> begin_gpu() const
		{
			return pitched_ptr<const _T>(&_data_gpu);
		}

		// Begining pitched_ptr of GPU data
		// = data_gpu()
		pitched_ptr<_T> begin_gpu()
		{
			return pitched_ptr<_T>(&_data_gpu);
		}

		// Ending pitched_ptr of GPU data
		// = data_gpu() + numel()
		pitched_ptr<const _T> end_gpu() const
		{
			return pitched_ptr<const _T>(&_data_gpu) + numel();
		}

		// Ending pitched_ptr of GPU data
		// = data_gpu() + numel()
		pitched_ptr<_T> end_gpu()
		{
			return pitched_ptr<_T>(&_data_gpu) + numel();
		}

		// Raw pointer of GPU data
		const _T *data_gpu_raw() const
		{
			return static_cast<const _T *>(_data_gpu.ptr);
		}

		// Raw pointer of GPU data
		_T *data_gpu_raw()
		{
			return static_cast<_T *>(_data_gpu.ptr);
		}

		// Return BlobWrapper of this Blob
		wrapper_t wrapper()
		{
			return wrapper_t{
				static_cast<_T *>(_data_gpu.ptr), (uint)(_data_gpu.pitch / sizeof(_T)),
					(uint)(_nx_in_bytes / sizeof(_T)), _ny, _nz };
		}

		// Return BlobWrapperConst of this Blob
		wrapper_const_t wrapper_const() const
		{
			return wrapper_const_t{
				static_cast<const _T *>(_data_gpu.ptr), (uint)(_data_gpu.pitch / sizeof(_T)),
					(uint)(_nx_in_bytes / sizeof(_T)), _ny, _nz };
		}
	};

	// A light-weighted helper class for easier kernel implementation using Blob
	template<typename _T>
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

	// A light-weighted helper class for easier kernel implementation using Blob
	template<typename _T>
	struct BlobWrapper
	{
		_T *ptr;
		uint pitch;
		uint nx;
		uint ny;
		uint nz;

		operator BlobWrapperConst<_T>() const
		{
			return BlobWrapperConst<_T>{ ptr, pitch, nx, ny, nz };
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
}

#endif // !__BLOB_H__