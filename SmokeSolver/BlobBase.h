#pragma once

#ifndef __BLOB_BASE_H__
#define __BLOB_BASE_H__

#include "common.h"

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
		void setSize(size_t nx, size_t ny, size_t nz = 1u, 
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

		// Create cudaTextureObject of GPU data in 2D
		// If the Blob is 3D, use layer_id to specify which layer should be sampled.
		// texDesc & channelDesc are optional
		// default texDesc: clamp addr mode, linear filter, not normalized
		// default channelDesc: byte
		cudaTextureObject_t createTexture2d(
			const cudaTextureDesc *texDesc = nullptr,
			const cudaChannelFormatDesc *channelDesc = nullptr,
			size_t layer_id = 0
		);

		// cudaTextureObject of GPU data in 3D
		// texDesc & channelDesc are optional
		// default texDesc: clamp addr mode, linear filter, not normalized
		// default channelDesc: byte
		cudaTextureObject_t createTexture3d(
			const cudaTextureDesc *texDesc = nullptr,
			const cudaChannelFormatDesc *channelDesc = nullptr
		);

	public:
		// Return nx (in elements)
		size_t nx() const { return _nx; }

		// Return ny (in elements)
		size_t ny() const { return _ny; }

		// Return nz (in elements)
		size_t nz() const { return _nz; }

		// Return total number of elements ( = nx * ny * nz)
		size_t numel() const { return _numel; }

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
		cudaTextureObject_t data_texture_2d() const
		{
			return _data_texture_2d;
		}

		// cudaTextureObject of GPU data in 3D
		// Call this method to re-obtain the texture after GPU data are modified
		// A memory copy is needed. Consider using linear memory for better performance.
		cudaTextureObject_t data_texture_3d() const;

	protected:
		void _InitCuda(int gpu_device);
		void _DestroyCuda();

	protected:
		size_t _nx, _ny, _nz, _numel;
		void *_data_cpu;

	protected:
		int _storage_gpu_device;
		cudaExtent _data_gpu_extent;
		cudaPitchedPtr _data_gpu;
		cudaTextureObject_t _data_texture_2d, _data_texture_3d;
		cudaArray_t _data_cuda_array;
	};
}

#endif // !__BLOB_BASE_H__
