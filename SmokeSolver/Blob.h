#pragma once

#ifndef __BLOB_H__
#define __BLOB_H__

#include "common.h"
#include "BlobBase.h"
#include "pitched_ptr.h"

#include <cuda_texture_types.h>

namespace ssv
{
	template <typename _T>
	class Blob
		: public BlobBase
	{
	public:
		Blob() : BlobBase() {}

		// Set or reset Blob parameters
		// nx, ny, nz: size in elements
		// gpu_device: cuda device id
		// cpu_copy: if true, copying from and to CPU is enabled
		void setSize(size_t nx, size_t ny, size_t nz = 1u, int gpu_device = 0, bool cpu_copy = true)
		{
			BlobBase::setSize(nx * sizeof(_T), ny, nz, gpu_device, cpu_copy);
		}

		// Create cudaTextureObject of GPU data in 2D
		// If the Blob is 3D, use layer_id to specify which layer should be sampled.
		// texDesc & channelDesc are optional
		// default texDesc: clamp addr mode, linear filter, not normalized
		// default channelDesc: byte
		cudaTextureObject_t createTexture2d(
			const cudaTextureDesc *texDesc = nullptr,
			size_t layer_id = 0
		)
		{
			cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<_T>();
			return BlobBase::createTexture2d(texDesc, &channelDesc, layer_id);
		}

		// cudaTextureObject of GPU data in 3D
		// texDesc & channelDesc are optional
		// default texDesc: clamp addr mode, linear filter, not normalized
		// default channelDesc: byte
		cudaTextureObject_t createTexture3d(
			const cudaTextureDesc *texDesc = nullptr
		)
		{
			cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<_T>();
			return BlobBase::createTexture3d(texDesc, &channelDesc);
		}

	public:		
		// Return pitch in elements
		// Total memory allocated = pitch_in_elements * ny * nz * sizeof(_T)
		size_t pitch_in_elements() const
		{
			return _data_gpu.pitch / sizeof(_T);
		}

		// Raw pointer of CPU data
		_T *data_cpu()
		{
			return static_cast<_T *>(_data_cpu);
		}

		// pitched_ptr of GPU data
		pitched_ptr<_T> data_gpu()
		{
			return pitched_ptr<_T>(&_data_gpu);
		}

		// Raw pointer of GPU data
		_T *data_gpu_raw()
		{
			return static_cast<_T *>(_data_gpu.ptr);
		}
	};
}


#endif // !__BLOB_H__