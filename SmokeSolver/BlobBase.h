#pragma once

#ifndef __BLOB_BASE_H__
#define __BLOB_BASE_H__

#include "common.h"
#include "hash_tuple.h"

#include <unordered_map>

namespace ssv
{
	typedef std::tuple<uint, uint, uint> BlobShape;

	// Base class of Blob
	class BlobBase
	{
	public:
		typedef BlobShape shape_t;
		enum class storage_t : unsigned char
		{
			CPU = 1,
			GPU = 2,
			Both = 3
		};

	public:
		BlobBase();
		// nx: size in bytes
		// ny, nz: size in elements
		// gpu_device: cuda device id for underlying storage
		// cpu_copy: if true, copying from and to CPU is enabled
		BlobBase(size_t nx_in_bytes, uint ny, uint nz = 1u,
			int gpu_device = 0, storage_t storage = storage_t::Both);
		BlobBase(const BlobBase &other);
		BlobBase &operator= (const BlobBase &other);
		BlobBase(BlobBase &&other);
		BlobBase &operator= (BlobBase &&other);
		~BlobBase();

	public:
		// Sync data from GPU to CPU
		void syncGpu2Cpu();

		// Sync data from CPU to GPU
		void syncCpu2Gpu();

		// Copy data to some buffer
		// dst:  destination
		// from: from CPU/GPU data of this Blob
		// to:   to CPU/GPU buffer (i.e. is dst a CPU/GPU pointer)
		void copyTo(void *dst, storage_t from, storage_t to) const;
		void copyTo(cudaPitchedPtr *dst, storage_t from, storage_t to) const;

		// Copy data from some buffer
		// src:  source
		// from: from CPU/GPU buffer (i.e. is src a CPU/GPU pointer)
		// to:   to CPU/GPU/Both data of this Blob
		void copyFrom(void *src, storage_t from, storage_t to);
		void copyFrom(cudaPitchedPtr *src, storage_t from, storage_t to);

		// Destroy the specific texture
		// if texture_object == 0, destroy all textures
		void destroyTexture(cudaTextureObject_t texture_object = 0);

		// Set all memory to zero
		void clear();

	public:
		// Return nx (in elements)
		virtual uint nx() const = 0;

		// Return nx (in bytes)
		size_t nx_in_bytes() const { return _nx_in_bytes; }

		// Return ny (in elements)
		uint ny() const { return _ny; }

		// Return nz (in elements)
		uint nz() const { return _nz; }

		// Return total number of elements 
		// ( = nx() * ny() * nz())
		virtual uint numel() const = 0;

		// Return shape 
		// ( = make_tuple(nx(), ny(), nz()))
		virtual shape_t shape() const = 0;

		// Return GPU device id of underlying storage
		int gpu_device() const { return _storage_gpu_device; }

		// Return total size in bytes on CPU
		// ( = nx_in_bytes() * ny() * nz())
		size_t size_cpu_in_bytes() const 
		{ 
			return _nx_in_bytes * _ny * _nz;
		}

		// Return total size of bytes on GPU
		// ( = pitch_in_bytes() * ny() * nz())
		size_t size_gpu_in_bytes() const
		{
			return _data_gpu.pitch * _ny * _nz;
		}

		// Return pitch in bytes
		// Total memory allocated = pitch_in_bytes() * ny() * nz()
		size_t pitch_in_bytes() const
		{
			return _data_gpu.pitch;
		}

		// Return pitch in elements
		virtual uint pitch_in_elements() const = 0;

		// Return raw pointer of CPU data
		const void *data_cpu() const
		{
			return _data_cpu;
		}

		// Return raw pointer of CPU data
		void *data_cpu()
		{
			return _data_cpu;
		}

		// Return cudaPitchedPtr of GPU data
		const cudaPitchedPtr *data_gpu_cuda_pitched_ptr() const
		{
			return &_data_gpu;
		}

		// Return cudaPitchedPtr of GPU data
		const cudaPitchedPtr *data_gpu_cuda_pitched_ptr()
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
			uint layer_id = 0) const = 0;

		// Return cudaTextureObject of GPU data in 3D
		// If no texture of specific parameters exists, a new texture object will be created.
		// Call this method to re-obtain the texture after GPU data are modified
		// A memory copy is needed. Consider using linear memory for better performance.
		// texDesc is optional
		// default texDesc: clamp addr mode, linear filter, not normalized
		virtual cudaTextureObject_t data_texture_3d(
			const cudaTextureDesc *texDesc = nullptr) const = 0;

	public:
		typedef std::tuple<cudaTextureDesc, cudaChannelFormatDesc, unsigned char, uint> texture_param_t;

	protected:
		// Create cudaTextureObject of GPU data in 2D
		cudaTextureObject_t _CreateTexture2d(const texture_param_t &params) const;

		// Create cudaTextureObject of GPU data in 3D
		cudaTextureObject_t _CreateTexture3d(const texture_param_t &params) const;

		// Copy data from pitched pointer to 3d CUDA array
		void _CopyToCudaArray() const;

		void _InitCuda(int gpu_device = -1);
		void _CopyCuda(const BlobBase &other, int gpu_device);
		void _MoveCuda(BlobBase &&other);
		void _DestroyCuda();


	protected:
		static texture_param_t _MakeTextureParam(
			const cudaTextureDesc * texDesc, const cudaChannelFormatDesc * channelDesc, 
			unsigned char dimension, uint layer_id);

	protected:
		size_t _nx_in_bytes;
		uint _ny, _nz;
		void *_data_cpu;

	protected:
		int _storage_gpu_device;
		cudaExtent _data_gpu_extent;
		cudaPitchedPtr _data_gpu;
		mutable cudaTextureObject_t _data_texture_default_2d, _data_texture_default_3d;
		mutable std::unordered_map<texture_param_t, cudaTextureObject_t, ssv::hash_tuple::hash<texture_param_t> > _data_textures;
		mutable cudaArray_t _data_cuda_array;
	};
}

#endif // !__BLOB_BASE_H__
