#pragma once

#ifndef __BLOB_BASE_H__
#define __BLOB_BASE_H__

#include "common.h"
#include "hash_tuple.h"

#include <unordered_map>

namespace ssv
{
	typedef std::tuple<uint, uint, uint> blob_shape_t;

	// Base class of Blob
	class BlobBase
	{
	public:
		typedef blob_shape_t shape_t;

		enum class storage_t : unsigned char
		{
			CPU = 1,
			GPU = 2,
			BOTH = 3
		};

	public:
		/**
		 * \brief Construct and allocate memory
		 * \param nx_in_bytes size in bytes
		 * \param ny size in elements
		 * \param nz size in elements
		 * \param gpu_device cuda device id for underlying storage
		 * \param storage if true, copying from and to CPU is enabled
		 */
		BlobBase(size_t nx_in_bytes, uint ny, uint nz = 1u,
		         int gpu_device = 0, storage_t storage = storage_t::BOTH);
		BlobBase(const BlobBase &other);
		BlobBase &operator=(const BlobBase &other);
		BlobBase(BlobBase &&other) noexcept;
		BlobBase &operator=(BlobBase &&other) noexcept;
	protected:
		BlobBase();
		~BlobBase();

	public:
		/**
		 * \brief Sync data from GPU to CPU
		 */
		void sync_gpu_to_cpu();

		/**
		 * \brief Sync data from CPU to GPU
		 */
		void sync_cpu_to_gpu();
		
		/**
		 * \brief Destroy the specific texture
		 * \param texture_object the texture to be destroyed. if texture_object == 0, destroy all textures
		 */
		void destroy_texture(cudaTextureObject_t texture_object = 0);

		/**
		 * \brief Set all memory to zero
		 */
		void clear();

	public:
		/** \return nx (in bytes) */
		size_t nx_in_bytes() const { return nx_in_bytes_; }

		/** \return ny (in elements) */
		uint ny() const { return ny_; }

		/** \return nz (in elements) */
		uint nz() const { return nz_; }

		/** \return GPU device id of underlying storage */
		int gpu_device() const { return storage_gpu_device_; }

		/** \return total size in bytes on CPU ( = nx_in_bytes() * ny() * nz()) */
		size_t size_cpu_in_bytes() const
		{
			return nx_in_bytes_ * ny_ * nz_;
		}

		/** \return total size of bytes on GPU ( = pitch_in_bytes() * ny() * nz()) */
		size_t size_gpu_in_bytes() const
		{
			return data_gpu_.pitch * ny_ * nz_;
		}

		/** 
		 * \return pitch in bytes.\n
		 * Total memory allocated = pitch_in_bytes() * ny() * nz() 
		 */
		size_t pitch_in_bytes() const
		{
			return data_gpu_.pitch;
		}
		
		/** \return raw pointer of CPU data */
		const void *data_cpu_void() const
		{
			return data_cpu_;
		}

		/** \return raw pointer of CPU data */
		void *data_cpu_void()
		{
			return data_cpu_;
		}

		/** \return cudaPitchedPtr of GPU data */
		const cudaPitchedPtr *data_gpu_cuda_pitched_ptr() const
		{
			return &data_gpu_;
		}

		/** \return cudaPitchedPtr of GPU data */
		const cudaPitchedPtr *data_gpu_cuda_pitched_ptr()
		{
			return &data_gpu_;
		}

	public:
		typedef std::tuple<cudaTextureDesc, cudaChannelFormatDesc, unsigned char, uint> texture_param_t;

	protected:
		/**
		* \brief Copy data to some buffer
		* \param dst destination
		* \param from from CPU/GPU data of this Blob
		* \param to to CPU/GPU buffer (i.e. is dst a CPU/GPU pointer)
		*/
		void _CopyTo(void *dst, storage_t from, storage_t to) const;

		/**
		* \brief Copy data to some buffer
		* \param dst destination pointer
		* \param from from CPU/GPU data of this Blob
		* \param to to CPU/GPU buffer (i.e. is dst a CPU/GPU pointer)
		*/
		void _CopyTo(cudaPitchedPtr *dst, storage_t from, storage_t to) const;


		/**
		* \brief Copy data from some buffer
		* \param src source pointer
		* \param from from CPU/GPU buffer (i.e. is src a CPU/GPU pointer)
		* \param to to CPU/GPU/Both data of this Blob
		*/
		void _CopyFrom(void *src, storage_t from, storage_t to);

		/**
		* \brief Copy data from some buffer
		* \param src source pointer
		* \param from from CPU/GPU buffer (i.e. is src a CPU/GPU pointer)
		* \param to to CPU/GPU/Both data of this Blob
		*/
		void _CopyFrom(cudaPitchedPtr *src, storage_t from, storage_t to);

		/** \brief Create cudaTextureObject of GPU data in 2D */
		cudaTextureObject_t _CreateTexture2D(const texture_param_t &params) const;

		/** \brief Create cudaTextureObject of GPU data in 3D */
		cudaTextureObject_t _CreateTexture3D(const texture_param_t &params) const;

		/** \brief Copy data from pitched pointer to 3d CUDA array */
		void _CopyToCudaArray() const;

		void _InitCuda(int gpu_device = -1);
		void _CopyCuda(const BlobBase &other, int gpu_device);
		void _MoveCuda(BlobBase &&other);
		void _DestroyCuda();

	protected:
		static texture_param_t _MakeTextureParam(
			const cudaTextureDesc *tex_desc, const cudaChannelFormatDesc *channel_desc,
			unsigned char dimension, uint layer_id);

	protected:
		size_t nx_in_bytes_;
		uint ny_, nz_;
		void *data_cpu_;

	protected:
		int storage_gpu_device_{};
		cudaExtent data_gpu_extent_{};
		cudaPitchedPtr data_gpu_{};
		mutable cudaTextureObject_t data_texture_default_2d_{}, data_texture_default_3d_{};
		mutable std::unordered_map<texture_param_t, cudaTextureObject_t, hash_tuple::hash<texture_param_t> > data_textures_;
		mutable cudaArray_t data_cuda_array_{};
	};
}

#endif // !__BLOB_BASE_H__
