#pragma once

#ifndef __BLOB_H__
#define __BLOB_H__

#include "common.h"
#include "BlobBase.h"
#include "pitched_ptr.h"

namespace ssv
{
	template <typename>
	class Blob;
	template <typename>
	struct BlobWrapper;
	template <typename>
	struct BlobWrapperConst;

	/**
	 * \brief Basic data structure, Contains 2D or 3D data on CPU/GPU and provides managed texture objects
	 * \tparam T element data type
	 */
	template <typename T>
	class Blob
		: public BlobBase
	{
	public:
		/** \brief A light-weighted helper class for easier kernel implementation using Blob */
		using wrapper_t = BlobWrapper<T>;
		/** \brief A light-weighted helper class for easier kernel implementation using Blob */
		using wrapper_const_t = BlobWrapperConst<T>;

		/** \brief Construct without allocating memory */
		Blob() = default;

		/**
		 * \brief 
		 * \param nx size in elements
		 * \param ny size in elements
		 * \param nz size in elements
		 * \param gpu_device cuda device id
		 * \param storage if true, copying from and to CPU is enabled
		 */
		Blob(uint nx, uint ny, uint nz = 1u, int gpu_device = 0, storage_t storage = storage_t::BOTH)
			: BlobBase(nx * sizeof(T), ny, nz, gpu_device, storage)
		{
		}

		/**
		 * \brief Construct Blob from tuple formed shape
		 * \param shape size in elements
		 * \param gpu_device cuda device id
		 * \param storage if true, copying from and to CPU is enabled
		 */
		explicit Blob(std::tuple<uint, uint, uint> shape, int gpu_device = 0, storage_t storage = storage_t::BOTH)
			: BlobBase(std::get<0>(shape) * sizeof(T), std::get<1>(shape), std::get<2>(shape),
			           gpu_device, storage)
		{
		}

	public:
		/**
		 * \brief Copy data to some buffer
		 * \param dst destination pointer
		 * \param from from CPU/GPU data of this Blob
		 * \param to to CPU/GPU buffer (i.e. is dst a CPU/GPU pointer)
		 */
		void copy_to(T *dst, storage_t from, storage_t to) const
		{
			_CopyTo(static_cast<void *>(dst), from, to);
		}

		/**
		 * \brief Copy data from some buffer
		 * \param src source pointer
		 * \param from from CPU/GPU buffer (i.e. is src a CPU/GPU pointer)
		 * \param to to CPU/GPU/Both data of this Blob
		 */
		void copy_from(T *src, storage_t from, storage_t to)
		{
			_CopyFrom(static_cast<void *>(src), from, to);
		}

		/**
		* \brief Get GPU data as texture in 2D.\n
		* If no texture of specific parameters exists, a new texture object will be created.\n
		* Each layer has a unique texture. Consider using 3D texture to avoid creating too many textures.\n
		* Note that a linear filter mode and a clamp address mode are used.\n
		* \code tex2D<T>(b.data_texture_2d(), x + 0.5, y + 0.5) = b(x, y); \endcode
		* \param tex_desc (optional) default: clamp addr mode, linear filter, not normalized
		* \param layer_id If the Blob is 3D, use layer_id to specify which layer should be sampled
		* \return cudaTextureObject of GPU data in 2D
		*/
		cudaTextureObject_t data_texture_2d(
			const cudaTextureDesc *tex_desc = nullptr,
			uint layer_id = 0
		) const
		{
			if (tex_desc == nullptr && layer_id == 0)
			{
				if (data_texture_default_2d_)
				{
					return data_texture_default_2d_;
				}
			}

			cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<T>();
			texture_param_t params = _MakeTextureParam(
				tex_desc, &channel_desc, 2u, layer_id
			);
			auto iter = data_textures_.find(params);
			if (iter != data_textures_.end())
			{
				return iter->second;
			}

			cudaTextureObject_t texture_object = _CreateTexture2D(params);

			if (tex_desc == nullptr && layer_id == 0)
			{
				data_texture_default_2d_ = texture_object;
			}
			else
			{
				data_textures_[params] = texture_object;
			}

			return texture_object;
		}

		/**
		* \brief Get GPU data as texture in 3D.\n
		* If no texture of specific parameters exists, a new texture object will be created.\n
		* Call this method to re-obtain the texture after GPU data are modified.\n
		* A memory copy is needed. Consider using linear memory for better performance.\n
		* Note that a linear filter mode and a clamp address mode are used.\n
		* \code tex3D<T>(b.data_texture_3d(), x + 0.5, y + 0.5, z + 0.5) = b(x, y, z); \endcode
		* \param tex_desc (optional) default: clamp addr mode, linear filter, not normalized
		* \return cudaTextureObject of GPU data in 3D.
		*/
		cudaTextureObject_t data_texture_3d(
			const cudaTextureDesc *tex_desc = nullptr
		) const
		{
			if (tex_desc == nullptr)
			{
				if (data_texture_default_3d_)
				{
					_CopyToCudaArray();
					return data_texture_default_3d_;
				}
			}
			cudaChannelFormatDesc s_channel_desc = cudaCreateChannelDesc<T>();
			texture_param_t params = _MakeTextureParam(
				tex_desc, &s_channel_desc, 3u, 0
			);
			auto iter = data_textures_.find(params);
			if (iter != data_textures_.end())
			{
				_CopyToCudaArray();
				return iter->second;
			}

			cudaTextureObject_t texture_object = _CreateTexture3D(params);

			if (tex_desc == nullptr)
			{
				data_texture_default_3d_ = texture_object;
			}
			else
			{
				data_textures_[params] = texture_object;
			}

			_CopyToCudaArray();
			return texture_object;
		}

	public:
		/** \brief Set data to \p value in cube [\p x0, \p x1, \p y0, \p y1, \p z0, \p z1] */
		void set_data_cube_cpu(T value, uint x0, uint x1, uint y0, uint y1, uint z0 = 0, uint z1 = 0)
		{
			uint nx = this->nx();
			if (x0 >= nx || x1 >= nx || x0 > x1
				|| y0 >= nx || y0 >= ny_ || y0 > y1
				|| z0 >= nx || z0 >= nz_ || z0 > z1)
			{
				throw error_t(error_t::SSV_ERROR_INVALID_VALUE);
			}
			T *pa = data_cpu();
			for (uint z = z0; z <= z1; z++)
			{
				T *pz = pa + z * ny_ * nx;
				for (uint y = y0; y <= y1; y++)
				{
					T *p = pz + y * nx + x0;
					for (uint x = x0; x <= x1; x++)
					{
						*(p++) = value;
					}
				}
			}
		}

	public:
		/** \return Return nx (in elements) */
		uint nx() const
		{
			return static_cast<uint>(nx_in_bytes_ / sizeof(T));
		}

		/** \return total number of elements ( = nx() * ny() * nz()) */
		uint numel() const
		{
			return static_cast<uint>(nx_in_bytes_ * ny_ * nz_ / sizeof(T));
		}

		/** \return shape ( = make_tuple(nx(), ny(), nz())) */
		shape_t shape() const
		{
			return std::make_tuple(static_cast<uint>(nx_in_bytes_ / sizeof(T)), ny_, nz_);
		}

		/**
		 * \return pitch in elements.\n
		 * Total memory allocated = pitch_in_elements * ny * nz * sizeof(_T)
		 */
		uint pitch_in_elements() const
		{
			return static_cast<uint>(data_gpu_.pitch / sizeof(T));
		}

		/** \return raw pointer of CPU data */
		const T *data_cpu() const
		{
			return static_cast<const T *>(data_cpu_);
		}

		/** \return raw pointer of CPU data */
		T *data_cpu()
		{
			return static_cast<T *>(data_cpu_);
		}

		/** \return Begining pointer of CPU data (= data_cpu_void()) */
		const T *begin_cpu() const
		{
			return static_cast<const T *>(data_cpu_);
		}

		/** \return begining pointer of CPU data (= data_cpu_void()) */
		T *begin_cpu()
		{
			return static_cast<T *>(data_cpu_);
		}

		/** \return ending pointer of CPU data (= data_cpu_void() + numel()) */
		const T *end_cpu() const
		{
			return static_cast<const T *>(data_cpu_) + numel();
		}

		/** \return ending pointer of CPU data (= data_cpu_void() + numel()) */
		T *end_cpu()
		{
			return static_cast<T *>(data_cpu_) + numel();
		}

		/** \return pitched_ptr of GPU data */
		pitched_ptr<const T> data_gpu() const
		{
			return pitched_ptr<const T>(&data_gpu_);
		}

		/** \return pitched_ptr of GPU data */
		pitched_ptr<T> data_gpu()
		{
			return pitched_ptr<T>(&data_gpu_);
		}

		/** \return begining pitched_ptr of GPU data (= data_gpu()) */
		pitched_ptr<const T> begin_gpu() const
		{
			return pitched_ptr<const T>(&data_gpu_);
		}

		/** \return begining pitched_ptr of GPU data (= data_gpu()) */
		pitched_ptr<T> begin_gpu()
		{
			return pitched_ptr<T>(&data_gpu_);
		}

		/** \return ending pitched_ptr of GPU data (= data_gpu() + numel()) */
		pitched_ptr<const T> end_gpu() const
		{
			return pitched_ptr<const T>(&data_gpu_) + numel();
		}

		/** \return ending pitched_ptr of GPU data (= data_gpu() + numel()) */
		pitched_ptr<T> end_gpu()
		{
			return pitched_ptr<T>(&data_gpu_) + numel();
		}

		/** \return raw pointer of GPU data */
		const T *data_gpu_raw() const
		{
			return static_cast<const T *>(data_gpu_.ptr);
		}

		/** \return raw pointer of GPU data */
		T *data_gpu_raw()
		{
			return static_cast<T *>(data_gpu_.ptr);
		}

		/** \return wrapper of this Blob */
		wrapper_t wrapper()
		{
			return wrapper_t{
				static_cast<T *>(data_gpu_.ptr), static_cast<uint>(data_gpu_.pitch / sizeof(T)),
				static_cast<uint>(nx_in_bytes_ / sizeof(T)), ny_, nz_
			};
		}

		/** \return const wrapper of this Blob */
		wrapper_const_t wrapper_const() const
		{
			return wrapper_const_t{
				static_cast<const T *>(data_gpu_.ptr), static_cast<uint>(data_gpu_.pitch / sizeof(T)),
				static_cast<uint>(nx_in_bytes_ / sizeof(T)), ny_, nz_
			};
		}
	};

	/**
	 * \brief A light-weighted helper class for easier kernel implementation using Blob
	 * \tparam T element data type
	 */
	template <typename T>
	struct BlobWrapperConst
	{
		const T *ptr;
		uint pitch;
		uint nx;
		uint ny;
		uint nz;

		__device__ const T &operator()(uint x, uint y, uint z)
		{
			return ptr[z * pitch * ny + y * pitch + x];
		}

		__device__ const T &operator()(uint x, uint y)
		{
			return ptr[y * pitch + x];
		}
	};

	/**
	* \brief A light-weighted helper class for easier kernel implementation using Blob
	* \tparam T element data type
	*/
	template <typename T>
	struct BlobWrapper
	{
		T *ptr;
		uint pitch;
		uint nx;
		uint ny;
		uint nz;

		operator BlobWrapperConst<T>() const
		{
			return BlobWrapperConst<T>{ptr, pitch, nx, ny, nz};
		}

		__device__ T &operator()(uint x, uint y, uint z)
		{
			return ptr[z * pitch * ny + y * pitch + x];
		}

		__device__ T &operator()(uint x, uint y)
		{
			return ptr[y * pitch + x];
		}
	};
}

#endif // !__BLOB_H__
