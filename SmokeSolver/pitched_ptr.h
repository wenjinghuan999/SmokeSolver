#pragma once

#ifndef __PITCHED_PTR_H__
#define __PITCHED_PTR_H__

#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/device_ptr.h>

namespace ssv
{
	template<typename Iterator>
	class pitched_iterator
		: public thrust::iterator_adaptor<
		pitched_iterator<Iterator>, // the first template parameter is the name of the iterator we're creating
		Iterator                    // the second template parameter is the name of the iterator we're adapting
									// we can use the default for the additional template parameters
		>
	{
	public:
		// shorthand for the name of the iterator_adaptor we're deriving from
		typedef thrust::iterator_adaptor<
			pitched_iterator<Iterator>,
			Iterator
		> super_t;

		__host__ __device__
			pitched_iterator(const Iterator &super, size_t pitch, size_t xsize)
			: super_t(super), begin(super), pitch(pitch), xsize(xsize) {}

		// befriend thrust::iterator_core_access to allow it access to the private interface below
		friend class thrust::iterator_core_access;
	private:
		// pitch and xsize
		size_t pitch;
		size_t xsize;
		// used to keep track of the beginning
		const Iterator begin;

		// it is private because only thrust::iterator_core_access needs access to it
		__host__ __device__
			typename super_t::reference dereference() const
		{
			size_t n = this->base() - begin;
			size_t nx = n / xsize;
			size_t ny = n % xsize;
			return *(begin + nx * pitch + ny);
		}
	};

	template<typename T>
	class pitched_ptr
		: public thrust::iterator_adaptor<
		pitched_ptr<T>,         // the first template parameter is the name of the iterator we're creating
		thrust::device_ptr<T>   // the second template parameter is the name of the iterator we're adapting
								// we can use the default for the additional template parameters          
		>
	{
	public:
		// shorthand for the name of the iterator_adaptor we're deriving from
		typedef thrust::iterator_adaptor<
			pitched_ptr<T>,
			thrust::device_ptr<T>
		> super_t;

		__host__ __device__
			pitched_ptr(T *devPtr, size_t pitch, size_t xsize)
			: super_t(thrust::device_pointer_cast(devPtr)), begin(thrust::device_ptr<T>(devPtr)),
			pitch(pitch / sizeof(T)), xsize(xsize / sizeof(T)) {}
		__host__ __device__
			pitched_ptr(const cudaPitchedPtr *pitched_ptr)
			: super_t(thrust::device_pointer_cast(static_cast<T *>(pitched_ptr->ptr))),
			begin(thrust::device_pointer_cast(static_cast<T *>(pitched_ptr->ptr))),
			pitch(pitched_ptr->pitch / sizeof(T)), xsize(pitched_ptr->xsize / sizeof(T)) {}

		// befriend thrust::iterator_core_access to allow it access to the private interface below
		friend class thrust::iterator_core_access;
	private:
		// pitch and xsize
		size_t pitch;
		size_t xsize;
		// used to keep track of the beginning
		const thrust::device_ptr<T> begin;

		// it is private because only thrust::iterator_core_access needs access to it
		__host__ __device__
			typename super_t::reference dereference() const
		{
			size_t n = this->base() - begin;
			size_t nx = n / xsize;
			size_t ny = n % xsize;
			return *(begin + nx * pitch + ny);
		}
	};
}
#endif // !__PITCHED_PTR_H__
