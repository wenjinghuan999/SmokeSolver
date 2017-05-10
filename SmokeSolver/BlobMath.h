#pragma once

#ifndef __BLOB_MATH_H__
#define __BLOB_MATH_H__

#include "common.h"
#include "Blob.h"


namespace ssv
{
	// Element-wise add
	template <typename _T>
	void add(Blob<_T> &qout, const Blob<_T> &q1, const Blob<_T> &q2);
	template <typename _T>
	void add(Blob<_T> &qout, const Blob<_T> &q1, _T v);
	template <typename _T>
	Blob<_T> &operator+=(Blob<_T> &q1, const Blob<_T> &q2);
	template <typename _T>
	Blob<_T> &operator+=(Blob<_T> &q1, _T v);

	// Element-wise sub
	template <typename _T>
	void sub(Blob<_T> &qout, const Blob<_T> &q1, const Blob<_T> &q2);
	template <typename _T>
	void sub(Blob<_T> &qout, const Blob<_T> &q1, _T v);
	template <typename _T>
	Blob<_T> &operator-=(Blob<_T> &q1, const Blob<_T> &q2);
	template <typename _T>
	Blob<_T> &operator-=(Blob<_T> &q1, _T v);

	// Element-wise mul
	template <typename _T>
	void mul(Blob<_T> &qout, const Blob<_T> &q1, const Blob<_T> &q2);
	template <typename _T>
	void mul(Blob<_T> &qout, const Blob<_T> &q1, _T v);
	template <typename _T>
	Blob<_T> &operator*=(Blob<_T> &q1, const Blob<_T> &q2);
	template <typename _T>
	Blob<_T> &operator*=(Blob<_T> &q1, _T v);

	// Element-wise div
	template <typename _T>
	void div(Blob<_T> &qout, const Blob<_T> &q1, const Blob<_T> &q2);
	template <typename _T>
	void div(Blob<_T> &qout, const Blob<_T> &q1, _T v);
	template <typename _T>
	Blob<_T> &operator/=(Blob<_T> &q1, const Blob<_T> &q2);
	template <typename _T>
	Blob<_T> &operator/=(Blob<_T> &q1, _T v);

	// Partial difference by x
	template <typename _T>
	void diff_x(Blob<_T> &d, const Blob<_T> &q);
	// Partial difference by y
	template <typename _T>
	void diff_y(Blob<_T> &d, const Blob<_T> &q);
	// Partial difference by z
	template <typename _T>
	void diff_z(Blob<_T> &d, const Blob<_T> &q);
	// Divergence 2D
	void divergence(Blob<T> &d, const Blob<T2> &q);
	// Divergence 3D
	void divergence(Blob<T> &d, const Blob<T4> &q);
	// Gradient 2D
	void gradient(Blob<T2> &d, const Blob<T> &q);
	// Gradient 3D
	void gradient(Blob<T4> &d, const Blob<T> &q);
	// Laplacian 2D
	template <typename _T>
	void laplacian2d(Blob<_T> &d, const Blob<_T> &q);
	// Laplacian 3D
	template <typename _T>
	void laplacian3d(Blob<_T> &d, const Blob<_T> &q);
}

#endif // !__BLOB_MATH_H__

