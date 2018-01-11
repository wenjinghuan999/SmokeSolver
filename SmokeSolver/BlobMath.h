#pragma once

#ifndef __BLOB_MATH_H__
#define __BLOB_MATH_H__

#include "common.h"
#include "Blob.h"


namespace ssv
{
	// ==================== Element-wise add ====================
	/** 
	 * \brief Element-wise add.\n
	 * \p qout[i] = \p q1[i] + \p q2[i] 
	 * \tparam T element data type, one of real, real2 or real4
	 */
	template <typename T>
	void add(Blob<T> &qout, const Blob<T> &q1, const Blob<T> &q2);
	/** 
	 * \brief Element-wise add.\n
	 * \p qout[i] = \p q1[i] + \p v 
	 * \tparam T element data type, one of real, real2 or real4
	 */
	template <typename T>
	void add(Blob<T> &qout, const Blob<T> &q1, T v);
	/** 
	 * \brief Element-wise add.\n
	 * \p q1[i] += \p q2[i] 
	 * \tparam T element data type, one of real, real2 or real4
	 */
	template <typename T>
	Blob<T> &operator+=(Blob<T> &q1, const Blob<T> &q2);
	/** 
	 * \brief Element-wise add.\n
	 * \p q1[i] += \p v 
	 * \tparam T element data type, one of real, real2 or real4
	 */
	template <typename T>
	Blob<T> &operator+=(Blob<T> &q1, T v);

	// ==================== Element-wise sub ====================
	/** 
	 * \brief Element-wise sub.\n
	 * \p qout[i] = \p q1[i] - \p q2[i] 
	 * \tparam T element data type, one of real, real2 or real4
	 */
	template <typename T>
	void sub(Blob<T> &qout, const Blob<T> &q1, const Blob<T> &q2);
	/** 
	 * \brief Element-wise sub.\n
	 * \p qout[i] = \p q1[i] - \p v 
	 * \tparam T element data type, one of real, real2 or real4
	 */
	template <typename T>
	void sub(Blob<T> &qout, const Blob<T> &q1, T v);
	/** 
	 * \brief Element-wise sub.\n
	 * \p q1[i] -= \p q2[i] 
	 * \tparam T element data type, one of real, real2 or real4
	 */
	template <typename T>
	Blob<T> &operator-=(Blob<T> &q1, const Blob<T> &q2);
	/** 
	 * \brief Element-wise sub.\n
	 * \p q1[i] -= \p v 
	 * \tparam T element data type, one of real, real2 or real4
	 */
	template <typename T>
	Blob<T> &operator-=(Blob<T> &q1, T v);

	// ==================== Element-wise mul ====================
	/** 
	 * \brief Element-wise, dimension-wise mul.\n
	 * \p qout[i] = \p q1[i] * \p q2[i] 
	 * \tparam T element data type, one of real, real2 or real4
	 */
	template <typename T>
	void mul(Blob<T> &qout, const Blob<T> &q1, const Blob<T> &q2);
	/** 
	 * \brief Element-wise, dimension-wise mul.\n
	 * \p qout[i] = \p q1[i] * \p v 
	 * \tparam T element data type, one of real, real2 or real4
	 */
	template <typename T>
	void mul(Blob<T> &qout, const Blob<T> &q1, T v);
	/** 
	 * \brief Element-wise, dimension-wise mul.\n
	 * \p q1[i] *= \p q2[i] 
	 * \tparam T element data type, one of real, real2 or real4
	 */
	template <typename T>
	Blob<T> &operator*=(Blob<T> &q1, const Blob<T> &q2);
	/** 
	 * \brief Element-wise, dimension-wise mul.\n
	 * \p q1[i] *= \p v 
	 * \tparam T element data type, one of real, real2 or real4
	 */
	template <typename T>
	Blob<T> &operator*=(Blob<T> &q1, T v);

	// ==================== Element-wise div ====================

	/** 
	 * \brief Element-wise, dimension-wise div.\n
	 * \p qout[i] = \p q1[i] / \p q2[i]
	 * \tparam T element data type, one of real, real2 or real4
	 */
	template <typename T>
	void div(Blob<T> &qout, const Blob<T> &q1, const Blob<T> &q2);
	/** 
	 * \brief Element-wise, dimension-wise div.\n
	 * \p qout[i] = \p q1[i] / \p v
	 * \tparam T element data type, one of real, real2 or real4
	 */
	template <typename T>
	void div(Blob<T> &qout, const Blob<T> &q1, T v);
	/** 
	 * \brief Element-wise, dimension-wise div.\n
	 * \p q1[i] /= \p q2[i]
	 * \tparam T element data type, one of real, real2 or real4
	 */
	template <typename T>
	Blob<T> &operator/=(Blob<T> &q1, const Blob<T> &q2);
	/** 
	 * \brief Element-wise, dimension-wise div.\n
	 * \p q1[i] /= \p v
	 * \tparam T element data type, one of real, real2 or real4
	 */
	template <typename T>
	Blob<T> &operator/=(Blob<T> &q1, T v);

	// ==================== Element-wise neg ====================

	/** 
	 * \brief Element-wise neg.\n
	 * \p q[i] = -\p q[i] 
	 * \tparam T element data type, one of real, real2 or real4
	 */
	template <typename T>
	void neg(Blob<T> &q);

	// ==================== Element-wise abs ====================
	// Element-wise abs
	/** \brief 
	 * Element-wise abs.\n
	 * \p q[i] = abs(\p q[i]) 
	 * \tparam T element data type, one of real, real2 or real4
	 */
	template <typename T>
	void abs(Blob<T> &q);

	// ==================== Element-wise norm ===================
	// Element-wise norm
	/** 
	 * \brief Element-wise norm.\n
	 * \p n[i] = norm(\p q[i]) = sqrt(\p q[i].x * \p q[i].x + \p q[i].y * \p q[i].y)
	 * \param n output Blob of scalar (real)
	 * \param q input Blob of 2D vector (real2)
	 */
	void norm(Blob<real> &n, const Blob<real2> &q);
	/** 
	 * \brief Element-wise norm.\n
	 * \p n[i] = norm(\p q[i]) = sqrt(\p q[i].x * \p q[i].x + \p q[i].y * \p q[i].y + \p q[i].z * \p q[i].z) 
	 * \param n output Blob of scalar (real)
	 * \param q input Blob of 3D vector (real4, last dimension unused)
	 */
	void norm(Blob<real> &n, const Blob<real4> &q);

	// ================= Element-wise normalize =================
	// Element-wise normalize
	/**
	 * \brief Element-wise normalize\n
	 * \p q[i].x = \p q[i].x / sqrt(\p q[i].x * \p q[i].x + \p q[i].y * \p q[i].y)\n
	 * \p q[i].y = \p q[i].y / sqrt(\p q[i].x * \p q[i].x + \p q[i].y * \p q[i].y)
	 * \param q input Blob of 2D vector (real2)
	 */
	void normalize(Blob<real2> &q);
	/**
	 * \brief Element-wise normalize\n
	 * \p q[i].x = \p q[i].x / sqrt(\p q[i].x * \p q[i].x + \p q[i].y * \p q[i].y + \p q[i].z * \p q[i].z)\n
	 * \p q[i].y = \p q[i].y / sqrt(\p q[i].x * \p q[i].x + \p q[i].y * \p q[i].y + \p q[i].z * \p q[i].z)\n
	 * \p q[i].z = \p q[i].z / sqrt(\p q[i].x * \p q[i].x + \p q[i].y * \p q[i].y + \p q[i].z * \p q[i].z)
	 * \param q input Blob of 3D vector (real4, last dimension unused)
	 */
	void normalize(Blob<real4> &q);

	// ==================== Element-wise zip ====================
	/**
	 * \brief Element-wise zip\n
	 * \p qout[i].x = \p qx[i], \p qout[i].y = \p qy[i]
	 * \param qout output Blob of 2D vector (real2)
	 * \param qx input Blob of scalar (real)
	 * \param qy input Blob of scalar (real)
	 */
	void zip(Blob<real2> &qout, const Blob<real> &qx, const Blob<real> &qy);
	/**
	 * \brief Element-wise zip\n
	 * \p qout[i].x = \p qx[i], \p qout[i].y = \p qy[i], \p qout[i].z = \p qz[i]
	 * \param qout output Blob of 3D vector (real4, last dimension set to 0)
	 * \param qx input Blob of scalar (real)
	 * \param qy input Blob of scalar (real)
	 * \param qz input Blob of scalar (real)
	 */
	void zip(Blob<real4> &qout, const Blob<real> &qx, const Blob<real> &qy, const Blob<real> &qz);
	/**
	 * \brief Element-wise zip\n
	 * \p qout[i].x = \p qx[i], \p qout[i].y = \p qy[i], \p qout[i].z = \p qz[i], \p qout[i].w = \p qw[i]
	 * \param qout output Blob of 4D vector (real4)
	 * \param qx input Blob of scalar (real)
	 * \param qy input Blob of scalar (real)
	 * \param qz input Blob of scalar (real)
	 * \param qw input Blob of scalar (real)
	 */
	void zip(Blob<real4> &qout, const Blob<real> &qx, const Blob<real> &qy, const Blob<real> &qz, const Blob<real> &qw);

	// =================== Element-wise unzip ===================
	/**
	 * \brief Element-wise unzip\n
	 * \p qxout[i] = \p q[i].x, \p qyout[i] = \p q[i].y
	 * \param qxout output Blob of scalar (real)
	 * \param qyout output Blob of scalar (real)
	 * \param q input Blob of 2D vector (real2)
	 */
	void unzip(Blob<real> &qxout, Blob<real> &qyout, const Blob<real2> &q);
	/**
	 * \brief Element-wise unzip\n
	 * \p qxout[i] = \p q[i].x, \p qyout[i] = \p q[i].y, \p qzout[i] = \p q[i].z
	 * \param qxout output Blob of scalar (real)
	 * \param qyout output Blob of scalar (real)
	 * \param qzout output Blob of scalar (real)
	 * \param q input Blob of 3D vector (real4, last dimension unused)
	 */
	void unzip(Blob<real> &qxout, Blob<real> &qyout, Blob<real> &qzout, const Blob<real4> &q);
	/**
	 * \brief Element-wise unzip\n
	 * \p qxout[i] = \p q[i].x, \p qyout[i] = \p q[i].y, \p qzout[i] = \p q[i].z, \p qwout[i] = \p q[i].w
	 * \param qxout output Blob of scalar (real)
	 * \param qyout output Blob of scalar (real)
	 * \param qzout output Blob of scalar (real)
	 * \param qwout output Blob of scalar (real)
	 * \param q input Blob of 4D vector (real4)
	 */
	void unzip(Blob<real> &qxout, Blob<real> &qyout, Blob<real> &qzout, Blob<real> &qwout, const Blob<real4> &q);

	// =================== Partial differences ==================
	/**
	 * \brief Partial difference by x\n
	 * \p d(x, y, z) = \p q(x + 1, y, z) + \p q(x - 1, y, z) / 2\n
	 * boudary: \p d(0, y, z) = \p q(1, y, z) - \p q(0, y, z)\n
	 * boudary: \p d(nx - 1, y, z) = \p q(nx - 1, y, z) - \p q(nx - 2, y, z)
	 * \tparam T element data type, one of real, real2 or real4
	 */
	template <typename T>
	void diff_x(Blob<T> &d, const Blob<T> &q);
	/**
	 * \brief Partial difference by y\n
	 * \p d(x, y, z) = \p q(x, y + 1, z) + \p q(x, y - 1, z) / 2\n
	 * boudary: \p d(x, 0, z) = \p q(x, 1, z) - \p q(x, 0, z)\n
	 * boudary: \p d(x, ny - 1, z) = \p q(x, ny - 1, z) - \p q(x, ny - 2, z)
	 * \tparam T element data type, one of real, real2 or real4
	 */
	template <typename T>
	void diff_y(Blob<T> &d, const Blob<T> &q);
	/**
	 * \brief Partial difference by z\n
	 * \p d(x, y, z) = \p q(x, y + 1, z) + \p q(x, y - 1, z) / 2\n
	 * boudary: \p d(x, y, 0) = \p q(x, y, 1) - \p q(x, y, 0)\n
	 * boudary: \p d(x, y, nz - 1) = \p q(x, y, nz - 1) - \p q(x, y, nz - 2)
	 * \tparam T element data type, one of real, real2 or real4
	 */
	template <typename T>
	void diff_z(Blob<T> &d, const Blob<T> &q);

	// =================== Vector differences ===================
	/**
	 * \brief Divergence 2D\n
	 * \p d = diff_x(\p q) + diff_y(\p q)
	 * \param d output Blob of scalar (real)
	 * \param q input Blob of 2D vector (real2)
	 */
	void divergence(Blob<real> &d, const Blob<real2> &q);
	/**
	 * \brief Divergence 3D\n
	 * \p d = diff_x(\p q) + diff_y(\p q) + diff_z(\p q)
	 * \param d output Blob of scalar (real)
	 * \param q input Blob of 3D vector (real4, last dimension unused)
	 */
	void divergence(Blob<real> &d, const Blob<real4> &q);
	/**
	 * \brief Curl 2D\n
	 * \p d = diff_x(\p q).y - diff_y(\p q).x
	 * \param d output Blob of scalar (real)
	 * \param q input Blob of 2D vector (real2)
	 */
	void curl(Blob<real> &d, const Blob<real2> &q);
	/**
	 * \brief Curl 3D\n
	 * \p d.x = diff_y(\p q).z - diff_z(\p q).y\n
	 * \p d.y = diff_z(\p q).x - diff_x(\p q).z\n
	 * \p d.z = diff_x(\p q).y - diff_y(\p q).x
	 * \param d output Blob of 3D vector (real4, last dimension set to 0)
	 * \param q input Blob of 3D vector (real4, last dimension unused)
	 */
	void curl(Blob<real4> &d, const Blob<real4> &q);
	/**
	 * \brief Gradient 2D\n
	 * \p d.x = diff_x(\p q)\n
	 * \p d.y = diff_y(\p q)
	 * \param d output Blob of 2D vector (real2)
	 * \param q input Blob of scalar (real)
	 */
	void gradient(Blob<real2> &d, const Blob<real> &q);
	/**
	 * \brief Gradient 3D\n
	 * \p d.x = diff_x(\p q)\n
	 * \p d.y = diff_y(\p q)\n
	 * \p d.z = diff_z(\p q)
	 * \param d output Blob of 3D vector (real4, last dimension set to 0)
	 * \param q input Blob of scalar (real)
	 */
	void gradient(Blob<real4> &d, const Blob<real> &q);
	/**
	 * \brief Laplacian 2D\n
	 * \p d(x, y) = \p q(x - 1, y) + \p q(x + 1, y) + \p q(x, y - 1) + \p q(x, y + 1) - 4 * q(x, y);
	 * \tparam T element data type, one of real, real2 or real4
	 */
	template <typename T>
	void laplacian_2d(Blob<T> &d, const Blob<T> &q);
	/**
	 * \brief Laplacian 3D\n
	 * \p d(x, y, z) = \p q(x - 1, y, z) + \p q(x + 1, y, z) + \p q(x, y - 1, z) + \p q(x, y + 1, z) + \p q(x, y, z - 1) + \p q(x, y, z + 1) - 6 * q(x, y, z);
	 * \tparam T element data type, one of real, real2 or real4
	 */
	template <typename T>
	void laplacian_3d(Blob<T> &d, const Blob<T> &q);

	/**
	 * \brief Generate 2D simplex noise\n
	 * \p q(x, y) = simplex(\p offset.x + x / \p factor, \p offset.y + y / \p factor)
	 * \param q output noise Blob of scalar (real)
	 * \param factor zoom factor (real2)
	 * \param offset offset to standard simplex noise (real2)
	 */
	void simplex_2d(Blob<real> &q, real2 factor, real2 offset = make_real2(0, 0));
	/**
	 * \brief Generate 2D simplex noise with gradient\n
	 * \p q(x, y) = simplex(\p offset.x + x / \p factor, \p offset.y + y / \p factor)
	 * \p dx(x, y) = -diff_y(\p q)
	 * \p dy(x, y) = diff_x(\p q)
	 * \param q output noise Blob of scalar (real)
	 * \param dx output noise gradient Blob of scalar (real)
	 * \param dy output noise gradient Blob of scalar (real)
	 * \param factor zoom factor (real2)
	 * \param offset offset to standard simplex noise (real2)
	 */
	void simplex_2d(Blob<real> &q, Blob<real> &dx, Blob<real> &dy, real2 factor, real2 offset = make_real2(0, 0));
}

#endif // !__BLOB_MATH_H__
