#pragma once

#ifndef __SMOKE2D_SOLVER_H__
#define __SMOKE2D_SOLVER_H__


#include "common.h"
#include "SmokeSolver.h"

#include "Blob.h"
#include "AdvectMethod.h"
#include "EulerMethod.h"
#include "PoissonMethod.h"
#include "BoundaryMethod.h"
#include "ForceMethod.h"

namespace ssv
{
	class Smoke2DSolver : public SmokeSolver
	{
	public:
		enum class CellType : byte
		{
			CELL_TYPE_EMPTY = 0,
			CELL_TYPE_WALL = 'w',
			CELL_TYPE_SOURCE = 's'
		};
		enum class Property : byte
		{
			PROPERTY_NONE = 0,
			PROPERTY_DENSITY = 1,
			PROPERTY_TEMPERATURE = 2,
			PROPERTY_VELOCITY = 4,
			PROPERTY_ALL = 7,
		};

	public:
		Smoke2DSolver() = default;
		virtual ~Smoke2DSolver() = default;

	public:
		/**
		 * \brief Set region size
		 * \param nx number of nodes in x
		 * \param ny number of nodes in y
		 */
		void set_size(uint nx, uint ny)
		{
			nx_ = nx;
			ny_ = ny;
		}

		/**
		 * \brief Set advect method for density and velocity
		 * \tparam MethodType derived class of AdvectMethod
		 * \param method an object of MethodType
		 */
		template <typename MethodType>
		void set_advect(MethodType method)
		{
			advect_den_ = method;
			advect_vel_ = method;
		}

		/**
		 * \brief Set euler method for density and velocity
		 * \tparam MethodType derived class of EulerMethod
		 * \param method an object of MethodType
		 */
		template <typename MethodType>
		void set_euler(MethodType method)
		{
			euler_den_ = method;
			euler_vel_ = method;
		}

		/**
		 * \brief Set poisson method for density
		 * \tparam MethodType derived class of PoissonMethod
		 * \param method an object of MethodType
		 */
		template <typename MethodType>
		void set_poisson(MethodType method)
		{
			poisson_ = method;
		}

		/**
		 * \brief Set boundary method for density
		 * \tparam MethodType derived class of BoundaryMethod
		 * \param method an object of MethodType
		 */
		template <typename MethodType>
		void set_boundary_density(MethodType method)
		{
			boundary_den_ = method;
		}

		/**
		 * \brief Set boundary method for velocity
		 * \tparam MethodType derived class of BoundaryMethod
		 * \param method an object of MethodType
		 */
		template <typename MethodType>
		void set_boundary_velocity(MethodType method)
		{
			boundary_vel_ = method;
		}

		/**
		 * \brief Set force method for velocity
		 * \tparam MethodType derived class of BoundaryMethod
		 * \param method an object of MethodType
		 */
		template <typename MethodType>
		void set_force(MethodType method)
		{
			force_ = method;
		}

		/** \brief Add source [\p x0:\p x1, \p y0:\p y1] to region */
		void add_source(uint x0, uint x1, uint y0, uint y1);
		/** \brief Generate random density field and velocity field in region */
		void gen_noise();

		void *get_data(Property property = Property::PROPERTY_ALL, size_t *size = nullptr);
		void save_data(const std::string &filename);

	public:
		void init() override;
		void step() override;
		void destory() override;

	private:
		uint nx_{}, ny_{};
		AdvectMethod::type<real, real2> advect_den_;
		AdvectMethod::type<real2, real2> advect_vel_;
		EulerMethod::type<real> euler_den_;
		EulerMethod::type<real2> euler_vel_;
		PoissonMethod::type<real> poisson_;
		BoundaryMethod::type<real, byte> boundary_den_;
		BoundaryMethod::type<real2, byte> boundary_vel_;
		ForceMethod::type<real2> force_;
	private:
		Blob<byte> tp_;
		Blob<real> rh_[2];
		Blob<real> tm_[2];
		Blob<real2> u_;
		Blob<real> w_;
		Blob<real2> f_;
		Blob<real> temp1_a_, temp1_b_;
		Blob<real2> temp2_c_, temp2_d_;
		int ping_{};
	};
}

#endif // !__SMOKE2D_SOLVER_H__
