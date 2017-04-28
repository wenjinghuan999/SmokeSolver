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

#include <memory>

namespace ssv
{
	class Smoke2dSolver : public SmokeSolver
	{
	public:
		Smoke2dSolver() {}
		~Smoke2dSolver() {}

	public:
		void setSize(uint nx, uint ny)
		{
			_nx = nx; _ny = ny;
		}
		template<typename MethodType>
		void setAdvectMethod(MethodType method)
		{
			_advect = method;
			_advect2 = method;
		}
		template<typename MethodType>
		void setEulerMethod(MethodType method)
		{
			_euler = method;
			_euler2 = method;
		}
		template<typename MethodType>
		void setPoissonMethod(MethodType method)
		{
			_poisson = method;
			_poisson2 = method;
		}
		template<typename MethodType>
		void setBoundaryMethod(MethodType method)
		{
			_boundary = method;
		}
		template<typename MethodType>
		void setBoundary2Method(MethodType method)
		{
			_boundary2 = method;
		}
		template<typename MethodType>
		void setForceMethod(MethodType method)
		{
			_force = method;
		}

	public:
		virtual void init();
		virtual void step();
		virtual void destory();
	private:
		void _InitCuda();
		void _StepCuda();
		void _DestroyCuda();
	
	private:
		uint _nx, _ny;
		Blob<T> _data;
		AdvectMethod::type<T, T2> _advect;
		AdvectMethod::type<T2, T2> _advect2;
		EulerMethod::type<T> _euler;
		EulerMethod::type<T2> _euler2;
		PoissonMethod::type<T>  _poisson;
		PoissonMethod::type<T2>  _poisson2;
		BoundaryMethod::type<T, byte> _boundary;
		BoundaryMethod::type<T2, byte> _boundary2;
		ForceMethod::type<T2> _force;
	};
}

#endif // !__SMOKE2D_SOLVER_H__
