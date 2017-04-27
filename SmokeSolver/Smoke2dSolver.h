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
		template<typename MethodType, typename ...ArgTypes>
		void setAdvectMethod(ArgTypes ...args)
		{
			_advect = std::make_unique<MethodType>(std::forward<ArgTypes>(args)...);
		}
		template<typename MethodType, typename ...ArgTypes>
		void setEulerMethod(ArgTypes ...args)
		{
			_euler = std::make_unique<MethodType>(std::forward<ArgTypes>(args)...);
		}
		template<typename MethodType, typename ...ArgTypes>
		void setPoissonMethod(ArgTypes ...args)
		{
			_poisson = std::make_unique<MethodType>(std::forward<ArgTypes>(args)...);
		}
		template<typename MethodType, typename ...ArgTypes>
		void setBoundaryMethod(ArgTypes ...args)
		{
			_boundary = std::make_unique<MethodType>(std::forward<ArgTypes>(args)...);
		}
		template<typename MethodType, typename ...ArgTypes>
		void setForceMethod(ArgTypes ...args)
		{
			_force = std::make_unique<MethodType>(std::forward<ArgTypes>(args)...);
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
		std::unique_ptr<AdvectMethod<T> > _advect;
		std::unique_ptr<EulerMethod<T> > _euler;
		std::unique_ptr<PoissonMethod<T> > _poisson;
		std::unique_ptr<BoundaryMethod<T, byte> > _boundary;
		std::unique_ptr<ForceMethod<T, T2> > _force;
	};
}

#endif // !__SMOKE2D_SOLVER_H__
