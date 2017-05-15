
#include <iostream>
#include <algorithm>
#include <thread>
using namespace std;

#include <grpc++/grpc++.h>
#include "Smoke2d.grpc.pb.h"

#include "debug_output.h"
#include "Smoke2dSolver.h"
using namespace ssv;

namespace
{
	const size_t DATA_CHUNK_SIZE = 65536u;

	/// ================== CLIENT ======================

	class Smoke2dClient
	{
	public:
		Smoke2dClient(std::shared_ptr<grpc::Channel> channel)
			: stub_(Smoke2d::NewStub(channel)) {}

		void init(uint nx, uint ny)
		{
			Smoke2dInitParams params;
			params.set_nx(nx);
			params.set_ny(ny);
			Result result;

			grpc::ClientContext context;
			stub_->Init(&context, params, &result);

			PrintGrpcError("init", result.status());
		}

		void step()
		{
			Smoke2dStepParams params;
			Result result;

			grpc::ClientContext context;
			grpc::Status status = stub_->Step(&context, params, &result);

			PrintGrpcError("step", result.status());
		}

		void destroy()
		{
			Smoke2dDestroyParams params;
			Result result;

			grpc::ClientContext context;
			grpc::Status status = stub_->Destroy(&context, params, &result);

			PrintGrpcError("destroy", result.status());
		}

		size_t getData(void *data_buffer)
		{
			Smoke2dGetDataParams params;
			grpc::ClientContext context;
			std::unique_ptr<grpc::ClientReader<DataChunk> > reader(
				stub_->GetData(&context, params));

			DataChunk chunk;
			byte *data = static_cast<byte *>(data_buffer);
			while (reader->Read(&chunk))
			{
				size_t size = chunk.data().size();
				memcpy(data, chunk.data().c_str(), size);
				data += size;
			}
			return data - static_cast<byte *>(data_buffer);
		}

	private:
		void PrintGrpcError(const std::string &func_name, uint status) const
		{
			std::cout << "Function " << func_name << " returned with status " << status	<< std::endl;
		}
	private:
		std::unique_ptr<Smoke2d::Stub> stub_;
	};

	/// ================== SERVER ======================
	class Smoke2dService final : public Smoke2d::Service
	{
		grpc::Status Init(grpc::ServerContext* context,
			const Smoke2dInitParams* params,
			Result* result) override
		{
			result->set_status(0);
			try
			{
				solver.setSize(params->nx(), params->ny());
				solver.setAdvectMethod(AdvectMethodSemiLagrangian());
				solver.setEulerMethod(EulerMethodForward());
				// solver.setPoissonMethod(PoissonMethodVCycle(3, 700, 0.4));
				solver.setPoissonMethod(PoissonMethodCG(100));
				solver.setBoundaryMethod(make_boundary_method_all(BoundaryOpClamp2<T, byte>{
					0, underlying(Smoke2dSolver::CellType::CellTypeWall),
					1.f, underlying(Smoke2dSolver::CellType::CellTypeSource)
				}));
				solver.setBoundary2Method(make_boundary_method_all(BoundaryOpClamp2<T2, byte>{
					make_T2(0.f, 0.f), underlying(Smoke2dSolver::CellType::CellTypeWall),
					make_T2(0.f, 0.f), underlying(Smoke2dSolver::CellType::CellTypeSource)
				}));
				solver.setForceMethod(ForceMethodSimple(0.0015f, 0.125f, 0.f));

				solver.init();
				solver.genNoise();
				//solver.addSource(params->nx()/2-2, params->nx()/2+1, 0, 2);
			}
			catch (error_t e)
			{
				result->set_status(underlying(e));
				return grpc::Status::CANCELLED;
			}

			return grpc::Status::OK;
		}
		grpc::Status Step(grpc::ServerContext* context,
			const Smoke2dStepParams* params,
			Result* result) override
		{
			result->set_status(0);
			try
			{
				solver.step();
			}
			catch (error_t e)
			{
				result->set_status(underlying(e));
				return grpc::Status::CANCELLED;
			}

			return grpc::Status::OK;
		}
		grpc::Status Reset(grpc::ServerContext* context,
			const Smoke2dResetParams* params,
			Result* result) override
		{
			result->set_status(0);
			try
			{
				solver.genNoise();
			}
			catch (error_t e)
			{
				result->set_status(underlying(e));
				return grpc::Status::CANCELLED;
			}

			return grpc::Status::OK;
		}
		grpc::Status Destroy(grpc::ServerContext* context,
			const Smoke2dDestroyParams* params,
			Result* result) override
		{
			result->set_status(0);
			try
			{
				solver.destory();
			}
			catch (error_t e)
			{
				result->set_status(underlying(e));
				return grpc::Status::CANCELLED;
			}

			return grpc::Status::OK;
		}
		grpc::Status GetData(grpc::ServerContext* context,
			const Smoke2dGetDataParams* params,
			grpc::ServerWriter<DataChunk>* writer) override
		{
			DataChunk chunk;
			size_t size;
			byte *data = nullptr;
			try
			{
				data = static_cast<byte *>(solver.getData(&size));
			}
			catch (error_t e)
			{
				return grpc::Status::CANCELLED;
			}

			while (size >= DATA_CHUNK_SIZE)
			{
				chunk.set_data(data, DATA_CHUNK_SIZE);
				writer->Write(chunk);
				data += DATA_CHUNK_SIZE;
				size -= DATA_CHUNK_SIZE;
			}
			if (size > 0)
			{
				chunk.set_data(data, size);
				writer->Write(chunk);
			}

			return grpc::Status::OK;
		}
	private:
		Smoke2dSolver solver;
	};
}

void RunServer() {
	std::string server_address("0.0.0.0:50077");
	Smoke2dService smoke2d_service;

	grpc::ServerBuilder builder;
	builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
	builder.RegisterService(&smoke2d_service);
	std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
	std::cout << "Server listening on " << server_address << std::endl;

	server->Wait();
}

/// =================================================

void Local(uint nx, uint ny) {
	Smoke2dSolver solver;

	solver.setSize(nx, ny);
	solver.setAdvectMethod(AdvectMethodSemiLagrangian());
	solver.setEulerMethod(EulerMethodForward());
	// solver.setPoissonMethod(PoissonMethodVCycle(3, 700, 0.4));
	solver.setPoissonMethod(PoissonMethodCG(100));
	solver.setBoundaryMethod(make_boundary_method_all(BoundaryOpClamp2<T, byte>{
		0, underlying(Smoke2dSolver::CellType::CellTypeWall),
			1.f, underlying(Smoke2dSolver::CellType::CellTypeSource)
	}));
	solver.setBoundary2Method(make_boundary_method_all(BoundaryOpClamp2<T2, byte>{
		make_T2(0.f, 0.f), underlying(Smoke2dSolver::CellType::CellTypeWall),
			make_T2(0.f, 0.f), underlying(Smoke2dSolver::CellType::CellTypeSource)
	}));
	solver.setForceMethod(ForceMethodSimple(0.0015f, 0.125f, 0.f));

	solver.init();
	solver.genNoise();

	int frame = 0;
	while (frame < 50000)
	{
		std::cout << frame << std::endl;
		solver.saveData("data/" + std::to_string(frame));
		frame++;
		if (frame % 100 == 0) solver.genNoise();
		else solver.step();
	}
	solver.destory();
}

int main()
{
	thread ts = thread(RunServer);

	//Smoke2dClient client(grpc::CreateChannel(
	//	"localhost:50077", grpc::InsecureChannelCredentials()));
	//client.init(16, 16);
	//client.step();
	//T *data_client = new T[16*16];
	//size_t size = client.getData(data_client);
	//std::cout << "Client received data: " << size << std::endl;
	//output::PrintRawCPU(data_client, 16 * 16, "data_client");

	ts.join();
	
	//Local(64, 64);

	return 0;
}
