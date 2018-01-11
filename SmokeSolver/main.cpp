#include <iostream>
#include <thread>
#include <chrono>
#include <iomanip>
#include <ctime>
using namespace std;

#include <grpc++/grpc++.h>
#include "Smoke2d.grpc.pb.h"

#include "Smoke2dSolver.h"
using namespace ssv;

namespace
{
	const size_t DATA_CHUNK_SIZE = 65536u;

	/// ================== CLIENT ======================

	class Smoke2DClient
	{
	public:
		explicit Smoke2DClient(const std::shared_ptr<grpc::Channel> &channel)
			: stub_(Smoke2d::NewStub(channel))
		{
		}

		void init(uint nx, uint ny)
		{
			Smoke2dInitParams params;
			params.set_nx(nx);
			params.set_ny(ny);
			Result result;

			grpc::ClientContext context;
			stub_->Init(&context, params, &result);

			_PrintGrpcError("init", result.status());
		}

		void step()
		{
			Smoke2dStepParams params;
			Result result;

			grpc::ClientContext context;
			grpc::Status status = stub_->Step(&context, params, &result);

			_PrintGrpcError("step", result.status());
		}

		void destroy()
		{
			Smoke2dDestroyParams params;
			Result result;

			grpc::ClientContext context;
			grpc::Status status = stub_->Destroy(&context, params, &result);

			_PrintGrpcError("destroy", result.status());
		}

		size_t get_data(void *data_buffer) const
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
		static void _PrintGrpcError(const std::string &func_name, uint status)
		{
			std::cout << "Function " << func_name << " returned with status " << status << std::endl;
		}

	private:
		std::unique_ptr<Smoke2d::Stub> stub_;
	};

	/// ================== SERVER ======================
	class Smoke2DService final : public Smoke2d::Service
	{
	public:
		static void PrintTime()
		{
			std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
			std::time_t now_c = std::chrono::system_clock::to_time_t(now);
			std::cout << "[" << std::put_time(std::localtime(&now_c), "%F %T") << "] ";
		}

		grpc::Status Init(grpc::ServerContext *context,
		                  const Smoke2dInitParams *params,
		                  Result *result) override
		{
			PrintTime();
			std::cout << "Connected: " << context->peer() << std::endl;
			result->set_status(0);
			try
			{
				solver_.set_size(params->nx(), params->ny());
				solver_.set_advect(AdvectMethodSemiLagrangian());
				solver_.set_euler(EulerMethodForward());
				// solver.setPoissonMethod(PoissonMethodVCycle(3, 700, 0.4));
				solver_.set_poisson(PoissonMethodCG(100));
				solver_.set_boundary_density(make_boundary_all(make_boundary_op_clamp(
					Smoke2DSolver::CellType::CELL_TYPE_WALL, 0.f
				)));
				solver_.set_boundary_density(make_boundary_all(make_boundary_op_clamp2(
					Smoke2DSolver::CellType::CELL_TYPE_WALL, 0.f, 
					Smoke2DSolver::CellType::CELL_TYPE_SOURCE, 1.f
				)));
				solver_.set_boundary_velocity(make_boundary_all(make_boundary_op_clamp2(
					Smoke2DSolver::CellType::CELL_TYPE_WALL, make_real2(0.f, 0.f), 
					Smoke2DSolver::CellType::CELL_TYPE_SOURCE, make_real2(0.f, 0.f)
				)));
				solver_.set_force(ForceMethodSimple(0.0015f, 0.125f, 0.f));

				solver_.init();
				solver_.gen_noise();
				//solver.add_source(params->nx()/2-2, params->nx()/2+1, 0, 2);
			}
			catch (ssv_error &e)
			{
				result->set_status(underlying(e.err));
				return grpc::Status::CANCELLED;
			}

			return grpc::Status::OK;
		}

		grpc::Status Step(grpc::ServerContext *context,
		                  const Smoke2dStepParams *params,
		                  Result *result) override
		{
			result->set_status(0);
			try
			{
				solver_.step();
			}
			catch (ssv_error &e)
			{
				result->set_status(underlying(e.err));
				return grpc::Status::CANCELLED;
			}

			return grpc::Status::OK;
		}

		grpc::Status Reset(grpc::ServerContext *context,
		                   const Smoke2dResetParams *params,
		                   Result *result) override
		{
			result->set_status(0);
			try
			{
				solver_.gen_noise();
			}
			catch (ssv_error &e)
			{
				result->set_status(underlying(e.err));
				return grpc::Status::CANCELLED;
			}

			return grpc::Status::OK;
		}

		grpc::Status Destroy(grpc::ServerContext *context,
		                     const Smoke2dDestroyParams *params,
		                     Result *result) override
		{
			result->set_status(0);
			try
			{
				solver_.destory();
			}
			catch (ssv_error &e)
			{
				result->set_status(underlying(e.err));
				return grpc::Status::CANCELLED;
			}

			return grpc::Status::OK;
		}

		grpc::Status GetData(grpc::ServerContext *context,
		                     const Smoke2dGetDataParams *params,
		                     grpc::ServerWriter<DataChunk> *writer) override
		{
			DataChunk chunk;
			size_t size;
			byte *data = nullptr;
			try
			{
				data = static_cast<byte *>(solver_.get_data(&size));
			}
			catch (ssv_error &e)
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
		Smoke2DSolver solver_;
	};
}

void run_server(int port)
{
	std::string server_address("0.0.0.0:" + std::to_string(port));
	Smoke2DService service;

	grpc::ServerBuilder builder;
	builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
	builder.RegisterService(&service);
	std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
	std::cout << "Server listening on " << server_address << std::endl;

	server->Wait();
}

/// =================================================

void local(uint nx, uint ny)
{
	Smoke2DSolver solver;

	solver.set_size(nx, ny);
	solver.set_advect(AdvectMethodSemiLagrangian());
	solver.set_euler(EulerMethodForward());
	// solver.setPoissonMethod(PoissonMethodVCycle(3, 700, 0.4));
	solver.set_poisson(PoissonMethodCG(100));
	solver.set_boundary_density(make_boundary_all(make_boundary_op_clamp2(
		Smoke2DSolver::CellType::CELL_TYPE_WALL, 0.f, 
		Smoke2DSolver::CellType::CELL_TYPE_SOURCE, 1.f
	)));
	solver.set_boundary_velocity(make_boundary_all(make_boundary_op_clamp2(
		Smoke2DSolver::CellType::CELL_TYPE_WALL, make_real2(0.f, 0.f), 
		Smoke2DSolver::CellType::CELL_TYPE_SOURCE, make_real2(0.f, 0.f)
	)));
	solver.set_force(ForceMethodSimple(0.0015f, 0.125f, 0.f));

	solver.init();
	solver.gen_noise();

	int frame = 0;
	while (frame < 50000)
	{
		std::cout << frame << std::endl;
		solver.save_data("data/" + std::to_string(frame));
		frame++;
		if (frame % 100 == 0) solver.gen_noise();
		else solver.step();
	}
	solver.destory();
}

int main(int argc, const char **argv)
{
	int port = 50077;
	if (argc > 1)
	{
		port = std::stoi(argv[1]);
	}
	thread ts;
	try
	{
		ts = thread(run_server, port);
	}
	catch (...)
	{
		std::cerr << "Cannot start server!" << endl;
	}

	//Smoke2dClient client(grpc::CreateChannel(
	//	"localhost:50077", grpc::InsecureChannelCredentials()));
	//client.init(16, 16);
	//client.step();
	//real *data_client = new real[16*16];
	//size_t size = client.getData(data_client);
	//std::cout << "Client received data: " << size << std::endl;
	//output::PrintRawCPU(data_client, 16 * 16, "data_client");

	ts.join();

	//local(64, 64);

	return 0;
}
