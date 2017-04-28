
#include <iostream>
#include <algorithm>
#include <thread>
using namespace std;

#include <grpc++/grpc++.h>
#include "Smoke2d.grpc.pb.h"

#include "Smoke2dSolver.h"
using namespace ssv;

/// ================== CLIENT ======================

class Client
{
public:
	Client(std::shared_ptr<grpc::Channel> channel)
		: stub_(Smoke2d::NewStub(channel)) {}

	unsigned int Init(unsigned int nx, unsigned int ny)
	{
		Smoke2dInitParams params;
		params.set_nx(0);
		params.set_ny(0);

		Result result;

		grpc::ClientContext context;
		grpc::Status status = stub_->Init(&context, params, &result);

		if (status.ok()) {
			return result.status();
		}
		else {
			PrintGrpcError(status);
			return 0xffffffff;
		}
	}

private:
	void PrintGrpcError(const grpc::Status &status) const
	{
		std::cout << status.error_code() << ": " << status.error_message()
			<< std::endl;
	}
private:
	std::unique_ptr<Smoke2d::Stub> stub_;
};

/// ================== SERVER ======================
class Smoke2dImpl final : public Smoke2d::Service
{
	grpc::Status Init(grpc::ServerContext* context, 
		const Smoke2dInitParams* params,
		Result* result) override
	{
		result->set_status(0);
		return grpc::Status::OK;
	}
};

void RunServer() {
	std::string server_address("0.0.0.0:50051");
	Smoke2dImpl service;

	grpc::ServerBuilder builder;
	// Listen on the given address without any authentication mechanism.
	builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
	// Register "service" as the instance through which we'll communicate with
	// clients. In this case it corresponds to an *synchronous* service.
	builder.RegisterService(&service);
	// Finally assemble the server.
	std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
	std::cout << "Server listening on " << server_address << std::endl;

	// Wait for the server to shutdown. Note that some other thread must be
	// responsible for shutting down the server for this call to ever return.
	server->Wait();
}

/// =================================================

int main()
{
	//thread ts = thread(RunServer);

	//Client client(grpc::CreateChannel(
	//	"localhost:50051", grpc::InsecureChannelCredentials()));

	//unsigned int status = client.Init(0, 0);
	//std::cout << "STATUS: " << status << std::endl;

	//ts.join();

	Smoke2dSolver solver;
	solver.setSize(16, 16);
	solver.setAdvectMethod(AdvectMethodSemiLagrangian());
	solver.setEulerMethod(EulerMethodForward());
	solver.setPoissonMethod(PoissonMethodVCycle(3, 7));
	solver.setBoundaryMethod(make_boundary_method_all(BoundaryOpClamp2<T, byte>{
		0,		underlying(Smoke2dSolver::CellType::CellTypeWall), 
		1.f,	underlying(Smoke2dSolver::CellType::CellTypeSource)
	}));
	solver.setBoundary2Method(make_boundary_method_all(BoundaryOpClamp2<T2, byte>{
		make_float2(0.f, 0.f), underlying(Smoke2dSolver::CellType::CellTypeWall),
		make_float2(0.f, 0.5f), underlying(Smoke2dSolver::CellType::CellTypeSource)
	}));
	solver.setForceMethod(ForceMethodSimple(0.003f, 0.25f, 0.f));

	solver.init();
	solver.addSource(6, 9, 0, 2);

	while(1)
	solver.step();


	return 0;
}
