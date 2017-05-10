// Generated by the gRPC C++ plugin.
// If you make any local change, they will be lost.
// source: Smoke2d.proto
#ifndef GRPC_Smoke2d_2eproto__INCLUDED
#define GRPC_Smoke2d_2eproto__INCLUDED

#include "Smoke2d.pb.h"

#include <grpc++/impl/codegen/async_stream.h>
#include <grpc++/impl/codegen/async_unary_call.h>
#include <grpc++/impl/codegen/method_handler_impl.h>
#include <grpc++/impl/codegen/proto_utils.h>
#include <grpc++/impl/codegen/rpc_method.h>
#include <grpc++/impl/codegen/service_type.h>
#include <grpc++/impl/codegen/status.h>
#include <grpc++/impl/codegen/stub_options.h>
#include <grpc++/impl/codegen/sync_stream.h>

namespace grpc {
class CompletionQueue;
class Channel;
class RpcService;
class ServerCompletionQueue;
class ServerContext;
}  // namespace grpc

namespace ssv {

class Smoke2d final {
 public:
  class StubInterface {
   public:
    virtual ~StubInterface() {}
    virtual ::grpc::Status Init(::grpc::ClientContext* context, const ::ssv::Smoke2dInitParams& request, ::ssv::Result* response) = 0;
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::ssv::Result>> AsyncInit(::grpc::ClientContext* context, const ::ssv::Smoke2dInitParams& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::ssv::Result>>(AsyncInitRaw(context, request, cq));
    }
    virtual ::grpc::Status Step(::grpc::ClientContext* context, const ::ssv::Smoke2dStepParams& request, ::ssv::Result* response) = 0;
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::ssv::Result>> AsyncStep(::grpc::ClientContext* context, const ::ssv::Smoke2dStepParams& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::ssv::Result>>(AsyncStepRaw(context, request, cq));
    }
    virtual ::grpc::Status Destroy(::grpc::ClientContext* context, const ::ssv::Smoke2dDestroyParams& request, ::ssv::Result* response) = 0;
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::ssv::Result>> AsyncDestroy(::grpc::ClientContext* context, const ::ssv::Smoke2dDestroyParams& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::ssv::Result>>(AsyncDestroyRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientReaderInterface< ::ssv::DataChunk>> GetData(::grpc::ClientContext* context, const ::ssv::Smoke2dGetDataParams& request) {
      return std::unique_ptr< ::grpc::ClientReaderInterface< ::ssv::DataChunk>>(GetDataRaw(context, request));
    }
    std::unique_ptr< ::grpc::ClientAsyncReaderInterface< ::ssv::DataChunk>> AsyncGetData(::grpc::ClientContext* context, const ::ssv::Smoke2dGetDataParams& request, ::grpc::CompletionQueue* cq, void* tag) {
      return std::unique_ptr< ::grpc::ClientAsyncReaderInterface< ::ssv::DataChunk>>(AsyncGetDataRaw(context, request, cq, tag));
    }
  private:
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::ssv::Result>* AsyncInitRaw(::grpc::ClientContext* context, const ::ssv::Smoke2dInitParams& request, ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::ssv::Result>* AsyncStepRaw(::grpc::ClientContext* context, const ::ssv::Smoke2dStepParams& request, ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::ssv::Result>* AsyncDestroyRaw(::grpc::ClientContext* context, const ::ssv::Smoke2dDestroyParams& request, ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientReaderInterface< ::ssv::DataChunk>* GetDataRaw(::grpc::ClientContext* context, const ::ssv::Smoke2dGetDataParams& request) = 0;
    virtual ::grpc::ClientAsyncReaderInterface< ::ssv::DataChunk>* AsyncGetDataRaw(::grpc::ClientContext* context, const ::ssv::Smoke2dGetDataParams& request, ::grpc::CompletionQueue* cq, void* tag) = 0;
  };
  class Stub final : public StubInterface {
   public:
    Stub(const std::shared_ptr< ::grpc::ChannelInterface>& channel);
    ::grpc::Status Init(::grpc::ClientContext* context, const ::ssv::Smoke2dInitParams& request, ::ssv::Result* response) override;
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::ssv::Result>> AsyncInit(::grpc::ClientContext* context, const ::ssv::Smoke2dInitParams& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::ssv::Result>>(AsyncInitRaw(context, request, cq));
    }
    ::grpc::Status Step(::grpc::ClientContext* context, const ::ssv::Smoke2dStepParams& request, ::ssv::Result* response) override;
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::ssv::Result>> AsyncStep(::grpc::ClientContext* context, const ::ssv::Smoke2dStepParams& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::ssv::Result>>(AsyncStepRaw(context, request, cq));
    }
    ::grpc::Status Destroy(::grpc::ClientContext* context, const ::ssv::Smoke2dDestroyParams& request, ::ssv::Result* response) override;
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::ssv::Result>> AsyncDestroy(::grpc::ClientContext* context, const ::ssv::Smoke2dDestroyParams& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::ssv::Result>>(AsyncDestroyRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientReader< ::ssv::DataChunk>> GetData(::grpc::ClientContext* context, const ::ssv::Smoke2dGetDataParams& request) {
      return std::unique_ptr< ::grpc::ClientReader< ::ssv::DataChunk>>(GetDataRaw(context, request));
    }
    std::unique_ptr< ::grpc::ClientAsyncReader< ::ssv::DataChunk>> AsyncGetData(::grpc::ClientContext* context, const ::ssv::Smoke2dGetDataParams& request, ::grpc::CompletionQueue* cq, void* tag) {
      return std::unique_ptr< ::grpc::ClientAsyncReader< ::ssv::DataChunk>>(AsyncGetDataRaw(context, request, cq, tag));
    }

   private:
    std::shared_ptr< ::grpc::ChannelInterface> channel_;
    ::grpc::ClientAsyncResponseReader< ::ssv::Result>* AsyncInitRaw(::grpc::ClientContext* context, const ::ssv::Smoke2dInitParams& request, ::grpc::CompletionQueue* cq) override;
    ::grpc::ClientAsyncResponseReader< ::ssv::Result>* AsyncStepRaw(::grpc::ClientContext* context, const ::ssv::Smoke2dStepParams& request, ::grpc::CompletionQueue* cq) override;
    ::grpc::ClientAsyncResponseReader< ::ssv::Result>* AsyncDestroyRaw(::grpc::ClientContext* context, const ::ssv::Smoke2dDestroyParams& request, ::grpc::CompletionQueue* cq) override;
    ::grpc::ClientReader< ::ssv::DataChunk>* GetDataRaw(::grpc::ClientContext* context, const ::ssv::Smoke2dGetDataParams& request) override;
    ::grpc::ClientAsyncReader< ::ssv::DataChunk>* AsyncGetDataRaw(::grpc::ClientContext* context, const ::ssv::Smoke2dGetDataParams& request, ::grpc::CompletionQueue* cq, void* tag) override;
    const ::grpc::RpcMethod rpcmethod_Init_;
    const ::grpc::RpcMethod rpcmethod_Step_;
    const ::grpc::RpcMethod rpcmethod_Destroy_;
    const ::grpc::RpcMethod rpcmethod_GetData_;
  };
  static std::unique_ptr<Stub> NewStub(const std::shared_ptr< ::grpc::ChannelInterface>& channel, const ::grpc::StubOptions& options = ::grpc::StubOptions());

  class Service : public ::grpc::Service {
   public:
    Service();
    virtual ~Service();
    virtual ::grpc::Status Init(::grpc::ServerContext* context, const ::ssv::Smoke2dInitParams* request, ::ssv::Result* response);
    virtual ::grpc::Status Step(::grpc::ServerContext* context, const ::ssv::Smoke2dStepParams* request, ::ssv::Result* response);
    virtual ::grpc::Status Destroy(::grpc::ServerContext* context, const ::ssv::Smoke2dDestroyParams* request, ::ssv::Result* response);
    virtual ::grpc::Status GetData(::grpc::ServerContext* context, const ::ssv::Smoke2dGetDataParams* request, ::grpc::ServerWriter< ::ssv::DataChunk>* writer);
  };
  template <class BaseClass>
  class WithAsyncMethod_Init : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithAsyncMethod_Init() {
      ::grpc::Service::MarkMethodAsync(0);
    }
    ~WithAsyncMethod_Init() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status Init(::grpc::ServerContext* context, const ::ssv::Smoke2dInitParams* request, ::ssv::Result* response) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestInit(::grpc::ServerContext* context, ::ssv::Smoke2dInitParams* request, ::grpc::ServerAsyncResponseWriter< ::ssv::Result>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(0, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  template <class BaseClass>
  class WithAsyncMethod_Step : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithAsyncMethod_Step() {
      ::grpc::Service::MarkMethodAsync(1);
    }
    ~WithAsyncMethod_Step() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status Step(::grpc::ServerContext* context, const ::ssv::Smoke2dStepParams* request, ::ssv::Result* response) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestStep(::grpc::ServerContext* context, ::ssv::Smoke2dStepParams* request, ::grpc::ServerAsyncResponseWriter< ::ssv::Result>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(1, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  template <class BaseClass>
  class WithAsyncMethod_Destroy : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithAsyncMethod_Destroy() {
      ::grpc::Service::MarkMethodAsync(2);
    }
    ~WithAsyncMethod_Destroy() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status Destroy(::grpc::ServerContext* context, const ::ssv::Smoke2dDestroyParams* request, ::ssv::Result* response) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestDestroy(::grpc::ServerContext* context, ::ssv::Smoke2dDestroyParams* request, ::grpc::ServerAsyncResponseWriter< ::ssv::Result>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(2, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  template <class BaseClass>
  class WithAsyncMethod_GetData : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithAsyncMethod_GetData() {
      ::grpc::Service::MarkMethodAsync(3);
    }
    ~WithAsyncMethod_GetData() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status GetData(::grpc::ServerContext* context, const ::ssv::Smoke2dGetDataParams* request, ::grpc::ServerWriter< ::ssv::DataChunk>* writer) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestGetData(::grpc::ServerContext* context, ::ssv::Smoke2dGetDataParams* request, ::grpc::ServerAsyncWriter< ::ssv::DataChunk>* writer, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncServerStreaming(3, context, request, writer, new_call_cq, notification_cq, tag);
    }
  };
  typedef WithAsyncMethod_Init<WithAsyncMethod_Step<WithAsyncMethod_Destroy<WithAsyncMethod_GetData<Service > > > > AsyncService;
  template <class BaseClass>
  class WithGenericMethod_Init : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithGenericMethod_Init() {
      ::grpc::Service::MarkMethodGeneric(0);
    }
    ~WithGenericMethod_Init() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status Init(::grpc::ServerContext* context, const ::ssv::Smoke2dInitParams* request, ::ssv::Result* response) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
  };
  template <class BaseClass>
  class WithGenericMethod_Step : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithGenericMethod_Step() {
      ::grpc::Service::MarkMethodGeneric(1);
    }
    ~WithGenericMethod_Step() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status Step(::grpc::ServerContext* context, const ::ssv::Smoke2dStepParams* request, ::ssv::Result* response) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
  };
  template <class BaseClass>
  class WithGenericMethod_Destroy : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithGenericMethod_Destroy() {
      ::grpc::Service::MarkMethodGeneric(2);
    }
    ~WithGenericMethod_Destroy() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status Destroy(::grpc::ServerContext* context, const ::ssv::Smoke2dDestroyParams* request, ::ssv::Result* response) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
  };
  template <class BaseClass>
  class WithGenericMethod_GetData : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithGenericMethod_GetData() {
      ::grpc::Service::MarkMethodGeneric(3);
    }
    ~WithGenericMethod_GetData() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status GetData(::grpc::ServerContext* context, const ::ssv::Smoke2dGetDataParams* request, ::grpc::ServerWriter< ::ssv::DataChunk>* writer) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
  };
  template <class BaseClass>
  class WithStreamedUnaryMethod_Init : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithStreamedUnaryMethod_Init() {
      ::grpc::Service::MarkMethodStreamed(0,
        new ::grpc::StreamedUnaryHandler< ::ssv::Smoke2dInitParams, ::ssv::Result>(std::bind(&WithStreamedUnaryMethod_Init<BaseClass>::StreamedInit, this, std::placeholders::_1, std::placeholders::_2)));
    }
    ~WithStreamedUnaryMethod_Init() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable regular version of this method
    ::grpc::Status Init(::grpc::ServerContext* context, const ::ssv::Smoke2dInitParams* request, ::ssv::Result* response) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    // replace default version of method with streamed unary
    virtual ::grpc::Status StreamedInit(::grpc::ServerContext* context, ::grpc::ServerUnaryStreamer< ::ssv::Smoke2dInitParams,::ssv::Result>* server_unary_streamer) = 0;
  };
  template <class BaseClass>
  class WithStreamedUnaryMethod_Step : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithStreamedUnaryMethod_Step() {
      ::grpc::Service::MarkMethodStreamed(1,
        new ::grpc::StreamedUnaryHandler< ::ssv::Smoke2dStepParams, ::ssv::Result>(std::bind(&WithStreamedUnaryMethod_Step<BaseClass>::StreamedStep, this, std::placeholders::_1, std::placeholders::_2)));
    }
    ~WithStreamedUnaryMethod_Step() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable regular version of this method
    ::grpc::Status Step(::grpc::ServerContext* context, const ::ssv::Smoke2dStepParams* request, ::ssv::Result* response) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    // replace default version of method with streamed unary
    virtual ::grpc::Status StreamedStep(::grpc::ServerContext* context, ::grpc::ServerUnaryStreamer< ::ssv::Smoke2dStepParams,::ssv::Result>* server_unary_streamer) = 0;
  };
  template <class BaseClass>
  class WithStreamedUnaryMethod_Destroy : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithStreamedUnaryMethod_Destroy() {
      ::grpc::Service::MarkMethodStreamed(2,
        new ::grpc::StreamedUnaryHandler< ::ssv::Smoke2dDestroyParams, ::ssv::Result>(std::bind(&WithStreamedUnaryMethod_Destroy<BaseClass>::StreamedDestroy, this, std::placeholders::_1, std::placeholders::_2)));
    }
    ~WithStreamedUnaryMethod_Destroy() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable regular version of this method
    ::grpc::Status Destroy(::grpc::ServerContext* context, const ::ssv::Smoke2dDestroyParams* request, ::ssv::Result* response) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    // replace default version of method with streamed unary
    virtual ::grpc::Status StreamedDestroy(::grpc::ServerContext* context, ::grpc::ServerUnaryStreamer< ::ssv::Smoke2dDestroyParams,::ssv::Result>* server_unary_streamer) = 0;
  };
  typedef WithStreamedUnaryMethod_Init<WithStreamedUnaryMethod_Step<WithStreamedUnaryMethod_Destroy<Service > > > StreamedUnaryService;
  template <class BaseClass>
  class WithSplitStreamingMethod_GetData : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithSplitStreamingMethod_GetData() {
      ::grpc::Service::MarkMethodStreamed(3,
        new ::grpc::SplitServerStreamingHandler< ::ssv::Smoke2dGetDataParams, ::ssv::DataChunk>(std::bind(&WithSplitStreamingMethod_GetData<BaseClass>::StreamedGetData, this, std::placeholders::_1, std::placeholders::_2)));
    }
    ~WithSplitStreamingMethod_GetData() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable regular version of this method
    ::grpc::Status GetData(::grpc::ServerContext* context, const ::ssv::Smoke2dGetDataParams* request, ::grpc::ServerWriter< ::ssv::DataChunk>* writer) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    // replace default version of method with split streamed
    virtual ::grpc::Status StreamedGetData(::grpc::ServerContext* context, ::grpc::ServerSplitStreamer< ::ssv::Smoke2dGetDataParams,::ssv::DataChunk>* server_split_streamer) = 0;
  };
  typedef WithSplitStreamingMethod_GetData<Service > SplitStreamedService;
  typedef WithStreamedUnaryMethod_Init<WithStreamedUnaryMethod_Step<WithStreamedUnaryMethod_Destroy<WithSplitStreamingMethod_GetData<Service > > > > StreamedService;
};

}  // namespace ssv


#endif  // GRPC_Smoke2d_2eproto__INCLUDED
