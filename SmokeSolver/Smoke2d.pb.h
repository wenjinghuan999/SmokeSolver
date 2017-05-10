// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: Smoke2d.proto

#ifndef PROTOBUF_Smoke2d_2eproto__INCLUDED
#define PROTOBUF_Smoke2d_2eproto__INCLUDED

#include <string>

#include <google/protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 3002000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 3002000 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/metadata.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)
namespace ssv {
class DataChunk;
class DataChunkDefaultTypeInternal;
extern DataChunkDefaultTypeInternal _DataChunk_default_instance_;
class Result;
class ResultDefaultTypeInternal;
extern ResultDefaultTypeInternal _Result_default_instance_;
class Smoke2dDestroyParams;
class Smoke2dDestroyParamsDefaultTypeInternal;
extern Smoke2dDestroyParamsDefaultTypeInternal _Smoke2dDestroyParams_default_instance_;
class Smoke2dGetDataParams;
class Smoke2dGetDataParamsDefaultTypeInternal;
extern Smoke2dGetDataParamsDefaultTypeInternal _Smoke2dGetDataParams_default_instance_;
class Smoke2dInitParams;
class Smoke2dInitParamsDefaultTypeInternal;
extern Smoke2dInitParamsDefaultTypeInternal _Smoke2dInitParams_default_instance_;
class Smoke2dStepParams;
class Smoke2dStepParamsDefaultTypeInternal;
extern Smoke2dStepParamsDefaultTypeInternal _Smoke2dStepParams_default_instance_;
}  // namespace ssv

namespace ssv {

namespace protobuf_Smoke2d_2eproto {
// Internal implementation detail -- do not call these.
struct TableStruct {
  static const ::google::protobuf::uint32 offsets[];
  static void InitDefaultsImpl();
  static void Shutdown();
};
void AddDescriptors();
void InitDefaults();
}  // namespace protobuf_Smoke2d_2eproto

// ===================================================================

class Smoke2dInitParams : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:ssv.Smoke2dInitParams) */ {
 public:
  Smoke2dInitParams();
  virtual ~Smoke2dInitParams();

  Smoke2dInitParams(const Smoke2dInitParams& from);

  inline Smoke2dInitParams& operator=(const Smoke2dInitParams& from) {
    CopyFrom(from);
    return *this;
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const Smoke2dInitParams& default_instance();

  static inline const Smoke2dInitParams* internal_default_instance() {
    return reinterpret_cast<const Smoke2dInitParams*>(
               &_Smoke2dInitParams_default_instance_);
  }

  void Swap(Smoke2dInitParams* other);

  // implements Message ----------------------------------------------

  inline Smoke2dInitParams* New() const PROTOBUF_FINAL { return New(NULL); }

  Smoke2dInitParams* New(::google::protobuf::Arena* arena) const PROTOBUF_FINAL;
  void CopyFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void MergeFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void CopyFrom(const Smoke2dInitParams& from);
  void MergeFrom(const Smoke2dInitParams& from);
  void Clear() PROTOBUF_FINAL;
  bool IsInitialized() const PROTOBUF_FINAL;

  size_t ByteSizeLong() const PROTOBUF_FINAL;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) PROTOBUF_FINAL;
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const PROTOBUF_FINAL;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* target) const PROTOBUF_FINAL;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output)
      const PROTOBUF_FINAL {
    return InternalSerializeWithCachedSizesToArray(
        ::google::protobuf::io::CodedOutputStream::IsDefaultSerializationDeterministic(), output);
  }
  int GetCachedSize() const PROTOBUF_FINAL { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const PROTOBUF_FINAL;
  void InternalSwap(Smoke2dInitParams* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return NULL;
  }
  inline void* MaybeArenaPtr() const {
    return NULL;
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const PROTOBUF_FINAL;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // uint32 nx = 1;
  void clear_nx();
  static const int kNxFieldNumber = 1;
  ::google::protobuf::uint32 nx() const;
  void set_nx(::google::protobuf::uint32 value);

  // uint32 ny = 2;
  void clear_ny();
  static const int kNyFieldNumber = 2;
  ::google::protobuf::uint32 ny() const;
  void set_ny(::google::protobuf::uint32 value);

  // @@protoc_insertion_point(class_scope:ssv.Smoke2dInitParams)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::uint32 nx_;
  ::google::protobuf::uint32 ny_;
  mutable int _cached_size_;
  friend struct  protobuf_Smoke2d_2eproto::TableStruct;
};
// -------------------------------------------------------------------

class Smoke2dStepParams : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:ssv.Smoke2dStepParams) */ {
 public:
  Smoke2dStepParams();
  virtual ~Smoke2dStepParams();

  Smoke2dStepParams(const Smoke2dStepParams& from);

  inline Smoke2dStepParams& operator=(const Smoke2dStepParams& from) {
    CopyFrom(from);
    return *this;
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const Smoke2dStepParams& default_instance();

  static inline const Smoke2dStepParams* internal_default_instance() {
    return reinterpret_cast<const Smoke2dStepParams*>(
               &_Smoke2dStepParams_default_instance_);
  }

  void Swap(Smoke2dStepParams* other);

  // implements Message ----------------------------------------------

  inline Smoke2dStepParams* New() const PROTOBUF_FINAL { return New(NULL); }

  Smoke2dStepParams* New(::google::protobuf::Arena* arena) const PROTOBUF_FINAL;
  void CopyFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void MergeFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void CopyFrom(const Smoke2dStepParams& from);
  void MergeFrom(const Smoke2dStepParams& from);
  void Clear() PROTOBUF_FINAL;
  bool IsInitialized() const PROTOBUF_FINAL;

  size_t ByteSizeLong() const PROTOBUF_FINAL;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) PROTOBUF_FINAL;
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const PROTOBUF_FINAL;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* target) const PROTOBUF_FINAL;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output)
      const PROTOBUF_FINAL {
    return InternalSerializeWithCachedSizesToArray(
        ::google::protobuf::io::CodedOutputStream::IsDefaultSerializationDeterministic(), output);
  }
  int GetCachedSize() const PROTOBUF_FINAL { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const PROTOBUF_FINAL;
  void InternalSwap(Smoke2dStepParams* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return NULL;
  }
  inline void* MaybeArenaPtr() const {
    return NULL;
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const PROTOBUF_FINAL;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // @@protoc_insertion_point(class_scope:ssv.Smoke2dStepParams)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  mutable int _cached_size_;
  friend struct  protobuf_Smoke2d_2eproto::TableStruct;
};
// -------------------------------------------------------------------

class Smoke2dDestroyParams : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:ssv.Smoke2dDestroyParams) */ {
 public:
  Smoke2dDestroyParams();
  virtual ~Smoke2dDestroyParams();

  Smoke2dDestroyParams(const Smoke2dDestroyParams& from);

  inline Smoke2dDestroyParams& operator=(const Smoke2dDestroyParams& from) {
    CopyFrom(from);
    return *this;
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const Smoke2dDestroyParams& default_instance();

  static inline const Smoke2dDestroyParams* internal_default_instance() {
    return reinterpret_cast<const Smoke2dDestroyParams*>(
               &_Smoke2dDestroyParams_default_instance_);
  }

  void Swap(Smoke2dDestroyParams* other);

  // implements Message ----------------------------------------------

  inline Smoke2dDestroyParams* New() const PROTOBUF_FINAL { return New(NULL); }

  Smoke2dDestroyParams* New(::google::protobuf::Arena* arena) const PROTOBUF_FINAL;
  void CopyFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void MergeFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void CopyFrom(const Smoke2dDestroyParams& from);
  void MergeFrom(const Smoke2dDestroyParams& from);
  void Clear() PROTOBUF_FINAL;
  bool IsInitialized() const PROTOBUF_FINAL;

  size_t ByteSizeLong() const PROTOBUF_FINAL;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) PROTOBUF_FINAL;
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const PROTOBUF_FINAL;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* target) const PROTOBUF_FINAL;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output)
      const PROTOBUF_FINAL {
    return InternalSerializeWithCachedSizesToArray(
        ::google::protobuf::io::CodedOutputStream::IsDefaultSerializationDeterministic(), output);
  }
  int GetCachedSize() const PROTOBUF_FINAL { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const PROTOBUF_FINAL;
  void InternalSwap(Smoke2dDestroyParams* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return NULL;
  }
  inline void* MaybeArenaPtr() const {
    return NULL;
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const PROTOBUF_FINAL;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // @@protoc_insertion_point(class_scope:ssv.Smoke2dDestroyParams)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  mutable int _cached_size_;
  friend struct  protobuf_Smoke2d_2eproto::TableStruct;
};
// -------------------------------------------------------------------

class Smoke2dGetDataParams : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:ssv.Smoke2dGetDataParams) */ {
 public:
  Smoke2dGetDataParams();
  virtual ~Smoke2dGetDataParams();

  Smoke2dGetDataParams(const Smoke2dGetDataParams& from);

  inline Smoke2dGetDataParams& operator=(const Smoke2dGetDataParams& from) {
    CopyFrom(from);
    return *this;
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const Smoke2dGetDataParams& default_instance();

  static inline const Smoke2dGetDataParams* internal_default_instance() {
    return reinterpret_cast<const Smoke2dGetDataParams*>(
               &_Smoke2dGetDataParams_default_instance_);
  }

  void Swap(Smoke2dGetDataParams* other);

  // implements Message ----------------------------------------------

  inline Smoke2dGetDataParams* New() const PROTOBUF_FINAL { return New(NULL); }

  Smoke2dGetDataParams* New(::google::protobuf::Arena* arena) const PROTOBUF_FINAL;
  void CopyFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void MergeFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void CopyFrom(const Smoke2dGetDataParams& from);
  void MergeFrom(const Smoke2dGetDataParams& from);
  void Clear() PROTOBUF_FINAL;
  bool IsInitialized() const PROTOBUF_FINAL;

  size_t ByteSizeLong() const PROTOBUF_FINAL;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) PROTOBUF_FINAL;
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const PROTOBUF_FINAL;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* target) const PROTOBUF_FINAL;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output)
      const PROTOBUF_FINAL {
    return InternalSerializeWithCachedSizesToArray(
        ::google::protobuf::io::CodedOutputStream::IsDefaultSerializationDeterministic(), output);
  }
  int GetCachedSize() const PROTOBUF_FINAL { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const PROTOBUF_FINAL;
  void InternalSwap(Smoke2dGetDataParams* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return NULL;
  }
  inline void* MaybeArenaPtr() const {
    return NULL;
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const PROTOBUF_FINAL;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // @@protoc_insertion_point(class_scope:ssv.Smoke2dGetDataParams)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  mutable int _cached_size_;
  friend struct  protobuf_Smoke2d_2eproto::TableStruct;
};
// -------------------------------------------------------------------

class Result : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:ssv.Result) */ {
 public:
  Result();
  virtual ~Result();

  Result(const Result& from);

  inline Result& operator=(const Result& from) {
    CopyFrom(from);
    return *this;
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const Result& default_instance();

  static inline const Result* internal_default_instance() {
    return reinterpret_cast<const Result*>(
               &_Result_default_instance_);
  }

  void Swap(Result* other);

  // implements Message ----------------------------------------------

  inline Result* New() const PROTOBUF_FINAL { return New(NULL); }

  Result* New(::google::protobuf::Arena* arena) const PROTOBUF_FINAL;
  void CopyFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void MergeFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void CopyFrom(const Result& from);
  void MergeFrom(const Result& from);
  void Clear() PROTOBUF_FINAL;
  bool IsInitialized() const PROTOBUF_FINAL;

  size_t ByteSizeLong() const PROTOBUF_FINAL;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) PROTOBUF_FINAL;
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const PROTOBUF_FINAL;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* target) const PROTOBUF_FINAL;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output)
      const PROTOBUF_FINAL {
    return InternalSerializeWithCachedSizesToArray(
        ::google::protobuf::io::CodedOutputStream::IsDefaultSerializationDeterministic(), output);
  }
  int GetCachedSize() const PROTOBUF_FINAL { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const PROTOBUF_FINAL;
  void InternalSwap(Result* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return NULL;
  }
  inline void* MaybeArenaPtr() const {
    return NULL;
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const PROTOBUF_FINAL;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // uint32 status = 1;
  void clear_status();
  static const int kStatusFieldNumber = 1;
  ::google::protobuf::uint32 status() const;
  void set_status(::google::protobuf::uint32 value);

  // @@protoc_insertion_point(class_scope:ssv.Result)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::uint32 status_;
  mutable int _cached_size_;
  friend struct  protobuf_Smoke2d_2eproto::TableStruct;
};
// -------------------------------------------------------------------

class DataChunk : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:ssv.DataChunk) */ {
 public:
  DataChunk();
  virtual ~DataChunk();

  DataChunk(const DataChunk& from);

  inline DataChunk& operator=(const DataChunk& from) {
    CopyFrom(from);
    return *this;
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const DataChunk& default_instance();

  static inline const DataChunk* internal_default_instance() {
    return reinterpret_cast<const DataChunk*>(
               &_DataChunk_default_instance_);
  }

  void Swap(DataChunk* other);

  // implements Message ----------------------------------------------

  inline DataChunk* New() const PROTOBUF_FINAL { return New(NULL); }

  DataChunk* New(::google::protobuf::Arena* arena) const PROTOBUF_FINAL;
  void CopyFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void MergeFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void CopyFrom(const DataChunk& from);
  void MergeFrom(const DataChunk& from);
  void Clear() PROTOBUF_FINAL;
  bool IsInitialized() const PROTOBUF_FINAL;

  size_t ByteSizeLong() const PROTOBUF_FINAL;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) PROTOBUF_FINAL;
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const PROTOBUF_FINAL;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* target) const PROTOBUF_FINAL;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output)
      const PROTOBUF_FINAL {
    return InternalSerializeWithCachedSizesToArray(
        ::google::protobuf::io::CodedOutputStream::IsDefaultSerializationDeterministic(), output);
  }
  int GetCachedSize() const PROTOBUF_FINAL { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const PROTOBUF_FINAL;
  void InternalSwap(DataChunk* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return NULL;
  }
  inline void* MaybeArenaPtr() const {
    return NULL;
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const PROTOBUF_FINAL;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // bytes data = 1;
  void clear_data();
  static const int kDataFieldNumber = 1;
  const ::std::string& data() const;
  void set_data(const ::std::string& value);
  #if LANG_CXX11
  void set_data(::std::string&& value);
  #endif
  void set_data(const char* value);
  void set_data(const void* value, size_t size);
  ::std::string* mutable_data();
  ::std::string* release_data();
  void set_allocated_data(::std::string* data);

  // @@protoc_insertion_point(class_scope:ssv.DataChunk)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::internal::ArenaStringPtr data_;
  mutable int _cached_size_;
  friend struct  protobuf_Smoke2d_2eproto::TableStruct;
};
// ===================================================================


// ===================================================================

#if !PROTOBUF_INLINE_NOT_IN_HEADERS
// Smoke2dInitParams

// uint32 nx = 1;
inline void Smoke2dInitParams::clear_nx() {
  nx_ = 0u;
}
inline ::google::protobuf::uint32 Smoke2dInitParams::nx() const {
  // @@protoc_insertion_point(field_get:ssv.Smoke2dInitParams.nx)
  return nx_;
}
inline void Smoke2dInitParams::set_nx(::google::protobuf::uint32 value) {
  
  nx_ = value;
  // @@protoc_insertion_point(field_set:ssv.Smoke2dInitParams.nx)
}

// uint32 ny = 2;
inline void Smoke2dInitParams::clear_ny() {
  ny_ = 0u;
}
inline ::google::protobuf::uint32 Smoke2dInitParams::ny() const {
  // @@protoc_insertion_point(field_get:ssv.Smoke2dInitParams.ny)
  return ny_;
}
inline void Smoke2dInitParams::set_ny(::google::protobuf::uint32 value) {
  
  ny_ = value;
  // @@protoc_insertion_point(field_set:ssv.Smoke2dInitParams.ny)
}

// -------------------------------------------------------------------

// Smoke2dStepParams

// -------------------------------------------------------------------

// Smoke2dDestroyParams

// -------------------------------------------------------------------

// Smoke2dGetDataParams

// -------------------------------------------------------------------

// Result

// uint32 status = 1;
inline void Result::clear_status() {
  status_ = 0u;
}
inline ::google::protobuf::uint32 Result::status() const {
  // @@protoc_insertion_point(field_get:ssv.Result.status)
  return status_;
}
inline void Result::set_status(::google::protobuf::uint32 value) {
  
  status_ = value;
  // @@protoc_insertion_point(field_set:ssv.Result.status)
}

// -------------------------------------------------------------------

// DataChunk

// bytes data = 1;
inline void DataChunk::clear_data() {
  data_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline const ::std::string& DataChunk::data() const {
  // @@protoc_insertion_point(field_get:ssv.DataChunk.data)
  return data_.GetNoArena();
}
inline void DataChunk::set_data(const ::std::string& value) {
  
  data_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value);
  // @@protoc_insertion_point(field_set:ssv.DataChunk.data)
}
#if LANG_CXX11
inline void DataChunk::set_data(::std::string&& value) {
  
  data_.SetNoArena(
    &::google::protobuf::internal::GetEmptyStringAlreadyInited(), std::move(value));
  // @@protoc_insertion_point(field_set_rvalue:ssv.DataChunk.data)
}
#endif
inline void DataChunk::set_data(const char* value) {
  
  data_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:ssv.DataChunk.data)
}
inline void DataChunk::set_data(const void* value, size_t size) {
  
  data_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:ssv.DataChunk.data)
}
inline ::std::string* DataChunk::mutable_data() {
  
  // @@protoc_insertion_point(field_mutable:ssv.DataChunk.data)
  return data_.MutableNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline ::std::string* DataChunk::release_data() {
  // @@protoc_insertion_point(field_release:ssv.DataChunk.data)
  
  return data_.ReleaseNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline void DataChunk::set_allocated_data(::std::string* data) {
  if (data != NULL) {
    
  } else {
    
  }
  data_.SetAllocatedNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), data);
  // @@protoc_insertion_point(field_set_allocated:ssv.DataChunk.data)
}

#endif  // !PROTOBUF_INLINE_NOT_IN_HEADERS
// -------------------------------------------------------------------

// -------------------------------------------------------------------

// -------------------------------------------------------------------

// -------------------------------------------------------------------

// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)


}  // namespace ssv

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_Smoke2d_2eproto__INCLUDED
