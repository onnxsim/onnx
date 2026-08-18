// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once
#include <memory>
#include <string>

#include "onnx/common/common.h"
#include "onnx/common/ir.h"

namespace ONNX_NAMESPACE {

class ConvertError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;

  explicit ConvertError(const std::string& message) : std::runtime_error(message) {}

  const char* what() const noexcept override {
    if (!expanded_message_.empty()) {
      return expanded_message_.c_str();
    }
    return std::runtime_error::what();
  }

  void AppendContext(const std::string& context) {
    expanded_message_ = MakeString(std::runtime_error::what(), "\n\n==> Context: ", context);
  }

 private:
  std::string expanded_message_;
};

#define fail_convert(...) ONNX_THROW_EX(ConvertError(MakeString(__VA_ARGS__)))

// Feature-detection macro for downstream code (onnxsim's onnx-optimizer
// fork) that wants to opt into the consuming/moving Import/Export overloads
// below when compiled against this onnx fork, and fall back to the ordinary
// copying-only API otherwise -- e.g. when linked against onnxruntime's own
// bundled, unpatched onnx copy instead of this fork (see onnxsim issue
// #633). Deliberately not an ONNX_NAMESPACE symbol, so it can be checked
// with #ifdef.
#define ONNX_IR_PB_CONVERTER_HAS_CONSUMING_OVERLOADS 1

// If consume_tensor_data is true, each initializer's raw bytes are moved out
// of ``g`` into ``p_m`` instead of copied. This roughly halves the memory
// traffic of a ModelProto <-> Graph round trip, but leaves ``g``'s
// initializer tensors with empty raw data afterward -- only pass true when
// the caller is about to discard ``g`` anyway (e.g. onnxoptimizer's
// Optimizer::optimize(), which owns ``g`` for exactly one Import/transform/
// Export call). Defaults to false so existing callers are unaffected.
void ExportModelProto(ModelProto* p_m, const std::shared_ptr<Graph>& g, bool consume_tensor_data = false);

std::unique_ptr<Graph> ImportModelProto(const ModelProto& mp);

// Consuming overload: moves each initializer's raw bytes out of ``mp``
// instead of copying them, leaving ``mp``'s initializer tensors with empty
// raw data afterward. Only call this when ``mp`` is about to be discarded or
// overwritten -- it is not safe to read ``mp``'s initializer data after this
// call. See ExportModelProto's consume_tensor_data for the matching Export
// side of the same optimization.
std::unique_ptr<Graph> ImportModelProto(ModelProto& mp);

ONNX_API ModelProto PrepareOutput(const ModelProto& mp_in);

void assertNonNull(const std::shared_ptr<Graph>& g);
} // namespace ONNX_NAMESPACE
