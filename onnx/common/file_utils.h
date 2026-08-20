// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem> // NOLINT(build/c++17)
#include <fstream>
#include <string>

#ifdef _WIN32
#include "onnx/common/path.h"
#endif
#include "onnx/checker.h"

namespace ONNX_NAMESPACE {

template <typename T>
void LoadProtoFromPath(const std::string& proto_path, T& proto) {
#ifdef _WIN32
  std::filesystem::path proto_u8_path(utf8str_to_wstring(proto_path));
#else
  std::filesystem::path proto_u8_path(proto_path);
#endif
  std::fstream proto_stream(proto_u8_path, std::ios::in | std::ios::binary);
  if (!proto_stream.good()) {
    fail_check("Unable to open proto file: ", proto_path, ". Please check if it is a valid proto. ");
  }
  // A single bulk read sized from the file's byte length, rather than
  // istreambuf_iterator's byte-at-a-time copy (each increment pays a
  // buffer-boundary check) -- on a large (hundreds of MB+) model file the
  // difference is seconds, not microseconds.
  std::error_code size_ec;
  const std::uintmax_t file_size = std::filesystem::file_size(proto_u8_path, size_ec);
  std::string data;
  bool read_ok = false;
  if (!size_ec) {
    data.resize(file_size);
    proto_stream.read(data.data(), static_cast<std::streamsize>(file_size));
    // read() may set eofbit alongside a fully successful read that consumes
    // exactly to the end of the file, so check the actual byte count rather
    // than the stream's good()/fail() flags.
    read_ok = static_cast<std::uintmax_t>(proto_stream.gcount()) == file_size;
  } else {
    // Fall back to the iterator-based read if the size could not be
    // determined (e.g. a non-regular file such as a pipe).
    data.assign(std::istreambuf_iterator<char>{proto_stream}, std::istreambuf_iterator<char>{});
    read_ok = true;
  }
  if (!read_ok) {
    fail_check("Unable to read proto file: ", proto_path, ". Please check if it is a valid proto. ");
  }
  if (!ParseProtoFromBytes(&proto, data.c_str(), data.size())) {
    fail_check(
        "Unable to parse proto from file: ", proto_path, ". Please check if it is a valid protobuf file of proto. ");
  }
}
} // namespace ONNX_NAMESPACE
