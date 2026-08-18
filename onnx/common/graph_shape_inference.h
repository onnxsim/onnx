// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/common/ir.h"
#include "onnx/defs/shape_inference.h"

namespace ONNX_NAMESPACE {

// Runs ONNX type-and-shape inference directly against the C++ IR (Graph `g`),
// with no ModelProto <-> Graph round trip at all: every op's existing,
// schema-registered TypeAndShapeInferenceFunction is invoked unmodified (via
// shape_inference::InferenceContextImpl, fed a lightweight per-node
// NodeProto/TypeProto/TensorProto view built on the fly from the Node/Value
// being visited), and results are merged back with onnx's own
// shape_inference::mergeShapesAndTypes -- so behavior matches
// onnx::shape_inference::InferShapes as closely as a from-scratch Graph-level
// driver reasonably can, without reimplementing any op's inference formula.
//
// Written for onnxsim issue #633's "remaining option 2": OptAndShape's fixed
// point alternates ModelProto-based InferShapes with a
// ModelProto<->Graph round trip for Optimize on every round; this lets a
// caller that already holds a resident Graph (e.g. via
// Optimizer::optimize(Graph&)/OptimizeGraphFixed, see
// onnxoptimizer/optimize.h) run shape inference on that same Graph, with no
// conversion in either direction.
//
// v1 scope -- these limitations are all *safe*: an affected node's outputs
// are simply left as they were, exactly as if the op had no registered
// schema (onnx's own protobuf-based InferShapes does the same for a
// genuinely unknown op), never producing incorrect information:
//  - Nodes with a GRAPH or GRAPHS attribute (If/Loop/Scan and similar
//    control-flow ops) are not inferred; their subgraphs are not walked.
//  - Function-body inference (schema->HasFunction(), for ops defined as an
//    onnx function rather than a native op) is not implemented.
//  - ShapeInferenceOptions::enable_data_propagation is not honored (v1
//    always runs as if it were false); onnxsim's own callers don't set it.
//  - Sparse tensor inputs are not fed to getInputSparseData().
//
// Returns whether any value's inferred type/shape actually changed anything,
// mirroring onnx::shape_inference::InferShapes's num_inferred_values
// out-param (see onnxsim's own _InferShapes(model, bool*) wrapper).
bool InferShapesOnGraph(Graph& g, const ShapeInferenceOptions& options = ShapeInferenceOptions());

} // namespace ONNX_NAMESPACE
