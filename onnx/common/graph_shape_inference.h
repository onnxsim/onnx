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
// Nodes with a GRAPH or GRAPHS attribute (If/Loop/Scan and similar
// control-flow ops) ARE inferred: the node's own attribute is exported to a
// real GraphProto (the one part of this file that does touch protobuf, once
// per node visit, scoped to that node's typically-small body -- not the
// whole graph) and wired up via a GraphInferenceContext, so the op's own
// schema-registered inference function recurses into its body through
// onnx's existing, unmodified GraphInferencerImpl/InferShapesImpl. A
// subgraph's *captured* references (a name used inside the body with no
// local definition) resolve against the enclosing graph's inputs,
// initializers and already-processed node outputs, mirroring onnx's own
// outer_scope_value_types_by_name mechanism.
//
// v1 scope -- these remaining limitations are all *safe*: an affected
// node's outputs are simply left as they were, exactly as if the op had no
// registered schema (onnx's own protobuf-based InferShapes does the same
// for a genuinely unknown op), never producing incorrect information:
//  - Function-body inference (schema->HasFunction(), for ops defined as an
//    onnx function rather than a native op) is not implemented.
//  - ShapeInferenceOptions::enable_data_propagation is not honored (v1
//    always runs as if it were false); onnxsim's own callers don't set it.
//  - Sparse tensor inputs are not fed to getInputSparseData().
//  - No SymbolTable is threaded through, so symbolic (dim_param) shapes
//    are not materialized/unified across a Loop's iterations the way
//    onnx::shape_inference::InferShapes's own SymbolTableImpl would; this
//    only affects the *precision* of some symbolic shapes, never their
//    correctness.
//
// Returns whether any value's inferred type/shape actually changed
// anything, mirroring onnx::shape_inference::InferShapes's
// num_inferred_values out-param.
bool InferShapesOnGraph(Graph& g, const ShapeInferenceOptions& options = ShapeInferenceOptions());

} // namespace ONNX_NAMESPACE
