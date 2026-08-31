// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx/common/graph_shape_inference.h"

#include <cstdint>

#include "gtest/gtest.h"
#include "onnx/common/ir.h"
#include "onnx/shape_inference/implementation.h"

namespace ONNX_NAMESPACE::Test {

// MeanVarianceNormalization (opset 13) has a schema-attached function body
// (OpSchema::HasFunction()) and no ordinary TypeAndShapeInferenceFunction --
// confirmed via `onnx.defs.get_all_schemas_with_history()` in Python: it's
// one of only two ops (besides GreaterOrEqual/LessOrEqual's older versions,
// which do have an ordinary inference function too and so never exercise
// this path) with that combination. Its function body (ReduceMean, Pow,
// Sub, Sqrt, Add, Div on X) broadcasts back to X's own shape, so a correct
// inference should leave Y with X's elem_type and sizes.
TEST(GraphShapeInference, SchemaAttachedFunctionBodyInfersOutputType) {
  Graph g;
  g.opset_versions_mutable().emplace_back(OpSetID("", 13));

  Value* x = g.addInput();
  x->setUniqueName("X");
  x->setElemType(TensorProto_DataType_FLOAT);
  x->setSizes({Dimension(int64_t{2}), Dimension(int64_t{3}), Dimension(int64_t{4})});

  Node* n = g.create(Symbol("MeanVarianceNormalization"), 1);
  n->addInput(x);
  g.appendNode(n);
  // The function body materializes `axes` via a Constant node holding a
  // ref_attr_name back to this attribute -- unlike the schema's own Attr()
  // registration, the FunctionProto itself carries no default value for it
  // (confirmed via Python: `function_body.attribute == ["axes"]`, empty
  // `attribute_proto`), so an explicit value is required here for the same
  // reason the official onnx.shape_inference.infer_shapes() also raises
  // "Constant node... must be specified" on a MeanVarianceNormalization
  // node with no axes attribute set.
  n->is_(Symbol("axes"), std::vector<int64_t>{0, 2});
  Value* y = n->outputs()[0];
  y->setUniqueName("Y");
  g.registerOutput(y);

  EXPECT_EQ(y->elemType(), TensorProto_DataType_UNDEFINED);
  EXPECT_FALSE(y->has_sizes());

  const bool changed = InferShapesOnGraph(g);
  EXPECT_TRUE(changed);
  EXPECT_EQ(y->elemType(), TensorProto_DataType_FLOAT);
  ASSERT_TRUE(y->has_sizes());
  ASSERT_EQ(y->sizes().size(), 3u);
  EXPECT_EQ(y->sizes()[0].dim, 2);
  EXPECT_EQ(y->sizes()[1].dim, 3);
  EXPECT_EQ(y->sizes()[2].dim, 4);
}

// A model-local function (ModelProto.functions(), not attached to any
// OpSchema) referenced from a custom-domain node with no registered schema
// at all. Graph carries no notion of model-local functions itself (see
// graph_shape_inference.h's own doc comment), so the caller must pass the
// ModelLocalFunctionsMap in explicitly -- verify both halves of that
// contract: inference runs the function body's Relu node and infers Y when
// the map is provided, and safely no-ops (leaves Y untouched, exactly as if
// the op had no registered schema) when it isn't.
TEST(GraphShapeInference, ModelLocalFunctionInfersOutputTypeOnlyWhenMapProvided) {
  Graph g;
  g.opset_versions_mutable().emplace_back(OpSetID("", 13));
  g.opset_versions_mutable().emplace_back(OpSetID("test.custom", 1));

  Value* x = g.addInput();
  x->setUniqueName("X");
  x->setElemType(TensorProto_DataType_FLOAT);
  x->setSizes({Dimension(int64_t{2}), Dimension(int64_t{3})});

  Node* n = g.create(Symbol("MyOp"), 1);
  n->setDomain("test.custom");
  n->addInput(x);
  g.appendNode(n);
  Value* y = n->outputs()[0];
  y->setUniqueName("Y");
  g.registerOutput(y);

  FunctionProto func;
  func.set_name("MyOp");
  func.set_domain("test.custom");
  func.add_input("X");
  func.add_output("Y");
  {
    auto* opset = func.add_opset_import();
    opset->set_domain("");
    opset->set_version(13);
  }
  {
    auto* body_node = func.add_node();
    body_node->set_op_type("Relu");
    body_node->add_input("X");
    body_node->add_output("Y");
  }

  shape_inference::ModelLocalFunctionsMap functions;
  functions["test.custom:MyOp"] = &func;

  // Without the map: unsupported op, v1-safe no-op.
  const bool changed_without_map = InferShapesOnGraph(g);
  EXPECT_FALSE(changed_without_map);
  EXPECT_EQ(y->elemType(), TensorProto_DataType_UNDEFINED);
  EXPECT_FALSE(y->has_sizes());

  // With the map: infers through the function body's Relu node.
  const bool changed_with_map = InferShapesOnGraph(g, ShapeInferenceOptions(), nullptr, functions);
  EXPECT_TRUE(changed_with_map);
  EXPECT_EQ(y->elemType(), TensorProto_DataType_FLOAT);
  ASSERT_TRUE(y->has_sizes());
  ASSERT_EQ(y->sizes().size(), 2u);
  EXPECT_EQ(y->sizes()[0].dim, 2);
  EXPECT_EQ(y->sizes()[1].dim, 3);
}

} // namespace ONNX_NAMESPACE::Test
