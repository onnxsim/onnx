// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx/common/graph_shape_inference.h"

#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "onnx/common/ir.h"
#include "onnx/defs/schema.h"
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

namespace {
// A tiny scratch domain/op whose only purpose is to prove getInputSparseData()
// sees real content: its inference function copies the input's SparseTensorProto
// dims onto the output's shape (so the assertion can check they came from the
// actual sparse data, not just "some shape got set").
constexpr const char* kSparseProbeDomain = "test.sparse";

class ScopedSparseProbeSchema {
 public:
  ScopedSparseProbeSchema() {
    // RegisterSchema below checks the domain against DomainToVersionRange
    // and rejects an unknown one; there's no matching "remove" API (the
    // registry doesn't support unregistering a whole domain, only
    // individual schemas via DeregisterSchema below), so this registration
    // stays for the process lifetime -- fine for a scratch, collision-free
    // test-only domain name.
    OpSchemaRegistry::DomainToVersionRange::Instance().AddDomainToVersion(kSparseProbeDomain, 1, 1);
    OpSchema schema;
    schema.SetName("SparseProbe")
        .SetDomain(kSparseProbeDomain)
        .SinceVersion(1)
        .Input(0, "x", "sparse input", "tensor(float)")
        .Output(0, "y", "probe result", "tensor(float)")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          const SparseTensorProto* sparse = ctx.getInputSparseData(0);
          auto* out = ctx.getOutputType(0)->mutable_tensor_type();
          out->set_elem_type(TensorProto_DataType_FLOAT);
          if (sparse == nullptr) {
            return;
          }
          auto* shape = out->mutable_shape();
          for (int64_t d : sparse->dims()) {
            shape->add_dim()->set_dim_value(d);
          }
        });
    RegisterSchema(schema);
  }
  ~ScopedSparseProbeSchema() {
    DeregisterSchema("SparseProbe", 1, kSparseProbeDomain);
  }
};
} // namespace

// Regression test for the "v1 scope" gap this file's own header used to
// document: input_sparse_data_by_name used to be hardcoded empty, so
// getInputSparseData() always returned nullptr regardless of what a node's
// input actually was. Covers both sources ProcessNode now feeds it from --
// a sparse graph initializer, and a Constant node's `sparse_value`
// attribute -- verifying the *content* (dims), not just non-nullness,
// reaches the inference function unmodified.
TEST(GraphShapeInference, SparseTensorInputsReachInferenceContext) {
  ScopedSparseProbeSchema scoped_schema;

  Graph g;
  g.opset_versions_mutable().emplace_back(OpSetID("", 13));
  g.opset_versions_mutable().emplace_back(OpSetID(kSparseProbeDomain, 1));

  // Node A: fed by a sparse graph initializer with dense shape [3, 5].
  SparseTensor init;
  init.values.setName("sparse_init");
  init.values.elem_type() = TensorProto_DataType_FLOAT;
  init.values.floats() = {1.0f, 2.0f};
  init.indices.elem_type() = TensorProto_DataType_INT64;
  init.indices.int64s() = {0, 4};
  init.dims = {3, 5};
  Value* init_value = g.addSparseInitializerAndCreateValue(init);

  Node* node_a = g.create(Symbol("SparseProbe"), 1);
  node_a->setDomain(kSparseProbeDomain);
  node_a->addInput(init_value);
  g.appendNode(node_a);
  Value* y_a = node_a->outputs()[0];
  y_a->setUniqueName("Y_A");
  g.registerOutput(y_a);

  // Node B: fed by a Constant node's sparse_value attribute, dense shape
  // [7, 11] -- deliberately different from node A's, so a bug that mixed
  // the two sources up (or fed the same data to both) would be caught.
  SparseTensor const_sparse;
  const_sparse.values.setName("const_sparse_value");
  const_sparse.values.elem_type() = TensorProto_DataType_FLOAT;
  const_sparse.values.floats() = {3.0f};
  const_sparse.indices.elem_type() = TensorProto_DataType_INT64;
  const_sparse.indices.int64s() = {0};
  const_sparse.dims = {7, 11};

  Node* const_node = g.create(Symbol("Constant"), 1);
  g.appendNode(const_node);
  const_node->z_(Symbol("sparse_value"), std::move(const_sparse));
  Value* const_out = const_node->outputs()[0];
  const_out->setUniqueName("ConstOut");

  Node* node_b = g.create(Symbol("SparseProbe"), 1);
  node_b->setDomain(kSparseProbeDomain);
  node_b->addInput(const_out);
  g.appendNode(node_b);
  Value* y_b = node_b->outputs()[0];
  y_b->setUniqueName("Y_B");
  g.registerOutput(y_b);

  const bool changed = InferShapesOnGraph(g);
  EXPECT_TRUE(changed);

  EXPECT_EQ(y_a->elemType(), TensorProto_DataType_FLOAT);
  ASSERT_TRUE(y_a->has_sizes());
  ASSERT_EQ(y_a->sizes().size(), 2u);
  EXPECT_EQ(y_a->sizes()[0].dim, 3);
  EXPECT_EQ(y_a->sizes()[1].dim, 5);

  EXPECT_EQ(y_b->elemType(), TensorProto_DataType_FLOAT);
  ASSERT_TRUE(y_b->has_sizes());
  ASSERT_EQ(y_b->sizes().size(), 2u);
  EXPECT_EQ(y_b->sizes()[0].dim, 7);
  EXPECT_EQ(y_b->sizes()[1].dim, 11);
}

} // namespace ONNX_NAMESPACE::Test
