// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "onnx/common/assertions.h"
#include "onnx/common/ir.h"
#include "onnx/common/ir_pb_converter.h"
#include "onnx/defs/parser.h"
#include "onnx/defs/tensor_util.h"

namespace ONNX_NAMESPACE::Test {

static bool IsValidIdentifier(const std::string& name) {
  if (name.empty()) {
    return false;
  }
  if (!IsAlpha(name[0]) && name[0] != '_') {
    return false;
  }
  for (size_t i = 1; i < name.size(); ++i) {
    if (!IsAlnum(name[i]) && name[i] != '_') {
      return false;
    }
  }
  return true;
}

TEST(IR, ValidIdentifierTest) {
  Graph* g = new Graph(); // NOLINT(cppcoreguidelines-owning-memory)
  g->setName("test");
  Value* x = g->addInput();
  x->setUniqueName("x");
  x->setElemType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  x->setSizes({Dimension("M"), Dimension("N")});
  Node* node1 = g->create(kNeg, 1);
  node1->addInput(x);
  g->appendNode(node1);
  Value* temp1 = node1->outputs()[0];
  Node* node2 = g->create(kNeg, 1);
  node2->addInput(temp1);
  g->appendNode(node2);
  Value* y = node2->outputs()[0];
  g->registerOutput(y);

  ModelProto model;
  ExportModelProto(&model, std::shared_ptr<Graph>(g));

  for (const auto& node : model.graph().node()) {
    for (const auto& name : node.output()) {
      EXPECT_TRUE(IsValidIdentifier(name));
    }
  }
}

// Regression: copyMetadata() must not turn "rank unknown" into rank 0.
TEST(IR, CopyMetadataPreservesUnknownRank) {
  Graph g;
  g.setName("test");
  Value* from = g.addInput();
  Value* to = g.addInput();
  to->setSizes({Dimension(1)});
  ASSERT_FALSE(from->has_sizes());
  ASSERT_TRUE(to->has_sizes());

  to->copyMetadata(from);
  EXPECT_FALSE(to->has_sizes());
}

// copyMetadata() must still copy a known rank/shape over.
TEST(IR, CopyMetadataCopiesKnownSizes) {
  Graph g;
  g.setName("test");
  Value* from = g.addInput();
  from->setSizes({Dimension(2), Dimension(3)});
  Value* to = g.addInput();
  ASSERT_FALSE(to->has_sizes());

  to->copyMetadata(from);
  ASSERT_TRUE(to->has_sizes());
  ASSERT_EQ(to->sizes().size(), 2u);
  EXPECT_EQ(to->sizes()[0].dim, 2);
  EXPECT_EQ(to->sizes()[1].dim, 3);
}

// Regression tests for Graph's name-uniqueness bookkeeping (used_names_ /
// subgraph_bearing_nodes_, backing isNameUnique()/getNextUniqueName()): a
// name can have more than one live holder at once (an initializer's Tensor
// entry and its mirrored graph Value, or an output value mid-rename in
// Value::replaceAllUsesWith), and releasing just one holder must not free
// the name while another is still displaying it.

// getNextUniqueName() only ever proposes "_v_<n>" candidates for a
// strictly-increasing, never-reused n, so a name it might mint again later
// has to itself be "_v_<n>"-shaped for some not-yet-reached n -- an
// arbitrary held name (e.g. "y") is never revisited regardless of whether
// it's correctly reserved. These tests reserve such a not-yet-reached slot
// explicitly, ahead of the counter's current position, then drive the
// counter up to it via ordinary getNextUniqueName() calls: with the name
// still correctly reserved, that exact "_v_<n>" is skipped over; with the
// bug, it gets minted a second time.

// Value::replaceAllUsesWith() on a registered graph output renames the old
// output value off of its name so the replacement can take it over. The old
// value's release of that name must not un-reserve it while the replacement
// is still live and displaying it.
TEST(IR, ReplaceAllUsesWithKeepsGraphOutputNameReserved) {
  Graph g;
  g.setName("test");
  Value* x = g.addInput();
  x->setUniqueName("x");

  Node* node1 = g.create(kNeg, 1);
  node1->addInput(x);
  g.appendNode(node1);
  Value* y = node1->outputs()[0];

  Node* node2 = g.create(kNeg, 1);
  node2->addInput(x);
  g.appendNode(node2);
  Value* replacement = node2->outputs()[0];

  const std::string y_name = toVarName(replacement->unique() + 5);
  y->setUniqueName(y_name);
  g.registerOutput(y);

  y->replaceAllUsesWith(replacement);
  ASSERT_EQ(replacement->uniqueName(), y_name);

  bool saw_reserved_name_again = false;
  for (int i = 0; i < 20; ++i) {
    if (g.getNextUniqueName() == y_name) {
      saw_reserved_name_again = true;
    }
  }
  EXPECT_FALSE(saw_reserved_name_again);
}

// isNameUnique() recurses into If/Loop/Scan subgraph bodies to avoid minting
// a name in the parent graph that a nested subgraph's own value already
// displays. Each Graph -- parent and subgraph alike -- keeps an
// independently-numbered id counter, so an unnamed ("_v_<n>") value inside a
// subgraph can share its default display name with a value the parent graph
// is about to mint, purely by counter coincidence.
TEST(IR, IsNameUniqueSeesDefaultNamesInsideSubgraphs) {
  Graph parent;
  parent.setName("parent");
  Value* cond = parent.addInput();
  cond->setUniqueName("cond");

  auto then_graph = std::make_shared<Graph>();
  then_graph->setName("then");
  Value* then_in = then_graph->addInput();
  then_in->setUniqueName("then_in");
  Node* then_node = then_graph->create(kNeg, 1);
  then_node->addInput(then_in);
  then_graph->appendNode(then_node);
  Value* then_unnamed = then_node->outputs()[0]; // never explicitly renamed
  then_graph->registerOutput(then_unnamed);
  const std::string collision_name = then_unnamed->uniqueName();

  Node* if_node = parent.create(kIf, 0);
  if_node->addInput(cond);
  parent.appendNode(if_node);
  if_node->g_(Symbol("then_branch"), then_graph);

  for (int i = 0; i < 20; ++i) {
    EXPECT_NE(parent.getNextUniqueName(), collision_name);
  }
}

// eraseInitializer() removes the Tensor-level bookkeeping for an
// initializer and, if it finds a matching output on initializer_node_,
// erases that too -- but for an IR<4 (or input-shadowed) initializer, the
// value holding that name is a *graph input* added separately via
// addInitializer() alone (mirroring ir_pb_converter.cc's import path: see
// its ir_version < 4 / "exists in input" branch), not a value under
// initializer_node_. eraseInitializer() must not touch that value's name.
TEST(IR, EraseInitializerKeepsNameReservedForLiveValue) {
  Graph g;
  g.setName("test");
  const std::string w_name = toVarName(5);

  Value* v = g.addInput();
  v->setUniqueName(w_name);
  Tensor t;
  t.setName(w_name);
  g.addInitializer(t);

  g.eraseInitializer(w_name);
  ASSERT_EQ(v->uniqueName(), w_name); // v itself must be untouched
  bool saw_reserved_name_again = false;
  for (int i = 0; i < 10; ++i) {
    if (g.getNextUniqueName() == w_name) {
      saw_reserved_name_again = true;
    }
  }
  EXPECT_FALSE(saw_reserved_name_again);
}

// clearInitializers() only clears the Tensor-level initializer bookkeeping;
// per its documented behavior, initializer_node_'s output Values survive and
// keep displaying their names, which must stay reserved.
TEST(IR, ClearInitializersKeepsSurvivingValueNamesReserved) {
  Graph g;
  g.setName("test");
  Tensor t;
  const std::string w_name = toVarName(5);
  t.setName(w_name);
  Value* v = g.addInitializerAndCreateValue(t);
  ASSERT_EQ(v->uniqueName(), w_name);

  g.clearInitializers();
  bool saw_reserved_name_again = false;
  for (int i = 0; i < 10; ++i) {
    if (g.getNextUniqueName() == w_name) {
      saw_reserved_name_again = true;
    }
  }
  EXPECT_FALSE(saw_reserved_name_again);
}

// forEachNode()'s subgraph walk must tolerate a callback that reaches back
// and mutates the attributes of a node it is currently recursing through --
// e.g. a rewrite pass editing its own enclosing If/Loop node while visiting
// a node inside that node's subgraph. This must not corrupt the walk (under
// ASAN: not a heap-use-after-free) and must still visit every node.
TEST(IR, ForEachNodeSurvivesSelfMutationDuringSubgraphWalk) {
  Graph parent;
  parent.setName("parent");
  Value* cond = parent.addInput();
  cond->setUniqueName("cond");

  auto body = std::make_shared<Graph>();
  body->setName("body");
  Value* body_in = body->addInput();
  body_in->setUniqueName("body_in");
  Node* inner = body->create(kNeg, 1);
  inner->addInput(body_in);
  body->appendNode(inner);
  body->registerOutput(inner->outputs()[0]);

  Node* if_node = parent.create(kIf, 0);
  if_node->addInput(cond);
  parent.appendNode(if_node);
  // A few unrelated attributes first, so the reentrant add below is more
  // likely to force a reallocation of if_node's own attribute storage.
  if_node->i_(Symbol("a1"), 1);
  if_node->i_(Symbol("a2"), 2);
  if_node->i_(Symbol("a3"), 3);
  if_node->g_(Symbol("then_branch"), body);

  int visited = 0;
  parent.forEachNode([&](Node* node) {
    ++visited;
    if (node == inner) {
      if_node->i_(Symbol("extra_attr"), 4);
    }
  });
  EXPECT_EQ(visited, 2);
  EXPECT_TRUE(if_node->hasAttribute(Symbol("extra_attr")));
}

// Regression test for Graph::subgraph_bearing_nodes_ (see ir.h's
// Node::onAttributeAdded()/onAttributeRemoved()): a node that gains a
// subgraph (g) attribute after the graph has already been walked once must
// still be found by a later setUniqueName() call's kCaptured search. A node
// that isn't correctly added to subgraph_bearing_nodes_ when it gains a g/gs
// attribute would silently skip the scan and leave the captured value's name
// out of sync with the value it captures -- corrupting, not just slowing
// down, the subgraph's reference.
TEST(IR, SetUniqueNamePropagatesIntoSubgraphAddedAfterFirstWalk) {
  Graph* g = new Graph(); // NOLINT(cppcoreguidelines-owning-memory)
  g->setName("outer");
  Value* v = g->addInput();
  v->setUniqueName("v0");
  // A second setUniqueName() call is what actually exercises
  // forEachNode()'s subgraph search (the first call has no previous name to
  // propagate) -- the graph has no subgraph nodes yet at this point.
  v->setUniqueName("v1");

  // Now give the graph a subgraph (mimicking an If/Loop node) containing a
  // kCaptured placeholder for v's current name -- the same construction
  // ir_pb_converter.cc's createDummyValue() uses for a real model's
  // captured references.
  auto subgraph = std::make_shared<Graph>();
  Node* captured = subgraph->create(kCaptured, 1);
  subgraph->appendNode(captured);
  Value* captured_value = captured->outputs()[0];
  captured_value->setUniqueName("v1");

  Node* control_flow_node = g->create(kIf, 0);
  g->appendNode(control_flow_node);
  control_flow_node->g_(kthen_branch, subgraph);

  // If control_flow_node was correctly added to subgraph_bearing_nodes_ when
  // the g attribute was set above, this rename must propagate into the
  // subgraph's captured placeholder.
  v->setUniqueName("v2");
  EXPECT_EQ(captured_value->uniqueName(), "v2");
}

// Companion to the above: once a node's only subgraph attribute is
// *removed*, removeAttribute() must remove it from subgraph_bearing_nodes_
// too, so a later setUniqueName() call correctly no-ops on the now-plain
// graph instead of (harmlessly, but as a sanity check on the removal path)
// still recursing into the detached subgraph.
TEST(IR, RemoveAttributeUpdatesSubgraphBearingNodes) {
  Graph* g = new Graph(); // NOLINT(cppcoreguidelines-owning-memory)
  g->setName("outer");
  Value* v = g->addInput();
  v->setUniqueName("v0");

  auto subgraph = std::make_shared<Graph>();
  Node* captured = subgraph->create(kCaptured, 1);
  subgraph->appendNode(captured);
  Value* captured_value = captured->outputs()[0];
  captured_value->setUniqueName("v0");

  Node* control_flow_node = g->create(kIf, 0);
  g->appendNode(control_flow_node);
  control_flow_node->g_(kthen_branch, subgraph);
  v->setUniqueName("v1");
  EXPECT_EQ(captured_value->uniqueName(), "v1");

  control_flow_node->removeAttribute(kthen_branch);
  // The graph now has no subgraphs; the outer value can still be renamed
  // freely, and the (now-detached) captured placeholder must not change.
  v->setUniqueName("v2");
  EXPECT_EQ(captured_value->uniqueName(), "v1");
}

// Regression test: Tensor::elem_num() and size_from_dim() must use 64-bit
// arithmetic. Previously, std::accumulate used `1` (int) as the initial value,
// causing 32-bit multiplication that silently overflowed for tensors whose
// element count exceeded INT_MAX (~2.1B). Fixed by using int64_t{1}.
TEST(Tensor, ElemNumLargeTensorNoOverflow) {
  Tensor t;
  // 50000 * 50000 = 2,500,000,000 which exceeds INT32_MAX (2,147,483,647)
  t.sizes() = {50000, 50000};
  const int64_t expected = static_cast<int64_t>(50000) * 50000;
  EXPECT_EQ(t.elem_num(), expected);
  EXPECT_EQ(t.size_from_dim(0), expected);
  EXPECT_EQ(t.size_from_dim(1), int64_t{50000});
}

// Build a raw_data string from native bytes of the given values.
template <typename T>
static std::string MakeRawData(const std::vector<T>& values) {
  std::string raw;
  raw.resize(values.size() * sizeof(T));
  std::memcpy(raw.data(), values.data(), raw.size());
  return raw;
}

// Regression: raw size not a multiple of the element size used to overflow.
#ifndef ONNX_NO_EXCEPTIONS
TEST(Tensor, ParseDataRawSizeNotMultipleThrows) {
  Tensor t;
  // 5 bytes is not a multiple of sizeof(int32_t) == 4.
  t.set_raw_data(std::string(5, '\0'));
  EXPECT_THROW(ParseData<int32_t>(&t), assert_error);
}
#endif

// Valid raw tensor round-trips; byte-symmetric values are endian-independent.
TEST(Tensor, ParseDataRawValid) {
  const std::vector<int32_t> values = {0, 0x01010101, 0x7F7F7F7F};
  Tensor t;
  t.set_raw_data(MakeRawData(values));
  EXPECT_EQ(ParseData<int32_t>(&t), values);
}

// Empty raw_data is a multiple of any element size and yields no elements.
TEST(Tensor, ParseDataRawEmpty) {
  Tensor t;
  t.set_raw_data(std::string());
  EXPECT_TRUE(ParseData<int32_t>(&t).empty());
}

namespace {
// Builds a 3x3-dense-shape SparseTensorProto with two non-default float
// elements at linear indices 0 and 4 (values 1.0 and 2.0) -- the [NNZ]
// linear-index format SparseTensorProto's own doc comment describes as
// format (b).
void FillSparseTensorProto(SparseTensorProto& stp, const std::string& values_name) {
  stp.mutable_values()->set_name(values_name);
  stp.mutable_values()->set_data_type(TensorProto_DataType_FLOAT);
  stp.mutable_values()->add_dims(2);
  stp.mutable_values()->add_float_data(1.0f);
  stp.mutable_values()->add_float_data(2.0f);
  stp.mutable_indices()->set_data_type(TensorProto_DataType_INT64);
  stp.mutable_indices()->add_dims(2);
  stp.mutable_indices()->add_int64_data(0);
  stp.mutable_indices()->add_int64_data(4);
  stp.add_dims(3);
  stp.add_dims(3);
}
} // namespace

// Sparse tensors used to be entirely unsupported by the C++ Graph IR: a
// sparse-tensor NODE ATTRIBUTE threw on Import (ir_pb_converter.cc's
// convertAttribute used to fail_convert("Sparse tensors not supported.")),
// and a top-level GRAPH INITIALIZER was silently dropped (graphProtoToGraph
// never read GraphProto.sparse_initializer at all). Regression test for
// both gaps: a full ModelProto -> Graph -> ModelProto round trip must now
// preserve a sparse initializer and a Constant node's `sparse_value`
// attribute exactly.
TEST(IR, SparseTensorRoundTripsThroughImportExport) {
  ModelProto model;
  model.set_ir_version(IR_VERSION);
  auto* opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(13);

  GraphProto& gp = *model.mutable_graph();
  gp.set_name("sparse_test");

  // A sparse graph initializer, consumed by the graph's only output --
  // exercises Import's addSparseInitializerAndCreateValue path (ir_version
  // >= 4) and Export's sparse_initializer loop.
  FillSparseTensorProto(*gp.add_sparse_initializer(), "sparse_init");
  ValueInfoProto* output = gp.add_output();
  output->set_name("sparse_init");
  output->mutable_type()->mutable_sparse_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);

  // A Constant node whose value is carried as a `sparse_value` attribute --
  // exercises Import/Export's AttributeKind::z handling.
  NodeProto* node = gp.add_node();
  node->set_op_type("Constant");
  node->add_output("const_out");
  AttributeProto* attr = node->add_attribute();
  attr->set_name("sparse_value");
  attr->set_type(AttributeProto_AttributeType_SPARSE_TENSOR);
  FillSparseTensorProto(*attr->mutable_sparse_tensor(), "const_sparse_value");
  ValueInfoProto* node_output = gp.add_output();
  node_output->set_name("const_out");
  node_output->mutable_type()->mutable_sparse_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);

  std::shared_ptr<Graph> g(ImportModelProto(model));
  // Not ASSERT_NE(g, nullptr): gtest's failure-message printer for
  // shared_ptr<Graph> requires an operator<<(ostream&, const Graph&),
  // which ir.h does not define.
  ASSERT_TRUE(g != nullptr);

  // Graph-side: the sparse initializer is queryable by name, and its
  // content matches what was fed in.
  const SparseTensor* init = g->getSparseInitializer("sparse_init");
  ASSERT_NE(init, nullptr);
  EXPECT_EQ(init->values.elem_type(), TensorProto_DataType_FLOAT);
  EXPECT_EQ(init->indices.elem_type(), TensorProto_DataType_INT64);
  EXPECT_EQ(init->dims, (std::vector<int64_t>{3, 3}));
  ASSERT_EQ(init->values.floats().size(), 2u);
  EXPECT_FLOAT_EQ(init->values.floats()[0], 1.0f);
  EXPECT_FLOAT_EQ(init->values.floats()[1], 2.0f);
  ASSERT_EQ(init->indices.int64s().size(), 2u);
  EXPECT_EQ(init->indices.int64s()[0], 0);
  EXPECT_EQ(init->indices.int64s()[1], 4);

  // A Value was also registered for the sparse initializer (ir_version >= 4).
  bool found_initializer_value = false;
  for (Value* out : g->outputs()) {
    if (out->uniqueName() == "sparse_init") {
      found_initializer_value = true;
    }
  }
  EXPECT_TRUE(found_initializer_value);

  // The Constant node's sparse_value attribute round-tripped too.
  Node* const_node = nullptr;
  for (Node* n : g->nodes()) {
    if (n->kind().toString() == std::string("Constant")) {
      const_node = n;
    }
  }
  ASSERT_NE(const_node, nullptr);
  ASSERT_EQ(const_node->kindOf(Symbol("sparse_value")), AttributeKind::z);
  const SparseTensor& node_sparse = const_node->z(Symbol("sparse_value"));
  EXPECT_EQ(node_sparse.dims, (std::vector<int64_t>{3, 3}));
  ASSERT_EQ(node_sparse.values.floats().size(), 2u);
  EXPECT_FLOAT_EQ(node_sparse.values.floats()[0], 1.0f);
  EXPECT_FLOAT_EQ(node_sparse.values.floats()[1], 2.0f);

  // Export back and check the round trip is faithful at the protobuf level
  // too.
  ModelProto exported;
  ExportModelProto(&exported, g);

  ASSERT_EQ(exported.graph().sparse_initializer_size(), 1);
  const SparseTensorProto& exported_init = exported.graph().sparse_initializer(0);
  EXPECT_EQ(exported_init.values().name(), "sparse_init");
  EXPECT_EQ(exported_init.values().data_type(), TensorProto_DataType_FLOAT);
  ASSERT_EQ(exported_init.values().float_data_size(), 2);
  EXPECT_FLOAT_EQ(exported_init.values().float_data(0), 1.0f);
  EXPECT_FLOAT_EQ(exported_init.values().float_data(1), 2.0f);
  ASSERT_EQ(exported_init.indices().int64_data_size(), 2);
  EXPECT_EQ(exported_init.indices().int64_data(0), 0);
  EXPECT_EQ(exported_init.indices().int64_data(1), 4);
  EXPECT_EQ(exported_init.dims_size(), 2);
  EXPECT_EQ(exported_init.dims(0), 3);
  EXPECT_EQ(exported_init.dims(1), 3);

  bool found_sparse_attr = false;
  for (const auto& exported_node : exported.graph().node()) {
    if (exported_node.op_type() != "Constant") {
      continue;
    }
    for (const auto& exported_attr : exported_node.attribute()) {
      if (exported_attr.name() != "sparse_value") {
        continue;
      }
      found_sparse_attr = true;
      EXPECT_EQ(exported_attr.type(), AttributeProto_AttributeType_SPARSE_TENSOR);
      EXPECT_EQ(exported_attr.sparse_tensor().values().name(), "const_sparse_value");
      ASSERT_EQ(exported_attr.sparse_tensor().values().float_data_size(), 2);
      EXPECT_FLOAT_EQ(exported_attr.sparse_tensor().values().float_data(0), 1.0f);
      EXPECT_FLOAT_EQ(exported_attr.sparse_tensor().values().float_data(1), 2.0f);
    }
  }
  EXPECT_TRUE(found_sparse_attr);
}

} // namespace ONNX_NAMESPACE::Test
