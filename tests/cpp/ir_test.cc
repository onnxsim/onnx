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

// Regression test for the hasAnySubgraphNode() cache added to
// forSelfAndEachSubGraphImpl (see ir.h's Graph::hasAnySubgraphNode() and
// invalidateSubgraphNodeCache()): a graph whose cache memoized "no
// subgraphs" BEFORE a node gained a subgraph (g) attribute must still find
// kCaptured placeholders inside that new subgraph on a later
// setUniqueName() call. A stale-false cache would silently skip the scan
// and leave the captured value's name out of sync with the value it
// captures -- corrupting, not just slowing down, the subgraph's reference.
TEST(IR, SetUniqueNamePropagatesIntoSubgraphAddedAfterCacheWarms) {
  Graph* g = new Graph(); // NOLINT(cppcoreguidelines-owning-memory)
  g->setName("outer");
  Value* v = g->addInput();
  v->setUniqueName("v0");
  // A second setUniqueName() call is what actually exercises
  // forEachNode()'s subgraph search (the first call has no previous name to
  // propagate), which is also what warms hasAnySubgraphNode()'s cache to
  // "false" -- the graph has no subgraph nodes yet at this point.
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

  // If the cache correctly invalidated when the g attribute was set above,
  // this rename must propagate into the subgraph's captured placeholder.
  v->setUniqueName("v2");
  EXPECT_EQ(captured_value->uniqueName(), "v2");
}

// Companion to the above: once a node's only subgraph attribute is
// *removed*, removeAttribute() must invalidate the cache too, so a later
// setUniqueName() call correctly no-ops on the now-plain graph instead of
// (harmlessly, but as a sanity check on the invalidation path) reusing a
// stale "has subgraphs" answer forever.
TEST(IR, RemoveAttributeInvalidatesSubgraphNodeCache) {
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

} // namespace ONNX_NAMESPACE::Test
