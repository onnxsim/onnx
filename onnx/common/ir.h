// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "onnx/common/array_ref.h"
#include "onnx/common/assertions.h"
#include "onnx/common/common.h"
#include "onnx/common/graph_node_list.h"
#include "onnx/common/interned_strings.h"
#include "onnx/common/tensor.h"
#include "onnx/string_utils.h"

namespace ONNX_NAMESPACE {

// internal/private API
static inline std::string toVarName(size_t i) {
  std::ostringstream oss;
  oss << "_v_" << i;
  return oss.str();
}

// Graph represents one "function" of computation.
// It uses a simple ownership model where the graph owns all the nodes inside it.
// All references inside the graph are raw pointers.
// Destroying the Graph will invalidate any pointers to nodes in the graph.
struct Graph;

// Node is the base class of the IR graph. It represents one computation
// and dependencies on a list of Values. The "prim-ops", so to speak.
struct Node;

// A Value represents an input or output to node that is either a
// Tensor or an opaque Handle object, as determined by type().
struct Value;

class ResourceGuard final {
  std::function<void()> destructor_;

 public:
  ONNX_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ResourceGuard);

  explicit ResourceGuard(std::function<void()> destructor) : destructor_(std::move(destructor)) {}

  ~ResourceGuard() {
    destructor_();
  }
};

struct Dimension final {
  Dimension() : is_unknown(true), is_int(false), dim(-1) {}
  explicit Dimension(std::string param) : is_unknown(false), is_int(false), dim(-1), param(std::move(param)) {}
  explicit Dimension(int64_t dim) : is_unknown(false), is_int(true), dim(dim) {}

  bool is_unknown;
  bool is_int;
  int64_t dim;
  std::string param;
};

enum class AttributeKind : uint8_t {
  // float, float list, int, int list, string, string list,
  // tensor, tensor list, subgraph, subgraph list. type proto, type proto list
  f,
  fs,
  i,
  is,
  s,
  ss,
  t,
  ts,
  g,
  gs,
  tp,
  tps
};

static inline const char* toString(AttributeKind kind) {
  // NOLINTNEXTLINE(modernize-avoid-c-arrays)
  static constexpr const char* names[] = {"f", "fs", "i", "is", "s", "ss", "t", "ts", "g", "gs", "tp", "tps"};
  ONNX_ASSERT(size_t(kind) < std::size(names))
  return names[static_cast<int>(kind)];
}

struct AttributeValue {
  explicit AttributeValue(Symbol name) : name(name) {}
  using Ptr = std::unique_ptr<AttributeValue>;
  Symbol name;
  virtual AttributeKind kind() const = 0;
  virtual Ptr clone() const = 0;
  virtual ~AttributeValue() = default;
};

template <typename T, AttributeKind Kind>
struct ScalarAttributeValue final : public AttributeValue {
  using ConstructorType = T;
  using ValueType = T;
  ScalarAttributeValue(Symbol name, ValueType value_) : AttributeValue(name), value_(std::move(value_)) {}
  ValueType& value() {
    return value_;
  }
  Ptr clone() const override {
    return std::make_unique<ScalarAttributeValue>(name, value_);
  }
  AttributeKind kind() const override {
    return Kind;
  }

 private:
  ValueType value_;
};

template <typename T, AttributeKind Kind>
struct VectorAttributeValue final : public AttributeValue {
  using ConstructorType = const std::vector<T>&&;
  using ValueType = std::vector<T>;
  VectorAttributeValue(Symbol name, ValueType value_) : AttributeValue(name), value_(std::move(value_)) {}
  ValueType& value() {
    return value_;
  }
  AttributeKind kind() const override {
    return Kind;
  }
  std::unique_ptr<AttributeValue> clone() const override {
    return std::make_unique<VectorAttributeValue>(name, ValueType(value_));
  }

 private:
  ValueType value_;
};

using FloatAttr = ScalarAttributeValue<double, AttributeKind::f>;
using FloatsAttr = VectorAttributeValue<double, AttributeKind::fs>;
using IntAttr = ScalarAttributeValue<int64_t, AttributeKind::i>;
using IntsAttr = VectorAttributeValue<int64_t, AttributeKind::is>;
using StringAttr = ScalarAttributeValue<std::string, AttributeKind::s>;
using StringsAttr = VectorAttributeValue<std::string, AttributeKind::ss>;
using TensorAttr = ScalarAttributeValue<Tensor, AttributeKind::t>;
using TensorsAttr = VectorAttributeValue<Tensor, AttributeKind::ts>;
using GraphAttr = ScalarAttributeValue<std::shared_ptr<Graph>, AttributeKind::g>;
using GraphsAttr = VectorAttributeValue<std::shared_ptr<Graph>, AttributeKind::gs>;
using TypeProtoAttr = ScalarAttributeValue<TypeProto, AttributeKind::tp>;
using TypeProtosAttr = VectorAttributeValue<TypeProto, AttributeKind::tps>;

// CRTP so that Node which inherits Attributes can be return for
// method chaining e.g:
// Node * n = g->create(kSelect)->set_i(kOffset,3)->set_f(kValue,3.5);
// we return Derived* pointers because Nodes are normally held as pointers.
template <typename Derived>
struct Attributes {
  // NOLINTNEXTLINE(bugprone-crtp-constructor-accessibility)
  Attributes() = default;

  void copyAttributes(const Attributes& rhs) {
    // Bulk attribute replacement -- may gain or lose a g/gs attribute (e.g.
    // copying an If/Loop node's attributes onto another node), so this
    // can't unconditionally skip invalidation the way a same-kind set()
    // does. But it also shouldn't invalidate unconditionally: compare
    // "had a subgraph attribute before" against "has one after" (via
    // hasSubgraphAttribute(), cheap -- see its own doc comment) and only
    // invalidate when that actually flips, exactly like set()/
    // removeAttribute() below now do for the single-attribute case.
    const bool had_subgraph_attr = hasSubgraphAttribute();
    values_.clear();
    values_.reserve(rhs.values_.size());
    for (const auto& i : rhs.values_) {
      values_.push_back(i->clone());
    }
    if (had_subgraph_attr || hasSubgraphAttribute()) {
      This()->owningGraph()->invalidateSubgraphNodeCache();
    }
  }
  bool hasAttribute(Symbol name) const {
    return find(name, false) != values_.end();
  }
  AttributeKind kindOf(Symbol name) const {
    return (*find(name, true))->kind();
  }
  Derived* removeAttribute(Symbol name) {
    auto it = find(name, true);
    // Only a g/gs attribute's removal can change this node's
    // hasSubgraphAttribute() answer -- see Graph::invalidateSubgraphNodeCache()'s
    // comment for why invalidating on every attribute removal regardless of
    // kind (the previous behavior here) made Graph::hasAnySubgraphNode()'s
    // cache nearly write-only for any graph under active rewriting.
    const AttributeKind kind = (*it)->kind();
    values_.erase(it);
    if (kind == AttributeKind::g || kind == AttributeKind::gs) {
      This()->owningGraph()->invalidateSubgraphNodeCache();
    }
    return This();
  }
  bool hasAttributes() const {
    return !values_.empty();
  }
  // The names are returned in order, since name actually is the index.
  std::vector<Symbol> attributeNames() const {
    std::vector<Symbol> names;
    names.reserve(values_.size());
    for (const auto& a : values_)
      names.push_back(a->name);
    return names;
  }
  // Equivalent to checking whether attributeNames() contains a Symbol whose
  // kindOf() is g or gs, but without attributeNames()'s std::vector<Symbol>
  // allocation -- values_ already stores each attribute's kind directly, so
  // this only needs to walk the (typically short) existing vector. Used by
  // Graph::forEachNode's subgraph search (see ir.h's forSelfAndEachSubGraphImpl),
  // which calls this once per node in the graph to find nodes with a nested
  // subgraph; that search runs on every Value::uses()/setUniqueName() call,
  // so avoiding an allocation here matters at scale.
  bool hasSubgraphAttribute() const {
    for (const auto& a : values_) {
      const auto kind = a->kind();
      if (kind == AttributeKind::g || kind == AttributeKind::gs) {
        return true;
      }
    }
    return false;
  }

#define CREATE_ACCESSOR(Kind, method)                                           \
  Derived* method##_(Symbol name, Kind##Attr::ConstructorType v) {              \
    return set<Kind##Attr>(name, std::forward<Kind##Attr::ConstructorType>(v)); \
  }                                                                             \
  const Kind##Attr::ValueType& method(Symbol name) const {                      \
    return get<Kind##Attr>(name);                                               \
  }
  CREATE_ACCESSOR(Float, f)
  CREATE_ACCESSOR(Floats, fs)
  CREATE_ACCESSOR(String, s)
  CREATE_ACCESSOR(Strings, ss)
  CREATE_ACCESSOR(Int, i)
  CREATE_ACCESSOR(Ints, is)
  CREATE_ACCESSOR(Tensor, t)
  CREATE_ACCESSOR(Tensors, ts)
  CREATE_ACCESSOR(Graph, g)
  CREATE_ACCESSOR(Graphs, gs)
  CREATE_ACCESSOR(TypeProto, tp)
  CREATE_ACCESSOR(TypeProtos, tps)

#undef CREATE_ACCESSOR

 private:
  Derived* This() {
    return static_cast<Derived*>(this);
  }
  template <typename T>
  Derived* set(Symbol name, typename T::ConstructorType v) {
    auto it = find(name, false);
    auto nv = std::make_unique<T>(name, std::forward<typename T::ConstructorType>(v));
    const AttributeKind new_kind = nv->kind();
    if (it == values_.end()) {
      values_.push_back(std::move(nv));
      // A brand-new attribute can only make hasSubgraphAttribute() flip
      // from false to true, and only if it's itself g/gs -- adding e.g. an
      // `axis` int attribute can't turn a node into one that has a
      // subgraph. Invalidating unconditionally here (the previous
      // behavior) meant Graph::hasAnySubgraphNode()'s memoized cache
      // (see its own doc comment) was invalidated by *any* node's *any*
      // attribute set anywhere in the graph -- e.g. every attribute
      // onnx-optimizer's own passes set while rewriting -- so the very
      // next Value::uses()/setUniqueName() call anywhere else in the
      // graph (both call this on every rename/use-check) paid a full
      // O(graph size) recompute just to answer "no, there's no If/Loop/
      // Scan here", the overwhelmingly common answer for a real model.
      // Confirmed via gperftools profiling: this one recompute path
      // dominated at 45%+ of total CPU time on a ~21k-node graph under
      // onnxsim's own fixed-point rewriting.
      if (new_kind == AttributeKind::g || new_kind == AttributeKind::gs) {
        This()->owningGraph()->invalidateSubgraphNodeCache();
      }
    } else {
      const AttributeKind old_kind = (*it)->kind();
      *it = std::move(nv);
      // Replacing an attribute can only change this node's
      // hasSubgraphAttribute() answer if the kind actually changes *and*
      // one side of that change is g/gs (same-kind replacement, e.g.
      // swapping an If node's `then_branch` for a different Graph, or
      // replacing one `axis` int with another, never changes whether this
      // node has *a* subgraph attribute -- only *which* Graph/value it
      // holds). See the new-attribute branch above for why this
      // conditional exists instead of invalidating unconditionally.
      if (old_kind != new_kind &&
          (old_kind == AttributeKind::g || old_kind == AttributeKind::gs || new_kind == AttributeKind::g ||
           new_kind == AttributeKind::gs)) {
        This()->owningGraph()->invalidateSubgraphNodeCache();
      }
    }
    return This();
  }
  template <typename T>
  typename T::ValueType& get(Symbol name) const {
    auto it = find(name, true);
    T* child = static_cast<T*>(it->get());
    return child->value();
  }
  using AVPtr = AttributeValue::Ptr;
  // NB: For determinism, we use a vector rather than a hash map.  This does
  // mean that lookups are O(n), so you shouldn't use Attributes to store
  // a big pile of messages.
  std::vector<AVPtr> values_;
  using iterator = std::vector<AVPtr>::iterator;
  iterator find(Symbol name, bool required) {
    auto it = std::find_if(values_.begin(), values_.end(), [&](const AVPtr& v) { return v->name == name; });
    ONNX_ASSERT(!required || it != values_.end())
    return it;
  }
  using const_iterator = std::vector<AVPtr>::const_iterator;
  const_iterator find(Symbol name, bool required) const {
    auto it = std::find_if(values_.begin(), values_.end(), [&](const AVPtr& v) { return v->name == name; });
    ONNX_ASSERTM(!required || it != values_.end(), "required undefined attribute '", name.toString(), "'")
    return it;
  }
};

// Each use is represented by this type, see Node::uses()
// 'user' is the consumer of the value, offset is the index into
// 'user's input this where the produces will be found.
struct Use final {
  Use(Node* user, size_t offset) : user(user), offset(offset) {}
  Node* user;
  size_t offset;
};

static inline bool operator==(const Use& a, const Use& b) {
  return a.user == b.user && a.offset == b.offset;
}

// the list types are intentionally simple, but we type-def
// them here so if we need to change them, refactoring will be easier
using node_list = std::vector<Node*>;
using value_list = std::vector<Value*>;
using use_list = std::vector<Use>;
using NodeKind = Symbol;

struct Value final {
  ONNX_DISALLOW_COPY_AND_ASSIGNMENT(Value);
  Value(Node& node, size_t offset);
  Value(Value&&) = default;
  Value& operator=(Value&&) = default;
  ~Value() = default;

 private:
  friend struct Node;
  friend struct Graph;
  Node* node_;
  size_t offset_;
  // Fallback numeric id backing uniqueName()'s no-real-name branch, minted
  // eagerly in the constructor (see Value::Value() below) via
  // Graph::getNextUnique(). Eager assignment is what keeps this
  // deterministic: it is tied to Value *construction* order, which is fixed
  // for a given input model, rather than to whichever code path first
  // happens to read uniqueName() on an unnamed Value -- e.g. an
  // unordered_map/set of Value* elsewhere whose iteration order depends on
  // heap addresses, which differ from run to run. (An earlier version of
  // this fix computed the id lazily, on first read, purely to dodge
  // getNextUnique()'s cost; that made two otherwise-identical runs of the
  // same input model produce different "_v_<n>" fallback names whenever
  // such a container's iteration order changed run to run -- see
  // Graph::used_names_ below for how getNextUnique() itself stays cheap
  // without needing to give up eager, deterministic assignment.)
  size_t unique_ = 0; // unique id
  size_t stage_ = 0; // 0-forward, 1-backward, 2-double-backward,...
  use_list uses_in_current_graph_;
  bool has_unique_name_{false};
  std::string unique_name_;
  bool has_doc_string_{false};
  std::string doc_string_;
  int32_t elem_type_{ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED};
  bool has_sizes_{false};
  std::vector<Dimension> sizes_;
  std::unique_ptr<TypeProto> type_;

 public:
  Value* setElemType(int32_t elem_type) {
    elem_type_ = elem_type;
    return this;
  }
  int32_t elemType() const {
    return elem_type_;
  }
  bool has_sizes() const {
    return has_sizes_;
  }
  Value* setSizes(std::vector<Dimension> sizes) {
    has_sizes_ = true;
    sizes_ = std::move(sizes);
    return this;
  }
  Value* wipeSizes() {
    has_sizes_ = false;
    sizes_ = std::vector<Dimension>();
    return this;
  }
  const std::vector<Dimension>& sizes() const {
    return sizes_;
  }
  size_t unique() const {
    return unique_;
  }
  bool has_unique_name() const {
    return has_unique_name_;
  }
  std::string uniqueName() const {
    if (has_unique_name())
      return unique_name_;
    return toVarName(unique());
  }
  Value* setUniqueName(const std::string& name, bool update_related_names = true);
  bool has_doc_string() const {
    return has_doc_string_;
  }
  const std::string& docString() const {
    return doc_string_;
  }
  Value* setDocString(std::string doc_string) {
    has_doc_string_ = true;
    doc_string_ = std::move(doc_string);
    return this;
  }
  Value* setStage(size_t s) {
    stage_ = s;
    return this;
  }
  size_t stage() const {
    return stage_;
  }
  Node* node() {
    return node_;
  }
  size_t offset() const {
    return offset_;
  }
  const Node* node() const {
    return node_;
  }
  Graph* owningGraph();
  const Graph* owningGraph() const;
  use_list uses() const;

  // Cheap (O(1)) equivalent of ``!uses().empty()`` that only sees uses from
  // nodes directly in this value's owning graph, not uses().'s additional
  // owningGraph()->forEachNode() scan for kCaptured placeholders in nested
  // If/Loop/Scan subgraph bodies (see uses()'s definition below). Safe to use
  // in place of ``!uses().empty()`` only when the caller has separately
  // established that no such capture is possible -- e.g. the whole graph
  // tree contains no node with a subgraph attribute at all, which a caller
  // visiting every node in a loop (as DCE/CSE do) can check once up front
  // instead of paying uses()'s full-graph subgraph scan on every node.
  bool hasUsesInCurrentGraph() const {
    return !uses_in_current_graph_.empty();
  }

  // Replaces all uses of this node with 'newValue'.
  //
  // Given:   %3 = f(%1, %2)
  //          %4 = g(%3)
  //          %5 = h(%3, %3)
  // Execute: %3.replaceAllUsesWith(%6)
  // Result:  %3 = f(%1, %2)
  //          %4 = g(%6)
  //          %5 = h(%6, %6)
  void replaceAllUsesWith(Value* newValue);

  Value* copyMetadata(const Value* from) {
    setElemType(from->elemType());
    if (from->has_sizes()) {
      setSizes(from->sizes());
    } else {
      // Unconditionally calling setSizes(from->sizes()) here would copy an
      // *empty* vector when `from` has no known rank (has_sizes() false),
      // which setSizes() would then mark as a definite, present rank-0
      // (scalar) shape -- silently corrupting "rank unknown" into "rank 0"
      // for any Value copied from one with no shape info yet.
      wipeSizes();
    }
    if (from->has_unique_name()) {
      setUniqueName(from->uniqueName());
    }
    if (from->has_doc_string()) {
      setDocString(from->docString());
    }
    return this;
  }

  std::unique_ptr<TypeProto>& type() {
    return type_;
  }
};

struct Node : public Attributes<Node> {
  ONNX_DISALLOW_COPY_AND_ASSIGNMENT(Node);
  friend struct Graph;
  friend struct Value;
  friend graph_node_list;
  friend const_graph_node_list;
  friend graph_node_list_iterator;
  friend const_graph_node_list_iterator;

 private:
  // each node but Return/Param
  // is associated with exactly one place in the node list...
  // of the graph_
  // this circular is a doubly-linked list, the Return node is used as the sentinel for the beginning and end of the
  // list such that the list never has null pointers next_in_graph[0] is next pointer next_in_graph[1] is prev pointer
  // using an array to allow the same iterator class for forward and reverse node lists
  // This list represents a topological sort

  std::array<Node*, 2> next_in_graph{nullptr, nullptr};
  Node*& next() {
    return next_in_graph[kNextDirection];
  }
  Node*& prev() {
    return next_in_graph[kPrevDirection];
  }
  Node* const& next() const {
    return next_in_graph[kNextDirection];
  }
  Node* const& prev() const {
    return next_in_graph[kPrevDirection];
  }

  const NodeKind kind_;
  std::vector<Value*> inputs_;
  std::vector<Value*> outputs_;
  Graph* graph_;
  size_t stage_;
  bool has_name_{false};
  std::string name_;
  bool has_domain_{false};
  std::string domain_;
  bool has_doc_string_{false};
  std::string doc_string_;
  bool has_overload_{false};
  std::string overload_;

  // Constructed only by the friend Graph factory, so every Node stays graph-owned.
  Node(Graph& graph, NodeKind kind); // defined after graph

 public:
  bool has_name() const {
    return has_name_;
  }
  const std::string& name() const {
    return name_;
  }
  void setName(std::string name) {
    has_name_ = true;
    name_ = std::move(name);
  }
  bool has_domain() const {
    return has_domain_;
  }
  const std::string& domain() const {
    return domain_;
  }
  void setDomain(std::string domain) {
    has_domain_ = true;
    domain_ = std::move(domain);
  }
  bool has_overload() const {
    return has_overload_;
  }
  const std::string& overload() const {
    return overload_;
  }
  void setOverload(std::string overload) {
    has_overload_ = true;
    overload_ = std::move(overload);
  }
  bool has_doc_string() const {
    return has_doc_string_;
  }
  const std::string& docString() const {
    return doc_string_;
  }
  void setDocString(std::string doc_string) {
    has_doc_string_ = true;
    doc_string_ = std::move(doc_string);
  }
  NodeKind kind() const {
    return kind_;
  }
  Graph* owningGraph() {
    return graph_;
  }
  const Graph* owningGraph() const {
    return graph_;
  }
  size_t stage() const {
    return stage_;
  }
  Node* setStage(size_t s) {
    stage_ = s;
    return this;
  }
  // NB: This returns an ArrayRef; that means that it will
  // get invalidated if you resize inputs (e.g., using addInput)
  // We can't return a std::vector<Node*>& because there's no
  // way to soundly cast to std::vector<const Node*> (an insane
  // implementation of std::vector could make this representationally
  // different.)
  ArrayRef<Value*> inputs() {
    return inputs_;
  }
  ArrayRef<const Value*> inputs() const {
    // Vectors are not convertible in const-ness of elements, but
    // raw pointers are.
    return {inputs_.data(), inputs_.size()};
  }
  // NB: This returns an ArrayRef; that means that it will
  // get invalidated if you resize inputs (e.g., using addInput)
  // We can't return a std::vector<Node*>& because there's no
  // way to soundly cast to std::vector<const Node*> (an insane
  // implementation of std::vector could make this representationally
  // different.)
  ArrayRef<Value*> outputs() {
    return outputs_;
  }
  ArrayRef<const Value*> outputs() const {
    // Vectors are not convertible in const-ness of elements, but
    // raw pointers are.
    return {outputs_.data(), outputs_.size()};
  }
  bool hasUses() const {
    for (const auto* o : outputs()) {
      if (!o->uses().empty())
        return true;
    }
    return false;
  }
  // Cheap equivalent of hasUses() -- see Value::hasUsesInCurrentGraph()'s
  // comment for exactly what this does and doesn't see.
  bool hasUsesInCurrentGraph() const {
    for (const auto* o : outputs()) {
      if (o->hasUsesInCurrentGraph())
        return true;
    }
    return false;
  }
  void replaceAllUsesWith(Node* n) {
    ONNX_ASSERT(outputs().size() == n->outputs().size())
    size_t nOutputs = outputs().size();
    for (size_t i = 0; i < nOutputs; i++) {
      outputs()[i]->replaceAllUsesWith(n->outputs()[i]);
    }
  }
  // lots of things like chunk have a single input or single output, so we have a
  // helper to make accessing it easier
  Value* input() {
    ONNX_ASSERT(inputs_.size() == 1)
    return inputs_.at(0);
  }
  Value* output() {
    ONNX_ASSERT(outputs_.size() == 1)
    return outputs_.at(0);
  }
  const Value* input() const {
    ONNX_ASSERT(inputs_.size() == 1)
    return inputs_.at(0);
  }
  const Value* output() const {
    ONNX_ASSERT(outputs_.size() == 1)
    return outputs_.at(0);
  }
  // Access a particular input.  This is a checked index.
  Value* input(size_t i) {
    return inputs_.at(i);
  }
  const Value* input(size_t i) const {
    return inputs_.at(i);
  }

  // Graphs

  // Note [Topological invariant]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // We always maintain an up-to-date topological ordering of all nodes via
  // the next()/prev() links.  All transformations to graphs must preserve
  // this topological ordering: for example, it is only valid to 'addInput'
  // with an input which is topologically before the current node.
  //
  // Usually, it is obvious whether or not topological order is maintained;
  // for example, if you are adding nodes to the end of the topsort, it's
  // impossible for them to refer to inputs that are not in the topsort.
  // If it is not obvious, please comment accordingly.

  // Add 'node' as an input to 'this' at the end of existing
  // arguments.  Returns the added node for ease of chaining.
  //
  // Given:   %3 = f(%1, %2)
  // Execute: %3.addInput(%4)
  // Result:  %3 = f(%1, %2, %4)
  Value* addInput(Value* node) {
    ONNX_ASSERT(graph_ == node->owningGraph())
    node->uses_in_current_graph_.emplace_back(this, inputs_.size());
    inputs_.push_back(node);
    return node;
  }

  // Replace the input of 'this' at position 'i' with
  // 'newValue', returning the old node.
  //
  // Given:   %3 = f(%1, %2)
  // Execute: %3.replaceInput(1, %4)
  // Result:  %3 = f(%1, %4)
  Value* replaceInput(size_t i, Value* newValue) {
    ONNX_ASSERT(newValue->owningGraph() == graph_)
    Value* old = dropInput(i);
    inputs_[i] = newValue;
    newValue->uses_in_current_graph_.emplace_back(this, i);
    return old;
  }

  // Replace all occurrences of 'from' in the inputs of this
  // node with 'to'. Corresponds to llvm's replaceUsesOfWith.
  //
  // Given:   %3 = f(%1, %2, %1)
  // Execute: %3.replaceInputWith(%1, %4)
  // Result:  %3 = f(%4, %2, %4)
  void replaceInputWith(Value* from, Value* to) {
    ONNX_ASSERT(from->owningGraph() == graph_)
    ONNX_ASSERT(to->owningGraph() == graph_)
    size_t i = 0;
    for (const auto* input : inputs()) {
      if (input == from)
        replaceInput(i, to);
      i++;
    }
  }

  Value* addOutput();

  void eraseOutput(size_t i);

  // Insert unattached 'this' node after 'n' in the topological order.
  // Returns this (for chaining).
  //
  // Given:   %3 = f(%1, %2)
  //          %4 = g(%3)
  // and unattached: %5 = h(%1)
  // Execute: %5.insertBefore(%4)
  // Result:  %3 = f(%1, %2)
  //          %5 = h(%1)
  //          %4 = g(%3)
  Node* insertBefore(Node* n) {
    ONNX_ASSERT(n->inGraphList())
    insertAfter(n->prev());
    return this;
  }

  // Insert unattached 'this' node after 'n' in the topological order.
  // Returns this (for chaining).
  //
  // Given: %3 = f(%1, %2)
  //        %4 = g(%3)
  // and unattached: %5 = h(%1)
  // Execute: %5.insertAfter(%4)
  // Result:  %3 = f(%1, %2)
  //          %4 = g(%3)
  //          %5 = h(%1)
  Node* insertAfter(Node* n) {
    ONNX_ASSERT(!inGraphList() && n->inGraphList())
    Node* next = n->next();
    n->next() = this;
    this->prev() = n;
    this->next() = next;
    next->prev() = this;
    return this;
  }

  // Move 'this' (already in the graph) after 'n' in the topological order.
  //
  // Given: %2 = f(%1)
  //        %3 = g(%1)
  // Execute: %2.moveAfter(%3)
  // Result: %3 = g(%1)
  //         %2 = f(%1)
  //
  void moveAfter(Node* n) {
    removeFromList();
    insertAfter(n);
  }

  // Move a node 'n' (already in the graph) before 'this' in the topological order.
  //
  // Given: %2 = f(%1)
  //        %3 = g(%1)
  // Execute: %3.moveBefore(%2)
  // Result: %3 = g(%1)
  //         %2 = f(%1)
  void moveBefore(Node* n) {
    removeFromList();
    insertBefore(n);
  }

  // Remove the input at 'i' from this node.
  //
  // WARNING: This is O(n) in the number of inputs, so avoid repeatedly calling
  // removeInput.
  //
  // Given: %3 = f(%1, %2)
  // Execute: %3.removeInput(1)
  // Result: %3 = f(%1)
  void removeInput(size_t i) {
    dropInput(i);
    // everything after this input shifts left,
    // so we need to update their use offsets to match
    for (size_t j = i + 1; j < inputs_.size(); j++) {
      auto it = findUseForInput(j);
      it->offset--;
    }
    inputs_.erase(inputs_.begin() + static_cast<std::ptrdiff_t>(i));
  }

  // Remove all inputs from a node.
  //
  // Given: %3 = f(%1, %2)
  // Execute: %3.removeAllInputs()
  // Result: %3 = f()
  void removeAllInputs() {
    for (size_t i = 0; i < inputs().size(); ++i)
      dropInput(i);
    inputs_.clear();
  }

  // Check whether this node is before node n in the graph.
  bool isBefore(const Node* n);

  // iterators of the node list starting at this node
  // useful for resuming a search starting at this node
  graph_node_list_iterator iterator();
  graph_node_list_iterator reverseIterator();
  const_graph_node_list_iterator iterator() const;
  const_graph_node_list_iterator reverseIterator() const;

  // Remove 'this' from the instruction list and deallocate it.
  //
  // Invariant: no outputs of 'this' may have any uses.
  //
  // Given: %2 = f(%1)
  //        %3 = g(%1)
  // Execute: %2.destroy()
  // Result: %3 = g(%1)
  void destroy();

  // Dynamically cast this node to the subclass indicated by the
  // template variable, returning nullptr if the cast is invalid..
  //
  // Example usage: if(auto s = n.cast<Select>()) { ... }
  //
  template <typename T>
  T* cast() {
    if (T::Kind == kind())
      return static_cast<T*>(this);
    return nullptr;
  }
  template <typename T>
  const T* cast() const {
    if (T::Kind == kind())
      return static_cast<const T*>(this);
    return nullptr;
  }
  template <typename T>
  T* expect() {
    ONNX_ASSERTM(T::Kind == kind(), "expected a ", T::Kind.toString(), " but found a ", kind().toString())
    return static_cast<T*>(this);
  }
  template <typename T>
  const T* expect() const {
    ONNX_ASSERTM(T::Kind == kind(), "expected a ", T::Kind.toString(), " but found a ", kind().toString())
    return static_cast<const T*>(this);
  }

  virtual ~Node() = default;

 private:
  // Lookup iterator in use list of _input i_ that corresponds to its use of _this_
  use_list::iterator findUseForInput(size_t i) {
    auto& input_uses = inputs_[i]->uses_in_current_graph_;
    // O(N) on the use list, but unless we get nodes with +100 uses
    // vector traversal still is probably faster than linked list
    auto use_it = std::find(input_uses.begin(), input_uses.end(), Use(this, i));
    ONNX_ASSERT(use_it != input_uses.end())
    return use_it;
  }

  // remove the use of input i, this sets input i to nullptr, but
  // is only used internally to Node before setting it to a new value
  // or erasing the entry from the list.
  Value* dropInput(size_t i) {
    ONNX_ASSERT(i < inputs_.size())
    auto* input_node = inputs_[i];
    auto use_it = findUseForInput(i);
    input_node->uses_in_current_graph_.erase(use_it);
    inputs_[i] = nullptr;
    return input_node;
  }

  bool inGraphList() const {
    ONNX_ASSERT(next() != nullptr || prev() == nullptr)
    return next() != nullptr;
  }
  void removeFromList() {
    ONNX_ASSERT(inGraphList())
    Node* next = this->next();
    Node* prev = this->prev();
    prev->next() = next;
    next->prev() = prev;
    this->next() = nullptr;
    this->prev() = nullptr;
  }
};

// A class with the same properties as OperatorSetIdProto, but without protobuf
// overhead, resulting in a simpler and more readable workflow.
class OpSetID final {
 private:
  std::string domain_;
  int64_t version_;

 public:
  explicit OpSetID(const OperatorSetIdProto& proto) : domain_(proto.domain()), version_(proto.version()) {}

  // Default Domain Constructor
  explicit OpSetID(const int64_t version) : version_(version) {}

  explicit OpSetID(std::string domain, int64_t version) : domain_(std::move(domain)), version_(version) {}

  // target must be in the form "<domain>$<version>"
  std::string toString() const {
    return domain_ + "$" + ONNX_NAMESPACE::to_string(version_);
  }

  // target must be in the form "<domain>$<version>"
  static OpSetID fromString(const std::string& target) {
    ONNX_TRY {
      auto pos = target.find('$');
      if (pos == std::string::npos) {
        ONNX_THROW("Invalid OpSetID string '", target, "': must be in the form \"<domain>$<version>\"");
      }
      std::string new_domain = target.substr(0, pos);
      const char* version_start = target.data() + pos + 1;
      const char* version_end = target.data() + target.size();
      int64_t new_version = 0;
      auto result = std::from_chars(version_start, version_end, new_version);
      if (result.ec != std::errc{} || result.ptr != version_end) {
        ONNX_THROW("Invalid OpSetID string '", target, "': must be in the form \"<domain>$<version>\"");
      }
      return OpSetID(new_domain, new_version);
    }
    ONNX_CATCH(const std::runtime_error& e) {
      ONNX_HANDLE_EXCEPTION([&]() { ONNX_ASSERTM(false, "Error in fromString: ", e.what()) });
    }

    // The control will never reach here.
    // In the default build where exceptions are turned on in case of any error
    // the control will enter catch block where an exception will be thrown again.
    // In case of "no exception build" the code aborts at the site of first exception.
    // Adding this to appease the warning "control may reach end of non-void function"
    // as the mac build fails when ONNX_WERROR==ON
    return OpSetID("", 0);
  }

  const std::string& domain() const {
    return domain_;
  }

  int64_t version() const {
    return version_;
  }

  void incrementVersion(int64_t step) {
    version_ += step;
  }

  void setVersion(int64_t newVal) {
    version_ = newVal;
  }
};

struct Graph final {
  ONNX_DISALLOW_COPY_AND_ASSIGNMENT(Graph);
  friend struct Node;
  friend struct Value;

 private:
  // only used to keep track of allocated nodes
  // actual representation of Graph is done with
  // inputs, outputs, nodes

  std::unordered_map<const Node*, std::unique_ptr<const Node>> all_nodes;
  std::unordered_map<const Value*, std::unique_ptr<const Value>> all_values;
  size_t next_unique_{0};

  // Reference-counts every name currently displayed by this graph's own
  // values (node inputs/outputs, graph inputs -- including their default,
  // never-explicitly-set "_v_<n>" display names, registered at Value
  // construction) and initializers, so isNameUnique() can answer with a
  // single hash lookup instead of the O(graph size) scan it used to run on
  // every call (a std::find over initializer_names_, plus, per node, two
  // more std::find_if scans over its inputs/outputs) -- the cost that made
  // Graph::getNextUnique() O(V) per call and graph Import O(V^2) overall.
  // A count (not a plain set) because one name can have more than one live
  // holder at once -- e.g. an initializer's own initializer_names_ entry
  // and the Graph Value that mirrors it (addInitializerAndCreateValue), or
  // an output value mid-rename in Value::replaceAllUsesWith -- and the name
  // must stay reserved until every holder releases it via useName()/
  // releaseName(), not just the first one to let go. This is NOT extended to
  // names used inside subgraph attributes (If/Loop/Scan bodies): each nested
  // subgraph is its own Graph with its own used_names_, and isNameUnique()
  // recurses into those explicitly below.
  std::unordered_map<std::string, size_t> used_names_;

  void useName(const std::string& name) {
    ++used_names_[name];
  }
  void releaseName(const std::string& name) {
    auto it = used_names_.find(name);
    if (it == used_names_.end()) {
      return;
    }
    if (--it->second == 0) {
      used_names_.erase(it);
    }
  }

  // Memoizes hasAnySubgraphNode() below (whether this graph's own all_nodes,
  // not recursing into subgraphs, contains a node with a g/gs attribute).
  // `subgraph_node_cache_valid_` false means "unknown, recompute";
  // invalidateSubgraphNodeCache() is the only writer of that -- see its own
  // comment for why every place that can add/remove a subgraph attribute
  // calls it. mutable: computed lazily from const-qualified callers
  // (forSelfAndEachSubGraphImpl's const overload).
  mutable bool subgraph_node_cache_valid_{false};
  mutable bool has_subgraph_node_cached_{false};

  size_t new_node_stage_{0};

  // holds outputs in a way that can be reflected
  // as a Use object
  // also used as the beginning/end of the circular node list to avoid
  // having corner cases where the list is empty.
  Node* const output_;
  Node* const input_;
  // Create an independent node list for those initializers do not exist in input
  Node* const initializer_node_;

  // unique_ptr<Tensor> (not a plain Tensor) so a Tensor's address is stable
  // across other initializers being inserted/erased in this vector -- other
  // code (e.g. onnx-optimizer's TensorContentDigest cache) keys a cache on
  // a Tensor* for the scope of one call, which a reallocating/shifting
  // std::vector<Tensor> would silently invalidate for any elements that
  // moved. See onnxsim issue #633's follow-up investigation.
  std::vector<std::unique_ptr<Tensor>> initializers_;
  std::vector<std::string> initializer_names_;

  bool has_name_{false};
  std::string name_;
  bool has_doc_string_{false};
  std::string doc_string_;

  std::vector<OpSetID> opset_versions_;

  bool isNameUnique(const std::string& name) const {
    // O(1) check against this graph's own initializer/value names, replacing
    // what used to be a linear scan of initializer_names_ plus, per node,
    // two more linear string-comparison scans over its inputs and outputs.
    if (used_names_.count(name)) {
      return false;
    }
    // hasAnySubgraphNode() is the same memoized "does this graph contain any
    // If/Loop/Scan-style node at all" check forSelfAndEachSubGraphImpl()
    // already uses -- O(1) for the overwhelming common case (no
    // control-flow ops), so this recursion is skipped entirely instead of
    // walking every node just to conclude there is nothing to recurse into.
    if (!hasAnySubgraphNode()) {
      return true;
    }
    for (const auto& node_entry : all_nodes) {
      const Node* node = node_entry.first;
      if (!node->hasSubgraphAttribute()) {
        continue;
      }
      for (const auto& attr : node->attributeNames()) {
        if (node->kindOf(attr) == AttributeKind::g) {
          if (!node->g(attr)->isNameUnique(name)) {
            return false;
          }
        } else if (node->kindOf(attr) == AttributeKind::gs) {
          for (const auto& subgraph : node->gs(attr)) {
            if (!subgraph->isNameUnique(name)) {
              return false;
            }
          }
        }
      }
    }
    return true;
  }

  // Same traversal as isNameUnique() above (initializer_names_, every node's
  // inputs/outputs via uniqueName() -- which falls back to a derived name for
  // an unnamed Value, so this must run for every Value, not just named ones
  // -- recursing into g/gs subgraph attributes), but collecting every name
  // seen into `out` instead of checking one candidate at a time. Used by
  // reserveUniqueNames() below to pay this scan once for a whole batch of
  // names instead of once per name.
  void collectAllNames(std::unordered_set<std::string>& out) const {
    out.insert(initializer_names_.cbegin(), initializer_names_.cend());
    for (const auto& node_entry : all_nodes) {
      const Node* node = node_entry.first;
      for (const auto& attr : node->attributeNames()) {
        if (node->kindOf(attr) == AttributeKind::g) {
          node->g(attr)->collectAllNames(out);
        } else if (node->kindOf(attr) == AttributeKind::gs) {
          for (const auto& subgraph : node->gs(attr)) {
            subgraph->collectAllNames(out);
          }
        }
      }
      for (const Value* v : node->inputs()) {
        out.insert(v->uniqueName());
      }
      for (const Value* v : node->outputs()) {
        out.insert(v->uniqueName());
      }
    }
  }

 public:
  Graph() : output_(initOutput(create(kReturn, 0))), input_(create(kParam, 0)), initializer_node_(create(kParam, 0)) {}

  // Called by Attributes<Derived>::set()/removeAttribute()/copyAttributes()
  // (via Node::owningGraph()) whenever a node's attributes change in a way
  // that could add or remove a g/gs (subgraph) attribute -- see
  // hasAnySubgraphNode()'s comment for what this protects.
  //
  // set()/removeAttribute()/copyAttributes() each narrow this to only fire
  // when a g/gs attribute is actually involved (never skip it when one
  // *might* have changed -- a stale "no subgraph nodes" cache would make
  // forSelfAndEachSubGraphImpl silently skip a real subgraph, e.g. losing a
  // kCaptured value's rename in setUniqueName, not just cost performance).
  // An earlier version of this function fired unconditionally, on *any*
  // attribute mutation regardless of kind, on the assumption that attribute
  // mutation is rare relative to the hot paths (Value::uses()/
  // setUniqueName()/replaceAllUsesWith()) this cache exists to speed up.
  // That assumption doesn't hold under active rewriting: onnx-optimizer's
  // own passes set/remove plenty of non-subgraph attributes per round, each
  // one invalidating the cache and forcing the next uses()/setUniqueName()
  // call anywhere in the graph to pay a full O(graph size) recompute just
  // to reconfirm "no If/Loop/Scan here" -- confirmed via gperftools
  // profiling to dominate at 45%+ of total CPU time on a ~21k-node graph
  // under onnxsim's fixed-point rewriting.
  void invalidateSubgraphNodeCache() const {
    subgraph_node_cache_valid_ = false;
  }

  bool has_doc_string() const {
    return has_doc_string_;
  }
  const std::string& docString() const {
    return doc_string_;
  }
  void setDocString(std::string doc_string) {
    has_doc_string_ = true;
    doc_string_ = std::move(doc_string);
  }

  void addInitializer(Tensor& initializer) {
    if (initializer.name().empty()) {
      initializer.setName(getNextUniqueName());
    }
    // Heap-allocated per element (not stored inline in the vector) so a
    // Tensor's address stays valid across other initializers being
    // inserted/erased -- callers (e.g. onnx-optimizer's
    // TensorContentDigest cache) may key a cache on a Tensor* for the
    // duration of a single pass call, which a reallocating/shifting
    // std::vector<Tensor> would silently invalidate.
    initializers_.push_back(std::make_unique<Tensor>(initializer));
    initializer_names_.push_back(initializer.name());
    useName(initializer.name());
  }

  // Move-taking overload. The lvalue overload above always copies
  // initializer's fields (raw_data included) into the heap-allocated Tensor
  // it stores; a caller whose Tensor is a scratch value it is discarding
  // right after this call (e.g. a pass rewriting a Constant node's tensor
  // into an initializer) can pass it as an rvalue instead and skip that
  // copy. Existing call sites are unaffected: passing an lvalue still binds
  // to the copying overload above, since it is a better match for an lvalue
  // than a Tensor&& parameter.
  void addInitializer(Tensor&& initializer) {
    if (initializer.name().empty()) {
      initializer.setName(getNextUniqueName());
    }
    // Read the name before the move below invalidates initializer's own
    // fields (Tensor's move constructor moves name_ too).
    initializer_names_.push_back(initializer.name());
    useName(initializer_names_.back());
    initializers_.push_back(std::make_unique<Tensor>(std::move(initializer)));
  }

  // For IR >= 4, initializer is not required to exist in input
  // Add initializer into initializer node list and return its Value
  Value* addInitializerAndCreateValue(Tensor& initializer) {
    addInitializer(initializer);
    auto* init_value = initializer_node_->addOutput();
    std::vector<Dimension> dim_sizes{initializer.sizes().cbegin(), initializer.sizes().cend()};
    init_value->setUniqueName(initializer.name());
    init_value->setSizes(dim_sizes);
    init_value->setElemType(initializer.elem_type());
    return init_value;
  }

  // Move-taking counterpart to the addInitializer(Tensor&&) overload above --
  // same rationale (skip the copy of raw_data / typed-data fields for a
  // Tensor the caller is discarding). Everything the returned Value needs
  // (name/sizes/elem_type) is read before initializer is moved from.
  Value* addInitializerAndCreateValue(Tensor&& initializer) {
    if (initializer.name().empty()) {
      initializer.setName(getNextUniqueName());
    }
    auto* init_value = initializer_node_->addOutput();
    std::vector<Dimension> dim_sizes{initializer.sizes().cbegin(), initializer.sizes().cend()};
    init_value->setUniqueName(initializer.name());
    init_value->setSizes(dim_sizes);
    init_value->setElemType(initializer.elem_type());
    initializer_names_.push_back(initializer.name());
    useName(initializer_names_.back());
    initializers_.push_back(std::make_unique<Tensor>(std::move(initializer)));
    return init_value;
  }

  void eraseInitializer(const std::string& name) {
    initializers_.erase(
        std::remove_if(
            initializers_.begin(),
            initializers_.end(),
            [&name](const std::unique_ptr<Tensor>& initializer) { return initializer->name() == name; }),
        initializers_.end());
    initializer_names_.erase(
        std::remove(initializer_names_.begin(), initializer_names_.end(), name), initializer_names_.end());
    releaseName(name);
    for (size_t i = 0; i < initializer_node_->outputs().size(); i++) {
      if (initializer_node_->outputs()[i]->uniqueName() == name) {
        initializer_node_->eraseOutput(i);
        break;
      }
    }
  }
  void clearInitializers() {
    for (const auto& name : initializer_names_) {
      releaseName(name);
    }
    initializers_.clear();
    initializer_names_.clear();
  }
  // Stored as unique_ptr<Tensor> now (see initializers_'s declaration), so
  // this is no longer a plain std::vector<Tensor>&: callers that used to
  // bind `const auto&` to a Tensor in a range-for now get a
  // `const unique_ptr<Tensor>&` and need one extra `->`/`*` to reach the
  // Tensor, same as any other owning-pointer container.
  const std::vector<std::unique_ptr<Tensor>>& initializers() const {
    return initializers_;
  }
  // Mutable access, used by the ir_pb_converter's "consuming" Export path to
  // move each initializer's raw bytes into the output TensorProto instead of
  // copying them. Only safe for a caller that discards this Graph right
  // after exporting it (see ExportModelProto's consume_tensor_data param).
  std::vector<std::unique_ptr<Tensor>>& initializers_mutable() {
    return initializers_;
  }
  const std::vector<std::string>& initializer_names() const {
    return initializer_names_;
  }
  // Returns nullptr if no initializer named `name` exists (previously an
  // iterator sentinel; every caller only ever unconditionally dereferenced
  // it, so a nullable pointer is both simpler and a closer match for how
  // it's actually used).
  const Tensor* getInitializer(const std::string& name) const {
    for (const auto& initializer : initializers_) {
      if (name == initializer->name()) {
        return initializer.get();
      }
    }
    return nullptr;
  }
  bool is_constant_initializer(const Value* value) const {
    return value->node() == initializer_node_;
  }
  ArrayRef<Value*> inputs() {
    return input_->outputs();
  }
  ArrayRef<const Value*> inputs() const {
    const auto& inputs = input_->outputs();
    return {inputs.data(), inputs.size()};
  }
  ArrayRef<Value*> outputs() {
    return output_->inputs();
  }
  ArrayRef<const Value*> outputs() const {
    return static_cast<const Node*>(output_)->inputs();
  }
  graph_node_list nodes() {
    return graph_node_list(output_, kNextDirection);
  }
  const_graph_node_list nodes() const {
    return const_graph_node_list(output_, kNextDirection);
  }

  std::vector<OpSetID>& opset_versions_mutable() {
    return opset_versions_;
  }

  size_t getNextUnique() {
    // isNameUnique() below is now a used_names_ hash lookup (plus, rarely, a
    // recursion into subgraph attributes) -- it no longer calls uniqueName()
    // on other Values, so this can't recurse back into getNextUnique() on
    // this same Graph. Still returns the locally-captured candidate rather
    // than re-reading next_unique_, simply because that's the id this call
    // actually verified unique.
    size_t candidate = ++next_unique_;
    std::string next_unique_name = toVarName(candidate);
    while (!isNameUnique(next_unique_name)) {
      candidate = ++next_unique_;
      next_unique_name = toVarName(candidate);
    }
    return candidate;
  }

  std::string getNextUniqueName() {
    return toVarName(getNextUnique());
  }

  // Reserves `count` distinct names, each guaranteed unique against every
  // name currently used anywhere in the graph tree -- matching exactly what
  // `count` consecutive getNextUniqueName() calls would produce, but paying
  // isNameUnique()'s full graph (and subgraph) scan once for the whole batch
  // (via collectAllNames() above) instead of once per name via
  // getNextUnique() -> isNameUnique(). Intended for passes that mint many
  // fresh names in one runPass() call (see onnxsim issue #651's follow-up,
  // where extract_constant_to_initializer's one-getNextUniqueName()-call-
  // per-match pattern made that pass accidentally quadratic); a single
  // getNextUniqueName() call remains simplest for the common one-name case.
  std::vector<std::string> reserveUniqueNames(size_t count) {
    std::vector<std::string> names;
    names.reserve(count);
    if (count == 0) {
      return names;
    }
    std::unordered_set<std::string> used;
    collectAllNames(used);
    for (size_t i = 0; i < count; i++) {
      std::string name = toVarName(++next_unique_);
      while (used.count(name) != 0) {
        name = toVarName(++next_unique_);
      }
      used.insert(name);
      names.push_back(std::move(name));
    }
    return names;
  }

  // These invocations of begin() on output of function are OK
  // because graph_node_list is non-owning, so it doesn't matter
  // if it immediately dies after the invocation.
  graph_node_list_iterator begin() {
    return nodes().begin();
  }
  const_graph_node_list_iterator begin() const {
    return nodes().begin();
  }
  graph_node_list_iterator end() {
    return nodes().end();
  }
  const_graph_node_list_iterator end() const {
    return nodes().end();
  }
  graph_node_list_iterator rbegin() {
    return nodes().rbegin();
  }
  const_graph_node_list_iterator rbegin() const {
    return nodes().rbegin();
  }
  graph_node_list_iterator rend() {
    return nodes().rend();
  }
  const_graph_node_list_iterator rend() const {
    return nodes().rend();
  }
  Node* return_node() {
    return output_;
  }
  const Node* return_node() const {
    return output_;
  }

  Value* addInput() {
    return input_->addOutput();
  }
  void eraseInput(size_t i) {
    input_->eraseOutput(i);
  }
  void advanceStage() {
    new_node_stage_++;
  }
  void setStage(size_t new_stage) {
    new_node_stage_ = new_stage;
  }
  size_t stage() const {
    return new_node_stage_;
  }
  ResourceGuard setStageTemporary(size_t s) {
    auto prev_stage = new_node_stage_;
    new_node_stage_ = s;
    return ResourceGuard([prev_stage, this]() { this->new_node_stage_ = prev_stage; });
  }

  size_t registerOutput(Value* n) {
    output_->addInput(n);
    return outputs().size() - 1;
  }

  Node* create(NodeKind kind, size_t num_outputs = 1) {
    std::unique_ptr<Node> node_owner(new Node(*this, kind));
    Node* n = node_owner.get();
    all_nodes.emplace(n, std::move(node_owner));
    for (size_t i = 0; i < num_outputs; i++)
      n->addOutput();
    return n;
  }

  // Allocate a Value owned by this graph.
  Value* createValue(Node& node, size_t offset);

  Node* create(NodeKind kind, ArrayRef<Value*> inputs, size_t num_outputs = 1) {
    auto* n = create(kind, num_outputs);
    for (auto* i : inputs)
      n->addInput(i);
    return n;
  }

  Node* appendNode(Node* n) {
    ONNX_ASSERT(n->graph_ == this && !n->inGraphList())
    n->insertBefore(output_);
    return n;
  }

  Node* prependNode(Node* n) {
    ONNX_ASSERT(n->graph_ == this && !n->inGraphList())
    n->insertAfter(output_);
    return n;
  }

  // Adds to graph initializer list, initializer names list, and as a graph input
  // Also syncs the initializer name, tensor name, and value name
  // Create an initializer whose value is stored in input
  Value* addInitializerAndInput(const Tensor& initializer, const std::string& name) {
    Tensor initializerCopy = initializer;
    std::vector<Dimension> dim_sizes{initializerCopy.sizes().cbegin(), initializerCopy.sizes().cend()};
    Value* new_init = addInput();
    initializerCopy.setName(name);
    new_init->setUniqueName(name);
    new_init->setSizes(dim_sizes);
    new_init->setElemType(initializerCopy.elem_type());
    addInitializer(initializerCopy);
    return new_init;
  }

  Value* addInitializerAndInput(const Tensor& initializer) {
    return addInitializerAndInput(initializer, getNextUniqueName());
  }

  // Erases from graph initializer list, initializer names list, and as a graph input
  // Must have no uses
  void eraseInitializerAndInput(Value* v) {
    Node* node = v->node();
    const size_t offset = v->offset();
    eraseInitializer(v->uniqueName());
    if (node == input_) {
      eraseInput(offset);
    }
  }

  ~Graph() = default;

  std::string toString() const {
    std::ostringstream oss;
    oss << *this;
    return oss.str();
  }

  bool has_name() const {
    return has_name_;
  }

  const std::string& name() const {
    return name_;
  }

  void setName(std::string name) {
    has_name_ = true;
    name_ = std::move(name);
  }

  friend std::ostream& operator<<(std::ostream& out, const Graph& g);

  void forSelfAndEachSubGraph(const std::function<void(Graph*)>& fn) {
    forSelfAndEachSubGraphImpl(this, fn);
  }

  void forSelfAndEachSubGraph(const std::function<void(const Graph*)>& fn) const {
    forSelfAndEachSubGraphImpl(this, fn);
  }

  void forEachNode(const std::function<void(Node*)>& fn) {
    forEachNodeImpl(this, fn);
  }

  void forEachNode(const std::function<void(const Node*)>& fn) const {
    forEachNodeImpl(this, fn);
  }

  // Like forEachNode(), but never visits self's own top-level nodes -- only
  // those in nested subgraphs, recursively. For a caller that only cares
  // about nested-subgraph nodes (see below), this is a real optimization
  // over forEachNode(), not just the same cost under a different name:
  // forEachNode() must still visit every one of self's own nodes to honor
  // its own (more general) "visit self too" contract, even when the caller's
  // own callback immediately discards every one of them -- an O(graph size)
  // cost paid for nothing when self has no subgraphs. This skips straight to
  // hasAnySubgraphNode()'s O(1) answer in that case instead.
  void forEachNodeInSubgraphs(const std::function<void(Node*)>& fn) {
    forEachNodeInSubgraphsImpl(this, fn);
  }

  void forEachNodeInSubgraphs(const std::function<void(const Node*)>& fn) const {
    forEachNodeInSubgraphsImpl(this, fn);
  }

 private:
  // Whether this graph's own all_nodes (not recursing into subgraphs)
  // contains at least one node with a g/gs attribute -- memoized, since
  // computing it is the same O(graph size) scan forSelfAndEachSubGraphImpl
  // below always paid on every call before this cache existed. That matters
  // at scale: Value::uses()/setUniqueName()/replaceAllUsesWith() each call
  // forEachNode(), which starts from forSelfAndEachSubGraphImpl -- so a
  // graph with, say, 5000 nodes and zero Loop/If/Scan subgraphs (the
  // overwhelming common case for a real exported model) previously re-paid
  // a 5000-entry scan on every one of those calls, an O(graph size²) cost
  // across a whole pass, purely to conclude "no subgraphs here" each time.
  // See invalidateSubgraphNodeCache() for how staleness is avoided.
  bool hasAnySubgraphNode() const {
    if (!subgraph_node_cache_valid_) {
      has_subgraph_node_cached_ = std::any_of(
          all_nodes.begin(), all_nodes.end(), [](const auto& entry) { return entry.first->hasSubgraphAttribute(); });
      subgraph_node_cache_valid_ = true;
    }
    return has_subgraph_node_cached_;
  }

  template <typename GraphPtr, typename Fn>
  static void forSelfAndEachSubGraphImpl(GraphPtr self, const Fn& fn) {
    fn(self);
    if (!self->hasAnySubgraphNode()) {
      return;
    }
    for (const auto& node_entry : self->all_nodes) {
      const Node* node = node_entry.first;
      // hasSubgraphAttribute() is a cheap, non-allocating check; skip the
      // allocating attributeNames() call entirely for the overwhelming
      // majority of nodes (no g/gs attribute at all -- i.e. every node
      // outside a Loop/If/Scan-style op). This traversal runs on every
      // Value::uses()/setUniqueName() call, so avoiding that allocation on
      // every one of a graph's nodes matters at scale.
      if (!node->hasSubgraphAttribute()) {
        continue;
      }
      for (const auto& attr : node->attributeNames()) {
        if (node->kindOf(attr) == AttributeKind::g) {
          forSelfAndEachSubGraphImpl(node->g(attr).get(), fn);
        } else if (node->kindOf(attr) == AttributeKind::gs) {
          for (const auto& subgraph : node->gs(attr)) {
            forSelfAndEachSubGraphImpl(subgraph.get(), fn);
          }
        }
      }
    }
  }

  template <typename GraphPtr, typename Fn>
  static void forEachNodeImpl(GraphPtr self, const Fn& fn) {
    forSelfAndEachSubGraphImpl(self, [&fn](auto* graph) {
      for (auto* node : graph->nodes()) {
        fn(node);
      }
    });
  }

  // Same subgraph scan as forSelfAndEachSubGraphImpl above, but never calls
  // fn(self) or visits self's own nodes -- only recurses into nested
  // subgraphs (via forEachNodeImpl, so a subgraph found here still has its
  // own nodes visited, and its own nested subgraphs recursed into in turn).
  // See forEachNodeInSubgraphs()'s comment for why this exists as a
  // separate traversal instead of forEachNode() plus a filter.
  template <typename GraphPtr, typename Fn>
  static void forEachNodeInSubgraphsImpl(GraphPtr self, const Fn& fn) {
    if (!self->hasAnySubgraphNode()) {
      return;
    }
    for (const auto& node_entry : self->all_nodes) {
      const Node* node = node_entry.first;
      if (!node->hasSubgraphAttribute()) {
        continue;
      }
      for (const auto& attr : node->attributeNames()) {
        if (node->kindOf(attr) == AttributeKind::g) {
          forEachNodeImpl(node->g(attr).get(), fn);
        } else if (node->kindOf(attr) == AttributeKind::gs) {
          for (const auto& subgraph : node->gs(attr)) {
            forEachNodeImpl(subgraph.get(), fn);
          }
        }
      }
    }
  }

  // should only be called in the constructor
  Node* initOutput(Node* p) {
    p->next() = p;
    p->prev() = p;
    p->setStage(std::numeric_limits<size_t>::max());
    return p;
  }

  void freeNode(Node* n) {
    auto it = all_nodes.find(n);
    ONNX_ASSERT(it != all_nodes.end())
    all_nodes.erase(it);
  }
  void freeValue(Value* v) {
    // v->unique_name_ directly, not v->uniqueName(): avoids a string copy in
    // the common (explicitly-named) case; the default-name branch still has
    // to materialize "_v_<n>" since there's no stored copy of it to borrow.
    releaseName(v->has_unique_name() ? v->unique_name_ : toVarName(v->unique_));
    auto it = all_values.find(v);
    ONNX_ASSERT(it != all_values.end())
    all_values.erase(it);
  }
};

inline Value::Value(Node& node, size_t offset)
    : node_(&node), offset_(offset), unique_(node.graph_->getNextUnique()), stage_(node.graph_->new_node_stage_) {
  // Every value occupies its default display name ("_v_<unique_>") in
  // used_names_ from construction, even before any explicit rename --
  // setUniqueName() and Graph::freeValue() release it. Without this, a
  // nested subgraph's own unnamed values would be invisible to
  // isNameUnique() calls recursing in from an enclosing graph, since each
  // Graph -- parent and subgraph alike -- keeps an independently-numbered
  // unique_ counter and so can mint the very same "_v_<n>" default name.
  node.graph_->useName(toVarName(unique_));
}

inline Graph* Value::owningGraph() {
  return node()->owningGraph();
}

inline const Graph* Value::owningGraph() const {
  return node()->owningGraph();
}

// `captured` nodes in subgraph determines which value it captures
// by storing the value's unique name, so old unique names in `captured` nodes
// should also be updated.
// Initializer names are also stored in graph.initializer_names_, it should be
// updated too.
inline Value* Value::setUniqueName(const std::string& name, bool update_related_names) {
  if (has_unique_name_ && unique_name_ == name) {
    // Already exactly this name; nothing to update. (replaceAllUsesWith's
    // captured-node fixup loop routinely re-applies a value's existing
    // name.)
    return this;
  }
  auto* graph = owningGraph();
  if (has_unique_name() && update_related_names) {
    auto old_name = unique_name_;
    for (size_t i = 0; i < owningGraph()->initializer_names_.size(); i++) {
      auto& initializer_name = owningGraph()->initializer_names_[i];
      if (initializer_name == old_name) {
        initializer_name = name;
        owningGraph()->initializers_[i]->setName(name);
        // This initializer_names_ entry is its own holder of the name,
        // independent of this Value's own registration below -- see
        // used_names_'s comment for why the same name can have more than
        // one live holder at once.
        graph->releaseName(old_name);
        graph->useName(name);
      }
    }
    // forEachNodeInSubgraphs(), not forEachNode(): this only ever cares
    // about kCaptured placeholders in a *nested* subgraph (never a node in
    // `graph` itself, hence no owningGraph() filter needed here anymore --
    // forEachNodeInSubgraphs() never visits graph's own nodes to begin
    // with).
    graph->forEachNodeInSubgraphs([this, &name, &old_name](Node* node) {
      if (node->kind() == kCaptured) {
        Value* output = node->output();
        if (output->uniqueName() == old_name) {
          output->setUniqueName(name, false);
        }
      }
    });
  }
  // Release whatever name this value currently displays -- explicit or
  // still just its default -- before claiming the new one, so a second
  // holder of the same name (e.g. an initializer's Tensor entry, or
  // newValue in Value::replaceAllUsesWith while this value is mid-rename
  // off of it) keeps its own claim alive in used_names_.
  graph->releaseName(has_unique_name_ ? unique_name_ : toVarName(unique_));
  unique_name_ = name;
  has_unique_name_ = true;
  graph->useName(name);
  return this;
}

inline void Value::replaceAllUsesWith(Value* newValue) {
  auto* graph = owningGraph();
  ONNX_ASSERT(graph == newValue->owningGraph())
  // propagate sizes and elem type
  if (this->has_sizes()) {
    newValue->setSizes(this->sizes());
  }
  if (this->elemType() != ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED) {
    newValue->setElemType(this->elemType());
  }
  const auto unique_name = this->uniqueName();
  // We do not want the optimization to change the graph output name
  if (std::find(graph->outputs().rbegin(), graph->outputs().rend(), this) != graph->outputs().rend()) {
    newValue->setUniqueName(unique_name);
    // The "unique" semantic of unique_name should be kept or uses()
    // will return an incorrect result when the value is used in subgraph
    this->setUniqueName(graph->getNextUniqueName(), false);
  }
  newValue->uses_in_current_graph_.reserve(this->uses_in_current_graph_.size());
  for (auto u : uses_in_current_graph_) {
    u.user->inputs_[u.offset] = newValue;
    newValue->uses_in_current_graph_.push_back(u);
  }
  // forEachNodeInSubgraphs(), not forEachNode(): see setUniqueName()'s own
  // comment on the same substitution.
  graph->forEachNodeInSubgraphs([this, &newValue, &unique_name](Node* node) {
    if (node->kind() == kCaptured) {
      Value* output = node->output();
      if (output->uniqueName() == unique_name) {
        output->setUniqueName(newValue->uniqueName());
      }
    }
  });
  uses_in_current_graph_.clear();
  assert(this->uses().empty());
}

inline Node::Node(Graph& graph, NodeKind kind) : kind_(kind), graph_(&graph), stage_(graph.new_node_stage_) {}

inline Value* Graph::createValue(Node& node, size_t offset) {
  auto value = std::make_unique<Value>(node, offset);
  Value* v = value.get();
  all_values.emplace(v, std::move(value));
  return v;
}

inline Value* Node::addOutput() {
  Value* v = graph_->createValue(*this, outputs_.size());
  outputs_.push_back(v);
  return v;
}

inline void Node::eraseOutput(size_t i) {
  ONNX_ASSERT(i < outputs_.size())
  ONNX_ASSERT(outputs_[i]->uses().empty())
  Value* n = outputs_[i];
  outputs_.erase(outputs_.begin() + static_cast<std::ptrdiff_t>(i));
  owningGraph()->freeValue(n);
  for (size_t j = i; j < outputs_.size(); j++) {
    outputs_[j]->offset_--;
  }
}

inline bool Node::isBefore(const Node* n) {
  if (n == nullptr || this == n) {
    // Bail out early.
    return false;
  }
  // return true if node is Param (in initializers)
  if (kind_ == kParam) {
    return true;
  }
  // return false if target node is Param (in initializers)
  if (n->kind() == kParam) {
    return false;
  }
  ONNX_ASSERT(n->inGraphList())
  for (Node* p = next(); p != *graph_->end(); p = p->next()) {
    if (p == n) {
      return true;
    }
  }
  return false;
}

inline void Node::destroy() {
  ONNX_ASSERT(inGraphList())
  while (!outputs().empty())
    eraseOutput(outputs().size() - 1);
  removeAllInputs();
  removeFromList();
  graph_->freeNode(this);
}

/************* All nodes not required to be defined before Graph **************/

inline graph_node_list_iterator Node::iterator() {
  return graph_node_list_iterator(this, 0);
}
inline graph_node_list_iterator Node::reverseIterator() {
  return iterator().reverse();
}
inline const_graph_node_list_iterator Node::iterator() const {
  return const_graph_node_list_iterator(this, 0);
}
inline const_graph_node_list_iterator Node::reverseIterator() const {
  return iterator().reverse();
}

// Returns a list about which nodes are using this value,
// nodes in subgraph are also included.
// This method is usually used to check whether it is
// safe to delete a Value.
inline use_list Value::uses() const {
  use_list all_uses = uses_in_current_graph_;
  // forEachNodeInSubgraphs(), not forEachNode(): see setUniqueName()'s own
  // comment on the same substitution.
  owningGraph()->forEachNodeInSubgraphs([this, &all_uses](const Node* node) {
    if (node->kind() == kCaptured) {
      const Value* output = node->outputs()[0];
      if (output->uniqueName() == this->uniqueName()) {
        const auto output_uses = output->uses();
        all_uses.insert(all_uses.end(), output_uses.begin(), output_uses.end());
      }
    }
  });
  return all_uses;
}

} // namespace ONNX_NAMESPACE
