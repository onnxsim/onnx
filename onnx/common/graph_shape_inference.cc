// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#include "onnx/common/graph_shape_inference.h"

#include <google/protobuf/arena.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "onnx/common/constants.h"
#include "onnx/common/ir_pb_converter_internal.h"
#include "onnx/defs/schema.h"
#include "onnx/shape_inference/implementation.h"

namespace ONNX_NAMESPACE {

namespace {

// Shape-describing inputs (Reshape's `shape`, Slice's starts/ends/axes,
// ConstantOfShape's `input`, ...) are always tiny by ONNX spec design: they
// describe ranks/dims/indices, not tensor contents. Anything above this
// threshold is presumably a weight/activation tensor that no standard op's
// shape-inference function actually reads via getInputData() in the first
// place -- so it is simply left out of input_data_by_name rather than paying
// to convert it to a TensorProto on every visit. That conversion cost, paid
// per round for every large initializer, is exactly the copy issue #633 is
// about; a node whose inference genuinely needs a large constant's value
// (there is no such standard op) just sees a null getInputData(), the same
// as any other "value not statically known" case shape inference already
// handles.
constexpr int64_t kMaxInputDataElements = 4096;

// Encodes `v`'s current type: the plain tensor_type case from
// elemType()/sizes() when that's all `v` carries, or a copy of v.type()
// verbatim for non-tensor types (Sequence/Optional/Map/...) -- the same
// choice encodeValueInfo makes in ir_pb_converter.cc.
void EncodeCurrentType(Value& v, TypeProto& out) {
  if (v.elemType() != 0 || v.has_sizes()) {
    encodeTypeProtoTensorType(*out.mutable_tensor_type(), v);
  } else if (v.type()) {
    out.CopyFrom(*v.type());
  }
}

// Applies a (possibly merged) inferred TypeProto back onto `v`: the tensor
// case updates elemType()/sizes() directly (matching graphProtoToGraph's own
// output/value_info handling on Import), anything else replaces v.type()
// wholesale.
void ApplyInferredType(const TypeProto& inferred, Value& v) {
  if (inferred.value_case() == TypeProto::VALUE_NOT_SET) {
    return;
  }
  if (inferred.has_tensor_type()) {
    const auto& tensor_type = inferred.tensor_type();
    if (tensor_type.has_elem_type()) {
      v.setElemType(tensor_type.elem_type());
    }
    if (tensor_type.has_shape()) {
      v.setSizes(tensorShapeProtoToDimensions(tensor_type.shape()));
    }
  } else {
    v.type() = std::make_unique<TypeProto>(inferred);
  }
}

bool ElementCountFits(const Tensor& t) {
  int64_t n = 1;
  for (int64_t d : t.sizes()) {
    if (d < 0) {
      return false;
    }
    n *= d;
    if (n > kMaxInputDataElements) {
      return false;
    }
  }
  return true;
}

// Returns the Tensor backing `v`'s statically-known constant value, if any: a
// Constant node's "value" attribute, or a graph initializer. Both are already
// resident as ir.h Tensor objects the Graph owns, so nothing is converted or
// copied here -- only the (size-gated) caller of this function pays for an
// actual encodeTensor() call, and only for tensors small enough to matter to
// shape inference in the first place (see kMaxInputDataElements above).
// `initializer_by_name` is a name -> Tensor* map built once per Run() (see
// GraphShapeInferenceRunner::Run) from g.initializers()/initializer_names().
// Graph::getInitializer() is a linear scan over that same vector; calling it
// once per input, per node, per round (as an earlier version of this
// function did) turned into an O(rounds * nodes * initializers) scan for
// models with hundreds of initializers -- exactly the kind of quadratic-ish
// blowup issue #633 is about, just moved from tensor-byte copies to name
// lookups. The map amortizes that scan to once per Run() call instead.
const Tensor* ConstantDataFor(Value& v, const std::unordered_map<std::string, const Tensor*>& initializer_by_name) {
  static const Symbol kConstant("Constant");
  static const Symbol kValue("value");

  const Node* producer = v.node();
  if (producer->kind() == kConstant && (!producer->has_domain() || producer->domain().empty()) &&
      producer->kindOf(kValue) == AttributeKind::t) {
    return &producer->t(kValue);
  }
  auto it = initializer_by_name.find(v.uniqueName());
  if (it != initializer_by_name.end()) {
    return it->second;
  }
  return nullptr;
}

// Encodes a Tensor's shape/dtype only, deliberately skipping the raw bytes
// -- used for a TENSOR/TENSORS-kind attribute value too large to be worth
// copying, mirroring ConstantDataFor's size gate on inputs (see
// kMaxInputDataElements): no standard op's shape-inference function needs a
// large attribute tensor's actual values, only its shape/dtype.
void EncodeShapeOnly(TensorProto& out, const Tensor& t) {
  out.set_data_type(t.elem_type());
  for (int64_t d : t.sizes()) {
    out.add_dims(d);
  }
}

// Adds one attribute of `node` to `np`, size-gating any TENSOR/TENSORS
// value exactly like ConstantDataFor gates constant/initializer inputs.
// addAttribute (ir_pb_converter_internal.h, shared with the real Export
// path, where full-fidelity conversion is required) always copies a
// tensor-valued attribute's raw bytes unconditionally; calling it here
// unconditionally too would silently reintroduce issue #633's per-round
// large-tensor-copy cost through node *attributes* rather than graph
// initializers -- e.g. a Constant node produced by folding, which is a
// per-node attribute rather than a graph initializer, but exactly as large.
//
// GRAPH/GRAPHS-kind attributes (If/Loop/Scan's bodies) go through
// addAttribute unconditionally, no size gate: a control-flow subgraph body
// is a small computational structure, not a weight tensor, so the copy cost
// this function otherwise guards against does not apply to it the same way.
// See ProcessNode's subgraph handling for why exporting a real GraphProto
// here (rather than skipping, as an earlier version of this file did) is
// what lets If/Loop/Scan's own inference functions recurse into their body
// via onnx's existing, unmodified GraphInferencerImpl/InferShapesImpl.
void AddAttributeForInference(NodeProto& np, Node& node, Symbol name) {
  AttributeKind kind = node.kindOf(name);
  if (kind == AttributeKind::t) {
    const Tensor& t = node.t(name);
    if (ElementCountFits(t)) {
      addAttribute(np, node, name, /*consume_tensor_data=*/false);
      return;
    }
    auto* attr = np.add_attribute();
    attr->set_name(name.toString());
    attr->set_type(AttributeProto_AttributeType_TENSOR);
    EncodeShapeOnly(*attr->mutable_t(), t);
    return;
  }
  if (kind == AttributeKind::ts) {
    bool any_large = false;
    for (const Tensor& t : node.ts(name)) {
      if (!ElementCountFits(t)) {
        any_large = true;
        break;
      }
    }
    if (!any_large) {
      addAttribute(np, node, name, /*consume_tensor_data=*/false);
      return;
    }
    auto* attr = np.add_attribute();
    attr->set_name(name.toString());
    attr->set_type(AttributeProto_AttributeType_TENSORS);
    for (const Tensor& t : node.ts(name)) {
      EncodeShapeOnly(*attr->add_tensors(), t);
    }
    return;
  }
  addAttribute(np, node, name, /*consume_tensor_data=*/false);
}

// A shape_inference::SymbolTable seeded directly from the Graph IR's own
// Values rather than a GraphProto. Without this, every dimension shape
// inference can't pin a concrete value to (Reshape's dynamic output rank,
// Resize's output spatial dims, ...) stays a bare, nameless "unknown" --
// which loses the identity onnx's own reference inference captures by
// naming it (e.g. "unk__11") and reusing that same name wherever the same
// unknown quantity flows to (Transpose, Cast, ...). Two independently
// unnamed dims are indistinguishable from each other and from "0", which
// is a real correctness gap for anything downstream that needs the merge
// logic to recognize "these are the same unknown" versus "these differ":
// left unfixed, this manifested as onnxsim producing shapes ONNX Runtime's
// own (name-carrying) inference then rejects as incompatible.
// GenerateSymbolicShape (called via MaterializeSymbolicShape, see
// ProcessNode) is what actually assigns names to bare dims; this class only
// supplies the naming authority + collision avoidance, mirroring
// SymbolTableImpl's addFromGraph but scanning Values instead of a
// GraphProto's value_info/input/output lists.
class GraphIrSymbolTable : public SymbolTable {
 public:
  // addFromGraph exists only to satisfy the abstract interface: a nested
  // subgraph's own recursive InferShapesImpl call (see ProcessNode's
  // GraphInferenceContext) may call it on the protobuf subgraph this file
  // exports, so symbols already used inside a subgraph body don't collide
  // with ones generated for the enclosing graph either.
  void addFromGraph(const GraphProto& g) override {
    AddExistingDims(g.input());
    AddExistingDims(g.output());
    AddExistingDims(g.value_info());
  }
  void AddExistingSymbol(const std::string& symbol) {
    existing_symbols_.insert(symbol);
  }
  std::string createNew(const std::string& symbol_prefix) override {
    std::string new_symbol;
    do {
      new_symbol = symbol_prefix + std::to_string(index_++);
    } while (existing_symbols_.count(new_symbol) > 0);
    existing_symbols_.insert(new_symbol);
    return new_symbol;
  }

 private:
  template <typename RepeatedValueInfo>
  void AddExistingDims(const RepeatedValueInfo& value_infos) {
    for (const auto& vi : value_infos) {
      if (!vi.type().has_tensor_type() || !vi.type().tensor_type().has_shape()) {
        continue;
      }
      for (const auto& dim : vi.type().tensor_type().shape().dim()) {
        if (dim.has_dim_param()) {
          existing_symbols_.insert(dim.dim_param());
        }
      }
    }
  }

  unsigned int index_ = 0;
  std::unordered_set<std::string> existing_symbols_;
};

class GraphShapeInferenceRunner {
 public:
  explicit GraphShapeInferenceRunner(const ShapeInferenceOptions& options,
                                      shape_inference::DataValueMap* generated_shape_data)
      : options_(options), generated_shape_data_(generated_shape_data) {
    if (options_.enable_data_propagation && generated_shape_data_ == nullptr) {
      fail_shape_inference(
          "Container for generated shape data cannot be nullptr when enable_data_propagation option is set.");
    }
  }

  // Returns whether anything changed.
  bool Run(Graph& g) {
    std::unordered_map<std::string, int> opset_imports;
    for (const OpSetID& opset : g.opset_versions_mutable()) {
      opset_imports[opset.domain()] = static_cast<int>(opset.version());
    }
    const ISchemaRegistry* registry = OpSchemaRegistry::Instance();

    // Built once per pass over the whole graph, not once per node visit --
    // see ConstantDataFor's doc comment for why that distinction matters.
    std::unordered_map<std::string, const Tensor*> initializer_by_name;
    const auto& initializers = g.initializers();
    const auto& initializer_names = g.initializer_names();
    initializer_by_name.reserve(initializers.size());
    for (size_t i = 0; i < initializers.size(); ++i) {
      initializer_by_name[initializer_names[i]] = initializers[i].get();
    }

    // Whole-graph accumulating type map, mirroring onnx's own protobuf-based
    // InferShapesImpl's "value_types_by_name": used only to resolve an
    // If/Loop/Scan subgraph's *captured* references (a name used inside the
    // body with no local definition -- see ir_pb_converter.cc's kCaptured)
    // against the enclosing scope. Ordinary per-node input lookups below
    // don't need this; only GraphInferenceContext does. outer_scope_storage
    // owns the TypeProto objects (must outlive every node visit in this
    // Run() call, unlike the per-node-local TypeProtos in ProcessNode);
    // outer_scope_types is the name->pointer view GraphInferenceContext
    // actually takes. unordered_map references stay valid across further
    // insertions, so pointers into outer_scope_storage handed out now stay
    // good as later nodes add their own entries.
    std::unordered_map<std::string, TypeProto> outer_scope_storage;
    std::unordered_map<std::string, TypeProto*> outer_scope_types;
    auto RecordOuterScopeType = [&](Value* v) {
      TypeProto& t = outer_scope_storage[v->uniqueName()];
      EncodeCurrentType(*v, t);
      if (t.value_case() != TypeProto::VALUE_NOT_SET) {
        outer_scope_types[v->uniqueName()] = &t;
      }
    };
    for (Value* input : g.inputs()) {
      RecordOuterScopeType(input);
    }
    for (size_t i = 0; i < initializers.size(); ++i) {
      // Input has priority over initializer of the same name, matching
      // onnx's own ProcessInitializer.
      if (outer_scope_types.count(initializer_names[i]) > 0) {
        continue;
      }
      TypeProto& t = outer_scope_storage[initializer_names[i]];
      auto* tensor_type = t.mutable_tensor_type();
      tensor_type->set_elem_type(initializers[i]->elem_type());
      auto* shape = tensor_type->mutable_shape();
      for (int64_t d : initializers[i]->sizes()) {
        shape->add_dim()->set_dim_value(d);
      }
      outer_scope_types[initializer_names[i]] = &t;
    }

    // Seed the symbol table with every dim_param already anywhere in the
    // graph -- both the original model's own named dims and any symbol a
    // previous Run() call materialized onto a Value that survived into this
    // round -- so a freshly-generated name (see GraphIrSymbolTable) can
    // never collide with (and be wrongly treated as identical to) an
    // unrelated existing one.
    auto SeedSymbolTable = [&](Value* v) {
      if (!v->has_sizes()) {
        return;
      }
      for (const Dimension& d : v->sizes()) {
        if (!d.is_unknown && !d.is_int && !d.param.empty()) {
          symbol_table_.AddExistingSymbol(d.param);
        }
      }
    };
    for (Value* input : g.inputs()) {
      SeedSymbolTable(input);
    }
    for (Node* node : g.nodes()) {
      for (Value* output : node->outputs()) {
        SeedSymbolTable(output);
      }
    }

    bool changed = false;
    for (Node* node : g.nodes()) {
      if (node->kind() == kUndefined || node->kind() == kCaptured) {
        continue;
      }
      changed |= ProcessNode(*node, opset_imports, registry, initializer_by_name, outer_scope_types);
      for (Value* output : node->outputs()) {
        RecordOuterScopeType(output);
      }
    }
    return changed;
  }

 private:
  bool ProcessNode(
      Node& node,
      const std::unordered_map<std::string, int>& opset_imports,
      const ISchemaRegistry* registry,
      const std::unordered_map<std::string, const Tensor*>& initializer_by_name,
      const std::unordered_map<std::string, TypeProto*>& outer_scope_types) {
    const std::vector<Symbol> attr_names = node.attributeNames();

    bool has_subgraph_attr = false;
    for (Symbol name : attr_names) {
      AttributeKind kind = node.kindOf(name);
      if (kind == AttributeKind::g || kind == AttributeKind::gs) {
        has_subgraph_attr = true;
        break;
      }
    }

    const std::string domain = node.has_domain() ? node.domain() : std::string(ONNX_DOMAIN);
    auto dit = opset_imports.find(domain);
    if (dit == opset_imports.end() && domain == ONNX_DOMAIN) {
      dit = opset_imports.find(AI_ONNX_DOMAIN);
    }
    if (dit == opset_imports.end()) {
      return false; // no opset import for this domain -- leave outputs as-is.
    }
    const std::string op_type = node.kind().toString();
    const OpSchema* schema = registry->GetSchema(op_type, dit->second, domain);
    if (schema == nullptr || !schema->has_type_and_shape_inference_function()) {
      // Unsupported op (no schema, or a function-body-only op -- see this
      // file's v1-scope doc comment): leave outputs as-is, same as onnx's
      // own protobuf-based InferShapes does for a genuinely unknown op.
      return false;
    }

    // Every protobuf message this one node visit builds -- np below plus its
    // attribute tree (including, for If/Loop/Scan, a full subgraph body
    // export via AddAttributeForInference) and the per-input
    // TypeProto/TensorProto adapters further down -- lives on `arena` and is
    // discarded when ProcessNode returns. ProcessNode runs once per node,
    // per Run() call, and Run() is itself driven to a fixed point over the
    // whole graph (see OptAndShapeOnGraph's FixedPointFn loop in onnxsim),
    // so this tree is rebuilt from scratch many times per node across a
    // typical simplification run. Without an arena, tearing one down after
    // every visit means walking whatever sub-message tree schema authoring
    // produced (recursively, for a node with a subgraph attribute) and
    // freeing each piece individually; on an arena the whole tree is
    // released in one bulk free when `arena` goes out of scope -- the same
    // rationale as RunOps's op_model in constant_folding.cpp. Nothing
    // allocated here is read past ProcessNode's own return:
    // InferenceContextImpl's per-call state, GraphInferencerImpl, and the
    // schema/data-propagation functions invoked below are all constructed
    // and fully consumed inside this function.
    google::protobuf::Arena arena;

    // A lightweight NodeProto shell: attribute conversion is the only part
    // worth sharing with the Export path (addAttribute, from
    // ir_pb_converter_internal.h), everything else here is small metadata
    // (names, counts) with no tensor bytes involved.
    NodeProto& np = *google::protobuf::Arena::Create<NodeProto>(&arena);
    np.set_op_type(op_type);
    if (node.has_domain()) {
      np.set_domain(node.domain());
    }
    if (node.has_name()) {
      np.set_name(node.name());
    }
    const auto& inputs = node.inputs();
    const auto& outputs = node.outputs();
    for (Value* input : inputs) {
      np.add_input(input->node()->kind() == kUndefined ? "" : input->uniqueName());
    }
    for (Value* output : outputs) {
      np.add_output(output->uniqueName());
    }
    for (Symbol attr_name : attr_names) {
      AddAttributeForInference(np, node, attr_name);
    }

    // Per-input TypeProto/TensorProto adapters, built fresh for this one
    // node visit and arena-allocated for the same reason as `np` above:
    // constructing a RepeatedPtrField with `&arena` makes every element
    // Add() creates live on it too. InferenceContextImpl captures pointers
    // into these into its own per-call state during construction and does
    // not use them past ProcessNode's return, so the arena's lifetime
    // (this function's scope) is all they need.
    google::protobuf::RepeatedPtrField<TypeProto> input_types(&arena);
    input_types.Reserve(static_cast<int>(inputs.size()));
    std::unordered_map<std::string, TypeProto*> value_types_by_name;
    google::protobuf::RepeatedPtrField<TensorProto> input_data_storage(&arena);
    input_data_storage.Reserve(static_cast<int>(inputs.size()));
    std::unordered_map<std::string, const TensorProto*> input_data_by_name;
    const std::unordered_map<std::string, const SparseTensorProto*> input_sparse_data_by_name; // always empty (v1)

    for (size_t i = 0; i < inputs.size(); ++i) {
      Value* input = inputs[i];
      if (input->node()->kind() == kUndefined) {
        continue; // absent optional input
      }
      TypeProto* input_type = input_types.Add();
      EncodeCurrentType(*input, *input_type);
      value_types_by_name[input->uniqueName()] = input_type;

      if (const Tensor* data = ConstantDataFor(*input, initializer_by_name)) {
        if (ElementCountFits(*data)) {
          TensorProto* tp = input_data_storage.Add();
          encodeTensor(*tp, *data);
          input_data_by_name[input->uniqueName()] = tp;
        }
      }
    }

    // Lets If/Loop/Scan's own (unmodified, schema-registered) inference
    // function recurse into its now-fully-exported body subgraph: it calls
    // ctx.getGraphAttributeInferencer(attr_name), which InferenceContextImpl
    // answers by building a GraphInferencerImpl over np's real GraphProto
    // attribute (see AddAttributeForInference) and this context -- entirely
    // onnx's own existing protobuf-based machinery, run once per node visit
    // on just that node's (typically small) body, not the whole graph.
    std::unique_ptr<shape_inference::GraphInferenceContext> graph_inference_context;
    if (has_subgraph_attr) {
      graph_inference_context =
          std::make_unique<shape_inference::GraphInferenceContext>(outer_scope_types, opset_imports, &symbol_table_);
    }

    shape_inference::InferenceContextImpl ctx(
        np,
        value_types_by_name,
        input_data_by_name,
        input_sparse_data_by_name,
        options_,
        generated_shape_data_,
        graph_inference_context.get());

    // Both the schema function itself and mergeShapesAndTypes below
    // (called via checkShapesAndTypes) can throw on a genuine conflict --
    // e.g. this value's shape was already set from some other source (the
    // original exporter's value_info, a previous round's inference) and
    // disagrees with what this round just inferred. onnx's own
    // protobuf-based InferShapesImpl catches exactly this span (schema call
    // through the corresponding UpdateType) as one unit, so mirror that
    // here: a node-level inference error doesn't abort the whole pass, it
    // just leaves that node's outputs as they were, exactly as if the op
    // had no registered schema.
    bool changed = false;
    ONNX_TRY {
      schema->GetTypeAndShapeInferenceFunction()(ctx);
      for (size_t i = 0; i < outputs.size(); ++i) {
        TypeProto* inferred = ctx.getOutputType(i);
        if (inferred == nullptr) {
          continue;
        }
        // Name any dim this node's own inference just produced with no
        // dim_value/dim_param (e.g. a data-dependent Reshape/Resize output)
        // before merging it against `existing` -- matching onnx's own
        // UpdateType, which calls this on the freshly-inferred type first
        // thing. Without it, two genuinely-different unknown dims are both
        // indistinguishable bare "unknowns", which downstream ops and
        // ONNX Runtime's own (name-carrying) reference inference do not
        // treat as equivalent -- see GraphIrSymbolTable's comment.
        shape_inference::MaterializeSymbolicShape(inferred, symbol_table_);
        Value* out = outputs[i];
        TypeProto existing;
        EncodeCurrentType(*out, existing);
        if (shape_inference::mergeShapesAndTypes(*inferred, &existing)) {
          changed = true;
          ApplyInferredType(existing, *out);
        }
      }
      // Mirrors ONNX_NAMESPACE::shape_inference's own ShapeInferenceImplBase::
      // Process: after this node's ordinary type/shape inference is merged
      // in, run its schema-registered data-propagation function (if any) to
      // populate generated_shape_data_ with this node's outputs' concrete
      // partial values, which later nodes' getSymbolicInput()/getInputData()
      // (via DataPropagationContextImpl) can then read back -- e.g. chaining
      // Shape -> Gather -> Concat end to end even though nothing here is a
      // graph initializer.
      if (options_.enable_data_propagation && schema->has_data_propagation_function()) {
        shape_inference::DataPropagationContextImpl data_propagation_ctx(
            np, value_types_by_name, input_data_by_name, *generated_shape_data_);
        schema->GetDataPropagationFunction()(data_propagation_ctx);
      }
    }
    ONNX_CATCH(const std::exception&) {
      return false;
    }
    return changed;
  }

  const ShapeInferenceOptions& options_;
  shape_inference::DataValueMap* generated_shape_data_;
  // Reset (default-constructed) for every Run() call and re-seeded from the
  // graph's current state each time -- see the seeding loop in Run().
  GraphIrSymbolTable symbol_table_;
};

} // namespace

bool InferShapesOnGraph(Graph& g, const ShapeInferenceOptions& options,
                         shape_inference::DataValueMap* out_generated_shape_data) {
  GraphShapeInferenceRunner runner(options, out_generated_shape_data);
  return runner.Run(g);
}

} // namespace ONNX_NAMESPACE
