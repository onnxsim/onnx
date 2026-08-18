// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#include "onnx/common/graph_shape_inference.h"

#include <string>
#include <unordered_map>
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

class GraphShapeInferenceRunner {
 public:
  explicit GraphShapeInferenceRunner(const ShapeInferenceOptions& options) : options_(options) {}

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
      initializer_by_name[initializer_names[i]] = &initializers[i];
    }

    bool changed = false;
    for (Node* node : g.nodes()) {
      if (node->kind() == kUndefined || node->kind() == kCaptured) {
        continue;
      }
      changed |= ProcessNode(*node, opset_imports, registry, initializer_by_name);
    }
    return changed;
  }

 private:
  bool ProcessNode(
      Node& node,
      const std::unordered_map<std::string, int>& opset_imports,
      const ISchemaRegistry* registry,
      const std::unordered_map<std::string, const Tensor*>& initializer_by_name) {
    const std::vector<Symbol> attr_names = node.attributeNames();

    // v1 limitation: control-flow ops (If/Loop/Scan) are not inferred here --
    // see graph_shape_inference.h's doc comment.
    for (Symbol name : attr_names) {
      AttributeKind kind = node.kindOf(name);
      if (kind == AttributeKind::g || kind == AttributeKind::gs) {
        return false;
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

    // A lightweight NodeProto shell: attribute conversion is the only part
    // worth sharing with the Export path (addAttribute, from
    // ir_pb_converter_internal.h), everything else here is small metadata
    // (names, counts) with no tensor bytes involved.
    NodeProto np;
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
    // node visit. InferenceContextImpl captures these pointers into its own
    // per-call state during construction and does not use them past
    // ProcessNode's return, so the backing vectors only need to outlive this
    // function.
    std::vector<TypeProto> input_types(inputs.size());
    std::unordered_map<std::string, TypeProto*> value_types_by_name;
    std::vector<TensorProto> input_data_storage;
    input_data_storage.reserve(inputs.size());
    std::unordered_map<std::string, const TensorProto*> input_data_by_name;
    const std::unordered_map<std::string, const SparseTensorProto*> input_sparse_data_by_name; // always empty (v1)

    for (size_t i = 0; i < inputs.size(); ++i) {
      Value* input = inputs[i];
      if (input->node()->kind() == kUndefined) {
        continue; // absent optional input
      }
      EncodeCurrentType(*input, input_types[i]);
      value_types_by_name[input->uniqueName()] = &input_types[i];

      if (const Tensor* data = ConstantDataFor(*input, initializer_by_name)) {
        if (ElementCountFits(*data)) {
          input_data_storage.emplace_back();
          encodeTensor(input_data_storage.back(), *data);
          input_data_by_name[input->uniqueName()] = &input_data_storage.back();
        }
      }
    }

    shape_inference::InferenceContextImpl ctx(
        np,
        value_types_by_name,
        input_data_by_name,
        input_sparse_data_by_name,
        options_,
        /*generatedShapeData=*/nullptr,
        /*graphInferenceContext=*/nullptr);

    ONNX_TRY {
      schema->GetTypeAndShapeInferenceFunction()(ctx);
    }
    ONNX_CATCH(const std::exception&) {
      // Matches onnx's own default (ShapeInferenceOptions::error_mode == 0):
      // a node-level inference error doesn't abort the whole pass, it just
      // leaves that node's outputs as they were.
      return false;
    }

    bool changed = false;
    for (size_t i = 0; i < outputs.size(); ++i) {
      TypeProto* inferred = ctx.getOutputType(i);
      if (inferred == nullptr) {
        continue;
      }
      Value* out = outputs[i];
      TypeProto existing;
      EncodeCurrentType(*out, existing);
      if (shape_inference::mergeShapesAndTypes(*inferred, &existing)) {
        changed = true;
        ApplyInferredType(existing, *out);
      }
    }
    return changed;
  }

  const ShapeInferenceOptions& options_;
};

} // namespace

bool InferShapesOnGraph(Graph& g, const ShapeInferenceOptions& options) {
  GraphShapeInferenceRunner runner(options);
  return runner.Run(g);
}

} // namespace ONNX_NAMESPACE
