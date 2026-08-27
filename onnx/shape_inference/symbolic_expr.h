// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "onnx/common/safe_math.h"

namespace ONNX_NAMESPACE {

// An integer-coefficient polynomial over named symbols (dimension-symbol
// names), used by symbolic-dimension algebra in shape inference to represent
// the result of combining two `dim_param`s -- e.g. `M + N`, `2 * M`, `M * N`
// -- as a real, reproducible value instead of an anonymous unknown dimension.
//
// This is deliberately narrow: no floor/ceil/min/max, no rational functions,
// no inequality solving. Every shape-inference rule this is meant to serve
// (reshape targets, pooling/conv output-length formulas, concat/broadcast
// axis sizes, cache-length arithmetic) only ever needs sums and products of
// dims and constants -- exactly what a polynomial expresses. It has no
// dependency beyond the standard library, so it is safe to link into every
// consumer of onnx's core library.
class SymbolicExpr {
 public:
  SymbolicExpr() = default;

  // Implicit on purpose: an int64_t constant is always a valid SymbolicExpr,
  // and this lets arithmetic mix bare integers into expressions naturally
  // (e.g. `expr - 1`).
  // NOLINTNEXTLINE(google-explicit-constructor)
  SymbolicExpr(int64_t constant) {
    if (constant != 0) {
      terms_.emplace(Monomial{}, constant);
    }
  }

  static SymbolicExpr Symbol(const std::string& name) {
    SymbolicExpr expr;
    expr.terms_.emplace(Monomial{{name, 1}}, 1);
    return expr;
  }

  // True if no symbol appears -- the expression is a plain integer.
  bool IsConstant() const {
    return terms_.empty() || (terms_.size() == 1 && terms_.begin()->first.empty());
  }

  // Precondition: IsConstant().
  int64_t ConstantValue() const {
    if (terms_.empty()) {
      return 0;
    }
    return terms_.begin()->second;
  }

  // If this expression is exactly one pre-existing symbol raised to the
  // first power with coefficient 1 (i.e. it denotes nothing more than that
  // symbol itself), returns its name; otherwise returns nullptr. Combining
  // dimensions with an identity element (e.g. `0 + M`, `1 * M`) naturally
  // produces this shape, and callers use it to recover the original
  // `dim_param` instead of minting a redundant fresh symbol for it.
  const std::string* AsBareSymbol() const {
    if (terms_.size() != 1) {
      return nullptr;
    }
    const auto& [monomial, coefficient] = *terms_.begin();
    if (coefficient != 1 || monomial.size() != 1) {
      return nullptr;
    }
    const auto& [name, exponent] = *monomial.begin();
    return exponent == 1 ? &name : nullptr;
  }

  bool operator==(const SymbolicExpr& other) const {
    return terms_ == other.terms_;
  }
  bool operator!=(const SymbolicExpr& other) const {
    return !(*this == other);
  }
  // Arbitrary but total ordering, so SymbolicExpr can key an ordered
  // container (e.g. for expression -> symbol deduplication).
  bool operator<(const SymbolicExpr& other) const {
    return terms_ < other.terms_;
  }

  friend SymbolicExpr operator+(SymbolicExpr lhs, const SymbolicExpr& rhs) {
    for (const auto& [monomial, coefficient] : rhs.terms_) {
      lhs.AddTerm(monomial, coefficient);
    }
    return lhs;
  }

  friend SymbolicExpr operator-(SymbolicExpr lhs, const SymbolicExpr& rhs) {
    for (const auto& [monomial, coefficient] : rhs.terms_) {
      int64_t negated = 0;
      if (checked_mul_overflow(coefficient, int64_t{-1}, &negated)) {
        throw std::overflow_error("Integer overflow while negating a symbolic dimension expression");
      }
      lhs.AddTerm(monomial, negated);
    }
    return lhs;
  }

  friend SymbolicExpr operator*(const SymbolicExpr& lhs, const SymbolicExpr& rhs) {
    SymbolicExpr result;
    for (const auto& [lhs_monomial, lhs_coefficient] : lhs.terms_) {
      for (const auto& [rhs_monomial, rhs_coefficient] : rhs.terms_) {
        Monomial product_monomial = lhs_monomial;
        for (const auto& [name, exponent] : rhs_monomial) {
          product_monomial[name] += exponent;
        }
        int64_t product_coefficient = 0;
        if (checked_mul_overflow(lhs_coefficient, rhs_coefficient, &product_coefficient)) {
          throw std::overflow_error("Integer overflow while multiplying symbolic dimension expressions");
        }
        result.AddTerm(product_monomial, product_coefficient);
      }
    }
    return result;
  }

  // Exact division only: succeeds when `divisor` is a single monomial (e.g.
  // a plain constant, a bare symbol, or a product of symbols with a
  // coefficient) that evenly divides every term of `dividend`. Returns
  // nullopt when the divisor is a genuine polynomial sum (no general
  // rational-function support is intended) or the division would not be
  // exact -- never a truncated/approximate result.
  friend std::optional<SymbolicExpr> TryDivide(const SymbolicExpr& dividend, const SymbolicExpr& divisor) {
    if (divisor.terms_.size() != 1) {
      return std::nullopt;
    }
    const auto& [divisor_monomial, divisor_coefficient] = *divisor.terms_.begin();
    if (divisor_coefficient == 0) {
      return std::nullopt;
    }
    SymbolicExpr result;
    for (const auto& [dividend_monomial, dividend_coefficient] : dividend.terms_) {
      if (dividend_coefficient % divisor_coefficient != 0) {
        return std::nullopt;
      }
      Monomial quotient_monomial = dividend_monomial;
      for (const auto& [name, divisor_exponent] : divisor_monomial) {
        auto it = quotient_monomial.find(name);
        const int dividend_exponent = it == quotient_monomial.end() ? 0 : it->second;
        if (dividend_exponent < divisor_exponent) {
          return std::nullopt;
        }
        if (dividend_exponent == divisor_exponent) {
          quotient_monomial.erase(name);
        } else {
          it->second = dividend_exponent - divisor_exponent;
        }
      }
      result.AddTerm(quotient_monomial, dividend_coefficient / divisor_coefficient);
    }
    return result;
  }

  // Human-readable form for diagnostics, e.g. "2*M + N", "M*N", "0".
  std::string ToString() const {
    if (terms_.empty()) {
      return "0";
    }
    std::ostringstream out;
    bool first = true;
    for (const auto& [monomial, coefficient] : terms_) {
      if (!first) {
        out << (coefficient < 0 ? " - " : " + ");
      } else if (coefficient < 0) {
        out << "-";
      }
      first = false;
      const int64_t magnitude = coefficient < 0 ? -coefficient : coefficient;
      if (monomial.empty()) {
        out << magnitude;
        continue;
      }
      if (magnitude != 1) {
        out << magnitude << "*";
      }
      bool first_symbol = true;
      for (const auto& [name, exponent] : monomial) {
        if (!first_symbol) {
          out << "*";
        }
        first_symbol = false;
        out << name;
        if (exponent != 1) {
          out << "^" << exponent;
        }
      }
    }
    return out.str();
  }

 private:
  // A sorted symbol-name -> exponent multiset, e.g. {"M": 1, "N": 2} for
  // `M*N^2`. The empty monomial denotes the constant term (symbol-free).
  using Monomial = std::map<std::string, int>;
  // Canonical form: no zero coefficients, so two expressions denote the same
  // polynomial iff their `terms_` compare equal.
  std::map<Monomial, int64_t> terms_;

  void AddTerm(const Monomial& monomial, int64_t coefficient) {
    auto it = terms_.find(monomial);
    if (it == terms_.end()) {
      if (coefficient != 0) {
        terms_.emplace(monomial, coefficient);
      }
      return;
    }
    int64_t sum = 0;
    if (checked_add_overflow(it->second, coefficient, &sum)) {
      throw std::overflow_error("Integer overflow while combining symbolic dimension expressions");
    }
    if (sum == 0) {
      terms_.erase(it);
    } else {
      it->second = sum;
    }
  }
};

} // namespace ONNX_NAMESPACE
