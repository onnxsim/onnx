// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <limits>

#include "gtest/gtest.h"
#include "onnx/shape_inference/symbolic_expr.h"

namespace ONNX_NAMESPACE::Test {

TEST(SymbolicExprTest, ConstantIsConstant) {
  SymbolicExpr zero;
  EXPECT_TRUE(zero.IsConstant());
  EXPECT_EQ(zero.ConstantValue(), 0);

  SymbolicExpr five(5);
  EXPECT_TRUE(five.IsConstant());
  EXPECT_EQ(five.ConstantValue(), 5);
}

TEST(SymbolicExprTest, SymbolIsNotConstantAndIsBare) {
  auto m = SymbolicExpr::Symbol("M");
  EXPECT_FALSE(m.IsConstant());
  ASSERT_NE(m.AsBareSymbol(), nullptr);
  EXPECT_EQ(*m.AsBareSymbol(), "M");
}

TEST(SymbolicExprTest, AdditionIsCommutative) {
  auto m = SymbolicExpr::Symbol("M");
  auto n = SymbolicExpr::Symbol("N");
  EXPECT_EQ(m + n, n + m);
}

TEST(SymbolicExprTest, AddingIdenticalSymbolsProducesCoefficient) {
  auto m = SymbolicExpr::Symbol("M");
  auto two_m = m + m;
  EXPECT_FALSE(two_m.IsConstant());
  EXPECT_EQ(two_m.AsBareSymbol(), nullptr);
  EXPECT_EQ(two_m.ToString(), "2*M");
}

TEST(SymbolicExprTest, DistinctSymbolsAreNotEqual) {
  auto m_plus_n = SymbolicExpr::Symbol("M") + SymbolicExpr::Symbol("N");
  auto m = SymbolicExpr::Symbol("M");
  EXPECT_NE(m_plus_n, m);
}

TEST(SymbolicExprTest, ZeroPlusSymbolIsBareSymbol) {
  auto result = SymbolicExpr(0) + SymbolicExpr::Symbol("M");
  ASSERT_NE(result.AsBareSymbol(), nullptr);
  EXPECT_EQ(*result.AsBareSymbol(), "M");
}

TEST(SymbolicExprTest, OnePlusMinusOneCancelsToZero) {
  auto m = SymbolicExpr::Symbol("M");
  auto result = (m + SymbolicExpr(1)) - (m + SymbolicExpr(1));
  EXPECT_TRUE(result.IsConstant());
  EXPECT_EQ(result.ConstantValue(), 0);
}

TEST(SymbolicExprTest, SubtractionOfSymbolFromItselfIsZero) {
  auto m = SymbolicExpr::Symbol("M");
  auto result = m - m;
  EXPECT_TRUE(result.IsConstant());
  EXPECT_EQ(result.ConstantValue(), 0);
}

TEST(SymbolicExprTest, MultiplicationOfDistinctSymbols) {
  auto product = SymbolicExpr::Symbol("M") * SymbolicExpr::Symbol("N");
  EXPECT_FALSE(product.IsConstant());
  EXPECT_EQ(product.AsBareSymbol(), nullptr);
  EXPECT_EQ(product.ToString(), "M*N");
}

TEST(SymbolicExprTest, MultiplicationOfSameSymbolProducesPower) {
  auto m = SymbolicExpr::Symbol("M");
  EXPECT_EQ((m * m).ToString(), "M^2");
}

TEST(SymbolicExprTest, DistributesOverAddition) {
  auto m = SymbolicExpr::Symbol("M");
  auto n = SymbolicExpr::Symbol("N");
  auto k = SymbolicExpr::Symbol("K");
  // K*(M+N) == K*M + K*N
  EXPECT_EQ(k * (m + n), (k * m) + (k * n));
}

TEST(SymbolicExprTest, TryDivideExactByConstant) {
  auto two_m = SymbolicExpr::Symbol("M") + SymbolicExpr::Symbol("M");
  auto result = TryDivide(two_m, SymbolicExpr(2));
  ASSERT_TRUE(result.has_value());
  ASSERT_NE(result->AsBareSymbol(), nullptr);
  EXPECT_EQ(*result->AsBareSymbol(), "M");
}

TEST(SymbolicExprTest, TryDivideInexactByConstantFails) {
  auto three_m = SymbolicExpr::Symbol("M") * SymbolicExpr(3);
  EXPECT_FALSE(TryDivide(three_m, SymbolicExpr(2)).has_value());
}

TEST(SymbolicExprTest, TryDividePolynomialSumDivisorFails) {
  auto m = SymbolicExpr::Symbol("M");
  auto n = SymbolicExpr::Symbol("N");
  // (M*N) / (M+N) is not a single-monomial divisor, so this must fail even
  // though it happens not to be a valid factorization anyway.
  EXPECT_FALSE(TryDivide(m * n, m + n).has_value());
}

TEST(SymbolicExprTest, TryDivideBySymbolThatDoesNotDivide) {
  auto m = SymbolicExpr::Symbol("M");
  auto n = SymbolicExpr::Symbol("N");
  EXPECT_FALSE(TryDivide(m, n).has_value());
}

TEST(SymbolicExprTest, TryDivideMonomialBySymbol) {
  auto m = SymbolicExpr::Symbol("M");
  auto n = SymbolicExpr::Symbol("N");
  auto result = TryDivide(m * n, n);
  ASSERT_TRUE(result.has_value());
  ASSERT_NE(result->AsBareSymbol(), nullptr);
  EXPECT_EQ(*result->AsBareSymbol(), "M");
}

TEST(SymbolicExprTest, ToStringFormatsMultipleTerms) {
  auto expr = SymbolicExpr::Symbol("M") + SymbolicExpr::Symbol("N") + SymbolicExpr(5);
  EXPECT_EQ(expr.ToString(), "5 + M + N");
}

TEST(SymbolicExprTest, ToStringOfZeroIsZero) {
  EXPECT_EQ(SymbolicExpr().ToString(), "0");
}

TEST(SymbolicExprTest, AdditionOverflowThrows) {
  auto max_expr = SymbolicExpr(std::numeric_limits<int64_t>::max()) + SymbolicExpr::Symbol("M");
  EXPECT_THROW({ auto _ = max_expr + SymbolicExpr(1); }, std::overflow_error);
}

TEST(SymbolicExprTest, MultiplicationOverflowThrows) {
  auto huge = SymbolicExpr(std::numeric_limits<int64_t>::max());
  EXPECT_THROW({ auto _ = huge * SymbolicExpr(2); }, std::overflow_error);
}

TEST(SymbolicExprTest, LessThanIsAStrictTotalOrderSuitableForMapKeys) {
  auto m = SymbolicExpr::Symbol("M");
  auto n = SymbolicExpr::Symbol("N");
  EXPECT_FALSE(m < m);
  EXPECT_TRUE((m < n) != (n < m) || m == n);
}

} // namespace ONNX_NAMESPACE::Test
