#include <DCORA/DCORA_types.h>
#include <DCORA/DCORA_solver.h>
#include <DCORA/DCORA_robust.h>
#include <DCORA/PoseGraph.h>
#include <DCORA/manifold/LiftedSEManifold.h>
#include <iostream>
#include <random>

#include "gtest/gtest.h"

using namespace DCORA;

TEST(testDCORA, testConvertMatrixTypeToVectorType) {
  Matrix A0 = Matrix::Random(3,1);
  Vector P0 = convertMatrixTypeToVectorType(A0);
  ASSERT_EQ(P0(0), A0(0,0));
  ASSERT_EQ(P0(1), A0(1,0));
  ASSERT_EQ(P0(2), A0(2,0));
  Matrix A1 = Matrix::Random(3,2);
  EXPECT_DEATH({
    Vector P1 = convertMatrixTypeToVectorType(A1);
  }, ".*");
}

TEST(testDCORA, testConvertVectorTypeToMatrixType) {
  Vector P0 = Vector::Random(3);
  Matrix A0 = convertVectorTypeToMatrixType(P0);
  ASSERT_EQ(P0(0), A0(0,0));
  ASSERT_EQ(P0(1), A0(1,0));
  ASSERT_EQ(P0(2), A0(2,0));
}

TEST(testDCORA, testStiefelGeneration) {
  Matrix Y = fixedStiefelVariable(5, 3);
  Matrix I = Matrix::Identity(3, 3);
  Matrix D = Y.transpose() * Y - I;
  ASSERT_LE(D.norm(), 1e-5);
}

TEST(testDCORA, testStiefelRepeat) {
  Matrix Y = fixedStiefelVariable(5, 3);
  for (size_t i = 0; i < 10; ++i) {
    Matrix Y_ = fixedStiefelVariable(5, 3);
    ASSERT_LE((Y_ - Y).norm(), 1e-5);
  }
}

TEST(testDCORA, testStiefelProjection) {
  size_t d = 3;
  size_t r = 5;
  Matrix I = Matrix::Identity(d, d);
  for (size_t j = 0; j < 50; ++j) {
    Matrix M = Matrix::Random(r, d);
    Matrix Y = projectToStiefelManifold(M);
    Matrix D = Y.transpose() * Y - I;
    ASSERT_LE(D.norm(), 1e-5);
  }
}

TEST(testDCORA, testObliqueProjection) {
  size_t d = 3;
  size_t r = 5;
  Matrix I = Matrix::Identity(d, d);
  for (size_t j = 0; j < 50; ++j) {
    Matrix M = Matrix::Random(r, d);
    Matrix Y = projectToObliqueManifold(M);
    Matrix D = (Y.transpose() * Y).diagonal() - I.diagonal();
    ASSERT_LE(D.norm(), 1e-5);
  }
}

TEST(testDCORA, testLiftedSEManifoldProjection) {
  int d = 3;
  int r = 5;
  int n = 100;
  LiftedSEManifold Manifold(r, d, n);
  Matrix M = Matrix::Random(r, (d + 1) * n);
  Matrix X = Manifold.project(M);
  ASSERT_EQ(X.rows(), r);
  ASSERT_EQ(X.cols(), (d + 1) * n);
  for (int i = 0; i < n; ++i) {
    Matrix Y = X.block(0, i * (d + 1), r, d);
    Matrix D = Y.transpose() * Y - Matrix::Identity(d, d);
    ASSERT_LE(D.norm(), 1e-5);
  }
}

TEST(testDCORA, testChi2Inv) {
  unsigned dof = 4;
  double quantile = 0.95;
  double threshold = chi2inv(quantile, dof);
  std::random_device rd;  // Will be used to obtain a seed for the random number engine
  std::mt19937 rng(rd());  // Standard mersenne_twister_engine seeded with rd()
  std::chi_squared_distribution<double> distribution(dof);
  int numTrials = 100000;
  int count = 0;
  for (int i = 0; i < numTrials; ++i) {
    double number = distribution(rng);
    if (number < threshold) count++;
  }
  double q = (double) count / numTrials;
  ASSERT_LE(abs(q - quantile), 0.01);
}
