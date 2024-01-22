#include <DCORA/DCORA_types.h>
#include <DCORA/DCORA_solver.h>
#include <DCORA/DCORA_robust.h>
#include <DCORA/Graph.h>
#include <DCORA/manifold/LiftedManifold.h>
#include <iostream>
#include <random>

#include "gtest/gtest.h"

using namespace DCORA;

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

TEST(testDCORA, testPartitionSEMatrix) {
  int d = 3;
  int r = 5;
  int n = 3;
  Matrix Y1 = randomStiefelVariable(r, d);
  Matrix Y2 = randomStiefelVariable(r, d);
  Matrix Y3 = randomStiefelVariable(r, d);
  Matrix p1 = randomEuclideanVariable(r);
  Matrix p2 = randomEuclideanVariable(r);
  Matrix p3 = randomEuclideanVariable(r);
  Matrix X_SE(r, (d + 1) * n);
  Matrix X_SE_R(r, d * n);
  Matrix X_SE_t(r, n);
  X_SE << Y1, p1, Y2, p2, Y3, p3;
  X_SE_R << Y1, Y2, Y3;
  X_SE_t << p1, p2, p3;
  auto [var_SE_R, var_SE_t] = partitionSEMatrix(X_SE, r, d, n);
  for (int i = 0; i < r; ++i) {
    for (int j = 0; j < d * n; ++j) {
      ASSERT_EQ(var_SE_R(i, j), X_SE_R(i, j));
    }
    for (int j = 0; j < n; ++j) {
      ASSERT_EQ(var_SE_t(i, j), X_SE_t(i, j));
    }
  }
}

TEST(testDCORA, testPartitionRAMatrix) {
  int d = 3;
  int r = 5;
  int n = 3;
  int l = 5;
  int b = 7;
  Matrix Y1 = randomStiefelVariable(r, d);
  Matrix Y2 = randomStiefelVariable(r, d);
  Matrix Y3 = randomStiefelVariable(r, d);
  Matrix p1 = randomEuclideanVariable(r);
  Matrix p2 = randomEuclideanVariable(r);
  Matrix p3 = randomEuclideanVariable(r);
  Matrix X_OB = randomObliqueVariable(r, l);
  Matrix X_E = randomEuclideanVariable(r, b);
  Matrix X_RA(r, (d + 1) * n + l + b);
  Matrix X_SE_R(r, d * n);
  Matrix X_SE_t(r, n);
  X_RA << Y1, Y2, Y3, X_OB, p1, p2, p3, X_E;
  X_SE_R << Y1, Y2, Y3;
  X_SE_t << p1, p2, p3;
  auto [var_SE_R, var_OB, var_SE_t, var_E] = partitionRAMatrix(X_RA, r, d, n, l, b);
  for (int i = 0; i < r; ++i) {
    for (int j = 0; j < d * n; ++j) {
      ASSERT_EQ(var_SE_R(i, j), X_SE_R(i, j));
    }
    for (int j = 0; j < l; ++j) {
      ASSERT_EQ(var_OB(i, j), X_OB(i, j));
    }
    for (int j = 0; j < n; ++j) {
      ASSERT_EQ(var_SE_t(i, j), X_SE_t(i, j));
    }
    for (int j = 0; j < b; ++j) {
      ASSERT_EQ(var_E(i, j), X_E(i, j));
    }
  }
}

TEST(testDCORA, testCreateSEMatrix) {
  int d = 3;
  int r = 5;
  int n = 3;
  Matrix Y1 = randomStiefelVariable(r, d);
  Matrix Y2 = randomStiefelVariable(r, d);
  Matrix Y3 = randomStiefelVariable(r, d);
  Matrix p1 = randomEuclideanVariable(r);
  Matrix p2 = randomEuclideanVariable(r);
  Matrix p3 = randomEuclideanVariable(r);
  Matrix X_SE(r, (d + 1) * n);
  Matrix X_SE_R(r, d * n);
  Matrix X_SE_t(r, n);
  X_SE << Y1, p1, Y2, p2, Y3, p3;
  X_SE_R << Y1, Y2, Y3;
  X_SE_t << p1, p2, p3;
  Matrix var_X_SE = createSEMatrix(X_SE_R, X_SE_t);
  for (int i = 0; i < r; ++i) {
    for (int j = 0; j < (d + 1) * n; ++j) {
      ASSERT_EQ(var_X_SE(i, j), X_SE(i, j));
    }
  }
}

TEST(testDCORA, testCreateRAMatrix) {
  int d = 3;
  int r = 5;
  int n = 3;
  int l = 5;
  int b = 7;
  Matrix Y1 = randomStiefelVariable(r, d);
  Matrix Y2 = randomStiefelVariable(r, d);
  Matrix Y3 = randomStiefelVariable(r, d);
  Matrix p1 = randomEuclideanVariable(r);
  Matrix p2 = randomEuclideanVariable(r);
  Matrix p3 = randomEuclideanVariable(r);
  Matrix X_OB = randomObliqueVariable(r, l);
  Matrix X_E = randomEuclideanVariable(r, b);
  Matrix X_RA(r, (d + 1) * n + l + b);
  Matrix X_SE_R(r, d * n);
  Matrix X_SE_t(r, n);
  X_RA << Y1, Y2, Y3, X_OB, p1, p2, p3, X_E;
  X_SE_R << Y1, Y2, Y3;
  X_SE_t << p1, p2, p3;
  Matrix var_X_RA = createRAMatrix(X_SE_R, X_OB, X_SE_t, X_E);
  for (int i = 0; i < r; ++i) {
    for (int j = 0; j < (d + 1) * n + l + b; ++j) {
      ASSERT_EQ(var_X_RA(i, j), X_RA(i, j));
    }
  }
}
