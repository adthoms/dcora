#include <DCORA/DCORA_robust.h>
#include <DCORA/DCORA_solver.h>
#include <DCORA/DCORA_types.h>
#include <DCORA/Graphs.h>
#include <DCORA/manifold/LiftedManifold.h>

#include <iostream>
#include <random>

#include "gtest/gtest.h"

TEST(testDCORA, testStiefelGeneration) {
  DCORA::Matrix Y = DCORA::fixedStiefelVariable(5, 3);
  DCORA::Matrix I = DCORA::Matrix::Identity(3, 3);
  DCORA::Matrix D = Y.transpose() * Y - I;
  ASSERT_LE(D.norm(), 1e-5);
}

TEST(testDCORA, testStiefelRepeat) {
  DCORA::Matrix Y = DCORA::fixedStiefelVariable(5, 3);
  for (size_t i = 0; i < 10; ++i) {
    DCORA::Matrix Y_ = DCORA::fixedStiefelVariable(5, 3);
    ASSERT_LE((Y_ - Y).norm(), 1e-5);
  }
}

TEST(testDCORA, testStiefelProjection) {
  size_t d = 3;
  size_t r = 5;
  DCORA::Matrix I = DCORA::Matrix::Identity(d, d);
  for (size_t j = 0; j < 50; ++j) {
    DCORA::Matrix M = DCORA::Matrix::Random(r, d);
    DCORA::Matrix Y = DCORA::projectToStiefelManifold(M);
    DCORA::Matrix D = Y.transpose() * Y - I;
    ASSERT_LE(D.norm(), 1e-5);
  }
}

TEST(testDCORA, testObliqueProjection) {
  size_t d = 3;
  size_t r = 5;
  DCORA::Matrix I = DCORA::Matrix::Identity(d, d);
  for (size_t j = 0; j < 50; ++j) {
    DCORA::Matrix M = DCORA::Matrix::Random(r, d);
    DCORA::Matrix Y = DCORA::projectToObliqueManifold(M);
    DCORA::Matrix D = (Y.transpose() * Y).diagonal() - I.diagonal();
    ASSERT_LE(D.norm(), 1e-5);
  }
}

TEST(testDCORA, testLiftedSEManifoldProjection) {
  int d = 3;
  int r = 5;
  int n = 100;
  DCORA::LiftedSEManifold Manifold(r, d, n);
  DCORA::Matrix M = DCORA::Matrix::Random(r, (d + 1) * n);
  DCORA::Matrix X = Manifold.project(M);
  ASSERT_EQ(X.rows(), r);
  ASSERT_EQ(X.cols(), (d + 1) * n);
  for (int i = 0; i < n; ++i) {
    DCORA::Matrix Y = X.block(0, i * (d + 1), r, d);
    DCORA::Matrix D = Y.transpose() * Y - DCORA::Matrix::Identity(d, d);
    ASSERT_LE(D.norm(), 1e-5);
  }
}

TEST(testDCORA, testChi2Inv) {
  unsigned dof = 4;
  double quantile = 0.95;
  double threshold = DCORA::chi2inv(quantile, dof);
  std::random_device rd;  // obtain a seed for the random number engine
  std::mt19937 rng(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::chi_squared_distribution<double> distribution(dof);
  int numTrials = 100000;
  int count = 0;
  for (int i = 0; i < numTrials; ++i) {
    double number = distribution(rng);
    if (number < threshold)
      count++;
  }
  double q = static_cast<double>(count) / numTrials;
  ASSERT_LE(abs(q - quantile), 0.01);
}

TEST(testDCORA, testPartitionSEMatrix) {
  int d = 3;
  int r = 5;
  int n = 3;
  DCORA::Matrix Y0 = DCORA::randomStiefelVariable(r, d);
  DCORA::Matrix Y1 = DCORA::randomStiefelVariable(r, d);
  DCORA::Matrix Y2 = DCORA::randomStiefelVariable(r, d);
  DCORA::Matrix p0 = DCORA::randomEuclideanVariable(r);
  DCORA::Matrix p1 = DCORA::randomEuclideanVariable(r);
  DCORA::Matrix p2 = DCORA::randomEuclideanVariable(r);
  DCORA::Matrix X_SE(r, (d + 1) * n);
  DCORA::Matrix X_SE_R(r, d * n);
  DCORA::Matrix X_SE_t(r, n);
  X_SE << Y0, p0, Y1, p1, Y2, p2;
  X_SE_R << Y0, Y1, Y2;
  X_SE_t << p0, p1, p2;
  auto [var_SE_R, var_SE_t] = DCORA::partitionSEMatrix(X_SE, r, d, n);
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
  DCORA::Matrix Y0 = DCORA::randomStiefelVariable(r, d);
  DCORA::Matrix Y1 = DCORA::randomStiefelVariable(r, d);
  DCORA::Matrix Y2 = DCORA::randomStiefelVariable(r, d);
  DCORA::Matrix p0 = DCORA::randomEuclideanVariable(r);
  DCORA::Matrix p1 = DCORA::randomEuclideanVariable(r);
  DCORA::Matrix p2 = DCORA::randomEuclideanVariable(r);
  DCORA::Matrix X_OB = DCORA::randomObliqueVariable(r, l);
  DCORA::Matrix X_E = DCORA::randomEuclideanVariable(r, b);
  DCORA::Matrix X_RA(r, (d + 1) * n + l + b);
  DCORA::Matrix X_SE_R(r, d * n);
  DCORA::Matrix X_SE_t(r, n);
  X_RA << Y0, Y1, Y2, X_OB, p0, p1, p2, X_E;
  X_SE_R << Y0, Y1, Y2;
  X_SE_t << p0, p1, p2;
  auto [var_SE_R, var_OB, var_SE_t, var_E] =
      DCORA::partitionRAMatrix(X_RA, r, d, n, l, b);
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
  DCORA::Matrix Y0 = DCORA::randomStiefelVariable(r, d);
  DCORA::Matrix Y1 = DCORA::randomStiefelVariable(r, d);
  DCORA::Matrix Y2 = DCORA::randomStiefelVariable(r, d);
  DCORA::Matrix p0 = DCORA::randomEuclideanVariable(r);
  DCORA::Matrix p1 = DCORA::randomEuclideanVariable(r);
  DCORA::Matrix p2 = DCORA::randomEuclideanVariable(r);
  DCORA::Matrix X_SE(r, (d + 1) * n);
  DCORA::Matrix X_SE_R(r, d * n);
  DCORA::Matrix X_SE_t(r, n);
  X_SE << Y0, p0, Y1, p1, Y2, p2;
  X_SE_R << Y0, Y1, Y2;
  X_SE_t << p0, p1, p2;
  DCORA::Matrix var_X_SE = DCORA::createSEMatrix(X_SE_R, X_SE_t);
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
  DCORA::Matrix Y0 = DCORA::randomStiefelVariable(r, d);
  DCORA::Matrix Y1 = DCORA::randomStiefelVariable(r, d);
  DCORA::Matrix Y2 = DCORA::randomStiefelVariable(r, d);
  DCORA::Matrix p0 = DCORA::randomEuclideanVariable(r);
  DCORA::Matrix p1 = DCORA::randomEuclideanVariable(r);
  DCORA::Matrix p2 = DCORA::randomEuclideanVariable(r);
  DCORA::Matrix X_OB = DCORA::randomObliqueVariable(r, l);
  DCORA::Matrix X_E = DCORA::randomEuclideanVariable(r, b);
  DCORA::Matrix X_RA(r, (d + 1) * n + l + b);
  DCORA::Matrix X_SE_R(r, d * n);
  DCORA::Matrix X_SE_t(r, n);
  X_RA << Y0, Y1, Y2, X_OB, p0, p1, p2, X_E;
  X_SE_R << Y0, Y1, Y2;
  X_SE_t << p0, p1, p2;
  DCORA::Matrix var_X_RA = DCORA::createRAMatrix(X_SE_R, X_OB, X_SE_t, X_E);
  for (int i = 0; i < r; ++i) {
    for (int j = 0; j < (d + 1) * n + l + b; ++j) {
      ASSERT_EQ(var_X_RA(i, j), X_RA(i, j));
    }
  }
}
