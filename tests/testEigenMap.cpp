#include <DCORA/DCORA_types.h>
#include <DCORA/DCORA_utils.h>
#include <DCORA/manifold/LiftedSEManifold.h>
#include <DCORA/manifold/LiftedSEVariable.h>
#include <DCORA/manifold/LiftedRAManifold.h>
#include <DCORA/manifold/LiftedRAVariable.h>
#include <iostream>
#include <random>

#include "gtest/gtest.h"

using namespace DCORA;

TEST(testDCORA, EigenMapSE) {
  size_t d = 3;
  size_t n = 10;
  LiftedSEVariable x(d, d, n);
  x.var()->RandInManifold();

  // View the internal memory of x as a read-only eigen matrix
  Eigen::Map<const Matrix> xMatConst((double *) x.var()->ObtainReadData(), d, (d + 1) * n);
  ASSERT_LE((xMatConst - x.getData()).norm(), 1e-4);

  // View the internal memory of x as a writable eigen matrix
  Eigen::Map<Matrix> xMat((double *) x.var()->ObtainWriteEntireData(), d, (d + 1) * n);

  // Modify x through eigen map
  for (size_t i = 0; i < n; ++i) {
    xMat.block(0, i * (d+1),     d, d) = Matrix::Identity(d, d);
    xMat.block(0, i * (d+1) + d, d, 1) = Matrix::Zero(d, 1);
  }

  // Check that the internal value of x is modified accordingly
  ASSERT_LE((xMat - x.getData()).norm(), 1e-4);

  xMat = Matrix::Random(d, (d + 1) *n);
  ASSERT_LE((xMat - x.getData()).norm(), 1e-4);
}

TEST(testDCORA, EigenMapRA) {
  size_t d = 3;
  size_t n = 10;
  size_t b = 5;
  size_t l = 5;
  LiftedRAVariable x(d, d, n, b, l);
  x.var()->RandInManifold();

  // View the internal memory of x as a read-only eigen matrix
  Eigen::Map<const Matrix> xMatConst((double *) x.var()->ObtainReadData(), d, (d + 1) * n + b + l);
  ASSERT_LE((xMatConst - x.getData()).norm(), 1e-4);

  // View the internal memory of x as a writable eigen matrix
  Eigen::Map<Matrix> xMat((double *) x.var()->ObtainWriteEntireData(), d, (d + 1) * n + b + l);

  // Modify x through eigen map
  auto [X_SE, X_E, X_OB] = partitionRAMatrix(xMat, d, d, n, b , l);
  for (size_t i = 0; i < n; ++i) {
    X_SE.block(0, i * (d+1),     d, d) = Matrix::Identity(d, d);
    X_SE.block(0, i * (d+1) + d, d, 1) = Matrix::Zero(d, 1);
  }
  for (size_t i = 0; i < b; ++i) {
    X_E.block(0, i, d, 1) = Matrix::Zero(d, 1);
  }
  for (size_t i = 0; i < l; ++i) {
    X_OB.block(0, i, d, 1) = Matrix::Zero(d, 1);
  }
  xMat = createRAMatrix(X_SE, X_E, X_OB);

  // Check that the internal value of x is modified accordingly
  ASSERT_LE((xMat - x.getData()).norm(), 1e-4);

  xMat = Matrix::Random(d, (d + 1) *n);
  ASSERT_LE((xMat - x.getData()).norm(), 1e-4);
}
