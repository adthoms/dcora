/* -----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * Copyright 2024, University of California Los Angeles, * Los Angeles, CA 90095
 * All Rights Reserved
 * Authors: Yulun Tian, Alexander Thoms, Alan Papalia, et al.
 *  - For dpgo's full author list, see:
 *  https://github.com/mit-acl/dpgo/blob/main/README.md
 *  - For dcora's full author list, see dcora/README.md
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DCORA/DCORA_types.h>
#include <DCORA/DCORA_utils.h>
#include <DCORA/manifold/LiftedManifold.h>
#include <DCORA/manifold/LiftedVariable.h>
#include <DCORA/manifold/LiftedVector.h>

#include <iostream>
#include <random>

#include "gtest/gtest.h"

TEST(testDCORA, testLiftedSEVariable) {
  size_t r = 5;
  size_t d = 3;
  size_t n = 10;
  DCORA::LiftedSEVariable var(r, d, n);
  // Get random data in SE manifold
  DCORA::LiftedPoseArray pose_array(r, d, n);
  pose_array.setRandomData();
  DCORA::Matrix Y = pose_array.getData();
  // Test copy constructor
  DCORA::LiftedSEVariable var2(var);
  ASSERT_LE((var.getData() - var2.getData()).norm(), 1e-6);
  // Test assignment from LiftedSEVariable
  DCORA::LiftedSEVariable var3(r, d, n);
  var3 = var;
  ASSERT_LE((var.getData() - var3.getData()).norm(), 1e-6);
  // Test assignment from pose array
  DCORA::LiftedSEVariable var4(pose_array);
  ASSERT_LE((Y - var4.getData()).norm(), 1e-6);
  // Test setter and getter methods
  var.setData(Y);
  ASSERT_LE((var.getData() - Y).norm(), 1e-6);
}

TEST(testDCORA, testLiftedRAVariable) {
  size_t r = 5;
  size_t d = 3;
  size_t n = 10;
  std::vector<size_t> l_cases = {0, 6, 0, 6};
  std::vector<size_t> b_cases = {0, 0, 7, 7};
  for (size_t i = 0; i < l_cases.size(); i++) {
    size_t l = l_cases.at(i);
    size_t b = b_cases.at(i);
    DCORA::LiftedRAVariable var(r, d, n, l, b);
    // Get random data in RA manifold
    DCORA::LiftedRangeAidedArray range_aided_array(r, d, n, l, b);
    range_aided_array.setRandomData();
    DCORA::Matrix Y = range_aided_array.getData();
    // Test copy constructor
    DCORA::LiftedRAVariable var2(var);
    ASSERT_LE((var.getData() - var2.getData()).norm(), 1e-6);
    // Test assignment from LiftedRAVariable
    DCORA::LiftedRAVariable var3(r, d, n, l, b);
    var3 = var;
    ASSERT_LE((var.getData() - var3.getData()).norm(), 1e-6);
    // Test assignment from range aided array
    DCORA::LiftedRAVariable var4(range_aided_array);
    ASSERT_LE((Y - var4.getData()).norm(), 1e-6);
    // Test setter and getter methods
    var.setData(Y);
    ASSERT_LE((var.getData() - Y).norm(), 1e-6);
  }
}

TEST(testDCORA, testLiftedSEVariableRandomManifold) {
  size_t r = 5;
  size_t d = 3;
  size_t n = 10;
  for (int trial = 0; trial < 50; ++trial) {
    DCORA::LiftedSEVariable var(r, d, n);
    var.setRandomData();
    DCORA::Matrix Y = var.getData();
    DCORA::Matrix M = DCORA::projectToSEMatrix(Y, r, d, n);
    ASSERT_LE((M - Y).norm(), 1e-6);
  }
}

TEST(testDCORA, testLiftedRAVariableRandomManifold) {
  size_t r = 5;
  size_t d = 3;
  size_t n = 10;
  std::vector<size_t> l_cases = {0, 6, 0, 6};
  std::vector<size_t> b_cases = {0, 0, 7, 7};
  for (size_t i = 0; i < l_cases.size(); i++) {
    size_t l = l_cases.at(i);
    size_t b = b_cases.at(i);
    for (int trial = 0; trial < 50; ++trial) {
      DCORA::LiftedRAVariable var(r, d, n, l, b);
      var.setRandomData();
      DCORA::Matrix Y = var.getData();
      DCORA::Matrix M = DCORA::projectToRAMatrix(Y, r, d, n, l, b);
      ASSERT_LE((M - Y).norm(), 1e-6);
    }
  }
}

TEST(testDCORA, testLiftedSEVariableEigenMap) {
  size_t r = 5;
  size_t d = 3;
  size_t n = 10;
  size_t k = (d + 1) * n;
  DCORA::LiftedSEVariable x(r, d, n);
  x.setRandomData();

  // View the internal memory of x as a read-only Eigen matrix
  Eigen::Map<const DCORA::Matrix> xMatConst(
      const_cast<double *>(x.var()->ObtainReadData()), r, k);
  ASSERT_LE((xMatConst - x.getData()).norm(), 1e-6);

  // View the internal memory of x as a writable Eigen matrix
  Eigen::Map<DCORA::Matrix> xMat(
      const_cast<double *>(x.var()->ObtainWriteEntireData()), r, k);

  // Modify x through Eigen map
  for (size_t i = 0; i < n; ++i) {
    xMat.block(0, i * (d + 1), r, d) = DCORA::randomStiefelVariable(r, d);
    xMat.col(i * (d + 1) + d) = DCORA::randomEuclideanVariable(r);
  }

  // Check that the internal value of x is modified accordingly
  ASSERT_LE((xMat - x.getData()).norm(), 1e-6);

  xMat = DCORA::Matrix::Random(r, k);
  ASSERT_LE((xMat - x.getData()).norm(), 1e-6);

  // Modify x through Eigen map
  for (size_t i = 0; i < n; ++i) {
    x.rotation(i) = DCORA::randomStiefelVariable(r, d);
    x.translation(i) = DCORA::randomEuclideanVariable(r);
  }

  // Check that the internal value of x is modified accordingly
  ASSERT_LE((xMat - x.getData()).norm(), 1e-6);

  // Check that the internal value of x remains in the SE manifold
  DCORA::Matrix X_proj = DCORA::projectToSEMatrix(x.getData(), r, d, n);
  ASSERT_LE((X_proj - x.getData()).norm(), 1e-6);
}

TEST(testDCORA, testLiftedRAVariableEigenMap) {
  size_t r = 5;
  size_t d = 3;
  size_t n = 10;
  std::vector<size_t> l_cases = {0, 6, 0, 6};
  std::vector<size_t> b_cases = {0, 0, 7, 7};
  for (size_t i = 0; i < l_cases.size(); i++) {
    size_t l = l_cases.at(i);
    size_t b = b_cases.at(i);
    size_t k = (d + 1) * n + l + b;
    DCORA::LiftedRAVariable x(r, d, n, l, b);
    x.setRandomData();

    // View the internal memory of x as a read-only Eigen matrix
    Eigen::Map<const DCORA::Matrix> xMatConst(
        const_cast<double *>(x.var()->ObtainReadData()), r, k);
    ASSERT_LE((xMatConst - x.getData()).norm(), 1e-6);

    // View the internal memory of x as a writable Eigen matrix
    Eigen::Map<DCORA::Matrix> xMat(
        const_cast<double *>(x.var()->ObtainWriteEntireData()), r, k);

    // Modify x through Eigen map
    for (size_t i = 0; i < n; ++i) {
      xMat.block(0, i * d, r, d) = DCORA::randomStiefelVariable(r, d);
      xMat.col(i + (d * n) + l) = DCORA::randomEuclideanVariable(r);
    }
    for (size_t i = 0; i < l; ++i) {
      xMat.col(i + (d * n)) = DCORA::randomObliqueVariable(r);
    }
    for (size_t i = 0; i < b; ++i) {
      xMat.col(i + (d * n) + l + n) = DCORA::randomEuclideanVariable(r);
    }

    // Check that the internal value of x is modified accordingly
    ASSERT_LE((xMat - x.getData()).norm(), 1e-6);

    xMat = DCORA::Matrix::Random(r, k);
    ASSERT_LE((xMat - x.getData()).norm(), 1e-6);

    // Modify x through Eigen map
    for (size_t i = 0; i < n; ++i) {
      x.rotation(i) = DCORA::randomStiefelVariable(r, d);
      x.translation(i) = DCORA::randomEuclideanVariable(r);
    }
    for (size_t i = 0; i < l; ++i) {
      x.unitSphere(i) = DCORA::randomObliqueVariable(r);
    }
    for (size_t i = 0; i < b; ++i) {
      x.landmark(i) = DCORA::randomEuclideanVariable(r);
    }

    // Check that the internal value of x is modified accordingly
    ASSERT_LE((xMat - x.getData()).norm(), 1e-6);

    // Check that the internal value of x remains in the RA manifold
    DCORA::Matrix X_proj = DCORA::projectToRAMatrix(x.getData(), r, d, n, l, b);
    ASSERT_LE((X_proj - x.getData()).norm(), 1e-6);
  }
}

TEST(testDCORA, testLiftedSEVariableStateGetters) {
  size_t r = 5;
  size_t d = 3;
  size_t n = 10;
  size_t k = (d + 1) * n;
  DCORA::LiftedSEVariable x(r, d, n);
  DCORA::Matrix M = DCORA::Matrix::Random(r, k);
  DCORA::Matrix X = DCORA::projectToSEMatrix(M, r, d, n);
  x.setData(X);

  // Check pose getters
  auto [X_SE_R, X_SE_t] = DCORA::partitionSEMatrix(X, r, d, n);
  for (size_t i = 0; i < n; ++i) {
    DCORA::Matrix X_SE_R_block = X_SE_R.block(0, i * d, r, d);
    DCORA::Matrix X_SE_t_block = X_SE_t.col(i);
    DCORA::Matrix X_SE_block =
        DCORA::createSEMatrix(X_SE_R_block, X_SE_t_block);
    ASSERT_LE((x.pose(i) - X_SE_block).norm(), 1e-6);
    ASSERT_LE((x.rotation(i) - X_SE_R_block).norm(), 1e-6);
    ASSERT_LE((x.translation(i) - X_SE_t_block).norm(), 1e-6);
  }
}

TEST(testDCORA, testLiftedRAVariableStateGetters) {
  size_t r = 5;
  size_t d = 3;
  size_t n = 10;
  std::vector<size_t> l_cases = {0, 6, 0, 6};
  std::vector<size_t> b_cases = {0, 0, 7, 7};
  for (size_t i = 0; i < l_cases.size(); i++) {
    size_t l = l_cases.at(i);
    size_t b = b_cases.at(i);
    size_t k = (d + 1) * n + l + b;
    DCORA::LiftedRAVariable x(r, d, n, l, b);
    DCORA::Matrix M = DCORA::Matrix::Random(r, k);
    DCORA::Matrix X = DCORA::projectToRAMatrix(M, r, d, n, l, b);
    x.setData(X);

    // Check pose getters
    auto [X_SE_R, X_OB, X_SE_t, X_E] =
        DCORA::partitionRAMatrix(X, r, d, n, l, b);
    for (size_t i = 0; i < n; ++i) {
      DCORA::Matrix X_SE_R_block = X_SE_R.block(0, i * d, r, d);
      DCORA::Matrix X_SE_t_block = X_SE_t.col(i);
      DCORA::Matrix X_SE_block =
          DCORA::createSEMatrix(X_SE_R_block, X_SE_t_block);
      ASSERT_LE((x.pose(i) - X_SE_block).norm(), 1e-6);
      ASSERT_LE((x.rotation(i) - X_SE_R_block).norm(), 1e-6);
      ASSERT_LE((x.translation(i) - X_SE_t_block).norm(), 1e-6);
    }
    // Check unit sphere getters
    for (size_t i = 0; i < l; ++i) {
      ASSERT_LE((x.unitSphere(i) - X_OB.col(i)).norm(), 1e-6);
    }
    // Check landmark getters
    for (size_t i = 0; i < b; ++i) {
      ASSERT_LE((x.landmark(i) - X_E.col(i)).norm(), 1e-6);
    }
  }
}

TEST(testDCORA, testLiftedSEVector) {
  size_t r = 5;
  size_t d = 3;
  size_t n = 10;
  size_t k = (d + 1) * n;
  DCORA::Matrix M = DCORA::Matrix::Random(r, k);
  DCORA::Matrix X = DCORA::projectToSEMatrix(M, r, d, n);
  // Test setter and getter methods
  DCORA::LiftedSEVector vec(r, d, n);
  vec.setData(X);
  ASSERT_LE((X - vec.getData()).norm(), 1e-6);
}

TEST(testDCORA, testLiftedRAVector) {
  size_t r = 5;
  size_t d = 3;
  size_t n = 10;
  std::vector<size_t> l_cases = {0, 6, 0, 6};
  std::vector<size_t> b_cases = {0, 0, 7, 7};
  for (size_t i = 0; i < l_cases.size(); i++) {
    size_t l = l_cases.at(i);
    size_t b = b_cases.at(i);
    size_t k = (d + 1) * n + l + b;
    DCORA::Matrix M = DCORA::Matrix::Random(r, k);
    DCORA::Matrix X = DCORA::projectToRAMatrix(M, r, d, n, l, b);
    // Test setter and getter methods
    DCORA::LiftedRAVector vec(r, d, n, l, b);
    vec.setData(X);
    ASSERT_LE((X - vec.getData()).norm(), 1e-6);
  }
}

TEST(testDCORA, testLiftedSEVectorReadWrite) {
  size_t r = 5;
  size_t d = 3;
  size_t n = 10;
  size_t k = (d + 1) * n;
  DCORA::Matrix M = DCORA::Matrix::Random(r, k);
  DCORA::Matrix X = DCORA::projectToSEMatrix(M, r, d, n);

  // Check that the internal value of vec is written to
  DCORA::LiftedSEVector vec(r, d, n);
  Eigen::Map<DCORA::Matrix> X_write(
      const_cast<double *>(vec.vec()->ObtainWriteEntireData()), r, k);
  X_write = X;
  ASSERT_LE((X - vec.getData()).norm(), 1e-6);

  // Check that the internal value of vec is read from
  Eigen::Map<const DCORA::Matrix> X_read(
      const_cast<double *>(vec.vec()->ObtainReadData()), r, k);
  ASSERT_LE((X_read - vec.getData()).norm(), 1e-6);
}

TEST(testDCORA, testLiftedRAVectorReadWrite) {
  size_t r = 5;
  size_t d = 3;
  size_t n = 10;
  std::vector<size_t> l_cases = {0, 6, 0, 6};
  std::vector<size_t> b_cases = {0, 0, 7, 7};
  for (size_t i = 0; i < l_cases.size(); i++) {
    size_t l = l_cases.at(i);
    size_t b = b_cases.at(i);
    size_t k = (d + 1) * n + l + b;
    DCORA::Matrix M = DCORA::Matrix::Random(r, k);
    DCORA::Matrix X = DCORA::projectToRAMatrix(M, r, d, n, l, b);

    // Check that the internal value of vec is written to
    DCORA::LiftedRAVector vec(r, d, n, l, b);
    Eigen::Map<DCORA::Matrix> X_write(
        const_cast<double *>(vec.vec()->ObtainWriteEntireData()), r, k);
    X_write = X;
    ASSERT_LE((X - vec.getData()).norm(), 1e-6);

    // Check that the internal value of vec is read from
    Eigen::Map<const DCORA::Matrix> X_read(
        const_cast<double *>(vec.vec()->ObtainReadData()), r, k);
    ASSERT_LE((X_read - vec.getData()).norm(), 1e-6);
  }
}

TEST(testDCORA, testLiftedSEManifoldProjection) {
  size_t r = 5;
  size_t d = 3;
  size_t n = 10;
  size_t k = (d + 1) * n;

  // Set lifted variable with underlying matrix Y
  DCORA::LiftedSEVariable var(r, d, n);
  var.setRandomData();
  DCORA::Matrix Y = var.getData();

  // Set lifted vector with underlying matrix V
  DCORA::LiftedSEVector inVec(r, d, n);
  inVec.setData(DCORA::Matrix::Random(r, k));
  DCORA::Matrix V = inVec.getData();

  // Compute projection of V as matrix R
  DCORA::LiftedSEVector outVec(r, d, n);
  DCORA::LiftedSEManifold M_SE(r, d, n);
  M_SE.projectToTangentSpace(var.var(), inVec.vec(), outVec.vec());
  DCORA::Matrix R = outVec.getData();

  for (size_t i = 0; i < n; ++i) {
    // Check Stiefel manifold projection
    DCORA::Matrix Y_Yi = Y.block(0, i * (d + 1), r, d);
    DCORA::Matrix V_Yi = V.block(0, i * (d + 1), r, d);
    DCORA::Matrix R_Yi = R.block(0, i * (d + 1), r, d);
    DCORA::Matrix R_Yi_ =
        DCORA::projectToStiefelManifoldTangentSpace(Y_Yi, V_Yi, r, d, 1);
    ASSERT_LE((R_Yi - R_Yi_).norm(), 1e-6);

    // Check euclidean vectors remain unchanged
    DCORA::Matrix V_Ei = V.col(i * (d + 1) + d);
    DCORA::Matrix R_Ei = R.col(i * (d + 1) + d);
    ASSERT_LE((R_Ei - V_Ei).norm(), 1e-6);
  }
}

TEST(testDCORA, testLiftedRAManifoldProjection) {
  size_t r = 3;
  size_t d = 3;
  size_t n = 1;
  std::vector<size_t> l_cases = {0, 6, 0, 6};
  std::vector<size_t> b_cases = {0, 0, 7, 7};
  for (size_t i = 0; i < l_cases.size(); i++) {
    size_t l = l_cases.at(i);
    size_t b = b_cases.at(i);
    size_t k = (d + 1) * n + l + b;

    // Set Matrix Y from lifted vector
    DCORA::LiftedRAVariable var(r, d, n, l, b);
    var.setRandomData();
    DCORA::Matrix Y = var.getData();

    // Set Matrix V from lifted vector
    DCORA::LiftedRAVector inVec(r, d, n, l, b);
    inVec.setData(DCORA::Matrix::Random(r, k));
    DCORA::Matrix V = inVec.getData();

    // Compute projection of V as matrix R
    DCORA::LiftedRAVector outVec(r, d, n, l, b);
    DCORA::LiftedRAManifold M_RA(r, d, n, l, b);
    M_RA.projectToTangentSpace(var.var(), inVec.vec(), outVec.vec());
    DCORA::Matrix R = outVec.getData();

    // Check Stiefel manifold projection
    DCORA::Matrix Y_Y = Y.block(0, 0, r, d * n);
    DCORA::Matrix V_Y = V.block(0, 0, r, d * n);
    DCORA::Matrix R_Y = R.block(0, 0, r, d * n);
    DCORA::Matrix R_Y_ =
        DCORA::projectToStiefelManifoldTangentSpace(Y_Y, V_Y, r, d, n);
    ASSERT_LE((R_Y - R_Y_).norm(), 1e-6);

    // Check Stiefel manifold projection
    DCORA::Matrix Y_OB = Y.block(0, d * n, r, l);
    DCORA::Matrix V_OB = V.block(0, d * n, r, l);
    DCORA::Matrix R_OB = R.block(0, d * n, r, l);
    DCORA::Matrix R_OB_ =
        DCORA::projectToObliqueManifoldTangentSpace(Y_OB, V_OB);
    ASSERT_LE((R_OB - R_OB_).norm(), 1e-6);

    // Check euclidean vectors remain unchanged
    DCORA::Matrix V_E = V.block(0, (d * n) + l, r, n + b);
    DCORA::Matrix R_E = R.block(0, (d * n) + l, r, n + b);
    ASSERT_LE((R_E - V_E).norm(), 1e-6);
  }
}
