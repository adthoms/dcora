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
  unsigned int r = 5;
  unsigned int d = 3;
  unsigned int n = 10;
  DCORA::LiftedSEVariable var(r, d, n);
  // Get random data in SE manifold
  DCORA::LiftedPoseArray pose_array(r, d, n);
  pose_array.setRandomData();
  DCORA::Matrix Y = pose_array.getData();
  // Test copy constructor
  DCORA::LiftedSEVariable var2(var);
  ASSERT_TRUE(var.getData().isApprox(var2.getData()));
  // Test assignment from LiftedSEVariable
  DCORA::LiftedSEVariable var3(r, d, n);
  var3 = var;
  ASSERT_TRUE(var.getData().isApprox(var3.getData()));
  // Test assignment from pose array
  DCORA::LiftedSEVariable var4(pose_array);
  ASSERT_TRUE(Y.isApprox(var4.getData()));
  // Test setter and getter methods
  var.setData(Y);
  ASSERT_TRUE(var.getData().isApprox(Y));
}

TEST(testDCORA, testLiftedRAVariable) {
  unsigned int r = 5;
  unsigned int d = 3;
  unsigned int n = 10;
  std::vector<unsigned int> l_cases = {0, 6, 0, 6};
  std::vector<unsigned int> b_cases = {0, 0, 7, 7};
  for (unsigned int i = 0; i < l_cases.size(); i++) {
    unsigned int l = l_cases.at(i);
    unsigned int b = b_cases.at(i);
    DCORA::LiftedRAVariable var(r, d, n, l, b);
    // Get random data in RA manifold
    DCORA::LiftedRangeAidedArray range_aided_array(r, d, n, l, b);
    range_aided_array.setRandomData();
    DCORA::Matrix Y = range_aided_array.getData();
    // Test copy constructor
    DCORA::LiftedRAVariable var2(var);
    ASSERT_TRUE(var.getData().isApprox(var2.getData()));
    // Test assignment from LiftedRAVariable
    DCORA::LiftedRAVariable var3(r, d, n, l, b);
    var3 = var;
    ASSERT_TRUE(var.getData().isApprox(var3.getData()));
    // Test assignment from range aided array
    DCORA::LiftedRAVariable var4(range_aided_array);
    ASSERT_TRUE(Y.isApprox(var4.getData()));
    // Test setter and getter methods
    var.setData(Y);
    ASSERT_TRUE(var.getData().isApprox(Y));
  }
}

TEST(testDCORA, testLiftedSEVariableRandomManifold) {
  unsigned int r = 5;
  unsigned int d = 3;
  unsigned int n = 10;
  for (int trial = 0; trial < 50; ++trial) {
    DCORA::LiftedSEVariable var(r, d, n);
    var.setRandomData();
    DCORA::Matrix Y = var.getData();
    DCORA::Matrix M = DCORA::projectToSEMatrix(Y, r, d, n);
    ASSERT_TRUE(M.isApprox(Y));
  }
}

TEST(testDCORA, testLiftedRAVariableRandomManifold) {
  unsigned int r = 5;
  unsigned int d = 3;
  unsigned int n = 10;
  std::vector<unsigned int> l_cases = {0, 6, 0, 6};
  std::vector<unsigned int> b_cases = {0, 0, 7, 7};
  for (unsigned int i = 0; i < l_cases.size(); i++) {
    unsigned int l = l_cases.at(i);
    unsigned int b = b_cases.at(i);
    for (int trial = 0; trial < 50; ++trial) {
      DCORA::LiftedRAVariable var(r, d, n, l, b);
      var.setRandomData();
      DCORA::Matrix Y = var.getData();
      DCORA::Matrix M = DCORA::projectToRAMatrix(Y, r, d, n, l, b);
      ASSERT_TRUE(M.isApprox(Y));
    }
  }
}

TEST(testDCORA, testLiftedSEVariableEigenMap) {
  unsigned int r = 5;
  unsigned int d = 3;
  unsigned int n = 10;
  unsigned int k = (d + 1) * n;
  DCORA::LiftedSEVariable x(r, d, n);
  x.setRandomData();

  // View the internal memory of x as a read-only Eigen matrix
  Eigen::Map<const DCORA::Matrix> xMatConst(
      const_cast<double *>(x.var()->ObtainReadData()), r, k);
  ASSERT_TRUE(xMatConst.isApprox(x.getData()));

  // View the internal memory of x as a writable Eigen matrix
  Eigen::Map<DCORA::Matrix> xMat(
      const_cast<double *>(x.var()->ObtainWriteEntireData()), r, k);

  // Modify x through Eigen map
  for (unsigned int i = 0; i < n; ++i) {
    xMat.block(0, i * (d + 1), r, d) = DCORA::randomStiefelVariable(r, d);
    xMat.col(i * (d + 1) + d) = DCORA::randomEuclideanVariable(r);
  }

  // Check that the internal value of x is modified accordingly
  ASSERT_TRUE(xMat.isApprox(x.getData()));

  xMat = DCORA::Matrix::Random(r, k);
  ASSERT_TRUE(xMat.isApprox(x.getData()));

  // Modify x through Eigen map
  for (unsigned int i = 0; i < n; ++i) {
    x.rotation(i) = DCORA::randomStiefelVariable(r, d);
    x.translation(i) = DCORA::randomEuclideanVariable(r);
  }

  // Check that the internal value of x is modified accordingly
  ASSERT_TRUE(xMat.isApprox(x.getData()));

  // Check that the internal value of x remains in the SE manifold
  DCORA::Matrix X_proj = DCORA::projectToSEMatrix(x.getData(), r, d, n);
  ASSERT_TRUE(X_proj.isApprox(x.getData()));
}

TEST(testDCORA, testLiftedRAVariableEigenMap) {
  unsigned int r = 5;
  unsigned int d = 3;
  unsigned int n = 10;
  std::vector<unsigned int> l_cases = {0, 6, 0, 6};
  std::vector<unsigned int> b_cases = {0, 0, 7, 7};
  for (unsigned int i = 0; i < l_cases.size(); i++) {
    unsigned int l = l_cases.at(i);
    unsigned int b = b_cases.at(i);
    unsigned int k = (d + 1) * n + l + b;
    DCORA::LiftedRAVariable x(r, d, n, l, b);
    x.setRandomData();

    // View the internal memory of x as a read-only Eigen matrix
    Eigen::Map<const DCORA::Matrix> xMatConst(
        const_cast<double *>(x.var()->ObtainReadData()), r, k);
    ASSERT_TRUE(xMatConst.isApprox(x.getData()));

    // View the internal memory of x as a writable Eigen matrix
    Eigen::Map<DCORA::Matrix> xMat(
        const_cast<double *>(x.var()->ObtainWriteEntireData()), r, k);

    // Modify x through Eigen map
    for (unsigned int i = 0; i < n; ++i) {
      xMat.block(0, i * d, r, d) = DCORA::randomStiefelVariable(r, d);
      xMat.col(i + (d * n) + l) = DCORA::randomEuclideanVariable(r);
    }
    for (unsigned int i = 0; i < l; ++i) {
      xMat.col(i + (d * n)) = DCORA::randomObliqueVariable(r);
    }
    for (unsigned int i = 0; i < b; ++i) {
      xMat.col(i + (d * n) + l + n) = DCORA::randomEuclideanVariable(r);
    }

    // Check that the internal value of x is modified accordingly
    ASSERT_TRUE(xMat.isApprox(x.getData()));

    xMat = DCORA::Matrix::Random(r, k);
    ASSERT_TRUE(xMat.isApprox(x.getData()));

    // Modify x through Eigen map
    for (unsigned int i = 0; i < n; ++i) {
      x.rotation(i) = DCORA::randomStiefelVariable(r, d);
      x.translation(i) = DCORA::randomEuclideanVariable(r);
    }
    for (unsigned int i = 0; i < l; ++i) {
      x.unitSphere(i) = DCORA::randomObliqueVariable(r);
    }
    for (unsigned int i = 0; i < b; ++i) {
      x.landmark(i) = DCORA::randomEuclideanVariable(r);
    }

    // Check that the internal value of x is modified accordingly
    ASSERT_TRUE(xMat.isApprox(x.getData()));

    // Check that the internal value of x remains in the RA manifold
    DCORA::Matrix X_proj = DCORA::projectToRAMatrix(x.getData(), r, d, n, l, b);
    ASSERT_TRUE(X_proj.isApprox(x.getData()));
  }
}

TEST(testDCORA, testLiftedSEVariableStateGetters) {
  unsigned int r = 5;
  unsigned int d = 3;
  unsigned int n = 10;
  unsigned int k = (d + 1) * n;
  DCORA::LiftedSEVariable x(r, d, n);
  DCORA::Matrix M = DCORA::Matrix::Random(r, k);
  DCORA::Matrix X = DCORA::projectToSEMatrix(M, r, d, n);
  x.setData(X);

  // Check pose getters
  auto [X_SE_R, X_SE_t] = DCORA::partitionSEMatrix(X, r, d, n);
  for (unsigned int i = 0; i < n; ++i) {
    DCORA::Matrix X_SE_R_block = X_SE_R.block(0, i * d, r, d);
    DCORA::Matrix X_SE_t_block = X_SE_t.col(i);
    DCORA::Matrix X_SE_block =
        DCORA::createSEMatrix(X_SE_R_block, X_SE_t_block);
    ASSERT_TRUE(x.pose(i).isApprox(X_SE_block));
    ASSERT_TRUE(x.rotation(i).isApprox(X_SE_R_block));
    ASSERT_TRUE(x.translation(i).isApprox(X_SE_t_block));
  }
}

TEST(testDCORA, testLiftedRAVariableStateGetters) {
  unsigned int r = 5;
  unsigned int d = 3;
  unsigned int n = 10;
  std::vector<unsigned int> l_cases = {0, 6, 0, 6};
  std::vector<unsigned int> b_cases = {0, 0, 7, 7};
  for (unsigned int i = 0; i < l_cases.size(); i++) {
    unsigned int l = l_cases.at(i);
    unsigned int b = b_cases.at(i);
    unsigned int k = (d + 1) * n + l + b;
    DCORA::LiftedRAVariable x(r, d, n, l, b);
    DCORA::Matrix M = DCORA::Matrix::Random(r, k);
    DCORA::Matrix X = DCORA::projectToRAMatrix(M, r, d, n, l, b);
    x.setData(X);

    // Check pose getters
    auto [X_SE_R, X_OB, X_SE_t, X_E] =
        DCORA::partitionRAMatrix(X, r, d, n, l, b);
    for (unsigned int i = 0; i < n; ++i) {
      DCORA::Matrix X_SE_R_block = X_SE_R.block(0, i * d, r, d);
      DCORA::Matrix X_SE_t_block = X_SE_t.col(i);
      DCORA::Matrix X_SE_block =
          DCORA::createSEMatrix(X_SE_R_block, X_SE_t_block);
      ASSERT_TRUE(x.pose(i).isApprox(X_SE_block));
      ASSERT_TRUE(x.rotation(i).isApprox(X_SE_R_block));
      ASSERT_TRUE(x.translation(i).isApprox(X_SE_t_block));
    }
    // Check unit sphere getters
    for (unsigned int i = 0; i < l; ++i) {
      ASSERT_TRUE(x.unitSphere(i).isApprox(X_OB.col(i)));
    }
    // Check landmark getters
    for (unsigned int i = 0; i < b; ++i) {
      ASSERT_TRUE(x.landmark(i).isApprox(X_E.col(i)));
    }
  }
}

TEST(testDCORA, testLiftedSEVector) {
  unsigned int r = 5;
  unsigned int d = 3;
  unsigned int n = 10;
  unsigned int k = (d + 1) * n;
  DCORA::Matrix M = DCORA::Matrix::Random(r, k);
  DCORA::Matrix X = DCORA::projectToSEMatrix(M, r, d, n);
  // Test setter and getter methods
  DCORA::LiftedSEVector vec(r, d, n);
  vec.setData(X);
  ASSERT_TRUE(X.isApprox(vec.getData()));
}

TEST(testDCORA, testLiftedRAVector) {
  unsigned int r = 5;
  unsigned int d = 3;
  unsigned int n = 10;
  std::vector<unsigned int> l_cases = {0, 6, 0, 6};
  std::vector<unsigned int> b_cases = {0, 0, 7, 7};
  for (unsigned int i = 0; i < l_cases.size(); i++) {
    unsigned int l = l_cases.at(i);
    unsigned int b = b_cases.at(i);
    unsigned int k = (d + 1) * n + l + b;
    DCORA::Matrix M = DCORA::Matrix::Random(r, k);
    DCORA::Matrix X = DCORA::projectToRAMatrix(M, r, d, n, l, b);
    // Test setter and getter methods
    DCORA::LiftedRAVector vec(r, d, n, l, b);
    vec.setData(X);
    ASSERT_TRUE(X.isApprox(vec.getData()));
  }
}

TEST(testDCORA, testLiftedSEVectorReadWrite) {
  unsigned int r = 5;
  unsigned int d = 3;
  unsigned int n = 10;
  unsigned int k = (d + 1) * n;
  DCORA::Matrix M = DCORA::Matrix::Random(r, k);
  DCORA::Matrix X = DCORA::projectToSEMatrix(M, r, d, n);

  // Check that the internal value of vec is written to
  DCORA::LiftedSEVector vec(r, d, n);
  Eigen::Map<DCORA::Matrix> X_write(
      const_cast<double *>(vec.vec()->ObtainWriteEntireData()), r, k);
  X_write = X;
  ASSERT_TRUE(X.isApprox(vec.getData()));

  // Check that the internal value of vec is read from
  Eigen::Map<const DCORA::Matrix> X_read(
      const_cast<double *>(vec.vec()->ObtainReadData()), r, k);
  ASSERT_TRUE(X_read.isApprox(vec.getData()));
}

TEST(testDCORA, testLiftedRAVectorReadWrite) {
  unsigned int r = 5;
  unsigned int d = 3;
  unsigned int n = 10;
  std::vector<unsigned int> l_cases = {0, 6, 0, 6};
  std::vector<unsigned int> b_cases = {0, 0, 7, 7};
  for (unsigned int i = 0; i < l_cases.size(); i++) {
    unsigned int l = l_cases.at(i);
    unsigned int b = b_cases.at(i);
    unsigned int k = (d + 1) * n + l + b;
    DCORA::Matrix M = DCORA::Matrix::Random(r, k);
    DCORA::Matrix X = DCORA::projectToRAMatrix(M, r, d, n, l, b);

    // Check that the internal value of vec is written to
    DCORA::LiftedRAVector vec(r, d, n, l, b);
    Eigen::Map<DCORA::Matrix> X_write(
        const_cast<double *>(vec.vec()->ObtainWriteEntireData()), r, k);
    X_write = X;
    ASSERT_TRUE(X.isApprox(vec.getData()));

    // Check that the internal value of vec is read from
    Eigen::Map<const DCORA::Matrix> X_read(
        const_cast<double *>(vec.vec()->ObtainReadData()), r, k);
    ASSERT_TRUE(X_read.isApprox(vec.getData()));
  }
}

TEST(testDCORA, testLiftedSEManifoldProjectToTangentSpace) {
  unsigned int r = 5;
  unsigned int d = 3;
  unsigned int n = 10;
  unsigned int k = (d + 1) * n;

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

  for (unsigned int i = 0; i < n; ++i) {
    // Check Stiefel manifold projection
    DCORA::Matrix Y_Yi = Y.block(0, i * (d + 1), r, d);
    DCORA::Matrix V_Yi = V.block(0, i * (d + 1), r, d);
    DCORA::Matrix R_Yi = R.block(0, i * (d + 1), r, d);
    DCORA::Matrix R_Yi_ =
        DCORA::projectToStiefelManifoldTangentSpace(Y_Yi, V_Yi, r, d, 1);
    ASSERT_TRUE(R_Yi.isApprox(R_Yi_));

    // Check euclidean vectors remain unchanged
    DCORA::Matrix V_Ei = V.col(i * (d + 1) + d);
    DCORA::Matrix R_Ei = R.col(i * (d + 1) + d);
    ASSERT_TRUE(R_Ei.isApprox(V_Ei));
  }
}

TEST(testDCORA, testLiftedRAManifoldProjectToTangentSpace) {
  unsigned int r = 3;
  unsigned int d = 3;
  unsigned int n = 1;
  std::vector<unsigned int> l_cases = {0, 6, 0, 6};
  std::vector<unsigned int> b_cases = {0, 0, 7, 7};
  for (unsigned int i = 0; i < l_cases.size(); i++) {
    unsigned int l = l_cases.at(i);
    unsigned int b = b_cases.at(i);
    unsigned int k = (d + 1) * n + l + b;

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
    ASSERT_TRUE(R_Y.isApprox(R_Y_));

    // Check Oblique manifold projection
    DCORA::Matrix Y_OB = Y.block(0, d * n, r, l);
    DCORA::Matrix V_OB = V.block(0, d * n, r, l);
    DCORA::Matrix R_OB = R.block(0, d * n, r, l);
    DCORA::Matrix R_OB_ =
        DCORA::projectToObliqueManifoldTangentSpace(Y_OB, V_OB);
    ASSERT_TRUE(R_OB.isApprox(R_OB));

    // Check euclidean vectors remain unchanged
    DCORA::Matrix V_E = V.block(0, (d * n) + l, r, n + b);
    DCORA::Matrix R_E = R.block(0, (d * n) + l, r, n + b);
    ASSERT_TRUE(R_E.isApprox(V_E));
  }
}
