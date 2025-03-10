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

#include <DCORA/DCORA_robust.h>
#include <DCORA/DCORA_solver.h>
#include <DCORA/DCORA_types.h>
#include <DCORA/DCORA_utils.h>
#include <DCORA/Graph.h>

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
  for (int i = 0; i < 10; ++i) {
    DCORA::Matrix Y_ = DCORA::fixedStiefelVariable(5, 3);
    ASSERT_TRUE(Y_.isApprox(Y));
  }
}

TEST(testDCORA, testStiefelProjection) {
  unsigned int r = 5;
  unsigned int d = 3;
  DCORA::Matrix I = DCORA::Matrix::Identity(d, d);
  for (int j = 0; j < 50; ++j) {
    DCORA::Matrix M = DCORA::Matrix::Random(r, d);
    DCORA::Matrix Y = DCORA::projectToStiefelManifold(M);
    DCORA::Matrix D = Y.transpose() * Y - I;
    ASSERT_LE(D.norm(), 1e-5);
  }
}

TEST(testDCORA, testEuclideanGeneration) {
  DCORA::Matrix E = DCORA::fixedEuclideanVariable(5, 3);
  ASSERT_TRUE(E.allFinite());
}

TEST(testDCORA, testEuclideanRepeat) {
  DCORA::Matrix E = DCORA::fixedEuclideanVariable(5, 3);
  for (int i = 0; i < 10; ++i) {
    DCORA::Matrix E_ = DCORA::fixedEuclideanVariable(5, 3);
    ASSERT_TRUE(E_.isApprox(E));
  }
}

TEST(testDCORA, testObliqueGeneration) {
  DCORA::Matrix OB = DCORA::fixedObliqueVariable(5, 3);
  DCORA::Matrix I = DCORA::Matrix::Identity(3, 3);
  DCORA::Matrix D = (OB.transpose() * OB).diagonal() - I.diagonal();
  ASSERT_LE(D.norm(), 1e-5);
  for (int i = 0; i < OB.cols(); ++i) {
    double d = OB.col(i).norm() - 1.0;
    ASSERT_LE(d, 1e-5);
  }
}

TEST(testDCORA, testObliqueRepeat) {
  DCORA::Matrix OB = DCORA::fixedObliqueVariable(5, 3);
  for (int i = 0; i < 10; ++i) {
    DCORA::Matrix OB_ = DCORA::fixedObliqueVariable(5, 3);
    ASSERT_TRUE(OB_.isApprox(OB));
  }
}

TEST(testDCORA, testObliqueProjection) {
  unsigned int r = 5;
  unsigned int d = 3;
  DCORA::Matrix I = DCORA::Matrix::Identity(d, d);
  for (int j = 0; j < 50; ++j) {
    DCORA::Matrix M = DCORA::Matrix::Random(r, d);
    DCORA::Matrix Y = DCORA::projectToObliqueManifold(M);
    DCORA::Matrix D = (Y.transpose() * Y).diagonal() - I.diagonal();
    ASSERT_LE(D.norm(), 1e-5);
  }
}

TEST(testDCORA, testProjectToSEMAtrix) {
  unsigned int d = 3;
  unsigned int r = 5;
  unsigned int n = 100;
  DCORA::Matrix M = DCORA::Matrix::Random(r, (d + 1) * n);
  DCORA::Matrix X = DCORA::projectToSEMatrix(M, r, d, n);
  DCORA::Matrix I = DCORA::Matrix::Identity(d, d);
  ASSERT_EQ(X.rows(), r);
  ASSERT_EQ(X.cols(), (d + 1) * n);
  for (unsigned int i = 0; i < n; ++i) {
    DCORA::Matrix Y = X.block(0, i * (d + 1), r, d);
    DCORA::Matrix D = Y.transpose() * Y - I;
    ASSERT_LE(D.norm(), 1e-5);
  }
}

TEST(testDCORA, testProjectToRAMatrix) {
  unsigned int d = 3;
  unsigned int r = 5;
  unsigned int n = 100;
  unsigned int l = 5;
  unsigned int b = 7;
  DCORA::Matrix M = DCORA::Matrix::Random(r, (d + 1) * n + l + b);
  DCORA::Matrix X = DCORA::projectToRAMatrix(M, r, d, n, l, b);
  DCORA::Matrix I_dxd = DCORA::Matrix::Identity(d, d);
  DCORA::Matrix I_1x1 = DCORA::Matrix::Identity(1, 1);
  ASSERT_EQ(X.rows(), r);
  ASSERT_EQ(X.cols(), (d + 1) * n + l + b);
  for (unsigned int i = 0; i < n; ++i) {
    DCORA::Matrix Y = X.block(0, i * d, r, d);
    DCORA::Matrix D = Y.transpose() * Y - I_dxd;
    ASSERT_LE(D.norm(), 1e-5);
  }
  for (unsigned int i = 0; i < l; ++i) {
    DCORA::Matrix OB = X.block(0, i + (d * n), r, 1);
    DCORA::Matrix D = (OB.transpose() * OB).diagonal() - I_1x1.diagonal();
    double d = OB.norm() - 1.0;
    ASSERT_LE(D.norm(), 1e-5);
    ASSERT_LE(d, 1e-5);
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
  unsigned int d = 3;
  unsigned int r = 5;
  unsigned int n = 3;
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
  ASSERT_TRUE(var_SE_R.isApprox(X_SE_R));
  ASSERT_TRUE(var_SE_t.isApprox(X_SE_t));
}

TEST(testDCORA, testPartitionRAMatrix) {
  unsigned int d = 3;
  unsigned int r = 5;
  unsigned int n = 3;
  unsigned int l = 5;
  unsigned int b = 7;
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
  ASSERT_TRUE(var_SE_R.isApprox(X_SE_R));
  ASSERT_TRUE(var_SE_t.isApprox(X_SE_t));
  ASSERT_TRUE(var_OB.isApprox(X_OB));
  ASSERT_TRUE(var_E.isApprox(X_E));
}

TEST(testDCORA, testCreateSEMatrix) {
  unsigned int d = 3;
  unsigned int r = 5;
  unsigned int n = 3;
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
  ASSERT_TRUE(var_X_SE.isApprox(X_SE));
}

TEST(testDCORA, testCreateRAMatrix) {
  unsigned int d = 3;
  unsigned int r = 5;
  unsigned int n = 3;
  unsigned int l = 5;
  unsigned int b = 7;
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
  ASSERT_TRUE(var_X_RA.isApprox(X_RA));
}

TEST(testDCORA, testProjectSolutionRASLAM) {
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
    DCORA::Matrix X_project = DCORA::projectSolutionRASLAM(X, r, d, n, l, b);
    ASSERT_EQ(X_project.rows(), d);
    ASSERT_EQ(X_project.cols(), k);
    // Check that the projection is within the feasible set
    ASSERT_TRUE(
        X_project.isApprox(DCORA::projectToRAMatrix(X_project, d, d, n, l, b)));
  }
}

TEST(testDCORA, testAlignTrajectoryToFrame) {
  unsigned int d = 3;
  unsigned int n = 10;
  // Set random data
  DCORA::PoseArray TrajectoryInit(d, n);
  TrajectoryInit.setRandomData();
  // Set random frame
  DCORA::Pose Tw0;
  Tw0.rotation() = DCORA::randomStiefelVariable(d, d);
  Tw0.translation() = DCORA::randomEuclideanVariable(d);
  // Transform
  DCORA::PoseArray TrajectoryTransformedTmp =
      alignTrajectoryToFrame(TrajectoryInit, Tw0);
  DCORA::PoseArray TrajectoryTransformed =
      alignTrajectoryToFrame(TrajectoryTransformedTmp, Tw0.inverse());
  ASSERT_TRUE(
      TrajectoryTransformed.getData().isApprox(TrajectoryInit.getData()));
}

TEST(testDCORA, testAlignUnitSpheresToFrame) {
  unsigned int d = 3;
  std::vector<unsigned int> l_cases = {0, 6};
  // Set random frame
  DCORA::Pose Tw0;
  Tw0.rotation() = DCORA::randomStiefelVariable(d, d);
  Tw0.translation() = DCORA::randomEuclideanVariable(d);
  for (unsigned int i = 0; i < l_cases.size(); i++) {
    unsigned int l = l_cases.at(i);
    // Set random data
    DCORA::PointArray UnitSpheresInit(d, l);
    UnitSpheresInit.setData(DCORA::randomObliqueVariable(d, l));
    // Transform
    DCORA::PointArray UnitSpheresTransformedTmp =
        alignUnitSpheresToFrame(UnitSpheresInit, Tw0);
    DCORA::PointArray UnitSpheresTransformed =
        alignUnitSpheresToFrame(UnitSpheresTransformedTmp, Tw0.inverse());
    ASSERT_TRUE(
        UnitSpheresTransformed.getData().isApprox(UnitSpheresInit.getData()));
    ASSERT_TRUE(UnitSpheresTransformed.getData().isApprox(
        DCORA::projectToObliqueManifold(UnitSpheresTransformed.getData())));
  }
}

TEST(testDCORA, testAlignLandmarksToFrame) {
  unsigned int d = 3;
  std::vector<unsigned int> b_cases = {0, 7};
  // Set random frame
  DCORA::Pose Tw0;
  Tw0.rotation() = DCORA::randomStiefelVariable(d, d);
  Tw0.translation() = DCORA::randomEuclideanVariable(d);
  for (unsigned int i = 0; i < b_cases.size(); i++) {
    unsigned int b = b_cases.at(i);
    // Set random data
    DCORA::PointArray LandmarksInit(d, b);
    LandmarksInit.setData(DCORA::randomEuclideanVariable(d, b));
    // Transform
    DCORA::PointArray LandmarksTransformedTmp =
        alignLandmarksToFrame(LandmarksInit, Tw0);
    DCORA::PointArray LandmarksTransformed =
        alignLandmarksToFrame(LandmarksTransformedTmp, Tw0.inverse());
    ASSERT_TRUE(
        LandmarksTransformed.getData().isApprox(LandmarksInit.getData()));
  }
}
