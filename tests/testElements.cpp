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

#include <DCORA/DCORA_utils.h>
#include <DCORA/manifold/Elements.h>

#include <iostream>

#include "gtest/gtest.h"

TEST(testDCORA, testLiftedPoseArray) {
  unsigned int r = 5;
  unsigned int d = 3;
  unsigned int n = 3;
  DCORA::LiftedPoseArray var(r, d, n);
  // Test setter and getter methods
  for (unsigned int i = 0; i < n; ++i) {
    DCORA::Matrix Y = DCORA::randomStiefelVariable(r, d + 1);
    var.pose(i) = Y;
    ASSERT_TRUE(Y.isApprox(var.pose(i)));
    DCORA::Matrix Yi = DCORA::randomStiefelVariable(r, d);
    DCORA::Matrix pi = DCORA::randomEuclideanVariable(r);
    var.rotation(i) = Yi;
    var.translation(i) = pi;
    ASSERT_TRUE(Yi.isApprox(var.rotation(i)));
    ASSERT_TRUE(pi.isApprox(var.translation(i)));
  }
  DCORA::Matrix M = DCORA::Matrix::Random(r, (d + 1) * n);
  DCORA::Matrix Y = DCORA::projectToSEMatrix(M, r, d, n);
  var.setData(Y);
  ASSERT_TRUE(var.getData().isApprox(Y));
  // Test copy constructor
  DCORA::LiftedPoseArray var2(var);
  ASSERT_TRUE(var.getData().isApprox(var2.getData()));
  // Test assignment
  DCORA::LiftedPoseArray var3(r, d, n);
  var3 = var;
  ASSERT_TRUE(var.getData().isApprox(var3.getData()));
  // Test random data
  var.setRandomData();
  Y = var.getData();
  ASSERT_TRUE(DCORA::projectToSEMatrix(Y, r, d, n).isApprox(Y));
}

TEST(testDCORA, testLiftedPointArray) {
  unsigned int r = 5;
  unsigned int d = 3;
  unsigned int n = 3;
  DCORA::LiftedPointArray var(r, d, n);
  // Test setter and getter methods
  for (unsigned int i = 0; i < n; ++i) {
    DCORA::Matrix pi = DCORA::randomEuclideanVariable(r);
    var.translation(i) = pi;
    ASSERT_TRUE(pi.isApprox(var.translation(i)));
  }
  DCORA::Matrix M = DCORA::Matrix::Random(r, n);
  var.setData(M);
  ASSERT_TRUE(var.getData().isApprox(M));
  // Test copy constructor
  DCORA::LiftedPointArray var2(var);
  ASSERT_TRUE(var.getData().isApprox(var2.getData()));
  // Test assignment
  DCORA::LiftedPointArray var3(r, d, n);
  var3 = var;
  ASSERT_TRUE(var.getData().isApprox(var3.getData()));
}

TEST(testDCORA, testLiftedRangeAidedArray) {
  unsigned int r = 5;
  unsigned int d = 3;
  unsigned int n = 3;
  std::vector<unsigned int> l_cases = {0, 6, 0, 6};
  std::vector<unsigned int> b_cases = {0, 0, 7, 7};
  for (unsigned int i = 0; i < l_cases.size(); i++) {
    unsigned int l = l_cases.at(i);
    unsigned int b = b_cases.at(i);
    DCORA::LiftedRangeAidedArray var(r, d, n, l, b);
    // Test setter and getter methods
    for (unsigned int i = 0; i < n; ++i) {
      DCORA::Matrix Y = DCORA::randomStiefelVariable(r, d + 1);
      var.pose(i) = Y;
      ASSERT_TRUE(Y.isApprox(var.pose(i)));
      DCORA::Matrix Yi = DCORA::randomStiefelVariable(r, d);
      DCORA::Matrix pi = DCORA::randomEuclideanVariable(r);
      var.rotation(i) = Yi;
      var.translation(i) = pi;
      ASSERT_TRUE(Yi.isApprox(var.rotation(i)));
      ASSERT_TRUE(pi.isApprox(var.translation(i)));
    }
    for (unsigned int i = 0; i < l; ++i) {
      DCORA::Matrix ri = DCORA::randomObliqueVariable(r);
      var.unitSphere(i) = ri;
      ASSERT_TRUE(ri.isApprox(var.unitSphere(i)));
    }
    for (unsigned int i = 0; i < b; ++i) {
      DCORA::Matrix li = DCORA::randomEuclideanVariable(r);
      var.landmark(i) = li;
      ASSERT_TRUE(li.isApprox(var.landmark(i)));
    }
    DCORA::Matrix M = DCORA::Matrix::Random(r, (d + 1) * n + l + b);
    DCORA::Matrix Y = DCORA::projectToRAMatrix(M, r, d, n, l, b);
    var.setData(Y);
    ASSERT_TRUE(var.getData().isApprox(Y));
    // Test copy constructor
    DCORA::LiftedRangeAidedArray var2(var);
    ASSERT_TRUE(var.getData().isApprox(var2.getData()));

    // Test assignment
    DCORA::LiftedRangeAidedArray var3(r, d, n, l, b);
    var3 = var;
    ASSERT_TRUE(var.getData().isApprox(var3.getData()));

    // Test random data
    var.setRandomData();
    Y = var.getData();
    ASSERT_TRUE(DCORA::projectToRAMatrix(Y, r, d, n, l, b).isApprox(Y));

    // Test lifted array setter and getter methods
    DCORA::LiftedPoseArray liftedPoseArray(r, d, n);
    DCORA::LiftedPointArray liftedUnitSphereArray(r, d, l);
    DCORA::LiftedPointArray liftedLandmarkArray(r, d, b);
    liftedPoseArray.setRandomData();
    DCORA::Matrix M1 = DCORA::Matrix::Random(r, l);
    liftedUnitSphereArray.setData(DCORA::projectToObliqueManifold(M1));
    DCORA::Matrix M2 = DCORA::Matrix::Random(r, b);
    liftedLandmarkArray.setData(M2);
    var.setLiftedPoseArray(liftedPoseArray);
    var.setLiftedUnitSphereArray(liftedUnitSphereArray);
    var.setLiftedLandmarkArray(liftedLandmarkArray);
    ASSERT_TRUE(liftedPoseArray.getData().isApprox(
        var.GetLiftedPoseArray()->getData()));
    ASSERT_TRUE(liftedUnitSphereArray.getData().isApprox(
        var.GetLiftedUnitSphereArray()->getData()));
    ASSERT_TRUE(liftedLandmarkArray.getData().isApprox(
        var.GetLiftedLandmarkArray()->getData()));
  }
  // Test SE setter and getter methods
  DCORA::LiftedRangeAidedArray var(r, d, n, 0, 0, DCORA::GraphType::PoseGraph);
  DCORA::Matrix M = DCORA::Matrix::Random(r, (d + 1) * n);
  DCORA::Matrix Y = DCORA::projectToSEMatrix(M, r, d, n);
  var.setData(Y);
  ASSERT_TRUE(var.getData().isApprox(Y));
}

TEST(testDCORA, testRangeAidedArray) {
  unsigned int d = 3;
  unsigned int n = 3;
  std::vector<unsigned int> l_cases = {0, 6, 0, 6};
  std::vector<unsigned int> b_cases = {0, 0, 7, 7};
  for (unsigned int i = 0; i < l_cases.size(); i++) {
    unsigned int l = l_cases.at(i);
    unsigned int b = b_cases.at(i);
    DCORA::RangeAidedArray var(d, n, l, b);
    // Test setter and getter methods
    DCORA::LiftedPoseArray liftedPoseArray(d, d, n);
    DCORA::LiftedPointArray liftedUnitSphereArray(d, d, l);
    DCORA::LiftedPointArray liftedLandmarkArray(d, d, b);
    liftedPoseArray.setRandomData();
    DCORA::Matrix M1 = DCORA::Matrix::Random(d, l);
    liftedUnitSphereArray.setData(DCORA::projectToObliqueManifold(M1));
    DCORA::Matrix M2 = DCORA::Matrix::Random(d, b);
    liftedLandmarkArray.setData(M2);
    var.setLiftedPoseArray(liftedPoseArray);
    var.setLiftedUnitSphereArray(liftedUnitSphereArray);
    var.setLiftedLandmarkArray(liftedLandmarkArray);
    ASSERT_TRUE(
        liftedPoseArray.getData().isApprox(var.getPoseArray().getData()));
    ASSERT_TRUE(liftedUnitSphereArray.getData().isApprox(
        var.getUnitSphereArray().getData()));
    ASSERT_TRUE(liftedLandmarkArray.getData().isApprox(
        var.getLandmarkArray().getData()));
  }
}

TEST(testDCORA, testLiftedPose) {
  unsigned int d = 3;
  unsigned int r = 5;
  for (int trial = 0; trial < 50; ++trial) {
    DCORA::Matrix Xi = DCORA::Matrix::Zero(r, d + 1);
    Xi.block(0, 0, r, d) = DCORA::randomStiefelVariable(r, d);
    Xi.col(d) = DCORA::randomEuclideanVariable(r);
    // Test constructor from Eigen matrix
    DCORA::LiftedPose var(Xi);
    ASSERT_TRUE(Xi.isApprox(var.getData()));
  }
}

TEST(testDCORA, testLiftedPoint) {
  unsigned int r = 5;
  for (int trial = 0; trial < 50; ++trial) {
    DCORA::Vector Pi = DCORA::randomEuclideanVariable(r);
    // Test constructor from Eigen vector
    DCORA::LiftedPoint var(Pi);
    ASSERT_TRUE(Pi.isApprox(var.getData()));
  }
}

TEST(testDCORA, testPoseIdentity) {
  unsigned int d = 3;
  DCORA::Pose T(d);
  ASSERT_TRUE(T.identity().rotation().isApprox(DCORA::Matrix::Identity(d, d)));
  ASSERT_TRUE(T.identity().translation().isApprox(DCORA::Vector::Zero(d)));
  ASSERT_TRUE(
      T.identity().matrix().isApprox(DCORA::Matrix::Identity(d + 1, d + 1)));
}

TEST(testDCORA, testTranslationZeroVector) {
  unsigned int d = 3;
  DCORA::Point P(d);
  ASSERT_TRUE(P.zeroVector().translation().isApprox(DCORA::Vector::Zero(d)));
}

TEST(testDCORA, testPoseInverse) {
  unsigned int d = 3;
  DCORA::Matrix I = DCORA::Matrix::Identity(d + 1, d + 1);
  for (int trial = 0; trial < 50; ++trial) {
    DCORA::Pose T(d);
    T.rotation() = Eigen::Quaterniond::UnitRandom().toRotationMatrix();
    T.translation() = DCORA::Vector::Random(d);
    DCORA::Pose TInv = T.inverse();
    ASSERT_TRUE((T.matrix() * TInv.matrix()).isApprox(I));
    ASSERT_TRUE((TInv.matrix() * T.matrix()).isApprox(I));
  }
}

TEST(testDCORA, testPoseMultiplication) {
  unsigned int d = 3;
  for (int trial = 0; trial < 50; ++trial) {
    DCORA::Pose T1(d);
    T1.rotation() = Eigen::Quaterniond::UnitRandom().toRotationMatrix();
    T1.translation() = DCORA::Vector::Random(d);
    DCORA::Pose T2(d);
    T2.rotation() = Eigen::Quaterniond::UnitRandom().toRotationMatrix();
    T2.translation() = DCORA::Vector::Random(d);
    DCORA::Pose T = T1 * T2;
    ASSERT_TRUE((T1.matrix() * T2.matrix()).isApprox(T.matrix()));
  }
}
