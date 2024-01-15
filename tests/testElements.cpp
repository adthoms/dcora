#include <DCORA/DCORA_utils.h>
#include <DCORA/manifold/Elements.h>

#include <iostream>

#include "gtest/gtest.h"

TEST(testDCORA, testLiftedPoseArray) {
  int r = 5;
  int d = 3;
  int n = 3;
  for (int trial = 0; trial < 50; ++trial) {
    DCORA::LiftedPoseArray var(r, d, n);
    // Test setter and getter methods
    for (int i = 0; i < n; ++i) {
      auto Yi = DCORA::randomStiefelVariable(r, d);
      auto pi = DCORA::randomEuclideanVariable(r);
      var.rotation(i) = Yi;
      var.translation(i) = pi;
      ASSERT_LE((Yi - var.rotation(i)).norm(), 1e-6);
      ASSERT_LE((pi - var.translation(i)).norm(), 1e-6);
    }
    // Test copy constructor
    DCORA::LiftedPoseArray var2(var);
    ASSERT_LE((var.getData() - var2.getData()).norm(), 1e-6);
    // Test assignment
    DCORA::LiftedPoseArray var3(r, d, n);
    var3 = var;
    ASSERT_LE((var.getData() - var3.getData()).norm(), 1e-6);
  }
}

TEST(testDCORA, testLiftedTranslationArray) {
  int r = 5;
  int d = 3;
  int n = 3;
  for (int trial = 0; trial < 50; ++trial) {
    DCORA::LiftedTranslationArray var(r, d, n);
    // Test setter and getter methods
    for (int i = 0; i < n; ++i) {
      auto pi = DCORA::randomEuclideanVariable(r);
      var.translation(i) = pi;
      ASSERT_LE((pi - var.translation(i)).norm(), 1e-6);
    }
    // Test copy constructor
    DCORA::LiftedTranslationArray var2(var);
    ASSERT_LE((var.getData() - var2.getData()).norm(), 1e-6);
    // Test assignment
    DCORA::LiftedTranslationArray var3(r, d, n);
    var3 = var;
    ASSERT_LE((var.getData() - var3.getData()).norm(), 1e-6);
  }
}

TEST(testDCORA, testLiftedRangeAidedArray) {
  int r = 5;
  int d = 3;
  int n = 3;
  int l = 5;
  int b = 7;
  for (int trial = 0; trial < 50; ++trial) {
    DCORA::LiftedRangeAidedArray var(r, d, n, l, b);
    // Test setter and getter methods
    for (int i = 0; i < n; ++i) {
      // poses
      auto Yi = DCORA::randomStiefelVariable(r, d);
      auto pi = DCORA::randomEuclideanVariable(r);
      var.GetLiftedPoseArray()->rotation(i) = Yi;
      var.GetLiftedPoseArray()->translation(i) = pi;
      ASSERT_LE((Yi - var.GetLiftedPoseArray()->rotation(i)).norm(), 1e-6);
      ASSERT_LE((pi - var.GetLiftedPoseArray()->translation(i)).norm(), 1e-6);
    }
    for (int i = 0; i < l; ++i) {
      // ranges
      auto ri = DCORA::randomObliqueVariable(r);
      var.GetLiftedRangeArray()->translation(i) = ri;
      ASSERT_LE((ri - var.GetLiftedRangeArray()->translation(i)).norm(), 1e-6);
    }
    for (int i = 0; i < b; ++i) {
      // landmarks
      auto li = DCORA::randomEuclideanVariable(r);
      var.GetLiftedLandmarkArray()->translation(i) = li;
      ASSERT_LE((li - var.GetLiftedLandmarkArray()->translation(i)).norm(),
                1e-6);
    }
    // Test copy constructor
    DCORA::LiftedRangeAidedArray var2(var);
    ASSERT_LE((var.getData() - var2.getData()).norm(), 1e-6);

    // Test assignment
    DCORA::LiftedRangeAidedArray var3(r, d, n, l, b);
    var3 = var;
    ASSERT_LE((var.getData() - var3.getData()).norm(), 1e-6);
  }
}

TEST(testDCORA, testLiftedPose) {
  int d = 3;
  int r = 5;
  for (int trial = 0; trial < 50; ++trial) {
    DCORA::Matrix Xi = DCORA::Matrix::Zero(r, d + 1);
    Xi.block(0, 0, r, d) = DCORA::randomStiefelVariable(r, d);
    Xi.col(d) = DCORA::randomEuclideanVariable(r);
    // Test constructor from Eigen matrix
    DCORA::LiftedPose var(Xi);
    ASSERT_LE((Xi - var.getData()).norm(), 1e-6);
  }
}

TEST(testDCORA, testLiftedTranslation) {
  int r = 5;
  for (int trial = 0; trial < 50; ++trial) {
    DCORA::Vector Pi = DCORA::randomEuclideanVariable(r);
    // Test constructor from Eigen vector
    DCORA::LiftedTranslation var(Pi);
    ASSERT_LE((Pi - var.getData()).norm(), 1e-6);
  }
}

TEST(testDCORA, testPoseIdentity) {
  int d = 3;
  DCORA::Pose T(d);
  ASSERT_LE((T.identity().rotation() - DCORA::Matrix::Identity(d, d)).norm(),
            1e-6);
  ASSERT_LE((T.identity().translation() - DCORA::Vector::Zero(d)).norm(), 1e-6);
  ASSERT_LE(
      (T.identity().matrix() - DCORA::Matrix::Identity(d + 1, d + 1)).norm(),
      1e-6);
}

TEST(testDCORA, testTranslationZeroVector) {
  int d = 3;
  DCORA::Translation P(d);
  ASSERT_LE((P.zeroVector().translation() - DCORA::Vector::Zero(d)).norm(),
            1e-6);
}

TEST(testDCORA, testPoseInverse) {
  for (int trial = 0; trial < 50; ++trial) {
    int d = 3;
    DCORA::Pose T(d);
    T.rotation() = Eigen::Quaterniond::UnitRandom().toRotationMatrix();
    T.translation() = DCORA::Vector::Random(d);
    auto TInv = T.inverse();
    ASSERT_LE(
        (T.matrix() * TInv.matrix() - DCORA::Matrix::Identity(d + 1, d + 1))
            .norm(),
        1e-6);
    ASSERT_LE(
        (TInv.matrix() * T.matrix() - DCORA::Matrix::Identity(d + 1, d + 1))
            .norm(),
        1e-6);
  }
}

TEST(testDCORA, testPoseMultiplication) {
  for (int trial = 0; trial < 50; ++trial) {
    int d = 3;
    DCORA::Pose T1(d);
    T1.rotation() = Eigen::Quaterniond::UnitRandom().toRotationMatrix();
    T1.translation() = DCORA::Vector::Random(d);
    DCORA::Pose T2(d);
    T2.rotation() = Eigen::Quaterniond::UnitRandom().toRotationMatrix();
    T2.translation() = DCORA::Vector::Random(d);
    auto T = T1 * T2;
    ASSERT_LE((T1.matrix() * T2.matrix() - T.matrix()).norm(), 1e-6);
  }
}
