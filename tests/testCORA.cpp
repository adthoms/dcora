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
#include <DCORA/Graph.h>
#include <DCORA/QuadraticOptimizer.h>
#include <DCORA/manifold/LiftedManifold.h>

#include <iostream>
#include <random>

#include "gtest/gtest.h"

TEST(testDCORA, testRobustSingleRotationAveragingTrivial) {
  for (int trial = 0; trial < 50; ++trial) {
    const DCORA::Matrix RTrue =
        Eigen::Quaterniond::UnitRandom().toRotationMatrix();
    const double cbar = DCORA::angular2ChordalSO3(0.5); // approximately 30 deg
    std::vector<DCORA::Matrix> RVec;
    RVec.push_back(RTrue);
    DCORA::Matrix ROpt;
    std::vector<size_t> inlierIndices;
    const auto kappa = DCORA::Vector::Ones(1);
    DCORA::robustSingleRotationAveraging(&ROpt, &inlierIndices, RVec, kappa,
                                         cbar);
    DCORA::checkRotationMatrix(ROpt);
    double distChordal = (ROpt - RTrue).norm();
    ASSERT_LE(distChordal, 1e-8);
    ASSERT_EQ(inlierIndices.size(), 1);
    ASSERT_EQ(inlierIndices[0], 0);
  }
}

TEST(testDCORA, testRobustSingleRotationAveraging) {
  for (int trial = 0; trial < 50; ++trial) {
    const double tol = DCORA::angular2ChordalSO3(0.02);
    const double cbar = DCORA::angular2ChordalSO3(0.3);
    const DCORA::Matrix RTrue =
        Eigen::Quaterniond::UnitRandom().toRotationMatrix();
    std::vector<DCORA::Matrix> RVec;
    // Push inliers
    for (int i = 0; i < 10; ++i) {
      RVec.emplace_back(RTrue);
    }
    // Push outliers
    while (RVec.size() < 50) {
      DCORA::Matrix RRand = Eigen::Quaterniond::UnitRandom().toRotationMatrix();
      // Make sure that outlier is separated from the true rotation
      if ((RRand - RTrue).norm() > 1.2 * cbar)
        RVec.emplace_back(RRand);
    }
    DCORA::Matrix ROpt;
    std::vector<size_t> inlierIndices;
    const auto kappa = DCORA::Vector::Ones(50);
    DCORA::robustSingleRotationAveraging(&ROpt, &inlierIndices, RVec, kappa,
                                         cbar);
    DCORA::checkRotationMatrix(ROpt);
    double distChordal = (ROpt - RTrue).norm();
    ASSERT_LE(distChordal, tol);
    ASSERT_EQ(inlierIndices.size(), 10);
    for (int i = 0; i < 10; ++i) {
      ASSERT_EQ(inlierIndices[i], i);
    }
  }
}

TEST(testDCORA, testRobustSinglePoseAveragingTrivial) {
  for (int trial = 0; trial < 50; ++trial) {
    const DCORA::Matrix RTrue =
        Eigen::Quaterniond::UnitRandom().toRotationMatrix();
    const DCORA::Vector tTrue = Eigen::Vector3d::Zero();
    std::vector<DCORA::Matrix> RVec;
    RVec.push_back(RTrue);
    std::vector<DCORA::Vector> tVec;
    tVec.push_back(tTrue);
    const auto kappa = 10000 * DCORA::Vector::Ones(1);
    const auto tau = 100 * DCORA::Vector::Ones(1);
    const double gnc_quantile = 0.9;
    const double gnc_barc =
        DCORA::RobustCost::computeErrorThresholdAtQuantile(gnc_quantile, 3);
    DCORA::Matrix ROpt;
    DCORA::Vector tOpt;
    std::vector<size_t> inlierIndices;
    DCORA::robustSinglePoseAveraging(&ROpt, &tOpt, &inlierIndices, RVec, tVec,
                                     kappa, tau, gnc_barc);
    DCORA::checkRotationMatrix(ROpt);
    ASSERT_LE((ROpt - RTrue).norm(), 1e-8);
    ASSERT_LE((tOpt - tTrue).norm(), 1e-8);
    ASSERT_EQ(inlierIndices.size(), 1);
    ASSERT_EQ(inlierIndices[0], 0);
  }
}

TEST(testDCORA, testRobustSinglePoseAveraging) {
  for (int trial = 0; trial < 50; ++trial) {
    const double RMaxError = DCORA::angular2ChordalSO3(0.02);
    const double tMaxError = 1e-2;
    const double gnc_quantile = 0.9;
    const double gnc_barc =
        DCORA::RobustCost::computeErrorThresholdAtQuantile(gnc_quantile, 3);
    const double kappa = 10000;
    const double tau = 100;
    const auto kappa_vec = kappa * DCORA::Vector::Ones(50);
    const auto tau_vec = tau * DCORA::Vector::Ones(50);

    const DCORA::Matrix RTrue =
        Eigen::Quaterniond::UnitRandom().toRotationMatrix();
    const DCORA::Vector tTrue = Eigen::Vector3d::Zero();
    std::vector<DCORA::Matrix> RVec;
    std::vector<DCORA::Vector> tVec;
    // Push inliers
    for (int i = 0; i < 10; ++i) {
      RVec.emplace_back(RTrue);
      tVec.emplace_back(tTrue);
    }
    // Push outliers
    while (RVec.size() < 50) {
      DCORA::Matrix RRand = Eigen::Quaterniond::UnitRandom().toRotationMatrix();
      DCORA::Matrix tRand = Eigen::Vector3d::Random();
      double rSq = kappa * (RTrue - RRand).squaredNorm() +
                   tau * (tTrue - tRand).squaredNorm();
      // Make sure that outliers are sufficiently far away from ground truth
      if (std::sqrt(rSq) > 1.2 * gnc_barc) {
        RVec.emplace_back(RRand);
        tVec.emplace_back(tRand);
      }
    }
    DCORA::Matrix ROpt;
    DCORA::Vector tOpt;
    std::vector<size_t> inlierIndices;
    DCORA::robustSinglePoseAveraging(&ROpt, &tOpt, &inlierIndices, RVec, tVec,
                                     kappa_vec, tau_vec, gnc_barc);
    DCORA::checkRotationMatrix(ROpt);
    ASSERT_LE((ROpt - RTrue).norm(), RMaxError);
    ASSERT_LE((tOpt - tTrue).norm(), tMaxError);
    ASSERT_EQ(inlierIndices.size(), 10);
    for (int i = 0; i < 10; ++i) {
      ASSERT_EQ(inlierIndices[i], i);
    }
  }
}

TEST(testDCORA, testPrior) {
  size_t dimension = 3;
  size_t num_poses = 2;
  size_t robot_id = 0;

  // Odometry measurement
  DCORA::RelativeSEMeasurement m;
  m.r1 = 0;
  m.p1 = 0;
  m.r2 = 0;
  m.p2 = 1;
  m.R = Eigen::Matrix3d::Identity();
  m.t = Eigen::Vector3d::Zero();
  m.kappa = 10000;
  m.tau = 100;
  m.weight = 1;
  m.fixedWeight = true;
  std::vector<DCORA::RelativeSEMeasurement> measurements;
  measurements.push_back(m);

  DCORA::PoseArray T(dimension, num_poses);
  T = DCORA::odometryInitialization(measurements);

  // Form pose graph and add a prior
  auto pose_graph =
      std::make_shared<DCORA::PoseGraph>(robot_id, dimension, dimension);
  pose_graph->setMeasurements(measurements);
  DCORA::Matrix prior_rotation(dimension, dimension);

  // clang-format off
  prior_rotation << 0.7236,  0.1817, 0.6658,
                   -0.6100,  0.6198, 0.4938,
                   -0.3230, -0.7634, 0.5594;
  // clang-format on

  prior_rotation = DCORA::projectToRotationGroup(prior_rotation);
  DCORA::Pose prior(dimension);
  prior.rotation() = prior_rotation;
  pose_graph->setPrior(1, prior);
  DCORA::QuadraticProblem problem(pose_graph);

  // The odometry initial guess does not respect the prior
  double error0 = (T.pose(0) - prior.pose()).norm();
  double error1 = (T.pose(1) - prior.pose()).norm();
  ASSERT_GT(error0, 1e-6);
  ASSERT_GT(error1, 1e-6);

  // Initialize optimizer object
  DCORA::ROptParameters params;
  params.verbose = false;
  params.RTR_iterations = 50;
  params.RTR_tCG_iterations = 500;
  params.gradnorm_tol = 1e-5;
  DCORA::QuadraticOptimizer optimizer(&problem, params);

  // Optimize!
  auto Topt_mat = optimizer.optimize(T.getData());
  T.setData(Topt_mat);

  // After optimization, the solution should be fixed on the prior
  error0 = (T.pose(0) - prior.pose()).norm();
  error1 = (T.pose(1) - prior.pose()).norm();
  ASSERT_LT(error0, 1e-6);
  ASSERT_LT(error1, 1e-6);
}

TEST(testDCORA, testRobustPGO) {
  int d = 3;
  int n = 4;
  double kappa = 10000;
  double tau = 100;
  std::vector<DCORA::Pose> poses_gt;
  for (int i = 0; i < n; ++i) {
    DCORA::Pose Ti(d);
    Ti.rotation() = Eigen::Quaterniond::UnitRandom().toRotationMatrix();
    Ti.translation() = i * Eigen::Vector3d::Ones();
    poses_gt.push_back(Ti);
  }
  std::vector<DCORA::RelativeSEMeasurement> measurements;
  // generate odometry
  for (int i = 0; i < n - 1; ++i) {
    int j = i + 1;
    DCORA::Pose Ti = poses_gt[i];
    DCORA::Pose Tj = poses_gt[j];
    DCORA::Pose Tij = Ti.inverse() * Tj;
    DCORA::RelativeSEMeasurement m;
    m.r1 = 0;
    m.r2 = 0;
    m.p1 = i;
    m.p2 = j;
    m.kappa = kappa;
    m.tau = tau;
    m.fixedWeight = true;
    m.R = Tij.rotation();
    m.t = Tij.translation();
    measurements.push_back(m);
  }
  // generate a single inlier loop closure
  DCORA::Pose Ti = poses_gt[0];
  DCORA::Pose Tj = poses_gt[3];
  DCORA::Pose Tij = Ti.inverse() * Tj;
  DCORA::RelativeSEMeasurement m_inlier;
  m_inlier.r1 = 0;
  m_inlier.r2 = 0;
  m_inlier.p1 = 0;
  m_inlier.p2 = 3;
  m_inlier.kappa = kappa;
  m_inlier.tau = tau;
  m_inlier.fixedWeight = false;
  m_inlier.R = Tij.rotation();
  m_inlier.t = Tij.translation();
  measurements.push_back(m_inlier);
  // generate a single outlier loop closure
  DCORA::RelativeSEMeasurement m_outlier;
  m_outlier.r1 = 0;
  m_outlier.r2 = 0;
  m_outlier.p1 = 1;
  m_outlier.p2 = 3;
  m_outlier.kappa = kappa;
  m_outlier.tau = tau;
  m_outlier.fixedWeight = false;
  m_outlier.R = Eigen::Quaterniond::UnitRandom().toRotationMatrix();
  m_outlier.t = Eigen::Vector3d::Zero();
  measurements.push_back(m_outlier);
  // Solve!
  auto pose_graph = std::make_shared<DCORA::PoseGraph>(0, d, d);
  pose_graph->setMeasurements(measurements);
  DCORA::solveRobustPGOParams params;
  params.verbose = false;
  params.opt_params.verbose = false;
  params.opt_params.gradnorm_tol = 1e-1;
  params.opt_params.RTR_iterations = 50;
  params.robust_params.GNCBarc = 7.0;
  DCORA::PoseArray TOdom =
      DCORA::odometryInitialization(pose_graph->odometry());
  auto mutable_measurements = measurements;
  DCORA::PoseArray T =
      DCORA::solveRobustPGO(&mutable_measurements, params, &TOdom);
  // Check classification of inlier vs outlier
  for (const auto &m : mutable_measurements) {
    if (!m.fixedWeight) {
      if (m.p1 == 0 && m.p2 == 3)
        CHECK_NEAR(m.weight, 1, 1e-6);
      if (m.p1 == 1 && m.p2 == 3)
        CHECK_NEAR(m.weight, 0, 1e-6);
    }
  }
}
