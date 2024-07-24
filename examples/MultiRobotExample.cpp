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

#include <DCORA/Agent.h>
#include <DCORA/DCORA_solver.h>
#include <DCORA/DCORA_types.h>
#include <DCORA/QuadraticProblem.h>

#include <cassert>
#include <cstdlib>
#include <iostream>

int main(int argc, char **argv) {
  /**
   * @brief Parse User Input
   */

  // Check for correct inputs
  if (argc < 3) {
    std::cout << "Multi-robot pose graph optimization example. " << std::endl;
    std::cout << "Usage: " << argv[0] << " [# robots] [input .g2o file]"
              << std::endl;
    exit(1);
  }
  std::cout << "Multi-robot pose graph optimization example. " << std::endl;

  // Set number of robots
  int num_robots = atoi(argv[1]);
  if (num_robots <= 0) {
    std::cout << "Number of robots must be positive!" << std::endl;
    exit(1);
  }
  std::cout << "Simulating " << num_robots << " robots." << std::endl;

  // Load G2O dataset
  size_t n, d;
  std::vector<DCORA::RelativePosePoseMeasurement> dataset =
      DCORA::read_g2o_file(argv[2], &n);
  if (dataset.empty()) {
    std::cout << "G2O Dataset is empty. Exiting program." << std::endl;
    exit(1);
  }
  d = dataset[0].t.size();
  std::cout << "Loaded dataset from file " << argv[2] << "." << std::endl;

  // Check valid number of robots
  unsigned int num_poses_per_robot = n / num_robots;
  if (num_poses_per_robot <= 0) {
    std::cout
        << "More robots than total number of poses! Decrease the number of "
           "robots"
        << std::endl;
    exit(1);
  }

  // Create mapping from global pose index to local pose index
  std::map<unsigned, DCORA::PoseID> PoseMap;
  for (unsigned robot = 0; robot < (unsigned)num_robots; ++robot) {
    unsigned startIdx = robot * num_poses_per_robot;
    unsigned endIdx = (robot + 1) * num_poses_per_robot; // non-inclusive
    if (robot == (unsigned)num_robots - 1)
      endIdx = n;
    for (unsigned idx = startIdx; idx < endIdx; ++idx) {
      unsigned localIdx = idx - startIdx; // this is the local ID of this pose
      DCORA::PoseID pose(robot, localIdx);
      PoseMap[idx] = pose;
    }
  }

  // Partition dataset among robots
  std::vector<std::vector<DCORA::RelativePosePoseMeasurement>> odometry(
      num_robots);
  std::vector<std::vector<DCORA::RelativePosePoseMeasurement>>
      private_loop_closures(num_robots);
  std::vector<std::vector<DCORA::RelativePosePoseMeasurement>>
      shared_loop_closure(num_robots);
  for (auto mIn : dataset) {
    DCORA::PoseID src = PoseMap[mIn.p1];
    DCORA::PoseID dst = PoseMap[mIn.p2];

    unsigned srcRobot = src.robot_id;
    unsigned srcIdx = src.frame_id;
    unsigned dstRobot = dst.robot_id;
    unsigned dstIdx = dst.frame_id;

    DCORA::RelativePosePoseMeasurement m(srcRobot, dstRobot, srcIdx, dstIdx,
                                         mIn.R, mIn.t, mIn.kappa, mIn.tau);

    if (srcRobot == dstRobot) {
      // private measurement
      if (srcIdx + 1 == dstIdx) {
        // Odometry
        odometry[srcRobot].push_back(m);
      } else {
        // private loop closure
        private_loop_closures[srcRobot].push_back(m);
      }
    } else {
      // shared measurement
      shared_loop_closure[srcRobot].push_back(m);
      shared_loop_closure[dstRobot].push_back(m);
    }
  }

  // Set optimization options
  bool acceleration = true;
  bool verbose = false;
  unsigned numIters = 1000;
  std::string logDirectory = "/home/alex/data/dcora_dpgo_examples/"
                             "dcora_examples/multi_robot_example_pgo/";
  std::string debug_filename;
  bool logData = true;
  unsigned int r_min = 5;
  unsigned int r_max = 100;
  double shift = -100;
  double RGradNormTol = 0.1;
  DCORA::InitializationMethod init_method =
      DCORA::InitializationMethod::Odometry;
  bool rbcd_only = true;

  /**
   * @brief DC2-PGO Algorithm
   */
  DCORA::Matrix TInit;
  switch (init_method) {
  case DCORA::InitializationMethod::Odometry: {
    std::vector<DCORA::RelativePosePoseMeasurement> odometry_central;
    for (auto mIn : dataset) {
      if (mIn.p1 + 1 == mIn.p2) {
        odometry_central.push_back(mIn);
      }
    }
    DCORA::PoseArray TOdom = DCORA::odometryInitialization(odometry_central);
    TInit = TOdom.getData();
    break;
  }
  case DCORA::InitializationMethod::Chordal: {
    DCORA::PoseArray TChordal = DCORA::chordalInitialization(dataset);
    TInit = TChordal.getData();
    break;
  }
  default: {
    DCORA::Matrix M = DCORA::Matrix::Random(d, (d + 1) * n);
    TInit = DCORA::projectToSEMatrix(M, d, d, n);
    break;
  }
  }

  // Initialize current state estimate
  DCORA::Matrix Xcurr;

  for (unsigned int r = r_min; r < r_max; ++r) {
    // All agents share a special, common matrix called the 'lifting matrix'
    DCORA::Matrix lifting_matrix = DCORA::fixedStiefelVariable(r, d);

    // Construct the centralized problem (used for evaluation)
    std::shared_ptr<DCORA::Graph> pose_graph_curr_rank =
        std::make_shared<DCORA::Graph>(0, r, d);
    pose_graph_curr_rank->setMeasurements(dataset);
    DCORA::QuadraticProblem problemCentralCurrRank(pose_graph_curr_rank);

    std::shared_ptr<DCORA::Graph> pose_graph_next_rank =
        std::make_shared<DCORA::Graph>(0, r + 1, d);
    pose_graph_next_rank->setMeasurements(dataset);
    DCORA::QuadraticProblem problemCentralNextRank(pose_graph_next_rank);

    // Initialize agents
    std::vector<DCORA::PGOAgent *> agents;
    for (unsigned robot = 0; robot < static_cast<unsigned int>(num_robots);
         ++robot) {
      DCORA::PGOAgentParameters options(d, r, num_robots);
      options.acceleration = acceleration;
      options.verbose = verbose;
      options.logDirectory = logDirectory;
      options.logData = logData;

      auto *agent = new DCORA::PGOAgent(robot, options);
      agent->setLiftingMatrix(lifting_matrix);
      agent->setMeasurements(odometry[robot], private_loop_closures[robot],
                             shared_loop_closure[robot]);
      agent->initialize();
      agents.push_back(agent);
    }

    // Set X
    if (r == r_min)
      Xcurr = lifting_matrix * TInit;

    for (unsigned robot = 0; robot < (unsigned)num_robots; ++robot) {
      unsigned startIdx = robot * num_poses_per_robot;
      unsigned endIdx = (robot + 1) * num_poses_per_robot; // non-inclusive
      if (robot == (unsigned)num_robots - 1)
        endIdx = n;
      agents[robot]->setX(
          Xcurr.block(0, startIdx * (d + 1), r, (endIdx - startIdx) * (d + 1)));
    }

    // Compute first-order critical point using RBCD
    DCORA::Matrix Xopt(r, n * (d + 1));
    unsigned selectedRobot = 0;
    std::cout << "Running " << numIters << " iterations..." << std::endl;
    for (unsigned iter = 0; iter < numIters; ++iter) {
      DCORA::PGOAgent *selectedRobotPtr = agents[selectedRobot];

      // Non-selected robots perform an iteration
      for (auto *robotPtr : agents) {
        assert(robotPtr->instance_number() == 0);
        assert(robotPtr->iteration_number() == iter);
        if (robotPtr->getID() != selectedRobot) {
          robotPtr->iterate(false);
        }
      }

      // Selected robot requests public poses from others
      for (auto *robotPtr : agents) {
        if (robotPtr->getID() == selectedRobot)
          continue;
        DCORA::PoseDict sharedPoses;
        if (!robotPtr->getSharedPoseDict(&sharedPoses)) {
          continue;
        }
        selectedRobotPtr->setNeighborStatus(robotPtr->getStatus());
        selectedRobotPtr->updateNeighborPoses(robotPtr->getID(), sharedPoses);
      }

      // When using acceleration, selected robot also requests auxiliary poses
      if (acceleration) {
        for (auto *robotPtr : agents) {
          if (robotPtr->getID() == selectedRobot)
            continue;
          DCORA::PoseDict auxSharedPoses;
          if (!robotPtr->getAuxSharedPoseDict(&auxSharedPoses)) {
            continue;
          }
          selectedRobotPtr->setNeighborStatus(robotPtr->getStatus());
          selectedRobotPtr->updateAuxNeighborPoses(robotPtr->getID(),
                                                   auxSharedPoses);
        }
      }

      // Selected robot update
      selectedRobotPtr->iterate(true);

      // Form centralized solution
      for (unsigned robot = 0; robot < (unsigned)num_robots; ++robot) {
        unsigned startIdx = robot * num_poses_per_robot;
        unsigned endIdx = (robot + 1) * num_poses_per_robot; // non-inclusive
        if (robot == (unsigned)num_robots - 1)
          endIdx = n;

        DCORA::Matrix XRobot;
        if (agents[robot]->getX(&XRobot)) {
          Xopt.block(0, startIdx * (d + 1), r, (endIdx - startIdx) * (d + 1)) =
              XRobot;
        }
      }
      DCORA::Matrix RGrad = problemCentralCurrRank.RieGrad(Xopt);
      double RGradNorm = RGrad.norm();
      std::cout << std::setprecision(5) << "Iter = " << iter << " | "
                << "robot = " << selectedRobotPtr->getID() << " | "
                << "cost = " << 2 * problemCentralCurrRank.f(Xopt) << " | "
                << "gradnorm = " << RGradNorm << std::endl;

      // Exit if gradient norm is sufficiently small
      if (RGradNorm < RGradNormTol) {
        break;
      }

      // Select next robot with largest gradient norm
      std::vector<unsigned> neighbors = selectedRobotPtr->getNeighbors();
      if (neighbors.empty()) {
        selectedRobot = selectedRobotPtr->getID();
      } else {
        std::vector<double> gradNorms;
        for (size_t robot = 0; robot < (unsigned)num_robots; ++robot) {
          unsigned startIdx = robot * num_poses_per_robot;
          unsigned endIdx = (robot + 1) * num_poses_per_robot; // non-inclusive
          if (robot == (unsigned)num_robots - 1)
            endIdx = n;
          DCORA::Matrix RGradRobot = RGrad.block(0, startIdx * (d + 1), r,
                                                 (endIdx - startIdx) * (d + 1));
          gradNorms.push_back(RGradRobot.norm());
        }
        selectedRobot = std::max_element(gradNorms.begin(), gradNorms.end()) -
                        gradNorms.begin();
      }
    }

    if (rbcd_only) {
      LOG(INFO) << "RBCD completed. Outputting agent trajectories.";
      // Share global anchor for rounding
      DCORA::Matrix M;
      agents[0]->getSharedPose(0, &M);
      for (auto agentPtr : agents) {
        agentPtr->setGlobalAnchor(M);
        agentPtr->reset();
      }
      break;
    }

    // Construct corresponding dual certificate matrix
    double lambda = 0.1;
    const DCORA::SparseMatrix &Q = pose_graph_curr_rank->quadraticMatrix();
    const DCORA::SparseMatrix dual_certificate_matrix =
        DCORA::constructDualCertificateMatrixPGO(Xopt, Q, d, n, lambda);

    // Check if dual certificate matrix is PSD
    if (DCORA::isSparseSymmetricMatrixPSD(dual_certificate_matrix)) {
      LOG(INFO) << "Z = (X*)^T(X*) is a global minimizer! Outputting agent "
                   "trajectories";
      // Share global anchor for rounding
      DCORA::Matrix M;
      agents[0]->getSharedPose(0, &M);
      for (auto agentPtr : agents) {
        agentPtr->setGlobalAnchor(M);
        agentPtr->reset();
      }
      break;
    }
    LOG(INFO) << "State estimate at rank " << r
              << " is not a global minimizer. Proceeding to next rank.";

    // Compute minimum eigen pair
    auto [eigenValue, eigenVector] =
        DCORA::computeMinimumEigenPair(dual_certificate_matrix, shift);
    shift = eigenValue;

    // Lift first-order critical point to next rank
    DCORA::Matrix X_plus = DCORA::Matrix::Zero(r + 1, n * (d + 1));
    X_plus.block(0, 0, r, n * (d + 1)) = Xopt;

    // Construct second-order decent direction
    DCORA::Matrix X_dot_plus = DCORA::Matrix::Zero(r + 1, n * (d + 1));
    X_dot_plus.row(r) = eigenVector.transpose();

    // Descend from suboptimal point Xopt
    double alpha = 1.0;
    DCORA::Matrix X_retract =
        problemCentralNextRank.Retract(X_plus, alpha * X_dot_plus);
    while (problemCentralNextRank.f(X_retract) >=
               problemCentralNextRank.f(X_plus) ||
           problemCentralNextRank.RieGradNorm(X_retract) == 0) {
      alpha = alpha / 2;
      X_retract = problemCentralNextRank.Retract(X_plus, alpha * X_dot_plus);
    }

    // Set current X for next rank
    Xcurr.conservativeResize(Xcurr.rows() + 1, Eigen::NoChange);
    Xcurr = X_retract;
  }

  exit(0);
}
