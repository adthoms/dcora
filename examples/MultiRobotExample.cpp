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
  unsigned int num_robots = atoi(argv[1]);
  if (num_robots <= 0) {
    std::cout << "Number of robots must be positive!" << std::endl;
    exit(1);
  }
  std::cout << "Simulating " << num_robots << " robots." << std::endl;

  // Load G2O dataset
  const DCORA::G2ODataset dataset = DCORA::read_g2o_file(argv[2]);
  const std::vector<DCORA::RelativePosePoseMeasurement> &measurements =
      dataset.pose_pose_measurements;
  if (measurements.empty()) {
    std::cout << "G2O Dataset is empty. Exiting program." << std::endl;
    exit(1);
  }
  size_t d = dataset.dim;
  size_t n = dataset.num_poses;
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

  // Create set of robot IDs
  // Note: this example assume consecutive robot IDs from 0 to num_robots - 1
  std::set<unsigned int> robot_IDs;
  for (unsigned int i = 0; i <= (num_robots - 1); ++i)
    robot_IDs.insert(i);

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
  for (auto mIn : measurements) {
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
  double min_eig_num_tol = 1e-3;
  double gradient_tolerance = 1e-6;
  double preconditioned_gradient_tolerance = 1e-6;
  double shift = -10;
  double RGradNormTol = 0.1;
  DCORA::InitializationMethod init_method = DCORA::InitializationMethod::Random;
  bool rbcd_only = false;

  /**
   * @brief DC2-PGO Algorithm
   */

  // Initialize current state estimate
  DCORA::Matrix Xcurr = DCORA::Matrix::Zero(r_max, n * (d + 1));
  switch (init_method) {
  case DCORA::InitializationMethod::Odometry: {
    std::vector<DCORA::RelativePosePoseMeasurement> odometryCentral;
    for (auto mIn : measurements) {
      if (mIn.p1 + 1 != mIn.p2)
        continue;

      odometryCentral.push_back(mIn);
    }
    DCORA::PoseArray TOdom = DCORA::odometryInitialization(odometryCentral);
    Xcurr.topRows(d) = TOdom.getData();
    break;
  }
  case DCORA::InitializationMethod::Chordal: {
    DCORA::PoseArray TChordal = DCORA::chordalInitialization(measurements);
    Xcurr.topRows(d) = TChordal.getData();
    break;
  }
  case DCORA::InitializationMethod::Random: {
    DCORA::Matrix M = DCORA::Matrix::Random(r_min, (d + 1) * n);
    Xcurr.topRows(r_min) = DCORA::projectToSEMatrix(M, r_min, d, n);
    break;
  }
  default:
    LOG(FATAL) << "Error: Invalid initialization method: "
               << InitializationMethodToString(init_method) << "!";
  }

  unsigned int totalIter = 0;
  for (unsigned int r = r_min; r < r_max; ++r) {
    // Construct the centralized problem (used for evaluation)
    std::shared_ptr<DCORA::Graph> poseGraphCurrRank =
        std::make_shared<DCORA::Graph>(0, r, d);
    poseGraphCurrRank->setMeasurements(measurements);
    DCORA::QuadraticProblem problemCentralCurrRank(poseGraphCurrRank);

    std::shared_ptr<DCORA::Graph> poseGraphNextRank =
        std::make_shared<DCORA::Graph>(0, r + 1, d);
    poseGraphNextRank->setMeasurements(measurements);
    DCORA::QuadraticProblem problemCentralNextRank(poseGraphNextRank);

    // Initialize agents
    std::vector<DCORA::Agent *> agents;
    for (unsigned robot = 0; robot < static_cast<unsigned int>(num_robots);
         ++robot) {
      DCORA::AgentParameters options(d, r, robot_IDs);
      options.acceleration = acceleration;
      options.verbose = verbose;
      options.logDirectory = logDirectory;
      options.logData = logData;

      auto *agent = new DCORA::Agent(robot, options);

      // All agents share a special, common matrix called the 'lifting matrix'
      // which the first agent will generate
      if (robot > 0) {
        DCORA::Matrix M;
        agents[0]->getLiftingMatrix(&M);
        agent->setLiftingMatrix(M);
      }
      agent->setMeasurements(odometry[robot], private_loop_closures[robot],
                             shared_loop_closure[robot]);
      agent->initialize();
      agents.push_back(agent);
    }

    // Set X
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
      DCORA::Agent *selectedRobotPtr = agents[selectedRobot];

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
        if (!robotPtr->getSharedStateDicts(&sharedPoses))
          continue;
        selectedRobotPtr->setNeighborStatus(robotPtr->getStatus());
        selectedRobotPtr->updateNeighborStates(robotPtr->getID(), sharedPoses);
      }

      // When using acceleration, selected robot also requests auxiliary poses
      if (acceleration) {
        for (auto *robotPtr : agents) {
          if (robotPtr->getID() == selectedRobot)
            continue;
          DCORA::PoseDict auxSharedPoses;
          if (!robotPtr->getSharedStateDicts(&auxSharedPoses))
            continue;
          selectedRobotPtr->setNeighborStatus(robotPtr->getStatus());
          selectedRobotPtr->updateNeighborStates(robotPtr->getID(),
                                                 auxSharedPoses, acceleration);
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
      std::cout << std::setprecision(5) << "Iter = " << totalIter << " | "
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
      totalIter++;
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
    const DCORA::SparseMatrix &Q = poseGraphCurrRank->quadraticMatrix();
    const DCORA::SparseMatrix S =
        DCORA::constructDualCertificateMatrixPGO(Xopt, Q, d, n);

    // Check if dual certificate matrix is PSD
    double theta;
    DCORA::Vector min_eigenvector;
    bool global_opt = DCORA::fastVerification(S, min_eig_num_tol, shift, &theta,
                                              &min_eigenvector);

    // Check eigenvalue convergence
    if (!global_opt && theta >= -min_eig_num_tol / 2)
      LOG(FATAL) << "Error: Escape direction computation did not converge to "
                    "desired precision!";

    if (global_opt) {
      LOG(INFO)
          << "Z = (X*)^T(X*) is a global minimizer! Outputting rounded agent "
             "trajectories.";
      // Share global anchor for rounding
      DCORA::Matrix M;
      agents[0]->getSharedPose(0, &M);
      for (auto agentPtr : agents) {
        agentPtr->setGlobalAnchor(M);
        agentPtr->reset();
      }
      break;
    } else {
      LOG(INFO) << "Saddle point detected at rank " << r
                << "! Curvature along escape direction: " << theta;
    }

    DCORA::Matrix X;
    bool escape_success = problemCentralNextRank.escapeSaddle(
        Xopt, theta, min_eigenvector, gradient_tolerance,
        preconditioned_gradient_tolerance, &X);
    if (escape_success) {
      // Update initialization point for next level in the Staircase
      Xcurr.topRows(r + 1) = X;
    } else {
      break;
    }
  }

  exit(0);
}
