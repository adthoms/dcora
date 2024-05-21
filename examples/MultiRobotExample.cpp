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
  ###########################################
  Parse input dataset
  ###########################################
  */

  if (argc < 3) {
    std::cout << "Multi-robot pose graph optimization example. " << std::endl;
    std::cout << "Usage: " << argv[0] << " [# robots] [input .g2o file]"
              << std::endl;
    exit(1);
  }

  std::cout << "Multi-robot pose graph optimization example. " << std::endl;

  int num_robots = atoi(argv[1]);
  if (num_robots <= 0) {
    std::cout << "Number of robots must be positive!" << std::endl;
    exit(1);
  }
  std::cout << "Simulating " << num_robots << " robots." << std::endl;

  size_t num_poses;
  std::vector<DCORA::RelativePosePoseMeasurement> dataset =
      DCORA::read_g2o_file(argv[2], &num_poses);
  std::cout << "Loaded dataset from file " << argv[2] << "." << std::endl;

  /**
  ###########################################
  Options
  ###########################################
  */
  unsigned int n, d, r;
  d = (!dataset.empty() ? dataset[0].t.size() : 0);
  n = num_poses;
  r = 5;
  bool acceleration = true;
  bool verbose = false;
  unsigned numIters = 1000;

  // Construct the centralized problem (used for evaluation)
  std::shared_ptr<DCORA::Graph> pose_graph =
      std::make_shared<DCORA::Graph>(0, r, d);
  pose_graph->setMeasurements(dataset);
  DCORA::QuadraticProblem problemCentral(pose_graph);

  /**
  ###########################################
  Partition dataset into robots
  ###########################################
  */
  unsigned int num_poses_per_robot = num_poses / num_robots;
  if (num_poses_per_robot <= 0) {
    std::cout
        << "More robots than total number of poses! Decrease the number of "
           "robots"
        << std::endl;
    exit(1);
  }

  // create mapping from global pose index to local pose index
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

  /**
  ###########################################
  Initialization
  ###########################################
  */
  std::vector<DCORA::PGOAgent *> agents;
  for (unsigned robot = 0; robot < static_cast<unsigned int>(num_robots);
       ++robot) {
    DCORA::PGOAgentParameters options(d, r, num_robots);
    options.acceleration = acceleration;
    options.verbose = verbose;

    auto *agent = new DCORA::PGOAgent(robot, options);

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

  /**
  ##########################################################################################
  For this demo, we initialize each robot's estimate from the centralized
  chordal relaxation
  ##########################################################################################
  */
  auto TChordal = DCORA::chordalInitialization(dataset);
  DCORA::Matrix XChordal =
      DCORA::fixedStiefelVariable(r, d) *
      TChordal.getData(); // Lift estimate to the correct relaxation rank
  for (unsigned robot = 0; robot < (unsigned)num_robots; ++robot) {
    unsigned startIdx = robot * num_poses_per_robot;
    unsigned endIdx = (robot + 1) * num_poses_per_robot; // non-inclusive
    if (robot == (unsigned)num_robots - 1)
      endIdx = n;
    agents[robot]->setX(XChordal.block(0, startIdx * (d + 1), r,
                                       (endIdx - startIdx) * (d + 1)));
  }

  /**
  ###########################################
  Optimization loop
  ###########################################
  */
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
    DCORA::Matrix RGrad = problemCentral.RieGrad(Xopt);
    double RGradNorm = RGrad.norm();
    std::cout << std::setprecision(5) << "Iter = " << iter << " | "
              << "robot = " << selectedRobotPtr->getID() << " | "
              << "cost = " << 2 * problemCentral.f(Xopt) << " | "
              << "gradnorm = " << RGradNorm << std::endl;

    // Exit if gradient norm is sufficiently small
    if (RGradNorm < 0.1) {
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

    // Share global anchor for rounding
    DCORA::Matrix M;
    agents[0]->getSharedPose(0, &M);
    for (auto agentPtr : agents) {
      agentPtr->setGlobalAnchor(M);
    }
  }

  for (auto agentPtr : agents) {
    agentPtr->reset();
  }

  exit(0);
}
