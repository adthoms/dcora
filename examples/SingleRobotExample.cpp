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

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>   // std::stringstream
#include <stdexcept> // std::runtime_error
#include <string>
#include <utility> // std::pair
#include <vector>

int main(int argc, char **argv) {
  /**
  ###########################################
  Parse input dataset
  ###########################################
  */

  if (argc < 2) {
    std::cout << "Single robot pose-graph optimization. " << std::endl;
    std::cout << "Usage: " << argv[0] << " [input .g2o file]" << std::endl;
    exit(1);
  }

  std::cout << "Single robot pose-graph optimization demo. " << std::endl;

  const DCORA::G2ODataset dataset = DCORA::read_g2o_file(argv[1]);
  const std::vector<DCORA::RelativePosePoseMeasurement> &measurements =
      dataset.pose_pose_measurements;

  /**
  ###########################################
  Set parameters for Agent
  ###########################################
  */

  unsigned int d = (!measurements.empty() ? dataset.dim : 0);
  unsigned int r = d;
  DCORA::AgentParameters options(d, r);
  options.verbose = true;
  options.logDirectory = "/home/alex/data/dcora_dpgo_examples/"
                         "dcora_examples/single_robot_example_pgo/";
  options.logData = true;

  std::vector<DCORA::RelativePosePoseMeasurement> odometry;
  std::vector<DCORA::RelativePosePoseMeasurement> private_loop_closures;
  std::vector<DCORA::RelativePosePoseMeasurement> shared_loop_closure;
  for (const auto &mIn : measurements) {
    unsigned srcIdx = mIn.p1;
    unsigned dstIdx = mIn.p2;

    DCORA::RelativePosePoseMeasurement m(0, 0, srcIdx, dstIdx, mIn.R, mIn.t,
                                         mIn.kappa, mIn.tau);

    if (srcIdx + 1 == dstIdx) {
      // Odometry
      odometry.push_back(m);
    } else {
      // private loop closure
      private_loop_closures.push_back(m);
    }
  }

  // Construct the centralized PGO problem (used for evaluation)
  auto pose_graph = std::make_shared<DCORA::Graph>(0, d, d);
  pose_graph->setMeasurements(measurements);
  DCORA::QuadraticProblem problemCentral(pose_graph);

  /**
  ###########################################
  Initialization
  ###########################################
  */

  auto *agent = new DCORA::Agent(0, options);
  agent->setMeasurements(odometry, private_loop_closures, shared_loop_closure);
  agent->initialize();

  const DCORA::Matrix T_WORLD = DCORA::LiftedPose(r, d).getData();
  agent->setGlobalAnchor(T_WORLD);

  /**
  ###########################################
  Local Pose Graph Optimization
  ###########################################
  */

  std::cout << "Running local pose graph optimization..." << std::endl;
  DCORA::Matrix X = agent->localPoseGraphOptimization();

  // Evaluate
  std::cout << "Cost = " << 2 * problemCentral.f(X) << std::endl;

  // Log trajectory
  agent->reset();

  exit(0);
}
