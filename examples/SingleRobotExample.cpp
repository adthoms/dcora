
/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DCORA/CORAAgent.h>
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

  size_t num_poses;
  std::vector<DCORA::RelativeSEMeasurement> dataset =
      DCORA::read_g2o_file(argv[1], &num_poses);

  /**
  ###########################################
  Set parameters for PGOAgent
  ###########################################
  */

  unsigned int d, r;
  d = (!dataset.empty() ? dataset[0].t.size() : 0);
  r = d;
  DCORA::PGOAgentParameters options(d, r, 1);
  options.verbose = true;

  std::vector<DCORA::RelativeSEMeasurement> odometry;
  std::vector<DCORA::RelativeSEMeasurement> private_loop_closures;
  std::vector<DCORA::RelativeSEMeasurement> shared_loop_closure;
  for (const auto &mIn : dataset) {
    unsigned srcIdx = mIn.p1;
    unsigned dstIdx = mIn.p2;

    DCORA::RelativeSEMeasurement m(0, 0, srcIdx, dstIdx, mIn.R, mIn.t,
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
  auto pose_graph = std::make_shared<DCORA::PoseGraph>(0, d, d);
  pose_graph->setMeasurements(dataset);
  DCORA::QuadraticProblem problemCentral(pose_graph);

  /**
  ###########################################
  Initialization
  ###########################################
  */

  auto *agent = new DCORA::PGOAgent(0, options);
  agent->setMeasurements(odometry, private_loop_closures, shared_loop_closure);
  agent->initialize();

  /**
  ###########################################
  Local Pose Graph Optimization
  ###########################################
  */

  std::cout << "Running local pose graph optimization..." << std::endl;
  DCORA::Matrix X = agent->localPoseGraphOptimization();

  // Evaluate
  std::cout << "Cost = " << 2 * problemCentral.f(X) << std::endl;

  exit(0);
}
