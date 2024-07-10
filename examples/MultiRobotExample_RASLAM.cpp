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

  if (argc < 2) {
    std::cout << "Multi-robot RA-SLAM demo. " << std::endl;
    std::cout << "Usage: " << argv[0] << " [input .pyfg file]" << std::endl;
    exit(1);
  }

  std::cout << "Multi-robot RA-SLAM demo. " << std::endl;

  DCORA::PyFGDataset dataset = DCORA::read_pyfg_file(argv[1]);
  DCORA::RobotMeasurements robot_measurements =
      DCORA::getRobotMeasurements(dataset);

  unsigned int d, r;
  d = dataset.dim;
  r = 5;
  bool acceleration = true;
  bool verbose = false;
  unsigned numIters = 1000;

  // TODO(Alex): Used for debugging purposes for Graph Milestone.
  for (const auto &[robot_id, measurements] : robot_measurements) {
    std::shared_ptr<DCORA::Graph> graph =
        std::make_shared<DCORA::Graph>(robot_id, r, d);
    graph->setMeasurements(measurements.relative_measurements);
  }

  // TODO(Alex): Implement remaining RA-SLAM demo similar to
  // MultiRobotExample.cpp

  exit(0);
}
