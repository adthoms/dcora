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

#include <DCORA/DCORA_solver.h>
#include <DCORA/DCORA_types.h>
#include <DCORA/QuadraticProblem.h>

#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>

int main(int argc, char **argv) {
  /**
  ###########################################
  Parse input dataset
  ###########################################
  */

  if (argc != 2) {
    std::cout << "Chordal initialization example. " << std::endl;
    std::cout << "Usage: " << argv[0] << " [input .g2o file]" << std::endl;
    exit(1);
  }

  std::cout << "Chordal Initialization Example. " << std::endl;

  const DCORA::G2ODataset dataset = DCORA::read_g2o_file(argv[1]);
  const std::vector<DCORA::RelativePosePoseMeasurement> &measurements =
      dataset.pose_pose_measurements;
  size_t d = (!measurements.empty() ? dataset.dim : 0);
  std::cout << "Loaded dataset from file " << argv[1] << "." << std::endl;

  // Construct optimization problem
  std::shared_ptr<DCORA::Graph> pose_graph =
      std::make_shared<DCORA::Graph>(0, d, d);
  pose_graph->setMeasurements(measurements);
  DCORA::QuadraticProblem problemCentral(pose_graph);

  // Compute chordal relaxation
  auto TChordal = DCORA::chordalInitialization(measurements);
  std::cout << "Chordal initialization cost: "
            << 2 * problemCentral.f(TChordal.getData()) << std::endl;

  exit(0);
}
