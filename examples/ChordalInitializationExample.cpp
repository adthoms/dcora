
/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DiCORA/DiCORA_types.h>
#include <DiCORA/DiCORA_solver.h>
#include <DiCORA/QuadraticProblem.h>

#include <cstdlib>
#include <cassert>
#include <iostream>
#include <fstream>

using namespace std;
using namespace DiCORA;

int main(int argc, char **argv) {
  /**
  ###########################################
  Parse input dataset
  ###########################################
  */

  if (argc != 2) {
    cout << "Chordal initialization example. " << endl;
    cout << "Usage: " << argv[0] << " [input .g2o file]" << endl;
    exit(1);
  }

  cout << "Chordal Initialization Example. " << endl;

  size_t n;
  vector<RelativeSEMeasurement> dataset = read_g2o_file(argv[1], n);
  size_t d = (!dataset.empty() ? dataset[0].t.size() : 0);
  cout << "Loaded dataset from file " << argv[1] << "." << endl;

  // Construct optimization problem
  std::shared_ptr<PoseGraph> pose_graph = std::make_shared<PoseGraph>(0, d, d);
  pose_graph->setMeasurements(dataset);
  QuadraticProblem problemCentral(pose_graph);

  // Compute chordal relaxation
  auto TChordal = chordalInitialization(dataset);
  std::cout << "Chordal initialization cost: " << 2 * problemCentral.f(TChordal.getData()) << std::endl;

  exit(0);
}
