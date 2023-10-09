#include <DiCORA/DiCORA_solver.h>
#include <DiCORA/PoseGraph.h>
#include <cstdlib>
#include <iostream>

using namespace std;
using namespace DiCORA;

#include <string>
#include <fstream>
#include <vector>
#include <glog/logging.h>

int main(int argc, char **argv) {
  /**
  ###########################################
  Parse input dataset
  ###########################################
  */

  if (argc < 2) {
    cout << "Single robot robust pose-graph optimization demo using graduated non-convexity (GNC). " << endl;
    cout << "Usage: " << argv[0] << " [input .g2o file]" << endl;
    exit(1);
  }

  cout << "Single robot robust pose-graph optimization demo using graduated non-convexity (GNC). " << endl;
  size_t num_poses;
  vector<RelativeSEMeasurement> measurements = read_g2o_file(argv[1], num_poses);
  CHECK(!measurements.empty());
  unsigned int dimension = measurements[0].t.size();
  auto pose_graph = std::make_shared<PoseGraph>(0, dimension, dimension);
  pose_graph->setMeasurements(measurements);

  solveRobustPGOParams params;
  params.opt_params.verbose = false;
  params.opt_params.gradnorm_tol = 1;
  params.opt_params.RTR_iterations = 50;
  params.verbose = true;
  PoseArray TOdom = odometryInitialization(pose_graph->odometry());
  PoseArray T = solveRobustPGO(measurements, params, &TOdom);
  exit(0);
}