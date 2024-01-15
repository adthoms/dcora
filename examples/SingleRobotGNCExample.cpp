#include <DCORA/DCORA_solver.h>
#include <DCORA/Graphs.h>

#include <glog/logging.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  /**
  ###########################################
  Parse input dataset
  ###########################################
  */

  if (argc < 2) {
    std::cout
        << "Single robot robust pose-graph optimization demo using graduated "
           "non-convexity (GNC). "
        << std::endl;
    std::cout << "Usage: " << argv[0] << " [input .g2o file]" << std::endl;
    exit(1);
  }

  std::cout
      << "Single robot robust pose-graph optimization demo using graduated "
         "non-convexity (GNC). "
      << std::endl;
  size_t num_poses;
  std::vector<DCORA::RelativeSEMeasurement> measurements =
      DCORA::read_g2o_file(argv[1], &num_poses);
  CHECK(!measurements.empty());
  unsigned int dimension = measurements[0].t.size();
  auto pose_graph = std::make_shared<DCORA::PoseGraph>(0, dimension, dimension);
  pose_graph->setMeasurements(measurements);

  DCORA::solveRobustPGOParams params;
  params.opt_params.verbose = false;
  params.opt_params.gradnorm_tol = 1;
  params.opt_params.RTR_iterations = 50;
  params.verbose = true;
  DCORA::PoseArray TOdom =
      DCORA::odometryInitialization(pose_graph->odometry());
  DCORA::PoseArray T = DCORA::solveRobustPGO(&measurements, params, &TOdom);
  exit(0);
}
