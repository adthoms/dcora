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
#include <DCORA/QuadraticOptimizer.h>
#include <DCORA/QuadraticProblem.h>

int main(int argc, char **argv) {
  /**
   * @brief Parse User Input
   */

  if (argc < 2) {
    std::cout << "Single robot RA-SLAM demo. " << std::endl;
    std::cout << "Usage: " << argv[0] << " [input .pyfg file]" << std::endl;
    exit(1);
  }

  std::cout << "Single robot RA-SLAM demo. " << std::endl;

  // Load PyFG dataset and get centralized measurements
  const DCORA::PyFGDataset dataset = DCORA::read_pyfg_file(argv[1]);
  const DCORA::Measurements global_measurements =
      DCORA::getGlobalMeasurements(dataset);
  const DCORA::RelativeMeasurements &measurements =
      global_measurements.relative_measurements;
  const DCORA::RangeAidedArray &ground_truth_init =
      *global_measurements.ground_truth_init;

  /**
   * @brief Settings
   */

  // Set problem dimensions
  unsigned int d = ground_truth_init.d();
  unsigned int n = ground_truth_init.n();
  unsigned int l = ground_truth_init.l();
  unsigned int b = ground_truth_init.b();

  // set minimum and maximum rank
  unsigned int r_min = d;
  unsigned int r_max = 20;

  // Set optimization parameters
  DCORA::ROptParameters params;
  params.verbose = false;
  params.RTR_iterations = 200;
  params.RTR_tCG_iterations = 200;
  params.gradnorm_tol = 1e-4;
  params.RGD_stepsize = 1e-5;

  // Logging
  bool logData = true;
  std::string logDirectory = "/home/alex/data/dcora_dpgo_examples/"
                             "dcora_examples/single_robot_example_ra_slam/";
  DCORA::Logger logger(logDirectory);

  // Hyperparameters
  double min_eig_num_tol = 1e-4;
  double gradient_tolerance = 1e-4;
  double preconditioned_gradient_tolerance = 1e-4;
  double shift = -2;

  /**
   * @brief CORA Algorithm
   */

  // Initialize current state estimate
  DCORA::Matrix Xcurr = DCORA::Matrix::Zero(r_max, (d + 1) * n + l + b);

  // TODO(Alex): Add other initialization methods
  Xcurr.topRows(d) = ground_truth_init.getData();

  // Log ground truth trajectory
  if (logData) {
    DCORA::Matrix TGroundTruth =
        ground_truth_init.GetLiftedPoseArray()->getData();
    LOG(INFO) << "Outputting ground truth centralized trajectory.";
    logger.logTrajectory(d, n, TGroundTruth, "dcora_gt.txt");
  }

  for (unsigned int r = r_min; r < r_max; ++r) {
    // Construct the centralized problem
    std::shared_ptr<DCORA::Graph> graphCurrRank =
        std::make_shared<DCORA::Graph>(0, r, d,
                                       DCORA::GraphType::RangeAidedSLAMGraph);
    graphCurrRank->setMeasurements(measurements);
    DCORA::QuadraticProblem problemCentralCurrRank(graphCurrRank);

    std::shared_ptr<DCORA::Graph> graphNextRank =
        std::make_shared<DCORA::Graph>(0, r + 1, d,
                                       DCORA::GraphType::RangeAidedSLAMGraph);
    graphNextRank->setMeasurements(measurements);
    DCORA::QuadraticProblem problemCentralNextRank(graphNextRank);

    // Perform Riemannian optimization
    DCORA::QuadraticOptimizer optimizer(&problemCentralCurrRank, params);
    DCORA::Matrix Xopt = optimizer.optimize(Xcurr.topRows(r));
    LOG(INFO) << "Objective value at rank " << r << ": "
              << problemCentralCurrRank.f(Xopt);

    // Construct corresponding dual certificate matrix
    const DCORA::SparseMatrix &Q = graphCurrRank->quadraticMatrix();
    const DCORA::SparseMatrix S =
        DCORA::constructDualCertificateMatrixRASLAM(Xopt, Q, d, n, l, b);

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
      LOG(INFO) << "Z = (X*)^T(X*) is a global minimizer!";

      // Project solution
      const DCORA::Matrix Xproject =
          (r == d) ? Xopt : DCORA::projectSolutionRASLAM(Xopt, r, d, n, l, b);

      // Refine solution
      std::shared_ptr<DCORA::Graph> graphRankD = std::make_shared<DCORA::Graph>(
          0, d, d, DCORA::GraphType::RangeAidedSLAMGraph);
      graphRankD->setMeasurements(measurements);
      DCORA::QuadraticProblem problemCentralRankD(graphRankD);

      DCORA::QuadraticOptimizer optimizer(&problemCentralRankD, params);
      const DCORA::Matrix Xrefine = optimizer.optimize(Xproject);

      if (logData) {
        LOG(INFO) << "Outputting rounded centralized trajectory.";

        // Set refined solution for trajectory output
        DCORA::RangeAidedArray X(d, n, l, b);
        X.setData(Xrefine);

        // Set global anchor to first lifted pose
        const DCORA::LiftedPoseArray &Xposes = *X.GetLiftedPoseArray();
        const DCORA::LiftedPose Xa = DCORA::LiftedPose(Xposes.pose(0));

        // Get rounded trajectory
        DCORA::PoseArray T(d, n);
        T.setData(Xa.rotation().transpose() * Xposes.getData());
        const DCORA::Vector t0 = Xa.rotation().transpose() * Xa.translation();
        for (unsigned int i = 0; i < n; ++i)
          T.translation(i) = T.translation(i) - t0;

        // Log rounded trajectory
        logger.logTrajectory(d, n, T.getData(), "dcora_0.txt");
      }

      break;
    } else {
      LOG(INFO) << "Saddle point detected at rank " << r
                << "! Curvature along escape direction: " << theta;
    }

    DCORA::Matrix X;
    bool isSecondOrder = true; // centralized problem is second order
    bool escape_success = problemCentralNextRank.escapeSaddle(
        Xopt, theta, min_eigenvector, gradient_tolerance,
        preconditioned_gradient_tolerance, &X, isSecondOrder);
    if (escape_success) {
      // Update initialization point for next level in the Staircase
      Xcurr.topRows(r + 1) = X;
    } else {
      LOG(WARNING) << "Warning: Backtracking line search failed to escape from "
                      "Saddle point. Try decreasing the preconditioned "
                      "Gradient norm tolerance and/or the numerical tolerance "
                      "for minimum eigenvalue nonnegativity.";
      break;
    }
  }

  exit(0);
}
