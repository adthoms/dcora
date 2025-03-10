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
    LOG(INFO) << "Single robot RA-SLAM demo. ";
    LOG(INFO) << "Usage: " << argv[0] << " [input .pyfg file]";
    exit(1);
  }

  LOG(INFO) << "Single robot RA-SLAM demo. ";

  // Load PyFG dataset and get centralized measurements
  const DCORA::PyFGDataset dataset = DCORA::read_pyfg_file(argv[1]);
  const DCORA::Measurements global_measurements =
      DCORA::getGlobalMeasurements(dataset);
  const DCORA::RelativeMeasurements &measurements =
      global_measurements.relative_measurements;
  const DCORA::RangeAidedArray &ground_truth_init =
      *global_measurements.ground_truth_init;
  const DCORA::RobotMeasurements robot_measurements =
      DCORA::getRobotMeasurements(dataset);
  const DCORA::LocalToGlobalStateDicts local_to_global_state_dicts =
      DCORA::getLocalToGlobalStateMapping(dataset, true);

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

  // Initialization method
  DCORA::InitializationMethod init_method =
      DCORA::InitializationMethod::Odometry;

  // Hyperparameters
  double min_eig_num_tol = 1e-4;
  double gradient_tolerance = 1e-4;
  double preconditioned_gradient_tolerance = 1e-4;

  /**
   * @brief CORA Algorithm
   */

  // Initialize current state estimate
  DCORA::Matrix Xcurr = DCORA::Matrix::Zero(r_max, (d + 1) * n + l + b);
  DCORA::Matrix Xlift = DCORA::fixedStiefelVariable(r_min, d);
  switch (init_method) {
  case DCORA::InitializationMethod::Odometry: {
    DCORA::RangeAidedArray XOdomInit(d, n, l, b);

    // Calculate odometry for each agent
    for (unsigned int robot_id : dataset.robot_IDs) {
      if (robot_id == DCORA::MAP_ID)
        continue;

      // Get relative measurements and ground truth
      const DCORA::RelativeMeasurements &robot_relative_measurements =
          robot_measurements.at(robot_id).relative_measurements;
      const DCORA::RangeAidedArray &robot_ground_truth =
          *robot_measurements.at(robot_id).ground_truth_init;

      // Get odometry measurements
      std::vector<DCORA::RelativePosePoseMeasurement> odometryAgent;
      for (const auto &mVariant : robot_relative_measurements.vec) {
        if (!std::holds_alternative<DCORA::RelativePosePoseMeasurement>(
                mVariant))
          continue;
        const DCORA::RelativePosePoseMeasurement &m =
            std::get<DCORA::RelativePosePoseMeasurement>(mVariant);
        if (m.p1 + 1 != m.p2)
          continue;

        odometryAgent.push_back(m);
      }

      // Calculate odometry
      const DCORA::PoseArray XAGentOdom =
          DCORA::odometryInitialization(odometryAgent);

      // Align odometry with ground truth of agent's first pose
      const DCORA::StateID firstAgentGlobalStateID =
          local_to_global_state_dicts.poses.at(DCORA::PoseID(robot_id, 0));
      const unsigned int firstAgentGlobalStateIdx =
          firstAgentGlobalStateID.frame_id;
      const DCORA::Pose Tw0(ground_truth_init.pose(firstAgentGlobalStateIdx));
      DCORA::PoseArray XAGentOdomAligned =
          alignTrajectoryToFrame(XAGentOdom, Tw0.inverse());

      // Set poses for odometry initialization
      unsigned int n = dataset.robot_id_to_num_poses.at(robot_id);
      for (unsigned int i = 0; i < n; ++i) {
        XOdomInit.pose(firstAgentGlobalStateIdx + i) =
            XAGentOdomAligned.pose(i);
      }
    }

    // Set ground truth unit spheres
    DCORA::PointArray XUnitSpheres(d, l);
    XUnitSpheres.setData(
        ground_truth_init.GetLiftedUnitSphereArray()->getData());
    XOdomInit.setLiftedUnitSphereArray(XUnitSpheres);

    // Set random landmarks
    DCORA::PointArray XLandmarks(d, b);
    XLandmarks.setData(DCORA::Matrix::Random(d, b));
    XOdomInit.setLiftedLandmarkArray(XLandmarks);

    // Set initial state estimate
    Xcurr.topRows(r_min) = Xlift * XOdomInit.getData();
    break;
  }
  case DCORA::InitializationMethod::Random: {
    DCORA::Matrix M = DCORA::Matrix::Random(d, (d + 1) * n + l + b);
    Xcurr.topRows(r_min) = Xlift * DCORA::projectToRAMatrix(M, d, d, n, l, b);
    break;
  }
  default:
    // TODO(AT): Add ground truth initialization type
    Xcurr.topRows(r_min) = Xlift * ground_truth_init.getData();
  }

  // Log ground truth trajectory for each agent
  if (logData) {
    LOG(INFO) << "Outputting ground truth trajectory for each agent.";
    for (unsigned int robot_id : dataset.robot_IDs) {
      if (robot_id == DCORA::MAP_ID)
        continue;
      unsigned int n = dataset.robot_id_to_num_poses.at(robot_id);
      const DCORA::RangeAidedArray &robot_ground_truth =
          *robot_measurements.at(robot_id).ground_truth_init;
      DCORA::Matrix AgentTrajectoryGroundTruth =
          robot_ground_truth.GetLiftedPoseArray()->getData();

      const std::string filename =
          "cora_" + std::string(1, DCORA::FIRST_AGENT_SYMBOL + robot_id) +
          "_gt.txt";
      logger.logTrajectory(d, n, AgentTrajectoryGroundTruth, filename);
    }
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
    LOG(INFO) << "Gradient norm at rank " << r << ": "
              << problemCentralCurrRank.RieGrad(Xopt).norm();

    // Construct corresponding dual certificate matrix
    const DCORA::SparseMatrix &Q = graphCurrRank->quadraticMatrix();
    const DCORA::SparseMatrix S =
        DCORA::constructDualCertificateMatrixRASLAM(Xopt, Q, d, n, l, b);

    // Check if dual certificate matrix is PSD
    double theta;
    DCORA::Vector min_eigenvector;
    bool global_opt =
        DCORA::fastVerification(S, min_eig_num_tol, &theta, &min_eigenvector);

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
        LOG(INFO)
            << "Outputting rounded centralized trajectory for each agent.";

        // Set refined solution for trajectory output
        DCORA::RangeAidedArray X(d, n, l, b);
        X.setData(Xrefine);

        // Log rounded trajectories for each agent
        for (unsigned int robot_id : dataset.robot_IDs) {
          if (robot_id == DCORA::MAP_ID)
            continue;

          unsigned int n = dataset.robot_id_to_num_poses.at(robot_id);
          DCORA::PoseArray XAgentTrajectory(d, n);

          for (const auto &[local_pose_id, global_pose_id] :
               local_to_global_state_dicts.poses) {
            if (robot_id != local_pose_id.robot_id)
              continue;
            XAgentTrajectory.pose(local_pose_id.frame_id) =
                X.pose(global_pose_id.frame_id);
          }
          const std::string filename =
              "cora_" + std::string(1, DCORA::FIRST_AGENT_SYMBOL + robot_id) +
              ".txt";
          logger.logTrajectory(d, n, XAgentTrajectory.getData(), filename);
        }
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
      break;
    }
  }

  exit(0);
}
