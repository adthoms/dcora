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

#include <CLI/CLI.hpp>

// Configuration structure to hold all parameters
struct Config {
  // Input/Output
  std::string input_file;
  std::string log_directory;
  bool log_data = true;
  bool verbose = false;

  // Problem dimensions
  unsigned int r_min = 0;
  unsigned int r_max = 100;

  // Optimization parameters
  unsigned int rtr_iterations = 200;
  unsigned int rtr_tcg_iterations = 200;
  double gradnorm_tol = 1e-4;
  double rgd_stepsize = 1e-5;

  // Algorithm parameters
  double min_eig_num_tol = 1e-4;
  double gradient_tolerance = 1e-4;
  double preconditioned_gradient_tolerance = 1e-4;

  // Initialization method
  std::string init_method = "ground_truth";
};

// Parse command line arguments
Config parseArguments(int argc, char **argv) {
  CLI::App app{"CORA"};
  Config config;

  // Required arguments
  app.add_option("input_file", config.input_file, "Input .pyfg file")
      ->required()
      ->check(CLI::ExistingFile);

  // Optional arguments
  app.add_option("--log-dir", config.log_directory, "Directory for output logs")
      ->check(CLI::ExistingDirectory);

  app.add_flag("--log-data,!--no-log-data", config.log_data,
               "Enable/disable data logging");
  app.add_flag("--verbose,!--quiet", config.verbose, "Enable verbose output");

  // Problem dimensions
  app.add_option("--r-min", config.r_min, "Minimum rank (0 = auto-set to d)");
  app.add_option("--r-max", config.r_max, "Maximum rank");

  // Optimization parameters
  app.add_option("--rtr-iterations", config.rtr_iterations, "RTR iterations");
  app.add_option("--rtr-tcg-iterations", config.rtr_tcg_iterations,
                 "RTR tCG iterations");
  app.add_option("--gradnorm-tol", config.gradnorm_tol,
                 "Gradient norm tolerance");
  app.add_option("--rgd-stepsize", config.rgd_stepsize, "RGD step size");

  // Algorithm parameters
  app.add_option("--min-eig-tol", config.min_eig_num_tol,
                 "Minimum eigenvalue numerical tolerance");
  app.add_option("--grad-tol", config.gradient_tolerance, "Gradient tolerance");
  app.add_option("--precon-grad-tol", config.preconditioned_gradient_tolerance,
                 "Preconditioned gradient tolerance");

  // Initialization method
  app.add_option("--init-method", config.init_method, "Initialization method")
      ->check(CLI::IsMember({"odometry", "random", "ground_truth"}));

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    exit(app.exit(e));
  }

  return config;
}

// Load and validate dataset
struct DatasetInfo {
  DCORA::PyFGDataset dataset;
  DCORA::Measurements global_measurements;
  DCORA::RobotMeasurements robot_measurements;
  DCORA::LocalToGlobalStateDicts local_to_global_state_dicts;
  unsigned int d, n, l, b;
};

DatasetInfo loadDataset(const Config &config) {
  LOG(INFO) << "Loading dataset from: " << config.input_file;

  DatasetInfo info;
  info.dataset = DCORA::read_pyfg_file(config.input_file);
  info.global_measurements = DCORA::getGlobalMeasurements(info.dataset);
  info.robot_measurements = DCORA::getRobotMeasurements(info.dataset);
  info.local_to_global_state_dicts =
      DCORA::getLocalToGlobalStateMapping(info.dataset, true);

  const DCORA::RangeAidedArray &ground_truth_init =
      *info.global_measurements.ground_truth_init;
  info.d = ground_truth_init.d();
  info.n = ground_truth_init.n();
  info.l = ground_truth_init.l();
  info.b = ground_truth_init.b();

  LOG(INFO) << "Dataset loaded successfully:";
  LOG(INFO) << "  Dimensions: d=" << info.d << ", n=" << info.n
            << ", l=" << info.l << ", b=" << info.b;
  LOG(INFO) << "  Number of robots: " << info.dataset.robot_IDs.size();

  return info;
}

// Setup optimization parameters
DCORA::ROptParameters setupOptimizationParameters(const Config &config) {
  DCORA::ROptParameters params;
  params.verbose = config.verbose;
  params.RTR_iterations = config.rtr_iterations;
  params.RTR_tCG_iterations = config.rtr_tcg_iterations;
  params.gradnorm_tol = config.gradnorm_tol;
  params.RGD_stepsize = config.rgd_stepsize;
  return params;
}

// Get initialization method enum
DCORA::InitializationMethod getInitializationMethod(const std::string &method) {
  if (method == "odometry") {
    return DCORA::InitializationMethod::Odometry;
  } else if (method == "random") {
    return DCORA::InitializationMethod::Random;
  } else {
    return DCORA::InitializationMethod::GNC_TLS; // Default for ground_truth
  }
}

// Initialize state estimate
DCORA::Matrix initializeStateEstimate(const Config &config,
                                      const DatasetInfo &info) {
  const auto &ground_truth_init = *info.global_measurements.ground_truth_init;
  unsigned int r_min = (config.r_min == 0) ? info.d : config.r_min;

  DCORA::Matrix Xcurr = DCORA::Matrix::Zero(
      config.r_max, (info.d + 1) * info.n + info.l + info.b);
  DCORA::Matrix Xlift = DCORA::fixedStiefelVariable(r_min, info.d);

  DCORA::InitializationMethod init_method =
      getInitializationMethod(config.init_method);

  switch (init_method) {
  case DCORA::InitializationMethod::Odometry: {
    LOG(INFO) << "Using odometry initialization";
    DCORA::RangeAidedArray XOdomInit(info.d, info.n, info.l, info.b);

    // Calculate odometry for each agent
    for (unsigned int robot_id : info.dataset.robot_IDs) {
      if (robot_id == DCORA::MAP_ID)
        continue;

      const DCORA::RelativeMeasurements &robot_relative_measurements =
          info.robot_measurements.at(robot_id).relative_measurements;
      const DCORA::RangeAidedArray &robot_ground_truth =
          *info.robot_measurements.at(robot_id).ground_truth_init;

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
          info.local_to_global_state_dicts.poses.at(DCORA::PoseID(robot_id, 0));
      const unsigned int firstAgentGlobalStateIdx =
          firstAgentGlobalStateID.frame_id;
      const DCORA::Pose Tw0(ground_truth_init.pose(firstAgentGlobalStateIdx));
      DCORA::PoseArray XAGentOdomAligned =
          alignTrajectoryToFrame(XAGentOdom, Tw0.inverse());

      // Set poses for odometry initialization
      unsigned int n = info.dataset.robot_id_to_num_poses.at(robot_id);
      for (unsigned int i = 0; i < n; ++i) {
        XOdomInit.pose(firstAgentGlobalStateIdx + i) =
            XAGentOdomAligned.pose(i);
      }
    }

    // Set ground truth unit spheres
    DCORA::PointArray XUnitSpheres(info.d, info.l);
    XUnitSpheres.setData(
        ground_truth_init.GetLiftedUnitSphereArray()->getData());
    XOdomInit.setLiftedUnitSphereArray(XUnitSpheres);

    // Set random landmarks
    DCORA::PointArray XLandmarks(info.d, info.b);
    XLandmarks.setData(DCORA::Matrix::Random(info.d, info.b));
    XOdomInit.setLiftedLandmarkArray(XLandmarks);

    Xcurr.topRows(r_min) = Xlift * XOdomInit.getData();
    break;
  }
  case DCORA::InitializationMethod::Random: {
    LOG(INFO) << "Using random initialization";
    DCORA::Matrix M =
        DCORA::Matrix::Random(info.d, (info.d + 1) * info.n + info.l + info.b);
    Xcurr.topRows(r_min) =
        Xlift *
        DCORA::projectToRAMatrix(M, info.d, info.d, info.n, info.l, info.b);
    break;
  }
  default:
    LOG(INFO) << "Using ground truth initialization";
    Xcurr.topRows(r_min) = Xlift * ground_truth_init.getData();
  }

  return Xcurr;
}

// Log ground truth trajectories
void logGroundTruthTrajectories(const Config &config, const DatasetInfo &info,
                                const DCORA::Logger &logger) {
  if (!config.log_data)
    return;

  LOG(INFO) << "Outputting ground truth trajectory for each agent.";
  for (unsigned int robot_id : info.dataset.robot_IDs) {
    if (robot_id == DCORA::MAP_ID)
      continue;

    unsigned int n = info.dataset.robot_id_to_num_poses.at(robot_id);
    const DCORA::RangeAidedArray &robot_ground_truth =
        *info.robot_measurements.at(robot_id).ground_truth_init;
    DCORA::Matrix AgentTrajectoryGroundTruth =
        robot_ground_truth.GetLiftedPoseArray()->getData();

    const std::string filename =
        "cora_" + std::string(1, DCORA::FIRST_AGENT_SYMBOL + robot_id) +
        "_gt.txt";
    logger.logTrajectory(info.d, n, AgentTrajectoryGroundTruth, filename);
  }
}

// Log refined trajectories
void logRefinedTrajectories(const Config &config, const DatasetInfo &info,
                            const DCORA::Matrix &Xrefine,
                            const DCORA::Logger &logger) {
  if (!config.log_data)
    return;

  LOG(INFO) << "Outputting rounded centralized trajectory for each agent.";

  DCORA::RangeAidedArray X(info.d, info.n, info.l, info.b);
  X.setData(Xrefine);

  for (unsigned int robot_id : info.dataset.robot_IDs) {
    if (robot_id == DCORA::MAP_ID)
      continue;

    unsigned int n = info.dataset.robot_id_to_num_poses.at(robot_id);
    DCORA::PoseArray XAgentTrajectory(info.d, n);

    for (const auto &[local_pose_id, global_pose_id] :
         info.local_to_global_state_dicts.poses) {
      if (robot_id != local_pose_id.robot_id)
        continue;
      XAgentTrajectory.pose(local_pose_id.frame_id) =
          X.pose(global_pose_id.frame_id);
    }

    const std::string filename =
        "cora_" + std::string(1, DCORA::FIRST_AGENT_SYMBOL + robot_id) + ".txt";
    logger.logTrajectory(info.d, n, XAgentTrajectory.getData(), filename);
  }
}

// Perform single rank optimization iteration
struct OptimizationResult {
  DCORA::Matrix Xopt;
  bool is_global_optimum;
  double theta;
  DCORA::Vector min_eigenvector;
  double objective_value;
  double gradient_norm;
};

OptimizationResult
performRankOptimization(unsigned int r, const Config &config,
                        const DatasetInfo &info, const DCORA::Matrix &Xcurr,
                        const DCORA::ROptParameters &params) {
  LOG(INFO) << "Optimizing at rank " << r;

  // Construct the centralized problem
  std::shared_ptr<DCORA::Graph> graphCurrRank = std::make_shared<DCORA::Graph>(
      0, r, info.d, DCORA::GraphType::RangeAidedSLAMGraph);
  graphCurrRank->setMeasurements(
      info.global_measurements.relative_measurements);
  DCORA::QuadraticProblem problemCentralCurrRank(graphCurrRank);

  // Perform Riemannian optimization
  DCORA::QuadraticOptimizer optimizer(&problemCentralCurrRank, params);
  DCORA::Matrix Xopt = optimizer.optimize(Xcurr.topRows(r));

  OptimizationResult result;
  result.Xopt = Xopt;
  result.objective_value = problemCentralCurrRank.f(Xopt);
  result.gradient_norm = problemCentralCurrRank.RieGrad(Xopt).norm();

  LOG(INFO) << "Objective value at rank " << r << ": "
            << result.objective_value;
  LOG(INFO) << "Gradient norm at rank " << r << ": " << result.gradient_norm;

  // Construct corresponding dual certificate matrix
  const DCORA::SparseMatrix &Q = graphCurrRank->quadraticMatrix();
  const DCORA::SparseMatrix S = DCORA::constructDualCertificateMatrixRASLAM(
      Xopt, Q, info.d, info.n, info.l, info.b);

  // Check if dual certificate matrix is PSD
  result.is_global_optimum = DCORA::fastVerification(
      S, config.min_eig_num_tol, &result.theta, &result.min_eigenvector);

  // Check eigenvalue convergence
  if (!result.is_global_optimum &&
      result.theta >= -config.min_eig_num_tol / 2) {
    LOG(FATAL) << "Error: Escape direction computation did not converge to "
                  "desired precision!";
  }

  return result;
}

// Refine solution at rank d
DCORA::Matrix refineSolution(const Config &config, const DatasetInfo &info,
                             const DCORA::Matrix &Xproject,
                             const DCORA::ROptParameters &params) {
  std::shared_ptr<DCORA::Graph> graphRankD = std::make_shared<DCORA::Graph>(
      0, info.d, info.d, DCORA::GraphType::RangeAidedSLAMGraph);
  graphRankD->setMeasurements(info.global_measurements.relative_measurements);
  DCORA::QuadraticProblem problemCentralRankD(graphRankD);

  DCORA::QuadraticOptimizer optimizer(&problemCentralRankD, params);
  return optimizer.optimize(Xproject);
}

// Main CORA algorithm
void runCORAAlgorithm(const Config &config, const DatasetInfo &info,
                      const DCORA::Logger &logger) {
  const DCORA::ROptParameters params = setupOptimizationParameters(config);
  DCORA::Matrix Xcurr = initializeStateEstimate(config, info);

  unsigned int r_min = (config.r_min == 0) ? info.d : config.r_min;

  for (unsigned int r = r_min; r < config.r_max; ++r) {
    OptimizationResult result =
        performRankOptimization(r, config, info, Xcurr, params);

    if (result.is_global_optimum) {
      LOG(INFO) << "Z = (X*)^T(X*) is a global minimizer!";

      // Project solution
      const DCORA::Matrix Xproject =
          (r == info.d) ? result.Xopt
                        : DCORA::projectSolutionRASLAM(result.Xopt, r, info.d,
                                                       info.n, info.l, info.b);

      // Refine solution
      const DCORA::Matrix Xrefine =
          refineSolution(config, info, Xproject, params);

      // Log refined trajectories
      logRefinedTrajectories(config, info, Xrefine, logger);
      break;
    } else {
      LOG(INFO) << "Saddle point detected at rank " << r
                << "! Curvature along escape direction: " << result.theta;
    }

    // Escape saddle point
    std::shared_ptr<DCORA::Graph> graphNextRank =
        std::make_shared<DCORA::Graph>(0, r + 1, info.d,
                                       DCORA::GraphType::RangeAidedSLAMGraph);
    graphNextRank->setMeasurements(
        info.global_measurements.relative_measurements);
    DCORA::QuadraticProblem problemCentralNextRank(graphNextRank);

    DCORA::Matrix X;
    bool isSecondOrder = true;
    bool escape_success = problemCentralNextRank.escapeSaddle(
        result.Xopt, result.theta, result.min_eigenvector,
        config.gradient_tolerance, config.preconditioned_gradient_tolerance, &X,
        isSecondOrder);

    if (escape_success) {
      Xcurr.topRows(r + 1) = X;
    } else {
      LOG(WARNING) << "Failed to escape saddle point at rank " << r;
      break;
    }
  }
}

int main(int argc, char **argv) {
  // Parse command line arguments
  Config config = parseArguments(argc, argv);

  LOG(INFO) << "CORA";
  LOG(INFO) << "Configuration:";
  LOG(INFO) << "  Input file: " << config.input_file;
  LOG(INFO) << "  Log directory: " << config.log_directory;
  LOG(INFO) << "  Initialization method: " << config.init_method;
  LOG(INFO) << "  Rank range: " << config.r_min << " to " << config.r_max;

  // Load dataset
  DatasetInfo dataset_info = loadDataset(config);

  // Update r_min if it was auto-set
  if (config.r_min == 0) {
    config.r_min = dataset_info.d;
  }

  // Setup logging
  const DCORA::Logger logger(config.log_directory);
  logGroundTruthTrajectories(config, dataset_info, logger);

  // Run CORA algorithm
  runCORAAlgorithm(config, dataset_info, logger);
  return 0;
}
