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
   * @brief Parse User Input
   */

  // Check for correct inputs
  if (argc < 2) {
    LOG(INFO) << "Multi-robot RA-SLAM demo. ";
    LOG(INFO) << "Usage: " << argv[0] << " [input .pyfg file]";
    exit(1);
  }
  LOG(INFO) << "Multi-robot RA-SLAM demo. ";

  // Load PyFG dataset and get centralized measurements
  const DCORA::PyFGDataset dataset = DCORA::read_pyfg_file(argv[1]);
  const std::set<unsigned int> &robot_IDs = dataset.robot_IDs;
  if (robot_IDs.find(DCORA::MAP_ID) != robot_IDs.end())
    LOG(FATAL)
        << "In distributed RA-SLAM, the Map cannot own any variables "
           "(i.e. poses, unit spheres, or landmark). State variables "
           "must be owned by active agents. Eventually, when prior "
           "measurements are implemented, the map agent will be allowed.";

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
  unsigned int r_max = 100;

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
                             "dcora_examples/multi_robot_example_ra_slam/";
  DCORA::Logger logger(logDirectory);
  bool rbcd_only = false;

  // Initialization method
  DCORA::InitializationMethod init_method =
      DCORA::InitializationMethod::Odometry;
  DCORA::BlockSelectionRule block_selection_rule =
      DCORA::BlockSelectionRule::Greedy;
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<std::mt19937::result_type> uniform_sampling(
      0, robot_IDs.size() - 1);

  // Hyperparameters
  double min_eig_num_tol = 1e-3;
  double gradient_tolerance = 1e-4;
  double preconditioned_gradient_tolerance = 1e-4;
  bool acceleration = true;
  bool verbose = false;
  unsigned num_iters = 1000;
  double RGradNormTol = 0.1;

  /**
   * @brief DCORA Algorithm
   */

  // Initialize current state estimate
  DCORA::Matrix Xcurr = DCORA::Matrix::Zero(r_max, (d + 1) * n + l + b);
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
    Xcurr.topRows(d) = XOdomInit.getData();
    break;
  }
  case DCORA::InitializationMethod::Random: {
    DCORA::Matrix M = DCORA::Matrix::Random(d, (d + 1) * n + l + b);
    Xcurr.topRows(d) = DCORA::projectToRAMatrix(M, d, d, n, l, b);
    break;
  }
  default:
    // TODO(AT): Add ground truth initialization type
    Xcurr.topRows(d) = ground_truth_init.getData();
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
          "dcora_" + std::string(1, DCORA::FIRST_AGENT_SYMBOL + robot_id) +
          "_gt.txt";
      logger.logTrajectory(d, n, AgentTrajectoryGroundTruth, filename);
    }
  }

  unsigned int totalIter = 0;
  for (unsigned int r = r_min; r < r_max; ++r) {
    LOG(INFO) << "Solving Riemannian staircase at rank " << r;

    // Construct the centralized problem (used for evaluation)
    std::shared_ptr<DCORA::Graph> raslamGraphCurrRank =
        std::make_shared<DCORA::Graph>(0, r, d,
                                       DCORA::GraphType::RangeAidedSLAMGraph);
    raslamGraphCurrRank->setMeasurements(measurements);
    DCORA::QuadraticProblem problemCentralCurrRank(raslamGraphCurrRank);

    std::shared_ptr<DCORA::Graph> raslamGraphNextRank =
        std::make_shared<DCORA::Graph>(0, r + 1, d,
                                       DCORA::GraphType::RangeAidedSLAMGraph);
    raslamGraphNextRank->setMeasurements(measurements);
    DCORA::QuadraticProblem problemCentralNextRank(raslamGraphNextRank);

    // Initialize agents
    DCORA::LiftedRangeAidedArray XCurrArray(r, d, n, l, b);
    XCurrArray.setData(Xcurr.topRows(r));
    unsigned int first_agent_id = *robot_IDs.begin();
    std::map<unsigned int, DCORA::Agent *> agents;
    for (unsigned int robot_id : robot_IDs) {
      const DCORA::Measurements &agent_measurements =
          robot_measurements.at(robot_id);
      DCORA::AgentParameters options(d, r, robot_IDs,
                                     DCORA::GraphType::RangeAidedSLAMGraph);
      options.acceleration = acceleration;
      options.verbose = verbose;
      options.logDirectory = logDirectory;
      options.logData = logData;

      auto *agent = new DCORA::Agent(robot_id, options);

      // All agents share a common lifting matrix generated by the first agent
      if (robot_id != first_agent_id) {
        DCORA::Matrix M;
        agents.at(first_agent_id)->getLiftingMatrix(&M);
        agent->setLiftingMatrix(M);
      }
      if (robot_id != DCORA::MAP_ID) {
        agent->setMeasurements(agent_measurements.relative_measurements);
      }
      agent->initialize();

      // Set current iterate
      if (robot_id == DCORA::MAP_ID)
        continue;
      unsigned int n = dataset.robot_id_to_num_poses.at(robot_id);
      unsigned int l = dataset.robot_id_to_num_unit_spheres.at(robot_id);
      unsigned int b = dataset.robot_id_to_num_landmarks.at(robot_id);
      DCORA::LiftedRangeAidedArray XAgentCurrArray(r, d, n, l, b);

      for (const auto &[local_pose_id, global_pose_id] :
           local_to_global_state_dicts.poses) {
        if (robot_id != local_pose_id.robot_id)
          continue;
        XAgentCurrArray.pose(local_pose_id.frame_id) =
            XCurrArray.pose(global_pose_id.frame_id);
      }
      for (const auto &[local_unit_sphere_id, global_unit_sphere_id] :
           local_to_global_state_dicts.unit_spheres) {
        if (robot_id != local_unit_sphere_id.robot_id)
          continue;
        XAgentCurrArray.unitSphere(local_unit_sphere_id.frame_id) =
            XCurrArray.unitSphere(global_unit_sphere_id.frame_id);
      }
      for (const auto &[local_landmark_id, global_landmark_id] :
           local_to_global_state_dicts.landmarks) {
        if (robot_id != local_landmark_id.robot_id)
          continue;
        XAgentCurrArray.landmark(local_landmark_id.frame_id) =
            XCurrArray.landmark(global_landmark_id.frame_id);
      }
      agent->setX(XAgentCurrArray.getData());

      agents[robot_id] = agent;
    }

    // Compute first-order critical point using RBCD
    DCORA::LiftedRangeAidedArray XOptArray(r, d, n, l, b);
    DCORA::Matrix XOpt;
    unsigned int selected_robot_id = *robot_IDs.begin();
    LOG(INFO) << "RBCD/RBCD++";
    LOG(INFO) << "Maximum number of iterations: " << num_iters;
    LOG(INFO) << "Block selection rule: "
              << DCORA::BlockSelectionRuleToString(block_selection_rule);
    LOG(INFO) << "# iter robot cost gradnorm";
    for (unsigned iter = 0; iter < num_iters; ++iter) {
      DCORA::Agent *selected_robot_ptr = agents.at(selected_robot_id);

      // Non-selected robots perform an iteration
      for (const auto &robot_id : robot_IDs) {
        DCORA::Agent *robot_ptr = agents.at(robot_id);
        assert(robot_ptr->instance_number() == 0);
        assert(robot_ptr->iteration_number() == iter);
        if (robot_ptr->getID() != selected_robot_id)
          robot_ptr->iterate(false);
      }

      // Selected robot requests public states from others
      for (const auto &robot_id : robot_IDs) {
        DCORA::Agent *robot_ptr = agents.at(robot_id);
        if (robot_ptr->getID() == selected_robot_id)
          continue;
        DCORA::PoseDict sharedPoses;
        DCORA::UnitSphereDict sharedUnitSpheres;
        DCORA::LandmarkDict sharedLandmarks;
        if (!robot_ptr->getSharedStateDicts(&sharedPoses, &sharedUnitSpheres,
                                            &sharedLandmarks))
          continue;
        selected_robot_ptr->setNeighborStatus(robot_ptr->getStatus());
        selected_robot_ptr->updateNeighborStates(
            robot_ptr->getID(), sharedPoses, false, sharedUnitSpheres,
            sharedLandmarks);
      }

      // When using acceleration, selected robot also requests auxiliary states
      if (acceleration) {
        for (const auto &robot_id : robot_IDs) {
          DCORA::Agent *robot_ptr = agents.at(robot_id);
          if (robot_ptr->getID() == selected_robot_id)
            continue;
          DCORA::PoseDict auxSharedPoses;
          DCORA::UnitSphereDict auxSharedUnitSpheres;
          DCORA::LandmarkDict auxSharedLandmarks;
          if (!robot_ptr->getSharedStateDicts(
                  &auxSharedPoses, &auxSharedUnitSpheres, &auxSharedLandmarks))
            continue;
          selected_robot_ptr->setNeighborStatus(robot_ptr->getStatus());
          selected_robot_ptr->updateNeighborStates(
              robot_ptr->getID(), auxSharedPoses, acceleration,
              auxSharedUnitSpheres, auxSharedLandmarks);
        }
      }

      // Selected robot performs an iteration
      selected_robot_ptr->iterate(true);

      // Form centralized solution
      for (const auto &robot_id : robot_IDs) {
        DCORA::Matrix XAgent;
        if (agents[robot_id]->getX(&XAgent)) {
          if (robot_id == DCORA::MAP_ID)
            continue;

          unsigned int n = dataset.robot_id_to_num_poses.at(robot_id);
          unsigned int l = dataset.robot_id_to_num_unit_spheres.at(robot_id);
          unsigned int b = dataset.robot_id_to_num_landmarks.at(robot_id);
          DCORA::LiftedRangeAidedArray XAgentOptArray(r, d, n, l, b);
          XAgentOptArray.setData(XAgent);

          for (const auto &[local_pose_id, global_pose_id] :
               local_to_global_state_dicts.poses) {
            if (robot_id != local_pose_id.robot_id)
              continue;
            XOptArray.pose(global_pose_id.frame_id) =
                XAgentOptArray.pose(local_pose_id.frame_id);
          }
          for (const auto &[local_unit_sphere_id, global_unit_sphere_id] :
               local_to_global_state_dicts.unit_spheres) {
            if (robot_id != local_unit_sphere_id.robot_id)
              continue;
            XOptArray.unitSphere(global_unit_sphere_id.frame_id) =
                XAgentOptArray.unitSphere(local_unit_sphere_id.frame_id);
          }
          for (const auto &[local_landmark_id, global_landmark_id] :
               local_to_global_state_dicts.landmarks) {
            if (robot_id != local_landmark_id.robot_id)
              continue;
            XOptArray.landmark(global_landmark_id.frame_id) =
                XAgentOptArray.landmark(local_landmark_id.frame_id);
          }
        }
      }

      XOpt = XOptArray.getData();
      DCORA::Matrix RGrad = problemCentralCurrRank.RieGrad(XOpt);
      double RGradNorm = RGrad.norm();
      DCORA::LiftedRangeAidedArray RGradArray(r, d, n, l, b);
      RGradArray.setData(RGrad);
      LOG(INFO) << std::fixed << totalIter << " " << selected_robot_ptr->getID()
                << " " << std::setprecision(6) << " "
                << problemCentralCurrRank.f(XOpt) << " " << RGradNorm;

      // Exit if gradient norm is sufficiently small
      if (RGradNorm < RGradNormTol)
        break;

      // Select next robot with largest gradient norm
      if (selected_robot_ptr->getNeighbors().empty()) {
        selected_robot_id = selected_robot_ptr->getID();
      } else {
        std::map<unsigned int, unsigned int> gradNorms;
        for (const auto &robot_id : robot_IDs) {
          if (robot_id == DCORA::MAP_ID)
            continue;

          unsigned int n = dataset.robot_id_to_num_poses.at(robot_id);
          unsigned int l = dataset.robot_id_to_num_unit_spheres.at(robot_id);
          unsigned int b = dataset.robot_id_to_num_landmarks.at(robot_id);
          DCORA::LiftedRangeAidedArray XAgentGradArray(r, d, n, l, b);

          for (const auto &[local_pose_id, global_pose_id] :
               local_to_global_state_dicts.poses) {
            if (robot_id != local_pose_id.robot_id)
              continue;
            XAgentGradArray.pose(local_pose_id.frame_id) =
                RGradArray.pose(global_pose_id.frame_id);
          }
          for (const auto &[local_unit_sphere_id, global_unit_sphere_id] :
               local_to_global_state_dicts.unit_spheres) {
            if (robot_id != local_unit_sphere_id.robot_id)
              continue;
            XAgentGradArray.unitSphere(local_unit_sphere_id.frame_id) =
                RGradArray.unitSphere(global_unit_sphere_id.frame_id);
          }
          for (const auto &[local_landmark_id, global_landmark_id] :
               local_to_global_state_dicts.landmarks) {
            if (robot_id != local_landmark_id.robot_id)
              continue;
            XAgentGradArray.landmark(local_landmark_id.frame_id) =
                RGradArray.landmark(global_landmark_id.frame_id);
          }
          gradNorms[robot_id] = XAgentGradArray.getData().norm();
        }

        switch (block_selection_rule) {
        case DCORA::BlockSelectionRule::Greedy: {
          auto it = std::max_element(
              gradNorms.begin(), gradNorms.end(),
              [](const std::pair<unsigned int, unsigned int> &a,
                 const std::pair<unsigned int, unsigned int> &b) {
                return a.second < b.second;
              });
          if (it == gradNorms.end())
            LOG(FATAL)
                << "Error: Failed to find agent with maximum gradient norm!";
          selected_robot_id = it->first;
          break;
        }
        case DCORA::BlockSelectionRule::Uniform: {
          auto it = gradNorms.begin();
          std::advance(it, uniform_sampling(rng));
          selected_robot_id = it->first;
          break;
        }
        }
      }
      totalIter++;
    }

    if (rbcd_only) {
      LOG(INFO) << "RBCD completed. Outputting agent trajectories.";
      // Share global anchor for rounding
      DCORA::Matrix M;
      agents[first_agent_id]->getSharedPose(0, &M);
      for (const auto &robot_id : robot_IDs) {
        DCORA::Agent *robot_ptr = agents.at(robot_id);
        robot_ptr->setGlobalAnchor(M);
        robot_ptr->reset();
      }
      break;
    }

    // Construct corresponding dual certificate matrix
    const DCORA::SparseMatrix &Q = raslamGraphCurrRank->quadraticMatrix();
    const DCORA::SparseMatrix S =
        DCORA::constructDualCertificateMatrixRASLAM(XOpt, Q, d, n, l, b);

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
      LOG(INFO)
          << "Z = (X*)^T(X*) is a global minimizer! Outputting rounded agent "
             "trajectories.";
      // Share global anchor for rounding
      DCORA::Matrix M;
      agents[first_agent_id]->getSharedPose(0, &M);
      for (const auto &robot_id : robot_IDs) {
        DCORA::Agent *robot_ptr = agents.at(robot_id);
        robot_ptr->setGlobalAnchor(M);
        robot_ptr->reset();
      }
      break;
    } else {
      LOG(INFO) << "Saddle point detected at rank " << r
                << "! Curvature along escape direction: " << theta;
    }

    DCORA::Matrix X;
    bool escape_success = problemCentralNextRank.escapeSaddle(
        XOpt, theta, min_eigenvector, gradient_tolerance,
        preconditioned_gradient_tolerance, &X);
    if (escape_success) {
      // Update initialization point for next level in the Staircase
      Xcurr.topRows(r + 1) = X;
    } else {
      break;
    }
  }

  exit(0);
}
