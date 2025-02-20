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

#include <filesystem>

#include "gtest/gtest.h"

// Set optimization parameters
const double OPTIMIZATION_TOL = 1e-9;
const double OPTIMIZATION_RUNTIME = 2.0;

// Set datasets (datasets are noiseless)
const std::vector<std::string> g2o_datasets = {
    "pose_graph_optimization_test_2d.g2o",
    "pose_graph_optimization_test_3d.g2o"};
const std::vector<std::string> pyfg_datasets = {
    "range_aided_slam_test_2d.pyfg", "range_aided_slam_test_3d.pyfg"};

/**
 * @brief Get the full file path of datasets available in the data folder
 * @param datasetName
 * @return
 */
std::string getFullFileDataPath(const std::string &datasetName) {
  std::filesystem::path relativePath = "data/" + datasetName;
  std::filesystem::path basePath =
      std::filesystem::absolute("..").parent_path().parent_path();
  std::filesystem::path fullFilePath = basePath / relativePath;
  return fullFilePath.string();
}

TEST(testDCORA, testAgentConstructionSE) {
  unsigned int id = DCORA::CENTRALIZED_AGENT_ID;
  unsigned int d = 3;
  unsigned int r = 3;
  DCORA::AgentParameters options(d, r, {id}, DCORA::GraphType::PoseGraph);
  DCORA::Agent agent(id, options);
  ASSERT_EQ(agent.getID(), id);
  ASSERT_EQ(agent.relaxation_rank(), r);
  ASSERT_EQ(agent.dimension(), d);
  ASSERT_EQ(agent.num_poses(), 0);
}

TEST(testDCORA, testAgentConstructionRA) {
  unsigned int id = DCORA::CENTRALIZED_AGENT_ID;
  unsigned int d = 3;
  unsigned int r = 3;
  DCORA::AgentParameters options(d, r, {id},
                                 DCORA::GraphType::RangeAidedSLAMGraph);
  DCORA::Agent agent(id, options);
  ASSERT_EQ(agent.getID(), id);
  ASSERT_EQ(agent.relaxation_rank(), r);
  ASSERT_EQ(agent.dimension(), d);
  ASSERT_EQ(agent.num_poses(), 0);
}

TEST(testDCORA, testAgentOptimizationThreadBasicSE) {
  unsigned int id = DCORA::CENTRALIZED_AGENT_ID;
  unsigned int d = 3;
  unsigned int r = 3;
  DCORA::AgentParameters options(d, r, {id}, DCORA::GraphType::PoseGraph);
  DCORA::Agent agent(id, options);
  ASSERT_FALSE(agent.isOptimizationRunning());
  for (unsigned int trial = 0; trial < 3; ++trial) {
    agent.startOptimizationLoop();
    sleep(0.1);
    ASSERT_TRUE(agent.isOptimizationRunning());
    agent.endOptimizationLoop();
    ASSERT_FALSE(agent.isOptimizationRunning());
  }
}

TEST(testDCORA, testAgentOptimizationThreadBasicRA) {
  unsigned int id = DCORA::CENTRALIZED_AGENT_ID;
  unsigned int d = 3;
  unsigned int r = 3;
  DCORA::AgentParameters options(d, r, {id},
                                 DCORA::GraphType::RangeAidedSLAMGraph);
  DCORA::Agent agent(id, options);
  ASSERT_FALSE(agent.isOptimizationRunning());
  for (unsigned int trial = 0; trial < 3; ++trial) {
    agent.startOptimizationLoop();
    sleep(0.1);
    ASSERT_TRUE(agent.isOptimizationRunning());
    agent.endOptimizationLoop();
    ASSERT_FALSE(agent.isOptimizationRunning());
  }
}

TEST(testDCORA, testAgentInitializeIterateOptimizeSE) {
  for (const std::string &fileName : g2o_datasets) {
    // Read dataset
    const std::string fullFileName = getFullFileDataPath(fileName);
    const DCORA::G2ODataset dataset = DCORA::read_g2o_file(fullFileName);

    // Parse dataset
    const std::vector<DCORA::RelativePosePoseMeasurement> &measurements =
        dataset.pose_pose_measurements;
    unsigned int id = DCORA::CENTRALIZED_AGENT_ID;
    unsigned int d = dataset.dim;
    unsigned int r = d;
    unsigned int n = dataset.num_poses;
    const DCORA::PoseArray TrajectoryGroundTruth =
        dataset.getGroundTruthPoseArray();

    // Construct and initialize
    DCORA::AgentParameters options(d, r);
    DCORA::Agent agent(id, options);
    agent.setMeasurements(measurements);
    agent.initialize();
    ASSERT_EQ(agent.getID(), id);
    ASSERT_EQ(agent.relaxation_rank(), r);
    ASSERT_EQ(agent.dimension(), d);
    ASSERT_EQ(agent.num_poses(), n);

    // Get aligned ground truth
    const DCORA::Pose Tw0(TrajectoryGroundTruth.pose(0));
    const DCORA::PoseArray TrajectoryGroundTruthAligned =
        DCORA::alignTrajectoryToFrame(TrajectoryGroundTruth, Tw0);

    // Check default trajectory initialization
    DCORA::Matrix TrajectoryEstimated;
    agent.getTrajectoryInLocalFrame(&TrajectoryEstimated);
    ASSERT_TRUE(TrajectoryGroundTruthAligned.getData().isApprox(
        TrajectoryEstimated, OPTIMIZATION_TOL));

    // Check trajectory after one iteration
    agent.iterate();
    agent.getTrajectoryInLocalFrame(&TrajectoryEstimated);
    ASSERT_TRUE(TrajectoryGroundTruthAligned.getData().isApprox(
        TrajectoryEstimated, OPTIMIZATION_TOL));

    // Check optimization thread
    agent.startOptimizationLoop();
    sleep(OPTIMIZATION_RUNTIME);
    agent.endOptimizationLoop();
    agent.getTrajectoryInLocalFrame(&TrajectoryEstimated);
    ASSERT_TRUE(TrajectoryGroundTruthAligned.getData().isApprox(
        TrajectoryEstimated, OPTIMIZATION_TOL));

    // Reset
    agent.reset();
  }
}

TEST(testDCORA, testAgentInitializeIterateOptimizeRA) {
  for (const std::string &fileName : pyfg_datasets) {
    // Read dataset
    const std::string fullFileName = getFullFileDataPath(fileName);
    const DCORA::PyFGDataset dataset = DCORA::read_pyfg_file(fullFileName);

    // Parse dataset
    const DCORA::Measurements global_measurements =
        DCORA::getGlobalMeasurements(dataset);
    unsigned int id = DCORA::CENTRALIZED_AGENT_ID;
    unsigned int d = global_measurements.ground_truth_init->d();
    unsigned int r = d;
    unsigned int n = global_measurements.ground_truth_init->n();
    unsigned int l = global_measurements.ground_truth_init->l();
    unsigned int b = global_measurements.ground_truth_init->b();
    const DCORA::PoseArray TrajectoryGroundTruth =
        global_measurements.ground_truth_init->getPoseArray();
    const DCORA::PointArray UnitShereGroundTruth =
        global_measurements.ground_truth_init->getUnitSphereArray();
    const DCORA::PointArray LandmarkGroundTruth =
        global_measurements.ground_truth_init->getLandmarkArray();

    // Construct and initialize
    DCORA::AgentParameters options(d, r, {id},
                                   DCORA::GraphType::RangeAidedSLAMGraph);
    DCORA::Agent agent(id, options);
    agent.setMeasurements(global_measurements.relative_measurements);
    agent.initialize(&TrajectoryGroundTruth, &UnitShereGroundTruth,
                     &LandmarkGroundTruth);
    ASSERT_EQ(agent.getID(), id);
    ASSERT_EQ(agent.relaxation_rank(), r);
    ASSERT_EQ(agent.dimension(), d);
    ASSERT_EQ(agent.num_poses(), n);
    ASSERT_EQ(agent.num_unit_spheres(), l);
    ASSERT_EQ(agent.num_landmarks(), b);

    // Get aligned ground truth
    const DCORA::Pose Tw0(TrajectoryGroundTruth.pose(0));
    const DCORA::PoseArray TrajectoryGroundTruthAligned =
        DCORA::alignTrajectoryToFrame(TrajectoryGroundTruth, Tw0);
    const DCORA::PointArray UnitSpheresGroundTruthAligned =
        DCORA::alignUnitSpheresToFrame(UnitShereGroundTruth, Tw0);
    const DCORA::PointArray LandmarksGroundTruthAligned =
        DCORA::alignLandmarksToFrame(LandmarkGroundTruth, Tw0);

    // Check default state initialization
    DCORA::Matrix TrajectoryEstimated;
    DCORA::Matrix UnitSphereEstimated;
    DCORA::Matrix LandmarksEstimated;
    agent.getStatesInLocalFrame(&TrajectoryEstimated, &UnitSphereEstimated,
                                &LandmarksEstimated);
    ASSERT_TRUE(TrajectoryGroundTruthAligned.getData().isApprox(
        TrajectoryEstimated, OPTIMIZATION_TOL));
    ASSERT_TRUE(UnitSpheresGroundTruthAligned.getData().isApprox(
        UnitSphereEstimated, OPTIMIZATION_TOL));
    ASSERT_TRUE(LandmarksGroundTruthAligned.getData().isApprox(
        LandmarksEstimated, OPTIMIZATION_TOL));

    // Check trajectory after one iteration
    agent.iterate();
    agent.getStatesInLocalFrame(&TrajectoryEstimated, &UnitSphereEstimated,
                                &LandmarksEstimated);
    ASSERT_TRUE(TrajectoryGroundTruthAligned.getData().isApprox(
        TrajectoryEstimated, OPTIMIZATION_TOL));
    ASSERT_TRUE(UnitSpheresGroundTruthAligned.getData().isApprox(
        UnitSphereEstimated, OPTIMIZATION_TOL));
    ASSERT_TRUE(LandmarksGroundTruthAligned.getData().isApprox(
        LandmarksEstimated, OPTIMIZATION_TOL));

    // Check optimization thread
    agent.startOptimizationLoop();
    sleep(OPTIMIZATION_RUNTIME);
    agent.endOptimizationLoop();
    agent.getStatesInLocalFrame(&TrajectoryEstimated, &UnitSphereEstimated,
                                &LandmarksEstimated);
    ASSERT_TRUE(TrajectoryGroundTruthAligned.getData().isApprox(
        TrajectoryEstimated, OPTIMIZATION_TOL));
    ASSERT_TRUE(UnitSpheresGroundTruthAligned.getData().isApprox(
        UnitSphereEstimated, OPTIMIZATION_TOL));
    ASSERT_TRUE(LandmarksGroundTruthAligned.getData().isApprox(
        LandmarksEstimated, OPTIMIZATION_TOL));

    // Reset
    agent.reset();
  }
}

TEST(testDCORA, testAgentMapRA) {
  // Set map id and dimensions
  unsigned int id = DCORA::MAP_ID;
  unsigned int d = 3;
  unsigned int r = d;

  // Construct, lift, and initialize
  DCORA::AgentParameters options(d, r, {id},
                                 DCORA::GraphType::RangeAidedSLAMGraph);
  DCORA::Agent agent(id, options);
  DCORA::Matrix liftedMatrix = DCORA::fixedStiefelVariable(r, d);
  agent.setLiftingMatrix(liftedMatrix);
  agent.initialize();
  ASSERT_EQ(agent.getID(), id);
  ASSERT_EQ(agent.relaxation_rank(), r);
  ASSERT_EQ(agent.dimension(), d);
  ASSERT_EQ(agent.num_poses(), 1);
  ASSERT_EQ(agent.num_unit_spheres(), 0);
  ASSERT_EQ(agent.num_landmarks(), 0);

  // Check that the fixed global coordinate frame set by map is rotated by the
  // lifting matrix
  DCORA::Matrix XMap;
  ASSERT_TRUE(agent.getX(&XMap));
  ASSERT_TRUE(XMap.topLeftCorner(r, d).isApprox(liftedMatrix));
  ASSERT_TRUE(XMap.topRightCorner(r, 1).isApprox(DCORA::Matrix::Zero(r, 1)));

  // Given that the map agent is passive, calling the iterate method will
  // simply advance the iteration number (and the robust optimization inner
  // iteration number, which for RA-SLAM has no meaning) and return true
  ASSERT_TRUE(agent.iterate());

  // Check shared states (the map does not require awareness of other agents)
  DCORA::PoseDict sharedPoseDict;
  DCORA::UnitSphereDict sharedUnitSphereDict;
  DCORA::LandmarkDict sharedLandmarkDict;
  agent.getSharedStateDicts(&sharedPoseDict, &sharedUnitSphereDict,
                            &sharedLandmarkDict);
  ASSERT_TRUE(sharedPoseDict.empty());
  ASSERT_TRUE(sharedUnitSphereDict.empty());
  ASSERT_TRUE(sharedLandmarkDict.empty());

  // Reset
  agent.reset();
}

TEST(testDCORA, testAgentMultiAgentRA) {
  for (const std::string &fileName : pyfg_datasets) {
    // Read dataset
    const std::string fullFileName = getFullFileDataPath(fileName);
    const DCORA::PyFGDataset dataset = DCORA::read_pyfg_file(fullFileName);

    // Parse dataset
    const std::set<unsigned int> &robot_IDs = dataset.robot_IDs;
    const DCORA::Measurements global_measurements =
        DCORA::getGlobalMeasurements(dataset);
    const DCORA::RobotMeasurements robot_measurements =
        DCORA::getRobotMeasurements(dataset);
    const DCORA::RangeAidedArray XCentralizedGroundTruth =
        *global_measurements.ground_truth_init;
    const DCORA::LocalToGlobalStateDicts local_to_global_state_dicts =
        DCORA::getLocalToGlobalStateMapping(dataset, true);
    unsigned int d = dataset.dim;
    unsigned int r = d;
    unsigned int n = XCentralizedGroundTruth.n();
    unsigned int l = XCentralizedGroundTruth.l();
    unsigned int b = XCentralizedGroundTruth.b();

    // Initialize agents
    bool acceleration = true;
    unsigned int first_agent_id = *robot_IDs.begin();
    std::map<unsigned int, DCORA::Agent *> agents;
    for (unsigned int robot_id : robot_IDs) {
      const DCORA::Measurements &agent_measurements =
          robot_measurements.at(robot_id);
      DCORA::AgentParameters options(d, r, robot_IDs,
                                     DCORA::GraphType::RangeAidedSLAMGraph);
      options.acceleration = acceleration;
      options.verbose = true;

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

      // Set current iterate to ground truth
      if (robot_id == DCORA::MAP_ID)
        continue;
      agent->setX(agent_measurements.ground_truth_init->getData());

      // Push agent to map
      agents[robot_id] = agent;
    }

    // Lambda function to check that agent state is within tolerance of the
    // centralized ground truth.
    auto checkAgentStateIsGroundTruth = [&](const unsigned int robot_id) {
      DCORA::Matrix XAgent;
      agents.at(robot_id)->getX(&XAgent);
      unsigned int n = agents.at(robot_id)->num_poses();
      unsigned int l = agents.at(robot_id)->num_unit_spheres();
      unsigned int b = agents.at(robot_id)->num_landmarks();
      DCORA::LiftedRangeAidedArray XAgentLiftedArray(r, d, n, l, b);
      XAgentLiftedArray.setData(XAgent);

      for (const auto &[local_pose_id, global_pose_id] :
           local_to_global_state_dicts.poses) {
        if (robot_id != local_pose_id.robot_id)
          continue;
        const DCORA::Matrix &XAgentPose =
            XAgentLiftedArray.pose(local_pose_id.frame_id);
        const DCORA::Matrix &XCentralizedPose =
            XCentralizedGroundTruth.pose(global_pose_id.frame_id);
        ASSERT_TRUE(XAgentPose.isApprox(XCentralizedPose, OPTIMIZATION_TOL));
      }
      for (const auto &[local_unit_sphere_id, global_unit_sphere_id] :
           local_to_global_state_dicts.unit_spheres) {
        if (robot_id != local_unit_sphere_id.robot_id)
          continue;
        const DCORA::Vector &XAgentUnitSphere =
            XAgentLiftedArray.unitSphere(local_unit_sphere_id.frame_id);
        const DCORA::Vector &XCentralizedUnitSphere =
            XCentralizedGroundTruth.unitSphere(global_unit_sphere_id.frame_id);
        ASSERT_TRUE(XAgentUnitSphere.isApprox(XCentralizedUnitSphere,
                                              OPTIMIZATION_TOL));
      }
      for (const auto &[local_landmark_id, global_landmark_id] :
           local_to_global_state_dicts.landmarks) {
        if (robot_id != local_landmark_id.robot_id)
          continue;
        const DCORA::Vector &XAgentLandmark =
            XAgentLiftedArray.landmark(local_landmark_id.frame_id);
        const DCORA::Vector &XCentralizedLandmark =
            XCentralizedGroundTruth.landmark(global_landmark_id.frame_id);
        ASSERT_TRUE(
            XAgentLandmark.isApprox(XCentralizedLandmark, OPTIMIZATION_TOL));
      }
    };

    // Check centralized ground truth vs agent iterates after ground truth
    // initialization
    for (unsigned int robot_id : robot_IDs)
      checkAgentStateIsGroundTruth(robot_id);

    // Perform one iteration of RBCD/RBCD++ for each agent
    unsigned int iter = 0;
    for (unsigned int selected_robot_id : robot_IDs) {
      DCORA::Agent *selected_robot_ptr = agents.at(selected_robot_id);

      // Non-selected robots perform an iteration
      for (const auto &robot_id : robot_IDs) {
        DCORA::Agent *robot_ptr = agents.at(robot_id);
        ASSERT_EQ(robot_ptr->instance_number(), 0);
        ASSERT_EQ(robot_ptr->iteration_number(), iter);
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

      // Check centralized ground truth vs agent iterate
      checkAgentStateIsGroundTruth(selected_robot_id);

      // Increment iteration number
      iter++;
    }
  }
}
