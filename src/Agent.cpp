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
#include <DCORA/QuadraticOptimizer.h>
#include <glog/logging.h>

#include <Eigen/CholmodSupport>
#include <algorithm>
#include <chrono> // NOLINT(build/c++11)
#include <cmath>
#include <iostream>
#include <memory>
#include <random>

namespace DCORA {

Agent::Agent(unsigned ID, const AgentParameters &params)
    : mID(ID),
      d(params.d),
      r(params.r),
      X(r, d, 1, 0, 0, params.graphType),
      mParams(params),
      mState(AgentState::WAIT_FOR_DATA),
      mStatus(ID, mState, 0, 0, false, 0),
      mRobustCost(params.robustCostParams),
      mGraph(std::make_shared<Graph>(mID, r, d, params.graphType)),
      mInstanceNumber(0),
      mIterationNumber(0),
      mLatestWeightUpdateIteration(0),
      mRobustOptInnerIter(0),
      mWeightUpdateCount(0),
      mTrajectoryResetCount(0),
      mLogger(params.logDirectory),
      gamma(0),
      alpha(0),
      Y(X),
      V(X),
      XPrev(X) {
  if (mID == 0)
    setLiftingMatrix(fixedStiefelVariable(r, d));
  for (const auto &robot_id : mParams.robotIDs) {
    if (!isAgentMap(robot_id))
      mTeamRobotActive[robot_id] = true;
    else
      mTeamRobotActive[robot_id] = false;
  }
}

Agent::~Agent() {
  // Make sure that optimization thread is not running, before exiting
  endOptimizationLoop();
}

void Agent::setX(const Matrix &Xin) {
  std::lock_guard<std::mutex> lock(mStatesMutex);
  CHECK_NE(mState, AgentState::WAIT_FOR_DATA);
  CHECK_EQ(Xin.rows(), relaxation_rank());
  CHECK_EQ(Xin.cols(), problem_dimension());
  mState = AgentState::INITIALIZED;
  X.setData(Xin);
  if (mParams.acceleration) {
    initializeAcceleration();
  }
  LOG_IF(INFO, mParams.verbose)
      << "Robot " << getID() << " resets trajectory with length "
      << num_poses();
}

void Agent::setXToInitialGuess() {
  CHECK_NE(mState, AgentState::WAIT_FOR_DATA);
  CHECK(XInit.has_value());
  std::lock_guard<std::mutex> lock(mStatesMutex);
  X = XInit.value();
}

bool Agent::getX(Matrix *Mout) {
  std::lock_guard<std::mutex> lock(mStatesMutex);
  *Mout = X.getData();
  return true;
}

bool Agent::getSharedPose(unsigned int index, Matrix *Mout) {
  if (mState != AgentState::INITIALIZED)
    return false;
  std::lock_guard<std::mutex> lock(mStatesMutex);
  if (index >= num_poses())
    return false;
  *Mout = X.pose(index);
  return true;
}

bool Agent::getAuxSharedPose(unsigned int index, Matrix *Mout) {
  CHECK(mParams.acceleration);
  if (mState != AgentState::INITIALIZED)
    return false;
  std::lock_guard<std::mutex> lock(mStatesMutex);
  if (index >= num_poses())
    return false;
  *Mout = Y.pose(index);
  return true;
}

bool Agent::getSharedStateDicts(PoseDict *poseDict,
                                UnitSphereDict *unitSphereDict,
                                LandmarkDict *landmarkDict) {
  if (isPGOCompatible())
    CHECK(!unitSphereDict && !landmarkDict);
  if (mState != AgentState::INITIALIZED)
    return false;
  poseDict->clear();
  if (unitSphereDict)
    unitSphereDict->clear();
  if (landmarkDict)
    landmarkDict->clear();

  // Lambda function for validating dictionary entries
  auto validateStateID = [this](const auto &state_id) {
    CHECK_EQ(state_id.robot_id, getID());
    return state_id.frame_id;
  };

  std::lock_guard<std::mutex> lock(mStatesMutex);
  for (const auto &pose_id : mGraph->myPublicPoseIDs()) {
    unsigned int frame_id = validateStateID(pose_id);
    const LiftedPose Xi(X.pose(frame_id));
    poseDict->emplace(pose_id, Xi);
  }
  for (const auto &unit_sphere_id : mGraph->myPublicUnitSphereIDs()) {
    unsigned int frame_id = validateStateID(unit_sphere_id);
    const LiftedPoint Xi(X.unitSphere(frame_id));
    if (unitSphereDict)
      unitSphereDict->emplace(unit_sphere_id, Xi);
  }
  for (const auto &landmark_id : mGraph->myPublicLandmarkIDs()) {
    unsigned int frame_id = validateStateID(landmark_id);
    const LiftedPoint Xi(X.landmark(frame_id));
    if (landmarkDict)
      landmarkDict->emplace(landmark_id, Xi);
  }

  return true;
}

bool Agent::getSharedPoseDictWithNeighbor(PoseDict *map, unsigned neighborID) {
  if (mState != AgentState::INITIALIZED)
    return false;
  map->clear();
  std::lock_guard<std::mutex> lock(mStatesMutex);
  const RelativeMeasurements measurements =
      mGraph->sharedLoopClosuresWithRobot(neighborID);
  for (const auto &m : measurements.GetRelativePosePoseMeasurements()) {
    if (m.r1 == getID()) {
      PoseID pose_id(m.r1, m.p1);
      LiftedPose Xi(X.pose(m.p1));
      map->emplace(pose_id, Xi);
    } else if (m.r2 == getID()) {
      PoseID pose_id(m.r2, m.p2);
      LiftedPose Xi(X.pose(m.p2));
      map->emplace(pose_id, Xi);
    }
  }
  return true;
}

bool Agent::getAuxSharedPoseDictWithNeighbor(PoseDict *map,
                                             unsigned neighborID) {
  if (mState != AgentState::INITIALIZED)
    return false;
  map->clear();
  std::lock_guard<std::mutex> lock(mStatesMutex);
  const RelativeMeasurements measurements =
      mGraph->sharedLoopClosuresWithRobot(neighborID);
  for (const auto &m : measurements.GetRelativePosePoseMeasurements()) {
    if (m.r1 == getID()) {
      PoseID pose_id(m.r1, m.p1);
      LiftedPose Yi(Y.pose(m.p1));
      map->emplace(pose_id, Yi);
    } else if (m.r2 == getID()) {
      PoseID pose_id(m.r2, m.p2);
      LiftedPose Yi(Y.pose(m.p2));
      map->emplace(pose_id, Yi);
    }
  }
  return true;
}

void Agent::setLiftingMatrix(const Matrix &M) {
  CHECK_EQ(M.rows(), r);
  CHECK_EQ(M.cols(), d);
  YLift.emplace(M);
}

void Agent::addMeasurement(const RelativePosePoseMeasurement &factor) {
  if (mState != AgentState::WAIT_FOR_DATA) {
    LOG(WARNING)
        << "Robot state is not WAIT_FOR_DATA. Ignore new measurements!";
    return;
  }
  std::lock_guard<std::mutex> mLock(mMeasurementsMutex);
  mGraph->addMeasurement(factor);
}

void Agent::setMeasurements(
    const std::vector<RelativePosePoseMeasurement> &inputOdometry,
    const std::vector<RelativePosePoseMeasurement> &inputPrivateLoopClosures,
    const std::vector<RelativePosePoseMeasurement> &inputSharedLoopClosures) {
  std::vector<RelativePosePoseMeasurement> inputMeasurements = inputOdometry;
  inputMeasurements.insert(inputMeasurements.end(),
                           inputPrivateLoopClosures.begin(),
                           inputPrivateLoopClosures.end());
  inputMeasurements.insert(inputMeasurements.end(),
                           inputSharedLoopClosures.begin(),
                           inputSharedLoopClosures.end());
  setMeasurements(inputMeasurements);
}

void Agent::setMeasurements(
    const std::vector<RelativePosePoseMeasurement> &inputMeasurements) {
  CHECK(!isOptimizationRunning());
  CHECK_EQ(mState, AgentState::WAIT_FOR_DATA);
  CHECK(isPGOCompatible()) << "Error: Robot << " << getID()
                           << "must have a graph compatible with PGO when "
                              "using relative pose-pose measurements only!";

  mGraph = std::make_shared<Graph>(mID, r, d, GraphType::PoseGraph);
  mGraph->setMeasurements(inputMeasurements);
  CHECK_GT(num_poses(), 0)
      << "Error: Robot << " << getID()
      << "has no poses! Each agent must have at least one pose!";
}

void Agent::setMeasurements(const RelativeMeasurements &inputMeasurements) {
  CHECK(!isOptimizationRunning());
  CHECK_EQ(mState, AgentState::WAIT_FOR_DATA);
  CHECK(!isPGOCompatible()) << "Error: Robot << " << getID()
                            << "must have a graph compatible with RA-SLAM when "
                               "using RelativeMeasurements!";

  mGraph = std::make_shared<Graph>(mID, r, d, GraphType::RangeAidedSLAMGraph);
  mGraph->setMeasurements(inputMeasurements);
  CHECK_GT(num_poses(), 0)
      << "Error: Robot << " << getID()
      << "has no poses! Each agent must have at least one pose!";
}

void Agent::initialize(const PoseArray *TrajectoryInitPtr,
                       const PointArray *UnitSphereInitPtr,
                       const PointArray *LandmarkInitPtr) {
  if (mState != AgentState::WAIT_FOR_DATA)
    return;

  // Optimization loop should not be running
  endOptimizationLoop();

  // Do nothing if local graph is empty
  if (num_poses() == 0) {
    LOG_IF(INFO, mParams.verbose)
        << "Local graph has no poses. Skipping initialization.";
    return;
  }

  // Check conditions for map in RA-SLAM
  if (isAgentMap()) {
    CHECK_EQ(mGraph->n(), 0) << "Error: Map cannot contain poses!";
    CHECK_EQ(num_unit_spheres(), 0)
        << "Error: Map cannot contain unit spheres!";
    CHECK_GT(num_landmarks(), 0) << "Error: Map must contain landmarks!";
  }

  // Check validity of initial estimates with respect to graph compatibility
  if (UnitSphereInitPtr && isPGOCompatible())
    LOG(FATAL) << "Error: Unit sphere initialization requires the local graph "
                  "to be RA-SLAM compatible!";
  if (LandmarkInitPtr && isPGOCompatible())
    LOG(FATAL) << "Error: Landmark initialization requires the local graph "
                  "to be RA-SLAM compatible!";
  if (!TrajectoryInitPtr && (UnitSphereInitPtr || LandmarkInitPtr) &&
      !isPGOCompatible())
    LOG(FATAL)
        << "Error: When providing initial unit sphere or landmark estimates, "
           "an initial trajectory estimate must also be provided!";

  // Check validity of initial estimates with respect to dimensionality
  bool trajectory_initialization_successful = false;
  if (TrajectoryInitPtr && TrajectoryInitPtr->d() == dimension() &&
      TrajectoryInitPtr->n() == num_poses()) {
    LOG(INFO) << "Using provided trajectory initialization.";
    TrajectoryLocalInit.emplace(*TrajectoryInitPtr);
    trajectory_initialization_successful = true;
  }
  if (!isPGOCompatible()) {
    bool unit_sphere_initialization_successful = false;
    if (UnitSphereInitPtr && UnitSphereInitPtr->d() == dimension() &&
        UnitSphereInitPtr->n() == num_unit_spheres()) {
      LOG(INFO) << "Using provided unit sphere initialization.";
      UnitSphereLocalInit.emplace(*UnitSphereInitPtr);
      unit_sphere_initialization_successful = true;
    }

    bool landmark_initialization_successful = false;
    if (LandmarkInitPtr && LandmarkInitPtr->d() == dimension() &&
        LandmarkInitPtr->n() == num_landmarks()) {
      LOG(INFO) << "Using provided landmark initialization.";
      LandmarkLocalInit.emplace(*LandmarkInitPtr);
      landmark_initialization_successful = true;
    }

    // Randomly initialize unit sphere estimate
    if (!unit_sphere_initialization_successful) {
      LOG(INFO) << "Computing local unit sphere random initialization.";
      PointArray UnitSphereRand(dimension(), num_unit_spheres());
      Matrix M = DCORA::Matrix::Random(dimension(), num_unit_spheres());
      UnitSphereRand.setData(projectToObliqueManifold(M));
      UnitSphereLocalInit.emplace(UnitSphereRand);
      unit_sphere_initialization_successful = true;
    }
    if (!unit_sphere_initialization_successful) {
      LOG(WARNING) << "Robot " << getID()
                   << " fails to initialize local unit spheres!";
      return;
    }

    // Randomly initialize landmark estimate
    if (!landmark_initialization_successful) {
      LOG(INFO) << "Computing local landmark random initialization.";
      PointArray LandmarkRand(dimension(), num_landmarks());
      Matrix M = DCORA::Matrix::Random(dimension(), num_landmarks());
      LandmarkRand.setData(M);
      LandmarkLocalInit.emplace(LandmarkRand);
      landmark_initialization_successful = true;
    }
    if (!landmark_initialization_successful) {
      LOG(WARNING) << "Robot " << getID()
                   << " fails to initialize local landmarks!";
      return;
    }
  }

  // Initialize trajectory estimate in an arbitrary frame
  if (!trajectory_initialization_successful) {
    PoseArray T(dimension(), num_poses());
    if (!isAgentMap()) {
      switch (mParams.localInitializationMethod) {
      case (InitializationMethod::Odometry): {
        LOG(INFO) << "Computing local trajectory odometry initialization.";
        T = odometryInitialization(mGraph->odometry());
        break;
      }
      case (InitializationMethod::Chordal): {
        if (!isPGOCompatible())
          LOG(FATAL) << "Error: "
                     << InitializationMethodToString(
                            mParams.localInitializationMethod)
                     << "initialization requires the local graph "
                        "to be PGO compatible!";
        LOG(INFO) << "Computing local trajectory chordal initialization.";
        const RelativeMeasurements &m = mGraph->localMeasurements();
        T = chordalInitialization(m.GetRelativePosePoseMeasurements());
        break;
      }
      case (InitializationMethod::Random): {
        LOG(INFO) << "Computing local trajectory random initialization.";
        T.setRandomData();
        break;
      }
      case (InitializationMethod::GNC_TLS): {
        if (!isPGOCompatible())
          LOG(FATAL) << "Error: "
                     << InitializationMethodToString(
                            mParams.localInitializationMethod)
                     << "initialization requires the local graph "
                        "to be PGO compatible!";
        LOG(INFO) << "Computing local trajectory GNC_TLS initialization.";
        solveRobustPGOParams params;
        params.verbose = mParams.verbose;
        // Standard L2 PGO params (GNC inner iters)
        params.opt_params.verbose = false;
        params.opt_params.gradnorm_tol = 1;
        params.opt_params.RTR_iterations = 20;
        // Robust optimization params (GNC outer iters)
        params.robust_params.costType = RobustCostParameters::Type::GNC_TLS;
        params.robust_params.GNCMaxNumIters = 10;
        params.robust_params.GNCBarc = 5.0;
        params.robust_params.GNCMuStep = 1.4;
        PoseArray TOdom = odometryInitialization(mGraph->odometry());
        const RelativeMeasurements &m = mGraph->localMeasurements();
        std::vector<RelativePosePoseMeasurement> mutable_local_measurements =
            m.GetRelativePosePoseMeasurements();
        // Solve for trajectory
        T = solveRobustPGO(&mutable_local_measurements, params, &TOdom);
        // Reject outlier local loop closures
        int reject_count = 0;
        for (const auto &m : mutable_local_measurements) {
          if (m.weight < 1e-8) {
            setMeasurementWeight(m.getEdgeID(), 0);
            reject_count++;
          }
        }
        LOG(INFO) << "Reject " << reject_count << " local loop closures.";
        break;
      }
      }
    }
    CHECK_EQ(T.d(), dimension());
    CHECK_EQ(T.n(), num_poses());
    TrajectoryLocalInit.emplace(T);
    trajectory_initialization_successful = true;
  }
  if (!trajectory_initialization_successful) {
    LOG(WARNING) << "Robot " << getID()
                 << " fails to initialize local trajectory!";
    return;
  }

  // Transform the local trajectory so that the first pose is identity
  const Pose Tw0(TrajectoryLocalInit.value().pose(0));
  const PoseArray TrajectoryLocalTransformed =
      alignTrajectoryToFrame(TrajectoryLocalInit.value(), Tw0);
  TrajectoryLocalInit.emplace(TrajectoryLocalTransformed);

  // Transform the local unit spheres and landmarks so that they remain in the
  // same frame as the trajectory
  if (!isPGOCompatible()) {
    const PointArray UnitSpheresLocalTransformed =
        alignUnitSpheresToFrame(UnitSphereLocalInit.value(), Tw0);
    const PointArray LandmarksLocalTransformed =
        alignLandmarksToFrame(LandmarkLocalInit.value(), Tw0);
    UnitSphereLocalInit.emplace(UnitSpheresLocalTransformed);
    LandmarkLocalInit.emplace(LandmarksLocalTransformed);
  }

  // Update dimension for internal iterate
  X = LiftedRangeAidedArray(relaxation_rank(), dimension(), num_poses(),
                            num_unit_spheres(), num_landmarks(),
                            mParams.graphType);

  // Waiting for initialization in the GLOBAL frame
  mState = AgentState::WAIT_FOR_INITIALIZATION;

  // If this robot has ID zero or if cross-robot initialization is off
  // We can initialize iterate in the global frame
  if (mID == 0 || !mParams.multirobotInitialization)
    initializeInGlobalFrame(Pose(d));

  // Start optimization thread in asynchronous mode
  if (mParams.asynchronous)
    startOptimizationLoop();
}

void Agent::initializeInGlobalFrame(const Pose &T_world_robot) {
  CHECK(YLift);
  CHECK_EQ(T_world_robot.d(), dimension());
  checkRotationMatrix(T_world_robot.rotation());

  // Halt optimization
  bool optimizationHalted = false;
  if (isOptimizationRunning()) {
    LOG_IF(INFO, mParams.verbose)
        << "Robot " << getID() << " halting optimization thread...";
    optimizationHalted = true;
    endOptimizationLoop();
  }

  // Halt insertion of new states
  std::lock_guard<std::mutex> tLock(mStatesMutex);

  // Clear cache
  clearNeighborStates();

  // Apply global transformation to local trajectory estimate
  const PoseArray TrajectoryGlobalInit = alignTrajectoryToFrame(
      TrajectoryLocalInit.value(), T_world_robot.inverse());

  // Apply global transformation to local unit sphere and landmark estimates
  PointArray UnitSpheresGlobalInit(dimension(), num_unit_spheres());
  PointArray LandmarksGlobalInit(dimension(), num_landmarks());
  if (!isPGOCompatible()) {
    UnitSpheresGlobalInit = alignUnitSpheresToFrame(UnitSphereLocalInit.value(),
                                                    T_world_robot.inverse());
    LandmarksGlobalInit = alignLandmarksToFrame(LandmarkLocalInit.value(),
                                                T_world_robot.inverse());
  }

  // Set initial global estimate
  RangeAidedArray GlobalInit(dimension(), num_poses(), num_unit_spheres(),
                             num_landmarks(), mParams.graphType);
  GlobalInit.setLiftedPoseArray(TrajectoryGlobalInit);
  GlobalInit.setLiftedUnitSphereArray(UnitSpheresGlobalInit);
  GlobalInit.setLiftedLandmarkArray(LandmarksGlobalInit);

  // Lift back to correct relaxation rank
  X.setData(YLift.value() * GlobalInit.getData());
  XInit.emplace(X);

  // Change state for this agent
  if (mState == AgentState::INITIALIZED) {
    LOG(INFO) << "Robot " << getID() << " re-initializes in global frame!";
  } else {
    LOG(INFO) << "Robot " << getID() << " initializes in global frame!";
    mState = AgentState::INITIALIZED;
  }

  // When doing robust optimization,
  // initialize all active and non-fixed edge weights to 1.0
  if (mParams.robustCostParams.costType != RobustCostParameters::Type::L2)
    initializeRobustOptimization();

  // Initialize auxiliary variables
  if (mParams.acceleration)
    initializeAcceleration();

  // Log initial trajectory
  if (mParams.logData && !isAgentMap()) {
    std::string filename = "dcora_" +
                           std::string(1, FIRST_AGENT_SYMBOL + getID()) +
                           "_initial.txt";
    mLogger.logTrajectory(dimension(), num_poses(),
                          TrajectoryGlobalInit.getData(), filename);
  }

  if (optimizationHalted)
    startOptimizationLoop();
}

bool Agent::iterate(bool doOptimization) {
  mIterationNumber++;
  if (mParams.robustCostParams.costType != RobustCostParameters::Type::L2)
    mRobustOptInnerIter++;

  // Perform iteration
  if (mState == AgentState::INITIALIZED && !isAgentMap()) {
    // Save current iterate
    XPrev = X;
    bool success;
    if (mParams.acceleration) {
      updateGamma();
      updateAlpha();
      updateY();
      success = updateX(doOptimization, true);
      updateV();
      // Check restart condition
      if (shouldRestart())
        restartNesterovAcceleration(doOptimization);
    } else {
      success = updateX(doOptimization, false);
    }

    // Update status after local optimization step
    if (doOptimization) {
      mStatus.agentID = getID();
      mStatus.state = mState;
      mStatus.instanceNumber = instance_number();
      mStatus.iterationNumber = iteration_number();
      // TODO(AT): ask AP and DR about setting the relative change condition in
      // RA-SLAM
      mStatus.relativeChange = LiftedArray::maxTranslationDistance(
          *X.GetLiftedPoseArray(), *XPrev.GetLiftedPoseArray());
      // Check local termination condition
      bool readyToTerminate = true;
      if (!success)
        readyToTerminate = false;
      double relative_change_tol = mParams.relChangeTol;
      // Use loose threshold during initial inner iters of robust opt
      if (mParams.robustCostParams.costType != RobustCostParameters::Type::L2 &&
          mWeightUpdateCount == 0) {
        relative_change_tol = 5;
      }
      if (mStatus.relativeChange > relative_change_tol)
        readyToTerminate = false;
      // Compute percentage of converged loop closures (i.e., either accepted or
      // rejected)
      const auto stat = mGraph->statistics();
      double ratio = (stat.accept_loop_closures + stat.reject_loop_closures) /
                     stat.total_loop_closures;
      if (ratio < mParams.robustOptMinConvergenceRatio)
        readyToTerminate = false;
      mStatus.readyToTerminate = readyToTerminate;
    }

    // Request to publish public states
    if (doOptimization || mParams.acceleration)
      mPublishPublicStatesRequested = true;

    mPublishAsynchronousRequested = true;
    return success;
  }
  return true;
}

void Agent::reset() {
  // Terminate optimization thread if running
  endOptimizationLoop();

  if (mParams.logData) {
    // Save measurements
    RelativeMeasurements m = mGraph->allMeasurements();
    mLogger.logMeasurements(m, "measurements.txt");

    // Save solution before rounding
    writeMatrixToFile(X.getData(), mParams.logDirectory + "X.txt");

    // Save trajectory estimates after rounding
    if (!isAgentMap()) {
      Matrix T;
      if (getTrajectoryInGlobalFrame(&T)) {
        std::string filename =
            "dcora_" + std::string(1, FIRST_AGENT_SYMBOL + getID()) + ".txt";
        mLogger.logTrajectory(dimension(), num_poses(), T, filename);
        LOG(INFO) << "Saved optimized trajectory to " << mParams.logDirectory;
      } else {
        LOG(WARNING) << "Global anchor not set for agent " << getID()
                     << ". Optimized trajectory not saved.";
      }
    }
  }

  mInstanceNumber++;
  mIterationNumber = 0;
  mLatestWeightUpdateIteration = 0;
  mRobustOptInnerIter = 0;
  mWeightUpdateCount = 0;
  mTrajectoryResetCount = 0;
  mState = AgentState::WAIT_FOR_DATA;
  mStatus =
      AgentStatus(getID(), mState, mInstanceNumber, mIterationNumber, false, 0);
  mTeamStatus.clear();
  for (const auto &robot_id : mParams.robotIDs)
    mTeamRobotActive.at(robot_id) = false;
  globalAnchor.reset();
  TrajectoryLocalInit.reset();
  UnitSphereLocalInit.reset();
  LandmarkLocalInit.reset();
  XInit.reset();
  mPublishPublicStatesRequested = false;
  mPublishAsynchronousRequested = false;

  // This function will activate all robots in graph again
  mGraph->reset();
  clearNeighborStates();
}

void Agent::startOptimizationLoop() {
  // Asynchronous updates currently restricted to non-accelerated updates
  CHECK(!mParams.acceleration)
      << "Asynchronous mode does not support acceleration!";
  if (isOptimizationRunning())
    return;

  LOG_IF(INFO, mParams.verbose)
      << "Robot " << getID() << " spins optimization thread at "
      << mParams.asynchronousOptimizationRate << " Hz.";
  mOptimizationThread =
      std::make_unique<std::thread>(&Agent::runOptimizationLoop, this);
}

void Agent::runOptimizationLoop() {
  // Create exponential distribution with the desired rate
  std::random_device
      rd; // Will be used to obtain a seed for the random number engine
  std::mt19937 rng(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::exponential_distribution<double> ExponentialDistribution(
      mParams.asynchronousOptimizationRate);
  while (true) {
    iterate(true);
    usleep(1e6 * ExponentialDistribution(rng));
    // Check if finish requested
    if (mEndLoopRequested)
      break;
  }
}

void Agent::endOptimizationLoop() {
  if (!isOptimizationRunning())
    return;
  mEndLoopRequested = true;
  // wait for thread to finish
  mOptimizationThread->join();
  mOptimizationThread.reset(nullptr);
  mEndLoopRequested = false; // reset request flag
  LOG_IF(INFO, mParams.verbose)
      << "Robot " << getID() << " optimization thread exits.";
}

bool Agent::isOptimizationRunning() { return mOptimizationThread != nullptr; }

Pose Agent::computeNeighborTransform(
    const RelativePosePoseMeasurement &measurement,
    const LiftedPose &neighbor_pose) {
  CHECK(YLift);
  CHECK_EQ(neighbor_pose.r(), r);
  CHECK_EQ(neighbor_pose.d(), d);

  // Notations:
  // world1: world frame before alignment
  // world2: world frame after alignment
  // frame1 : frame associated to my public pose
  // frame2 : frame associated to neighbor's public pose
  auto dT = Pose::Identity(d);
  dT.rotation() = measurement.R;
  dT.translation() = measurement.t;

  auto T_world2_frame2 = Pose::Identity(d);
  T_world2_frame2.pose() = YLift.value().transpose() * neighbor_pose.pose();

  auto T = TrajectoryLocalInit.value();
  auto T_frame1_frame2 = Pose::Identity(d);
  auto T_world1_frame1 = Pose::Identity(d);
  if (measurement.r2 == getID()) {
    // Incoming edge
    T_frame1_frame2 = dT.inverse();
    T_world1_frame1.setData(T.pose(measurement.p2));
  } else {
    // Outgoing edge
    T_frame1_frame2 = dT;
    T_world1_frame1.setData(T.pose(measurement.p1));
  }
  auto T_world2_frame1 = T_world2_frame2 * T_frame1_frame2.inverse();
  auto T_world2_world1 = T_world2_frame1 * T_world1_frame1.inverse();
  checkRotationMatrix(T_world2_world1.rotation());
  return T_world2_world1;
}

bool Agent::computeRobustNeighborTransformTwoStage(unsigned int neighborID,
                                                   const PoseDict &poseDict,
                                                   Pose *T_world_robot) {
  std::vector<Matrix> RVec;
  std::vector<Vector> tVec;
  // Populate candidate alignments
  // Each alignment corresponds to a single inter-robot loop closure
  const RelativeMeasurements measurements =
      mGraph->sharedLoopClosuresWithRobot(neighborID);
  for (const auto &m : measurements.GetRelativePosePoseMeasurements()) {
    PoseID nbr_pose_id;
    nbr_pose_id.robot_id = neighborID;
    if (m.r1 == neighborID)
      nbr_pose_id.frame_id = m.p1;
    else
      nbr_pose_id.frame_id = m.p2;
    const auto &it = poseDict.find(nbr_pose_id);
    if (it != poseDict.end()) {
      const auto T = computeNeighborTransform(m, it->second);
      RVec.emplace_back(T.rotation());
      tVec.emplace_back(T.translation());
    }
  }
  if (RVec.empty())
    return false;
  int m = static_cast<int>(RVec.size());
  const Vector kappa = Vector::Ones(m);
  const Vector tau = Vector::Ones(m);
  Matrix ROpt;
  Vector tOpt;
  std::vector<size_t> inlierIndices;

  // Perform robust single rotation averaging
  double maxRotationError = angular2ChordalSO3(0.5); // approximately 30 deg
  robustSingleRotationAveraging(&ROpt, &inlierIndices, RVec, kappa,
                                maxRotationError);
  int inlierSize = static_cast<int>(inlierIndices.size());
  printf("Robot %u attempts initialization from neighbor %u: finds %i/%i "
         "inliers.\n",
         getID(), neighborID, inlierSize, m);

  // Return if robust rotation averaging fails to find any inlier
  if (inlierIndices.size() < mParams.robustInitMinInliers)
    return false;

  // Perform single translation averaging on the inlier set
  std::vector<Vector> tVecInliers;
  for (const auto index : inlierIndices) {
    tVecInliers.emplace_back(tVec[index]);
  }
  singleTranslationAveraging(&tOpt, tVecInliers);

  // Return transformation as matrix
  CHECK_NOTNULL(T_world_robot);
  CHECK_EQ(T_world_robot->d(), dimension());
  T_world_robot->rotation() = ROpt;
  T_world_robot->translation() = tOpt;
  return true;
}

bool Agent::computeRobustNeighborTransform(unsigned int neighborID,
                                           const PoseDict &poseDict,
                                           Pose *T_world_robot) {
  std::vector<Matrix> RVec;
  std::vector<Vector> tVec;
  // Populate candidate alignments
  // Each alignment corresponds to a single inter-robot loop closure
  const RelativeMeasurements measurements =
      mGraph->sharedLoopClosuresWithRobot(neighborID);
  for (const auto &m : measurements.GetRelativePosePoseMeasurements()) {
    PoseID nbr_pose_id;
    nbr_pose_id.robot_id = neighborID;
    if (m.r1 == neighborID)
      nbr_pose_id.frame_id = m.p1;
    else
      nbr_pose_id.frame_id = m.p2;
    const auto &it = poseDict.find(nbr_pose_id);
    if (it != poseDict.end()) {
      const auto T = computeNeighborTransform(m, it->second);
      RVec.emplace_back(T.rotation());
      tVec.emplace_back(T.translation());
    }
  }
  if (RVec.empty())
    return false;
  // Perform robust single pose averaging
  int m = static_cast<int>(RVec.size());
  const Vector kappa =
      1.82 * Vector::Ones(m); // rotation stddev approximately 30 degree
  const Vector tau = 0.01 * Vector::Ones(m); // translation stddev 10 m
  const double cbar = RobustCost::computeErrorThresholdAtQuantile(0.9, 3);
  Matrix ROpt;
  Vector tOpt;
  std::vector<size_t> inlierIndices;
  robustSinglePoseAveraging(&ROpt, &tOpt, &inlierIndices, RVec, tVec, kappa,
                            tau, cbar);
  int inlierSize = static_cast<int>(inlierIndices.size());
  printf("Robot %u attempts initialization from neighbor %u: finds %i/%i "
         "inliers.\n",
         getID(), neighborID, inlierSize, m);

  // Return if fails to identify any inlier
  if (inlierIndices.size() < mParams.robustInitMinInliers)
    return false;

  // Return transformation as matrix
  CHECK_NOTNULL(T_world_robot);
  CHECK_EQ(T_world_robot->d(), dimension());
  T_world_robot->rotation() = ROpt;
  T_world_robot->translation() = tOpt;
  return true;
}

void Agent::updateNeighborStates(unsigned neighborID, const PoseDict &poseDict,
                                 bool areNeighborStatesAux,
                                 const UnitSphereDict &unitSphereDict,
                                 const LandmarkDict &landmarkDict) {
  CHECK(neighborID != mID);
  if (isPGOCompatible())
    CHECK(unitSphereDict.empty() && landmarkDict.empty());
  if (!YLift)
    return;
  if (!hasNeighborStatus(neighborID))
    return;
  if (getNeighborStatus(neighborID).state != AgentState::INITIALIZED)
    return;
  if (mState == AgentState::WAIT_FOR_INITIALIZATION) {
    Pose T_world_robot(dimension());
    if (computeRobustNeighborTransformTwoStage(neighborID, poseDict,
                                               &T_world_robot)) {
      initializeInGlobalFrame(T_world_robot);
    }
  }
  if (mState != AgentState::INITIALIZED)
    return;

  // Lambda function for validating dictionary entries
  auto validateDictionaryEntry = [&](const auto &nID, const auto &var) {
    CHECK_EQ(nID.robot_id, neighborID);
    CHECK_EQ(var.r(), r);
    CHECK_EQ(var.d(), d);
  };

  // Save neighbor public states in local cache
  std::lock_guard<std::mutex> lock(mNeighborStatesMutex);
  for (const auto &it : poseDict) {
    const PoseID &nID = it.first;
    const LiftedPose &var = it.second;
    validateDictionaryEntry(nID, var);
    if (!mGraph->requireNeighborPose(nID))
      continue;
    if (areNeighborStatesAux)
      neighborAuxPoseDict[nID] = var;
    else
      neighborPoseDict[nID] = var;
  }
  for (const auto &it : unitSphereDict) {
    const UnitSphereID &nID = it.first;
    const LiftedPoint &var = it.second;
    validateDictionaryEntry(nID, var);
    if (!mGraph->requireNeighborUnitSphere(nID))
      continue;
    if (areNeighborStatesAux)
      neighborAuxUnitSphereDict[nID] = var;
    else
      neighborUnitSphereDict[nID] = var;
  }
  for (const auto &it : landmarkDict) {
    const LandmarkID &nID = it.first;
    const LiftedPoint &var = it.second;
    validateDictionaryEntry(nID, var);
    if (!mGraph->requireNeighborLandmark(nID))
      continue;
    if (areNeighborStatesAux)
      neighborAuxLandmarkDict[nID] = var;
    else
      neighborLandmarkDict[nID] = var;
  }
}

void Agent::clearNeighborStates() {
  std::lock_guard<std::mutex> lock(mNeighborStatesMutex);
  neighborPoseDict.clear();
  neighborUnitSphereDict.clear();
  neighborLandmarkDict.clear();
  neighborAuxPoseDict.clear();
  neighborAuxUnitSphereDict.clear();
  neighborAuxLandmarkDict.clear();
}

void Agent::clearActiveNeighborStates() {
  std::lock_guard<std::mutex> lock(mNeighborStatesMutex);
  for (const auto &pose_id : mGraph->activeNeighborPublicPoseIDs()) {
    neighborPoseDict.erase(pose_id);
    neighborAuxPoseDict.erase(pose_id);
  }
  for (const auto &unit_sphere_id :
       mGraph->activeNeighborPublicUnitSphereIDs()) {
    neighborUnitSphereDict.erase(unit_sphere_id);
    neighborAuxUnitSphereDict.erase(unit_sphere_id);
  }
  for (const auto &landmark_id : mGraph->activeNeighborPublicLandmarkIDs()) {
    neighborLandmarkDict.erase(landmark_id);
    neighborAuxLandmarkDict.erase(landmark_id);
  }
}

bool Agent::getTrajectoryInLocalFrame(Matrix *Trajectory) {
  if (mState != AgentState::INITIALIZED)
    return false;
  std::lock_guard<std::mutex> lock(mStatesMutex);

  // Get estimated lifted trajectory
  const Matrix &TrajectoryLifted = X.GetLiftedPoseArray()->getData();

  // Align lifted trajectory to its first pose
  const LiftedPose Tw0(X.pose(0));
  const PoseArray T = alignLiftedTrajectoryToFrame(
      TrajectoryLifted, Tw0, dimension(), num_poses(), false);

  *Trajectory = T.getData();
  return true;
}

bool Agent::getStatesInLocalFrame(Matrix *Trajectory, Matrix *UnitSpheres,
                                  Matrix *Landmarks) {
  if (mState != AgentState::INITIALIZED)
    return false;
  std::lock_guard<std::mutex> lock(mStatesMutex);

  // Get estimated lifted state estimates
  const Matrix &TrajectoryLifted = X.GetLiftedPoseArray()->getData();
  const Matrix &UnitSpheresLifted = X.GetLiftedUnitSphereArray()->getData();
  const Matrix &LandmarksLifted = X.GetLiftedLandmarkArray()->getData();

  // Get first pose of trajectory
  const LiftedPose Tw0(X.pose(0));
  const Matrix &R0T = Tw0.rotation().transpose();

  // Rotate trajectory to local frame
  PoseArray liftedTrajectoryTransformed(dimension(), num_poses());
  liftedTrajectoryTransformed.setData(R0T * TrajectoryLifted);
  const Vector &t0 = liftedTrajectoryTransformed.translation(0);

  // Project each rotation block to the rotation group, and make the first
  // translation zero
  for (unsigned int i = 0; i < num_poses(); ++i) {
    liftedTrajectoryTransformed.rotation(i) =
        projectToRotationGroup(liftedTrajectoryTransformed.rotation(i));
    liftedTrajectoryTransformed.translation(i) =
        liftedTrajectoryTransformed.translation(i) - t0;
  }

  // Rotate unit spheres to local frame. As unit sphere variables do not depend
  // on the translation of the frame, we simply apply a rotation.
  PointArray liftedUnitSpheresTransformed(dimension(), num_unit_spheres());
  liftedUnitSpheresTransformed.setData(R0T * UnitSpheresLifted);

  // Rotate landmarks to local frame
  PointArray liftedLandmarksTransformed(dimension(), num_landmarks());
  liftedLandmarksTransformed.setData(R0T * LandmarksLifted);

  // Translate landmarks into the local frame set by the trajectory
  for (unsigned int i = 0; i < num_landmarks(); ++i) {
    liftedLandmarksTransformed.translation(i) =
        liftedLandmarksTransformed.translation(i) - t0;
  }

  // Set state estimates in local frame
  *Trajectory = liftedTrajectoryTransformed.getData();
  if (UnitSpheres)
    *UnitSpheres = liftedUnitSpheresTransformed.getData();
  if (Landmarks)
    *Landmarks = liftedLandmarksTransformed.getData();

  return true;
}

bool Agent::getTrajectoryInGlobalFrame(Matrix *Trajectory) {
  PoseArray T(d, num_poses());
  if (!getTrajectoryInGlobalFrame(&T))
    return false;
  *Trajectory = T.getData();
  return true;
}

bool Agent::getTrajectoryInGlobalFrame(PoseArray *Trajectory) {
  if (!globalAnchor)
    return false;
  auto Xa = globalAnchor.value();
  CHECK(Xa.r() == relaxation_rank());
  CHECK(Xa.d() == dimension());
  if (mState != AgentState::INITIALIZED)
    return false;
  std::lock_guard<std::mutex> lock(mStatesMutex);

  // Get estimated lifted trajectory
  const Matrix &TrajectoryLifted = X.GetLiftedPoseArray()->getData();

  // Align lifted trajectory to globl anchor
  const PoseArray T = alignLiftedTrajectoryToFrame(
      TrajectoryLifted, Xa, dimension(), num_poses(), true);

  *Trajectory = T;
  return true;
}

bool Agent::getPoseInGlobalFrame(unsigned int poseID, Matrix *T) {
  if (!globalAnchor)
    return false;
  auto Xa = globalAnchor.value();
  CHECK(Xa.r() == relaxation_rank());
  CHECK(Xa.d() == dimension());
  if (mState != AgentState::INITIALIZED)
    return false;
  std::lock_guard<std::mutex> lock(mStatesMutex);
  if (poseID < 0 || poseID >= num_poses())
    return false;
  Matrix Ya = Xa.rotation();
  Matrix pa = Xa.translation();
  Matrix t0 = Ya.transpose() * pa;
  Matrix Xi = X.pose(poseID);
  Matrix Ti = Ya.transpose() * Xi;
  Ti.col(d) -= t0;
  CHECK(Ti.rows() == d);
  CHECK(Ti.cols() == d + 1);
  *T = Ti;
  return true;
}

bool Agent::getNeighborPoseInGlobalFrame(unsigned int neighborID,
                                         unsigned int poseID, Matrix *T) {
  if (!globalAnchor)
    return false;
  auto Xa = globalAnchor.value();
  CHECK(Xa.r() == relaxation_rank());
  CHECK(Xa.d() == dimension());
  if (mState != AgentState::INITIALIZED)
    return false;
  std::lock_guard<std::mutex> lock(mNeighborStatesMutex);
  PoseID nID(neighborID, poseID);
  if (neighborPoseDict.find(nID) != neighborPoseDict.end()) {
    Matrix Ya = Xa.rotation();
    Matrix pa = Xa.translation();
    Matrix t0 = Ya.transpose() * pa;
    auto Xi = neighborPoseDict.at(nID);
    CHECK(Xi.r() == r);
    CHECK(Xi.d() == d);
    Pose Ti(Ya.transpose() * Xi.pose());
    Ti.translation() -= t0;
    *T = Ti.pose();
    return true;
  }
  return false;
}

bool Agent::hasNeighbor(unsigned neighborID) const {
  return mGraph->hasNeighbor(neighborID);
}

std::vector<unsigned> Agent::getNeighbors() const {
  auto neighborRobotIDs = mGraph->neighborIDs();
  std::vector<unsigned> v(neighborRobotIDs.size());
  std::copy(neighborRobotIDs.begin(), neighborRobotIDs.end(), v.begin());
  return v;
}

Matrix Agent::localPoseGraphOptimization() {
  if (!isPGOCompatible())
    LOG(FATAL)
        << "Local PGO requires all measurements to be relative pose-pose "
           "measurements!";
  ROptParameters pgo_params;
  pgo_params.verbose = true;
  const RelativeMeasurements &m = mGraph->localMeasurements();
  const auto T = solvePGO(m.GetRelativePosePoseMeasurements(), pgo_params);
  return T.getData();
}

bool Agent::getLiftingMatrix(Matrix *M) const {
  if (YLift.has_value()) {
    *M = YLift.value();
    return true;
  }
  return false;
}

void Agent::setGlobalAnchor(const Matrix &M) {
  CHECK(M.rows() == relaxation_rank());
  CHECK(M.cols() == dimension() + 1);
  LiftedPose Xa(r, d);
  Xa.pose() = M;
  globalAnchor.emplace(Xa);
}

bool Agent::shouldTerminate() {
  // terminate if reached maximum iterations
  if (iteration_number() >= mParams.maxNumIters) {
    LOG(INFO) << "Reached maximum iterations.";
    return true;
  }

  // Do not terminate if not update measurement weights for sufficiently many
  // times
  if (mParams.robustCostParams.costType != RobustCostParameters::Type::L2) {
    if (mWeightUpdateCount < mParams.robustOptNumWeightUpdates)
      return false;
  }

  // terminate only if all robots meet conditions
  for (const auto &robot_id : mParams.robotIDs) {
    if (!isRobotActive(robot_id))
      continue;
    const auto &it = mTeamStatus.find(robot_id);
    // ready false if robot status not available
    if (it == mTeamStatus.end())
      return false;
    const auto &robot_status = it->second;
    CHECK_EQ(robot_status.agentID, robot_id);
    // return false if this robot is not initialized
    if (robot_status.state != AgentState::INITIALIZED)
      return false;
    // return false if this robot is not ready to terminate
    if (!robot_status.readyToTerminate)
      return false;
  }

  return true;
}

bool Agent::shouldRestart() const {
  if (mParams.acceleration) {
    return ((mIterationNumber + 1) % mParams.restartInterval == 0);
  }
  return false;
}

void Agent::restartNesterovAcceleration(bool doOptimization) {
  if (mParams.acceleration && mState == AgentState::INITIALIZED) {
    LOG_IF(INFO, mParams.verbose)
        << "Robot " << getID() << " restarts acceleration.";
    X = XPrev;
    updateX(doOptimization, false);
    V = X;
    Y = X;
    gamma = 0;
    alpha = 0;
  }
}

void Agent::initializeAcceleration() {
  CHECK(mParams.acceleration);
  if (mState == AgentState::INITIALIZED) {
    XPrev = X;
    gamma = 0;
    alpha = 0;
    V = X;
    Y = X;
  }
}

void Agent::updateGamma() {
  CHECK(mParams.acceleration);
  CHECK(mState == AgentState::INITIALIZED);
  gamma = (1 + sqrt(1 + 4 * pow(mParams.numRobots, 2) * pow(gamma, 2))) /
          (2 * mParams.numRobots);
}

void Agent::updateAlpha() {
  CHECK(mParams.acceleration);
  CHECK(mState == AgentState::INITIALIZED);
  alpha = 1 / (gamma * mParams.numRobots);
}

void Agent::updateY() {
  CHECK(mParams.acceleration);
  CHECK(mState == AgentState::INITIALIZED);
  const Matrix M = (1 - alpha) * X.getData() + alpha * V.getData();
  Y.setData(projectToManifold(M));
}

void Agent::updateV() {
  CHECK(mParams.acceleration);
  CHECK(mState == AgentState::INITIALIZED);
  const Matrix M = V.getData() + gamma * (X.getData() - Y.getData());
  V.setData(projectToManifold(M));
}

bool Agent::updateX(bool doOptimization, bool acceleration) {
  // Lock during local optimization
  std::unique_lock<std::mutex> tLock(mStatesMutex);
  std::unique_lock<std::mutex> mLock(mMeasurementsMutex);
  std::unique_lock<std::mutex> nLock(mNeighborStatesMutex);
  if (!doOptimization) {
    if (acceleration) {
      X = Y;
    }
    return true;
  }
  LOG_IF(INFO, mParams.verbose)
      << "Robot " << getID() << " optimizes at iteration "
      << iteration_number();
  if (acceleration)
    CHECK(mParams.acceleration);
  CHECK(mState == AgentState::INITIALIZED);

  // Initialize graph for optimization
  if (acceleration) {
    mGraph->setNeighborStates(neighborAuxPoseDict, neighborAuxUnitSphereDict,
                              neighborAuxLandmarkDict);
  } else {
    mGraph->setNeighborStates(neighborPoseDict, neighborUnitSphereDict,
                              neighborLandmarkDict);
  }

  // Skip optimization if cannot construct data matrices for some reason
  if (!mGraph->constructDataMatrices()) {
    LOG(WARNING) << "Robot " << getID()
                 << " cannot construct data matrices... Skip optimization.";
    mLocalOptResult = ROPTResult(false);
    return false;
  }

  // Initialize optimizer
  QuadraticProblem problem(mGraph);
  QuadraticOptimizer optimizer(&problem, mParams.localOptimizationParams);
  optimizer.setVerbose(mParams.verbose);

  // Starting solution
  Matrix X0;
  if (acceleration) {
    X0 = Y.getData();
  } else {
    X0 = X.getData();
  }
  CHECK(X0.rows() == relaxation_rank());
  CHECK(X0.cols() == problem_dimension());

  // Optimize!
  X.setData(optimizer.optimize(X0));

  // Print optimization statistics
  mLocalOptResult = optimizer.getOptResult();
  if (mParams.verbose) {
    printf("df: %f, init_gradnorm: %f, opt_gradnorm: %f. \n",
           mLocalOptResult.fInit - mLocalOptResult.fOpt,
           mLocalOptResult.gradNormInit, mLocalOptResult.gradNormOpt);
  }

  return true;
}

bool Agent::shouldUpdateMeasurementWeights() const {
  // No need to update weight if using L2 cost
  if (mParams.robustCostParams.costType == RobustCostParameters::Type::L2)
    return false;

  if (mWeightUpdateCount >= mParams.robustOptNumWeightUpdates) {
    LOG_IF(INFO, mParams.verbose) << "Reached maximum weight update steps.";
    return false;
  }

  // Return true if number of inner iterations exceeds threshold
  if (mRobustOptInnerIter >= mParams.robustOptInnerIters) {
    LOG_IF(INFO, mParams.verbose)
        << "Exceeds max inner iterations. Update weights.";
    return true;
  }

  // Only update if all agents sufficiently converged
  bool should_update = true;
  for (const auto &robot_id : mParams.robotIDs) {
    if (!isRobotActive(robot_id))
      continue;
    const auto &it = mTeamStatus.find(robot_id);
    if (it == mTeamStatus.end()) {
      should_update = false;
      break;
    }
    const auto &robot_status = it->second;
    CHECK_EQ(robot_status.agentID, robot_id);
    // return false if robot status is outdated
    if (robot_status.iterationNumber < mLatestWeightUpdateIteration) {
      should_update = false;
      break;
    }
    if (robot_status.state != AgentState::INITIALIZED) {
      should_update = false;
      break;
    }
    // return false if this robot is not ready to terminate
    if (!robot_status.readyToTerminate) {
      should_update = false;
      break;
    }
  }

  if (should_update) {
    LOG_IF(INFO, mParams.verbose) << "Ready to update weights.";
  }

  return should_update;
}

void Agent::initializeRobustOptimization() {
  if (mParams.robustCostParams.costType == RobustCostParameters::Type::L2) {
    LOG(WARNING) << "Using standard least squares cost function and shouldn't "
                    "need to initialize measurement weights!";
  }
  mRobustCost.reset();
  std::unique_lock<std::mutex> lock(mMeasurementsMutex);
  for (auto &m : mGraph->activeLoopClosures()) {
    std::visit(
        [](auto &&m) {
          if (!m->fixedWeight)
            m->weight = 1.0;
        },
        m);
  }
}

bool Agent::computeMeasurementResidual(const RelativeMeasurement &measurement,
                                       double *residual) const {
  if (measurement.measurementType != MeasurementType::PosePose)
    LOG(FATAL) << "Error: computeMeasurementResidual only supports "
                  "relative pose-pose measurements!";
  if (mState != AgentState::INITIALIZED) {
    return false;
  }
  CHECK_NOTNULL(residual);
  Matrix Y1, Y2, p1, p2;
  // Evaluate private loop closure
  if (measurement.r1 == measurement.r2) {
    Y1 = X.rotation(measurement.p1);
    p1 = X.translation(measurement.p1);
    Y2 = X.rotation(measurement.p2);
    p2 = X.translation(measurement.p2);
  } else {
    // Evaluate shared (inter-robot) loop closure
    if (measurement.r1 == getID()) {
      Y1 = X.rotation(measurement.p1);
      p1 = X.translation(measurement.p1);
      const PoseID nbrPoseID(measurement.r2, measurement.p2);
      auto KVpair = neighborPoseDict.find(nbrPoseID);
      if (KVpair == neighborPoseDict.end()) {
        return false;
      }
      Y2 = KVpair->second.rotation();
      p2 = KVpair->second.translation();
    } else {
      Y2 = X.rotation(measurement.p2);
      p2 = X.translation(measurement.p2);
      const PoseID nbrPoseID(measurement.r1, measurement.p1);
      auto KVpair = neighborPoseDict.find(nbrPoseID);
      if (KVpair == neighborPoseDict.end()) {
        return false;
      }
      Y1 = KVpair->second.rotation();
      p1 = KVpair->second.translation();
    }
  }
  // Dynamically cast to relative pose-pose measurement
  const RelativePosePoseMeasurement &pose_pose_measurement =
      dynamic_cast<const RelativePosePoseMeasurement &>(measurement);
  *residual =
      std::sqrt(computeMeasurementError(pose_pose_measurement, Y1, p1, Y2, p2));
  return true;
}

void Agent::updateMeasurementWeights() {
  if (mState != AgentState::INITIALIZED) {
    LOG(WARNING) << "Robot " << getID()
                 << " attempts to update weights but is not initialized.";
    return;
  }
  std::unique_lock<std::mutex> lock(mMeasurementsMutex);
  double residual = 0;
  for (auto &m : mGraph->activeLoopClosures()) {
    std::visit(
        [&residual, this](auto &&m) {
          if (!m->fixedWeight) {
            if (computeMeasurementResidual(*m, &residual))
              m->weight = mRobustCost.weight(residual);
            else
              LOG(WARNING) << "Failed to update weight for edge: \n" << *m;
          }
        },
        m);
  }
  mWeightUpdateCount++;
  mLatestWeightUpdateIteration = iteration_number();
  mRobustOptInnerIter = 0;
  mGraph->clearDataMatrices();
  mRobustCost.update();
  mTeamStatus.clear();
  mStatus.readyToTerminate = false;
  mStatus.relativeChange = 0;

  // Reset trajectory estimate to initial guess
  // after the first round of GNC variable update
  // or if warm start is disabled
  if (mTrajectoryResetCount < mParams.robustOptNumResets) {
    mTrajectoryResetCount++;
    LOG(INFO) << "Robot " << getID()
              << " resets trajectory estimates after weight updates.";
    setXToInitialGuess();
    clearNeighborStates();
  }

  // Reset acceleration
  if (mParams.acceleration) {
    initializeAcceleration();
  }
}

bool Agent::setMeasurementWeight(const EdgeID &edgeID, double weight,
                                 bool fixed_weight) {
  RelativeMeasurement *m = mGraph->findMeasurement(edgeID);
  if (m) {
    std::unique_lock<std::mutex> lock(mMeasurementsMutex);
    m->weight = weight;
    m->fixedWeight = fixed_weight;
    return true;
  }
  LOG(WARNING) << "[setMeasurementWeight] Measurement does not exist!";
  return false;
}

bool Agent::isRobotInitialized(unsigned robot_id) const {
  if (robot_id == getID())
    return mState == AgentState::INITIALIZED;

  if (!hasNeighborStatus(robot_id))
    return false;

  return getNeighborStatus(robot_id).state == AgentState::INITIALIZED;
}

bool Agent::isRobotActive(unsigned robot_id) const {
  if (robot_id >= mParams.numRobots)
    return false;
  return mTeamRobotActive.at(robot_id);
}

void Agent::setRobotActive(unsigned robot_id, bool active) {
  if (robot_id >= mParams.numRobots)
    LOG(FATAL) << "Input robot ID " << robot_id
               << " bigger than number of robots!";
  if (isAgentMap(robot_id))
    return;

  mTeamRobotActive.at(robot_id) = active;
  // If this robot is a neighbor (and not the MAP in RA-SLAM), activate or
  // deactivate corresponding measurements
  if (mGraph->hasNeighbor(robot_id) && !isAgentMap(robot_id))
    mGraph->setNeighborActive(robot_id, active);
}

size_t Agent::numActiveRobots() const {
  size_t num_active = 0;
  for (const auto &robot_id : mParams.robotIDs) {
    if (isRobotActive(robot_id))
      num_active++;
  }
  return num_active;
}

bool Agent::anchorFirstPose() {
  if (num_poses() > 0) {
    LiftedPose prior(relaxation_rank(), dimension());
    prior.setData(X.pose(0));
    mGraph->setPrior(0, prior);
  } else {
    return false;
  }
  return true;
}

bool Agent::anchorFirstPose(const LiftedPose &prior) {
  CHECK_EQ(prior.d(), dimension());
  CHECK_EQ(prior.r(), relaxation_rank());
  mGraph->setPrior(0, prior);
  return true;
}

Matrix Agent::projectToManifold(const Matrix &M) {
  // Get problem dimensions
  unsigned int r = relaxation_rank();
  unsigned int d = dimension();
  unsigned int n = num_poses();
  unsigned int l = num_unit_spheres();
  unsigned int b = num_landmarks();

  // Delegate to specific projection method based on manifold underlying local
  // graph
  return isPGOCompatible() ? projectToSEMatrix(M, r, d, n)
                           : projectToRAMatrix(M, r, d, n, l, b);
}

} // namespace DCORA
