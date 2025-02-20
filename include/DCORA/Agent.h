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

#pragma once

#include <DCORA/DCORA_robust.h>
#include <DCORA/DCORA_types.h>
#include <DCORA/Logger.h>
#include <DCORA/Measurements.h>
#include <DCORA/QuadraticProblem.h>
#include <DCORA/manifold/Elements.h>

#include <Eigen/Dense>
#include <glog/logging.h>
#include <map>
#include <memory>
#include <mutex> // NOLINT(build/c++11)
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread> // NOLINT(build/c++11)
#include <unordered_map>
#include <utility>
#include <vector>

namespace DCORA {

/**
 * @brief This class contains parameter settings for Agent
 */
class AgentParameters {
public:
  // Problem dimension
  unsigned int d;

  // Relaxed rank in Riemannian optimization
  unsigned int r;

  // Set of robot IDs
  std::set<unsigned int> robotIDs;

  // Total number of robots
  unsigned int numRobots;

  // Type of graph to use
  GraphType graphType;

  // Run in asynchronous mode
  bool asynchronous;

  // Frequency of optimization loop in asynchronous mode
  double asynchronousOptimizationRate;

  // Riemannian optimization settings for solving local subproblem
  ROptParameters localOptimizationParams;

  // Method to use to initialize single-robot trajectory estimates
  InitializationMethod localInitializationMethod;

  // Cross-robot initialization
  bool multirobotInitialization;

  // Use Nesterov acceleration
  bool acceleration;

  // Interval for fixed (periodic) restart
  unsigned restartInterval;

  // Parameter settings over robust cost functions
  RobustCostParameters robustCostParams;

  // Number of weight updates for robust optimization
  int robustOptNumWeightUpdates;

  // Warm start iterate during robust optimization
  int robustOptNumResets;

  // Number of inner iterations to apply before updating measurement weights
  // during robust optimization
  int robustOptInnerIters;

  // Minimum ratio of converged weights before terminating robust optimization
  double robustOptMinConvergenceRatio;

  // Minimum number of inliers for robust distributed initialization
  unsigned robustInitMinInliers;

  // Maximum number of global iterations
  unsigned maxNumIters;

  // Tolerance on relative change
  double relChangeTol;

  // Verbose flag
  bool verbose;

  // Flag to enable data logging
  bool logData;

  // Directory to log data
  std::string logDirectory;

  // Default constructor
  AgentParameters(unsigned int dIn, unsigned int rIn,
                  std::set<unsigned int> robotIDsIn = {0},
                  GraphType graphTypeIn = GraphType::PoseGraph,
                  ROptParameters local_opt_params = ROptParameters(),
                  bool accel = false, unsigned restartInt = 30,
                  RobustCostParameters costParams = RobustCostParameters(),
                  int robust_opt_num_weight_updates = 10,
                  int robust_opt_num_resets = 0,
                  int robust_opt_inner_iters = 30,
                  double robust_opt_min_convergence_ratio = 0.8,
                  unsigned robust_init_min_inliers = 2, unsigned maxIters = 500,
                  double changeTol = 5e-3, bool v = false, bool log = false,
                  std::string logDir = "")
      : d(dIn),
        r(rIn),
        robotIDs(robotIDsIn),
        numRobots(robotIDsIn.size()),
        graphType(graphTypeIn),
        asynchronous(false),
        asynchronousOptimizationRate(1),
        localOptimizationParams(local_opt_params),
        localInitializationMethod(InitializationMethod::Odometry),
        multirobotInitialization(true),
        acceleration(accel),
        restartInterval(restartInt),
        robustCostParams(costParams),
        robustOptNumWeightUpdates(robust_opt_num_weight_updates),
        robustOptNumResets(robust_opt_num_resets),
        robustOptInnerIters(robust_opt_inner_iters),
        robustOptMinConvergenceRatio(robust_opt_min_convergence_ratio),
        robustInitMinInliers(robust_init_min_inliers),
        maxNumIters(maxIters),
        relChangeTol(changeTol),
        verbose(v),
        logData(log),
        logDirectory(std::move(logDir)) {}

  inline friend std::ostream &operator<<(std::ostream &os,
                                         const AgentParameters &params) {
    // clang-format off
    os << "Agent parameters: " << std::endl;
    os << "Dimension: " << params.d << std::endl;
    os << "Relaxation rank: " << params.r << std::endl;
    os << "Robot IDs: ";
    for (const auto &robot_id : params.robotIDs)
      os << robot_id << " ";
    os << std::endl;
    os << "Number of robots: " << params.numRobots << std::endl;
    os << "Graph type: " << GraphTypeToString(params.graphType) << std::endl;
    os << "Asynchronous: " << params.asynchronous << std::endl;
    os << "Asynchronous optimization rate: " << params.asynchronousOptimizationRate << std::endl; // NOLINT
    os << "Local trajectory initialization method: " << InitializationMethodToString(params.localInitializationMethod) << std::endl; // NOLINT
    os << "Use multi-robot initialization: " << params.multirobotInitialization << std::endl; // NOLINT
    os << "Use Nesterov acceleration: " << params.acceleration << std::endl;
    os << "Fixed restart interval: " << params.restartInterval << std::endl;
    os << "Robust optimization num weight updates: " << params.robustOptNumWeightUpdates << std::endl; // NOLINT
    os << "Robust optimization num resets: " << params.robustOptNumResets << std::endl; // NOLINT
    os << "Robust optimization inner iterations: " << params.robustOptInnerIters << std::endl; // NOLINT
    os << "Robust optimization weight convergence min ratio: " << params.robustOptMinConvergenceRatio << std::endl; // NOLINT
    os << "Robust initialization minimum inliers: " << params.robustInitMinInliers << std::endl; // NOLINT
    os << "Max iterations: " << params.maxNumIters << std::endl;
    os << "Relative change tol: " << params.relChangeTol << std::endl;
    os << "Verbose: " << params.verbose << std::endl;
    os << "Log data: " << params.logData << std::endl;
    os << "Log directory: " << params.logDirectory << std::endl;
    os << std::endl;
    os << params.localOptimizationParams << std::endl;
    os << std::endl;
    os << params.robustCostParams << std::endl;
    // clang-format on
    return os;
  }
};

/**
 * @brief Defines the possible states of an Agent. Each state can only
 * transition to the state below
 */
enum AgentState {
  WAIT_FOR_DATA,           // waiting to receive graph
  WAIT_FOR_INITIALIZATION, // waiting to initialize state estimate
  INITIALIZED,             // state initialized and ready to update
};

/**
 * @brief Status of an agent to be shared with its peers
 */
struct AgentStatus {
  // Unique ID of this agent
  unsigned agentID;

  // Current state of this agent
  AgentState state;

  // Current problem instance number
  unsigned instanceNumber;

  // Current global iteration number
  unsigned iterationNumber;

  // True if the agent passes its local termination condition
  bool readyToTerminate;

  // The relative change of the agent's estimate
  double relativeChange;

  // Constructor
  explicit AgentStatus(unsigned id = 0,
                       AgentState s = AgentState::WAIT_FOR_DATA,
                       unsigned instance = 0, unsigned iteration = 0,
                       bool ready_to_terminate = false,
                       double relative_change = 0)
      : agentID(id),
        state(s),
        instanceNumber(instance),
        iterationNumber(iteration),
        readyToTerminate(ready_to_terminate),
        relativeChange(relative_change) {}

  inline friend std::ostream &operator<<(std::ostream &os,
                                         const AgentStatus &status) {
    os << "Agent status: " << std::endl;
    os << "ID: " << status.agentID << std::endl;
    os << "State: " << status.state << std::endl;
    os << "Instance number: " << status.instanceNumber << std::endl;
    os << "Iteration number: " << status.iterationNumber << std::endl;
    os << "Ready to terminate: " << status.readyToTerminate << std::endl;
    os << "Relative change: " << status.relativeChange << std::endl;
    return os;
  }
} __attribute__((aligned(32)));

/**
 * @brief This class implements a single robot in the distributed pose graph or
 * RA-SLAM optimization problem
 */
class Agent {
public:
  /**
   * @brief Constructor
   * @param ID
   * @param params
   */
  Agent(unsigned ID, const AgentParameters &params);

  /**
   * @brief Destructor
   */
  ~Agent();

  /**
   * @brief Set measurements for this agent
   * @param inputOdometry
   * @param inputPrivateLoopClosures
   * @param inputSharedLoopClosures
   */
  void setMeasurements(
      const std::vector<RelativePosePoseMeasurement> &inputOdometry,
      const std::vector<RelativePosePoseMeasurement> &inputPrivateLoopClosures,
      const std::vector<RelativePosePoseMeasurement> &inputSharedLoopClosures);

  /**
   * @brief Set measurements for this agent
   * @param inputMeasurements
   */
  void setMeasurements(
      const std::vector<RelativePosePoseMeasurement> &inputMeasurements);

  /**
   * @brief Set measurements for this agent
   * @param inputMeasurements
   */
  void setMeasurements(const RelativeMeasurements &inputMeasurements);

  /**
   * @brief Add a single measurement to this agent's graph. Do nothing if
   * the input factor already exists.
   * @param factor
   */
  void addMeasurement(const RelativePosePoseMeasurement &factor);

  /**
   * @brief Perform local initialization for this robot. After this function
   * call, the robot is initialized in its LOCAL frame where its first pose is
   * set to identity. Initialization in the global frame is still needed by
   * calling initializeInGlobalFrame(). If the dimension and number of states of
   * a provided initial guess does not match what is expected, the initial
   * guess will be ignored and this function will instead use its built in local
   * initialization methods for trajectory and random initialization for unit
   * spheres and landmarks. If initial guesses for unit spheres or landmarks
   * are provided and the local graph is not PGO compatible, an error will be
   * thrown. Further, if initial guesses for unit spheres or landmarks
   * are provided, an initial guess for the trajectory must also be provided.
   * @param TrajectoryInitPtr an optional trajectory estimate in an arbitrary
   * local frame.
   * @param UnitSphereInitPtr an optional estimate of the unit sphere variables
   * in the same local frame as the trajectory estimate (this condition is not
   * enforced).
   * @param LandmarkInitPtr an optional estimate of the landmark variables in
   * the same local frame as the trajectory estimate (this condition is not
   * enforced).
   */
  void initialize(const PoseArray *TrajectoryInitPtr = nullptr,
                  const PointArray *UnitSphereInitPtr = nullptr,
                  const PointArray *LandmarkInitPtr = nullptr);

  /**
   * @brief Initialize this robot's trajectory estimate (and unit sphere and
   * landmark estimates if performing distributed RA-SLAM optimization) in the
   * global frame. This function must be called after initialize().
   * @param T_world_robot d+1 by d+1 transformation from robot (local) frame
   * to the world frame. By convention, the robot local frame is one in
   * which the first pose of this robot is set to identity
   */
  void initializeInGlobalFrame(const Pose &T_world_robot);

  /**
   * @brief perform a single iteration. Return true if iteration is successful
   * @param doOptimization: if true, this robot is selected to perform local
   * optimization at this iteration
   * @return
   */
  bool iterate(bool doOptimization = true);

  /**
   * @brief Reset this agent to have an empty graph
   */
  virtual void reset();

  /**
   * @brief Return ID of this robot
   * @return
   */
  inline unsigned getID() const { return mID; }

  /**
   * @brief Get relaxation rank
   * @return
   */
  inline unsigned relaxation_rank() const { return r; }

  /**
   * @brief Get dimension (2 or 3)
   * @return
   */
  inline unsigned dimension() const { return d; }

  /**
   * @brief Get number of poses of this robot
   * @return
   */
  inline unsigned num_poses() const { return isAgentMap() ? 1 : mGraph->n(); }

  /**
   * @brief Get number of unit spheres of this robot
   * @return
   */
  inline unsigned num_unit_spheres() const { return mGraph->l(); }

  /**
   * @brief Get number of landmarks of this robot
   * @return
   */
  inline unsigned num_landmarks() const { return mGraph->b(); }

  /**
   * @brief Get problem dimension
   * @return
   */
  inline unsigned problem_dimension() const {
    return isAgentMap() ? (d + 1) : mGraph->k();
  }

  /**
   * @brief Get current instance number
   * @return
   */
  inline unsigned instance_number() const { return mInstanceNumber; }

  /**
   * @brief Get current global iteration number
   * @return
   */
  inline unsigned iteration_number() const { return mIterationNumber; }

  /**
   * @brief Return true if this agent is compatible with PGO
   * @return
   */
  inline bool isPGOCompatible() const {
    return mGraph->isPGOCompatible() &&
           mParams.graphType == GraphType::PoseGraph;
  }

  /**
   * @brief Return true if this agent is the map in RA-SLAM
   * @return
   */
  inline bool isAgentMap() const {
    return !mGraph->isPGOCompatible() && mID == MAP_ID;
  }

  /**
   * @brief Return true if the provided agent is the map in RA-SLAM
   * @return
   */
  inline bool isAgentMap(unsigned int ID) const {
    return !mGraph->isPGOCompatible() && ID == MAP_ID;
  }

  /**
   * @brief Get the current status of this agent
   * @return
   */
  inline AgentStatus getStatus() {
    mStatus.agentID = getID();
    mStatus.state = mState;
    mStatus.instanceNumber = instance_number();
    mStatus.iterationNumber = iteration_number();
    return mStatus;
  }

  /**
   * @brief Return true if the status of a neighbor robot is available locally
   * @return
   */
  bool hasNeighborStatus(unsigned neighborID) const {
    return mTeamStatus.find(neighborID) != mTeamStatus.end();
  }

  /**
   * @brief Get current state of a neighbor
   * @param neighborID
   * @return
   */
  inline AgentStatus getNeighborStatus(unsigned neighborID) const {
    CHECK(hasNeighborStatus(neighborID));
    return mTeamStatus.at(neighborID);
  }

  /**
   * @brief Set the status of a neighbor
   * @param status
   */
  inline void setNeighborStatus(const AgentStatus &status) {
    mTeamStatus[status.agentID] = status;
  }

  /**
   * @brief Return true if the input robot is a neighbor (i.e., has shared loop
   * closure with this robot)
   * @return
   */
  bool hasNeighbor(unsigned neighborID) const;

  /**
   * @brief Get vector of neighbor robot IDs.
   * @return
   */
  std::vector<unsigned> getNeighbors() const;

  /**
   * @brief Trajectory estimate of this robot in local frame, with its first
   * pose set to identity
   * @param Trajectory
   * @return
   */
  bool getTrajectoryInLocalFrame(Matrix *Trajectory);

  /**
   * @brief State estimate of this robot in local frame, with its first
   * pose set to identity
   * @param Trajectory
   * @param UnitSpheres
   * @param Landmarks
   * @return
   */
  bool getStatesInLocalFrame(Matrix *Trajectory, Matrix *UnitSpheres = nullptr,
                             Matrix *Landmarks = nullptr);

  /**
   * @brief Trajectory estimate of this robot in global frame, with the first
   * pose of robot 0 set to identity
   * @param Trajectory
   * @return
   */
  bool getTrajectoryInGlobalFrame(Matrix *Trajectory);

  /**
   * @brief Return trajectory estimate of this robot in global frame, with the
   * first pose of robot 0 set to identity
   * @param Trajectory
   * @return
   */
  bool getTrajectoryInGlobalFrame(PoseArray *Trajectory);

  /**
   * @brief Return a single pose in the global frame
   * @param poseID
   * @param T
   * @return
   */
  bool getPoseInGlobalFrame(unsigned poseID, Matrix *T);

  /**
   * @brief Get the pose of a neighbor in global frame
   * @param neighborID
   * @param poseID
   * @param T
   * @return
   */
  bool getNeighborPoseInGlobalFrame(unsigned neighborID, unsigned poseID,
                                    Matrix *T);

  /**
   * @brief Get a single public pose of this robot. Note that currently, this
   * method does not check that the requested pose is a public pose. Return true
   * if the requested pose exists
   * @param index index of the requested pose
   * @param Mout actual value of the pose
   * @return
   */
  bool getSharedPose(unsigned index, Matrix *Mout);

  /**
   * @brief Get auxiliary variable associated with a single public pose. Return
   * true if the requested pose exists
   * @param index
   * @param Mout
   * @return
   */
  bool getAuxSharedPose(unsigned index, Matrix *Mout);

  /**
   * @brief Get maps for of all public states of this robot. Return true if the
   * agent is initialized
   * @param poseDict PoseDict object whose content will be filled
   * @param unitSphereDict (optional) UnitSphereDict object whose content will
   * be filled
   * @param landmarkDict (optional) LandmarkDict object whose content will be
   * filled
   * @return
   */
  bool getSharedStateDicts(PoseDict *poseDict,
                           UnitSphereDict *unitSphereDict = nullptr,
                           LandmarkDict *landmarkDict = nullptr);

  /**
   * @brief Get a map of all public poses of this robot with the specified
   * neighbor
   * @param map
   * @param neighborID
   * @return
   */
  bool getSharedPoseDictWithNeighbor(PoseDict *map, unsigned neighborID);

  /**
   * @brief Get a map of all auxiliary public poses of this robot with the
   * specified neighbor
   * @param map
   * @param neighborID
   * @return
   */
  bool getAuxSharedPoseDictWithNeighbor(PoseDict *map, unsigned neighborID);

  /**
   * @brief Helper function to reset internal solution. Currently only for
   * debugging
   * @param Xin
   */
  void setX(const Matrix &Xin);

  /**
   * @brief Reset internal solution to initial guess X = Xinit
   */
  void setXToInitialGuess();

  /**
   * @brief Helper function to get internal solution. Note that this method
   * disregards whether the agent is initialized
   * @param Mout
   * @return
   */
  bool getX(Matrix *Mout);

  /**
   * @brief Return true if termination condition is satisfied
   * @return
   */
  bool shouldTerminate();

  /**
   * @brief Return true if restart condition is satisfied
   * @return
   */
  bool shouldRestart() const;

  /**
   * @brief Restart Nesterov acceleration sequence
   * @param doOptimization true if perform optimization after restart
   */
  void restartNesterovAcceleration(bool doOptimization);

  /**
   * @brief Initiate a new thread that runs runOptimizationLoop()
   */
  void startOptimizationLoop();

  /**
   * @brief Request to terminate optimization loop, if running. This function
   * also waits until the optimization loop is finished
   */
  void endOptimizationLoop();

  /**
   * @brief Check if the optimization thread is running
   * @return
   */
  bool isOptimizationRunning();

  /**
   * @brief Get lifting matrix
   * @param M
   * @return
   */
  bool getLiftingMatrix(Matrix *M) const;

  /**
   * @brief Set the lifting matrix
   * @param M
   */
  void setLiftingMatrix(const Matrix &M);

  /**
   * @brief Set the global anchor
   * @param M
   */
  void setGlobalAnchor(const Matrix &M);

  /**
   * @brief Update local copy of a neighbor agent's states or auxiliary states
   * @param neighborID the ID of the neighbor agent
   * @param poseDict the pose dictionary of the neighbor agent
   * @param areNeighborStatesAux (optional) true if the neighbor states are
   * auxiliary
   * @param unitSphereDict (optional) the unit sphere dictionary of the neighbor
   * agent
   * @param landmarkDict (optional) the landmark dictionary of the neighbor
   * agent
   */
  void
  updateNeighborStates(unsigned neighborID, const PoseDict &poseDict,
                       bool areNeighborStatesAux = false,
                       const UnitSphereDict &unitSphereDict = UnitSphereDict(),
                       const LandmarkDict &landmarkDict = LandmarkDict());

  /**
   * @brief Clear local caches of all neighbors' states
   */
  void clearNeighborStates();

  /**
   * @brief Clear local caches of all active neighbors' states
   */
  void clearActiveNeighborStates();

  /**
   * @brief Perform local PGO using the standard L2 (least-squares) cost
   * function. Return trajectory estimate in matrix form T = [R1 t1 ... Rn tn]
   * in an arbitrary frame
   * @return
   */
  Matrix localPoseGraphOptimization();

protected:
  // The unique ID associated to this robot
  unsigned int mID;

  // Dimension
  unsigned int d;

  // Relaxed rank in Riemannian optimization problem
  unsigned int r;

  // Internal optimization iterate (before rounding)
  LiftedRangeAidedArray X;

  // Parameter settings
  const AgentParameters mParams;

  // Current state of this agent
  AgentState mState;

  // Current status of this agent (to be shared with others)
  AgentStatus mStatus;

  // Robust cost function
  RobustCost mRobustCost;

  // Pointer to graph
  std::shared_ptr<Graph> mGraph;

  // Current optimization instance
  unsigned mInstanceNumber;

  // Current global iteration counter (this is only meaningful in synchronous
  // mode)
  unsigned mIterationNumber;

  // Iteration number of the latest weight update
  unsigned mLatestWeightUpdateIteration;

  // Number of inner iterations performed for robust optimization
  int mRobustOptInnerIter;

  // Number of times measurement weights are updated
  int mWeightUpdateCount;

  // Number of times solutions are reset to initial guess
  int mTrajectoryResetCount;

  // Latest local optimization result
  ROPTResult mLocalOptResult;

  // Logging
  Logger mLogger;

  // Store status of peer agents
  std::unordered_map<unsigned, AgentStatus> mTeamStatus;

  // Store if robots are actively participating in optimization
  std::map<unsigned, bool> mTeamRobotActive;

  // Request to publish public states
  bool mPublishPublicStatesRequested = false;

  // Request to publish in asynchronous mode
  bool mPublishAsynchronousRequested = false;

  // Request to terminate optimization thread
  bool mEndLoopRequested = false;

  // Initial iterate
  std::optional<LiftedRangeAidedArray> XInit;

  // Initial trajectory [R1 t1 ... Rn tn] in an arbitrary coordinate frame
  std::optional<PoseArray> TrajectoryLocalInit;

  // Initial unit spheres [r1 ... rl] in the same local frame as the trajectory
  // estimate.
  std::optional<PointArray> UnitSphereLocalInit;

  // Initial landmarks [L1 ... Lb] in the same local frame as the trajectory
  // estimate.
  std::optional<PointArray> LandmarkLocalInit;

  // Lifting matrix shared by all agents
  std::optional<Matrix> YLift;

  // Anchor matrix shared by all agents
  std::optional<LiftedPose> globalAnchor;

  // This dictionary stores poses owned by other robots that are connected to
  // this robot by loop closure
  PoseDict neighborPoseDict;

  // This dictionary stores unit spheres owned by other robots that are
  // connected to this robot by loop closure
  UnitSphereDict neighborUnitSphereDict;

  // This dictionary stores landmarks owned by other robots that are connected
  // to this robot by loop closure
  LandmarkDict neighborLandmarkDict;

  // Implement locking to synchronize read & write of state estimate
  std::mutex mStatesMutex;

  // Implement locking to synchronize read & write of shared states from
  // neighbors
  std::mutex mNeighborStatesMutex;

  // Implement locking on measurements
  std::mutex mMeasurementsMutex;

  // Thread that runs optimization loop in asynchronous mode
  std::unique_ptr<std::thread> mOptimizationThread;

  /**
   * @brief Reset variables used in Nesterov acceleration
   */
  void initializeAcceleration();

  /**
   * @brief Compute a robust relative transform estimate between this robot and
   * neighbor robot, using a two-stage method which first perform robust single
   * rotation averaging, and then performs translation averaging on the inlier
   * set. Return true if transformation is computed successfully
   * @param neighborID
   * @param poseDict
   * @param T_world_robot output transformation from current local (robot) frame
   * to world frame
   * @return
   */
  bool computeRobustNeighborTransformTwoStage(unsigned neighborID,
                                              const PoseDict &poseDict,
                                              Pose *T_world_robot);

  /**
   * @brief Compute a robust relative transform estimate between this robot and
   * neighbor robot, by solving a robust single pose averaging problem using
   * GNC. Return true if transformation is computed successfully
   * @param neighborID
   * @param poseDict
   * @param T_world_robot output transformation from current local (robot) frame
   * to world frame
   * @return
   */
  bool computeRobustNeighborTransform(unsigned neighborID,
                                      const PoseDict &poseDict,
                                      Pose *T_world_robot);

  /**
   * @brief Spawn a separate thread that optimizes the local graph in a loop
   */
  void runOptimizationLoop();

  /**
   * @brief Initialize robust optimization.
   * This function sets all active loop closure weights to one
   * in preparation for GNC.
   */
  void initializeRobustOptimization();

  /**
   * @brief Return true if should update loop closure weights
   * @return
   */
  bool shouldUpdateMeasurementWeights() const;

  /**
   * @brief Update loop closure weights.
   */
  void updateMeasurementWeights();

  /**
   * @brief Compute the residual of a measurement (square root of weighted
   * square error). Return true if computation is successful
   * @param measurement The measurement to evaluate
   * @param residual The output residual
   * @return
   */
  bool computeMeasurementResidual(const RelativeMeasurement &measurement,
                                  double *residual) const;

  /**
   * @brief Set weight for measurement in the graph. Return false if the
   * specified public measurement does not exist
   * @param edgeID
   * @param weight
   * @param fixed_weight True if the weight is fixed (i.e. cannot be changed by
   * GNC)
   * @return
   */
  bool setMeasurementWeight(const EdgeID &edgeID, double weight,
                            bool fixed_weight = false);

  /**
   * @brief Return true if the robot is initialized in global frame
   * @param robot_id
   * @return
   */
  bool isRobotInitialized(unsigned robot_id) const;

  /**
   * @brief Return true if the robot is currently active
   * @param robot_id
   * @return
   */
  bool isRobotActive(unsigned robot_id) const;

  /**
   * @brief Set robot to be active
   * @param robot_id
   * @param active
   */
  void setRobotActive(unsigned robot_id, bool active = true);

  /**
   * @brief Return the number of currently active robots
   * @return
   */
  size_t numActiveRobots() const;

  /**
   * @brief Add a prior to the first pose of this robot
   * @return
   */
  bool anchorFirstPose();

  /**
   * @brief Add a prior to the first pose of this robot
   * @param prior
   * @return
   */
  bool anchorFirstPose(const LiftedPose &prior);

private:
  // Stores the auxiliary pose variables from neighbors (only used in
  // acceleration)
  PoseDict neighborAuxPoseDict;

  // Stores the auxiliary unit sphere variables from neighbors (only used in
  // acceleration)
  UnitSphereDict neighborAuxUnitSphereDict;

  // Stores the auxiliary landmark variables from neighbors (only used in
  // acceleration)
  LandmarkDict neighborAuxLandmarkDict;

  // Auxiliary scalar used in acceleration
  double gamma;

  // Auxiliary scalar used in acceleration
  double alpha;

  // Auxiliary variable used in acceleration
  LiftedRangeAidedArray Y;

  // Auxiliary variable used in acceleration
  LiftedRangeAidedArray V;

  // Save previous iteration (for restarting)
  LiftedRangeAidedArray XPrev;

  /**
   * @brief Update gamma variable
   */
  void updateGamma();

  /**
   * @brief Update alpha variable
   */
  void updateAlpha();

  /**
   * @brief Update X variable
   * @param doOptimization Whether this agent is selected to perform
   * optimization
   * @param acceleration true to use acceleration
   * @return true if update is successful
   */
  bool updateX(bool doOptimization = false, bool acceleration = false);

  /**
   * @brief Update Y variable
   */
  void updateY();

  /**
   * @brief Update V variable
   */
  void updateV();

  /**
   * @brief Compute the relative transformation to a neighboring robot using a
   * single inter-robot loop closure
   * @param measurement
   * @param neighbor_pose
   * @return
   */
  Pose computeNeighborTransform(const RelativePosePoseMeasurement &measurement,
                                const LiftedPose &neighbor_pose);

  /**
   * @brief Project matrix M to the manifold underlying the local graph
   * @param M
   * @return
   */
  Matrix projectToManifold(const Matrix &M);
};

} // namespace DCORA
