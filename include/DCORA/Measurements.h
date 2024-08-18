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

#include <DCORA/DCORA_types.h>
#include <DCORA/manifold/Elements.h>

#include <Eigen/Dense>
#include <glog/logging.h>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace DCORA {

/**
 * @brief A simple struct that contains the elements of a prior measurement for
 * (robot, state).
 */
struct PriorMeasurement {
  // Measurement type
  MeasurementType measurementType;

  // 0-based index of robot
  size_t r;

  // 0-based index of state
  size_t p;

  // State type for (robot, state) pair
  StateType stateType;

  // If measurement weight is fixed
  bool fixedWeight;

  // Weight between (0,1) used in Graduated Non-Convexity
  double weight;

  // Simple default constructor; does nothing
  PriorMeasurement() = default;

  // Virtual destructor to enforce abstract class
  virtual ~PriorMeasurement() {}

  // Basic constructor
  PriorMeasurement(MeasurementType type, size_t robot, size_t state,
                   StateType stateType, bool fixedWeight, double weight)
      : measurementType(type),
        r(robot),
        p(state),
        stateType(stateType),
        fixedWeight(fixedWeight),
        weight(weight) {}

  // Check the dimensions of the measurement
  virtual void checkDim(unsigned int d) const = 0;

  // Get the source state ID
  virtual StateID getSrcID() const = 0;

  // Print measurement information
  virtual void print(std::ostream &os) const = 0;

  // A utility function for streaming this struct to cout
  inline friend std::ostream &operator<<(std::ostream &os,
                                         const PriorMeasurement &measurement) {
    os << "MeasurementType: "
       << MeasurementTypeToString(measurement.measurementType) << std::endl;
    os << "StateType: " << StateTypeToString(measurement.stateType)
       << std::endl;
    os << "FixedWeight: " << measurement.fixedWeight << std::endl;
    os << "Weight: " << measurement.weight << std::endl;
    os << "Robot: " << measurement.r << std::endl;
    os << "State: " << measurement.p << std::endl;
    measurement.print(os);
    return os;
  }
};

/**
 * @brief A simple struct that contains the elements of a single pose prior for
 * (robot, pose).
 */
struct PosePrior : PriorMeasurement {
  // Rotational measurement
  Matrix R;

  // Translational measurement
  Vector t;

  // Rotational measurement precision
  double kappa;

  // Translational measurement precision
  double tau;

  // Default constructor
  PosePrior()
      : PriorMeasurement(MeasurementType::PosePrior, 0, 0, StateType::Pose,
                         false, 1.0) {}

  // Basic constructor
  PosePrior(size_t robot, size_t pose, const Matrix &priorRotation,
            const Vector &priorTranslation, double rotationalPrecision,
            double translationalPrecision, bool fixedWeight = false,
            double weight = 1.0)
      : PriorMeasurement(MeasurementType::PosePrior, robot, pose,
                         StateType::Pose, fixedWeight, weight),
        R(priorRotation),
        t(priorTranslation),
        kappa(rotationalPrecision),
        tau(translationalPrecision) {}

  void checkDim(unsigned int d) const override {
    CHECK(d == 2 || d == 3);
    CHECK(R.rows() == d && R.cols() == d);
    CHECK(t.size() == d);
  }

  StateID getSrcID() const override { return PoseID(r, p); }

  void print(std::ostream &os) const override {
    os << "R: " << std::endl << R << std::endl;
    os << "t: " << std::endl << t << std::endl;
    os << "Kappa: " << kappa << std::endl;
    os << "Tau: " << tau << std::endl;
  }
};

/**
 * @brief A simple struct that contains the elements of a single landmark prior
 * for (robot, landmark).
 */
struct LandmarkPrior : PriorMeasurement {
  // Translational measurement
  Vector t;

  // Translational measurement precision
  double tau;

  // Default constructor
  LandmarkPrior()
      : PriorMeasurement(MeasurementType::LandmarkPrior, 0, 0,
                         StateType::Landmark, false, 1.0) {}

  // Basic constructor
  LandmarkPrior(size_t robot, size_t landmark, const Vector &priorTranslation,
                double translationalPrecision, bool fixedWeight = false,
                double weight = 1.0)
      : PriorMeasurement(MeasurementType::LandmarkPrior, robot, landmark,
                         StateType::Landmark, fixedWeight, weight),
        t(priorTranslation),
        tau(translationalPrecision) {}

  void checkDim(unsigned int d) const override {
    CHECK(d == 2 || d == 3);
    CHECK(t.size() == d);
  }

  StateID getSrcID() const override { return LandmarkID(r, p); }

  void print(std::ostream &os) const override {
    os << "t: " << std::endl << t << std::endl;
    os << "Tau: " << tau << std::endl;
  }
};

/**
 * @brief A simple struct that contains the elements of a relative measurement
 * from (robot, state) pairs: (robot1, state1) to (robot2, state2).
 */
struct RelativeMeasurement {
  // Measurement type
  MeasurementType measurementType;

  // 0-based index of first robot
  size_t r1;

  // 0-based index of second robot
  size_t r2;

  // 0-based index of first state
  size_t p1;

  // 0-based index of second state
  size_t p2;

  // State type for first (robot, state) pair
  StateType stateType1;

  // State type for second (robot, state) pair
  StateType stateType2;

  // If measurement weight is fixed
  bool fixedWeight;

  // Weight between (0,1) used in Graduated Non-Convexity
  double weight;

  // Simple default constructor; does nothing
  RelativeMeasurement() = default;

  // Virtual destructor to enforce abstract class
  virtual ~RelativeMeasurement() {}

  // Basic constructor
  RelativeMeasurement(MeasurementType type, size_t firstRobot,
                      size_t secondRobot, size_t firstState, size_t secondState,
                      StateType stateType1, StateType stateType2,
                      bool fixedWeight, double weight)
      : measurementType(type),
        r1(firstRobot),
        r2(secondRobot),
        p1(firstState),
        p2(secondState),
        stateType1(stateType1),
        stateType2(stateType2),
        fixedWeight(fixedWeight),
        weight(weight) {}

  // Check the dimensions of the measurement
  virtual void checkDim(unsigned int d) const = 0;

  // Get the source state ID
  virtual StateID getSrcID() const = 0;

  // Get the destination state ID
  virtual StateID getDstID() const = 0;

  // Get the edge ID
  virtual EdgeID getEdgeID() const = 0;

  // Print relative measurement information
  virtual void print(std::ostream &os) const = 0;

  // A utility function for streaming this struct to cout
  inline friend std::ostream &
  operator<<(std::ostream &os, const RelativeMeasurement &measurement) {
    os << "MeasurementType: "
       << MeasurementTypeToString(measurement.measurementType) << std::endl;
    os << "StateType1: " << StateTypeToString(measurement.stateType1)
       << std::endl;
    os << "StateType2: " << StateTypeToString(measurement.stateType2)
       << std::endl;
    os << "FixedWeight: " << measurement.fixedWeight << std::endl;
    os << "Weight: " << measurement.weight << std::endl;
    os << "Robot1: " << measurement.r1 << std::endl;
    os << "State1: " << measurement.p1 << std::endl;
    os << "Robot2: " << measurement.r2 << std::endl;
    os << "State2: " << measurement.p2 << std::endl;
    measurement.print(os);
    return os;
  }
};

/**
 * @brief A simple struct that contains the elements of a relative
 * pose measurement from (robot, pose) pairs: (robot1, pose1) to (robot2,
 * pose2).
 */
struct RelativePosePoseMeasurement : RelativeMeasurement {
  // Rotational measurement
  Matrix R;

  // Translational measurement
  Vector t;

  // Rotational measurement precision
  double kappa;

  // Translational measurement precision
  double tau;

  // Default constructor
  RelativePosePoseMeasurement()
      : RelativeMeasurement(MeasurementType::PosePose, 0, 0, 0, 0,
                            StateType::Pose, StateType::Pose, false, 1.0) {}

  // Basic constructor
  RelativePosePoseMeasurement(size_t firstRobot, size_t secondRobot,
                              size_t firstPose, size_t secondPose,
                              const Matrix &relativeRotation,
                              const Vector &relativeTranslation,
                              double rotationalPrecision,
                              double translationalPrecision,
                              bool fixedWeight = false, double weight = 1.0)
      : RelativeMeasurement(MeasurementType::PosePose, firstRobot, secondRobot,
                            firstPose, secondPose, StateType::Pose,
                            StateType::Pose, fixedWeight, weight),
        R(relativeRotation),
        t(relativeTranslation),
        kappa(rotationalPrecision),
        tau(translationalPrecision) {}

  // Copy constructor
  RelativePosePoseMeasurement(const RelativePosePoseMeasurement &other)
      : RelativeMeasurement(other.measurementType, other.r1, other.r2, other.p1,
                            other.p2, other.stateType1, other.stateType2,
                            other.fixedWeight, other.weight),
        R(other.R),
        t(other.t),
        kappa(other.kappa),
        tau(other.tau) {}

  // Equality operator
  bool operator==(const RelativePosePoseMeasurement &other) const {
    return std::tie(r1, r2, p1, p2, fixedWeight, weight, R, t, kappa, tau) ==
           std::tie(other.r1, other.r2, other.p1, other.p2, other.fixedWeight,
                    other.weight, other.R, other.t, other.kappa, other.tau);
  }

  void checkDim(unsigned int d) const override {
    CHECK(d == 2 || d == 3);
    CHECK(R.rows() == d && R.cols() == d);
    CHECK(t.size() == d);
  }

  StateID getSrcID() const override { return PoseID(r1, p1); }

  StateID getDstID() const override { return PoseID(r2, p2); }

  EdgeID getEdgeID() const override {
    return EdgeID(getSrcID(), getDstID(), MeasurementType::PosePose);
  }

  void print(std::ostream &os) const override {
    os << "R: " << std::endl << R << std::endl;
    os << "t: " << std::endl << t << std::endl;
    os << "Kappa: " << kappa << std::endl;
    os << "Tau: " << tau << std::endl;
  }
};

/**
 * @brief A simple struct that contains the elements of a relative pose-landmark
 * measurement from (robot, state) pairs: (robot1, pose1) to (robot2,
 * landmark2).
 */
struct RelativePoseLandmarkMeasurement : RelativeMeasurement {
  // Translational measurement
  Vector t;

  // Translational measurement precision
  double tau;

  // Default constructor
  RelativePoseLandmarkMeasurement()
      : RelativeMeasurement(MeasurementType::PoseLandmark, 0, 0, 0, 0,
                            StateType::Pose, StateType::Landmark, false, 1.0) {}

  // Basic constructor
  RelativePoseLandmarkMeasurement(size_t firstRobot, size_t secondRobot,
                                  size_t firstPose, size_t secondLandmark,
                                  const Vector &relativeTranslation,
                                  double translationalPrecision,
                                  bool fixedWeight = false, double weight = 1.0)
      : RelativeMeasurement(MeasurementType::PoseLandmark, firstRobot,
                            secondRobot, firstPose, secondLandmark,
                            StateType::Pose, StateType::Landmark, fixedWeight,
                            weight),
        t(relativeTranslation),
        tau(translationalPrecision) {}

  // Copy constructor
  RelativePoseLandmarkMeasurement(const RelativePoseLandmarkMeasurement &other)
      : RelativeMeasurement(other.measurementType, other.r1, other.r2, other.p1,
                            other.p2, other.stateType1, other.stateType2,
                            other.fixedWeight, other.weight),
        t(other.t),
        tau(other.tau) {}

  // Equality operator
  bool operator==(const RelativePoseLandmarkMeasurement &other) const {
    return std::tie(r1, r2, p1, p2, fixedWeight, weight, t, tau) ==
           std::tie(other.r1, other.r2, other.p1, other.p2, other.fixedWeight,
                    other.weight, other.t, other.tau);
  }

  void checkDim(unsigned int d) const override {
    CHECK(d == 2 || d == 3);
    CHECK(t.size() == d);
  }

  StateID getSrcID() const override { return PoseID(r1, p1); }

  StateID getDstID() const override { return LandmarkID(r2, p2); }

  EdgeID getEdgeID() const override {
    return EdgeID(getSrcID(), getDstID(), MeasurementType::PoseLandmark);
  }

  void print(std::ostream &os) const override {
    os << "t: " << std::endl << t << std::endl;
    os << "Tau: " << tau << std::endl;
  }
};

/**
 * @brief A simple struct that contains the elements of a range measurement from
 * (robot, state) pairs: (robot1, state1) to (robot2, state2).
 */
struct RangeMeasurement : RelativeMeasurement {
  // Unit sphere variable index
  size_t l;

  // Range measurement
  double range;

  // Range measurement precision
  double precision;

  // Default constructor
  RangeMeasurement()
      : RelativeMeasurement(MeasurementType::Range, 0, 0, 0, 0, StateType::None,
                            StateType::None, false, 1.0) {}

  // Basic constructor
  RangeMeasurement(size_t firstRobot, size_t secondRobot, size_t firstState,
                   size_t secondState, size_t unitSphereVarIdx,
                   const double rangeMeasurement, double rangePrecision,
                   StateType stateType1, StateType stateType2,
                   bool fixedWeight = false, double weight = 1.0)
      : RelativeMeasurement(MeasurementType::Range, firstRobot, secondRobot,
                            firstState, secondState, stateType1, stateType2,
                            fixedWeight, weight),
        l(unitSphereVarIdx),
        range(rangeMeasurement),
        precision(rangePrecision) {}

  // Copy constructor
  RangeMeasurement(const RangeMeasurement &other)
      : RelativeMeasurement(other.measurementType, other.r1, other.r2, other.p1,
                            other.p2, other.stateType1, other.stateType2,
                            other.fixedWeight, other.weight),
        l(other.l),
        range(other.range),
        precision(other.precision) {}

  // Equality operator
  bool operator==(const RangeMeasurement &other) const {
    return std::tie(r1, r2, p1, p2, stateType1, stateType2, fixedWeight, weight,
                    l, range, precision) ==
           std::tie(other.r1, other.r2, other.p1, other.p2, other.stateType1,
                    other.stateType2, other.fixedWeight, other.weight, other.l,
                    other.range, other.precision);
  }

  void checkDim(unsigned int d) const override { CHECK(d == 2 || d == 3); }

  StateID getSrcID() const override {
    if (stateType1 == StateType::Pose)
      return PoseID(r1, p1);
    else if (stateType1 == StateType::Landmark)
      return LandmarkID(r1, p1);
    else
      LOG(FATAL) << "StateType1 is " << StateTypeToString(stateType1) << " in "
                 << MeasurementTypeToString(measurementType) << "!";
  }

  StateID getDstID() const override {
    if (stateType2 == StateType::Pose)
      return PoseID(r2, p2);
    else if (stateType2 == StateType::Landmark)
      return LandmarkID(r2, p2);
    else
      LOG(FATAL) << "StateType2 is " << StateTypeToString(stateType2) << " in "
                 << MeasurementTypeToString(measurementType) << "!";
  }

  EdgeID getEdgeID() const override {
    return EdgeID(getSrcID(), getDstID(), MeasurementType::Range);
  }

  UnitSphereID getUnitSphereID() const { return UnitSphereID(r1, l); }

  void print(std::ostream &os) const override {
    os << "l: " << l << std::endl;
    os << "range: " << range << std::endl;
    os << "precision: " << precision << std::endl;
  }
};

// Type-safe unions of relative measurements
typedef std::variant<RelativePosePoseMeasurement,
                     RelativePoseLandmarkMeasurement, RangeMeasurement>
    RelativeMeasurementVariant;
typedef std::variant<RelativePosePoseMeasurement *,
                     RelativePoseLandmarkMeasurement *, RangeMeasurement *>
    RelativeMeasurementPointerVariant;

/**
 * @brief A class that contains all relative measurements within a common
 * vector.
 */
class RelativeMeasurements {
public:
  // Simple default constructor; does nothing
  RelativeMeasurements() = default;

  // Copy constructor
  RelativeMeasurements(const RelativeMeasurements &other) : vec(other.vec) {}

  // Assignment operator
  RelativeMeasurements &operator=(const RelativeMeasurements &other) {
    if (this == &other)
      return *this;
    vec = other.vec;
    return *this;
  }

  // Getters
  std::vector<RelativePosePoseMeasurement>
  GetRelativePosePoseMeasurements() const {
    std::vector<RelativePosePoseMeasurement> pose_pose_measurements;
    for (const auto &m : vec) {
      if (std::holds_alternative<RelativePosePoseMeasurement>(m))
        pose_pose_measurements.emplace_back(
            std::get<RelativePosePoseMeasurement>(m));
    }
    return pose_pose_measurements;
  }

  std::vector<RelativePoseLandmarkMeasurement>
  GetRelativePoseLandmarkMeasurements() const {
    std::vector<RelativePoseLandmarkMeasurement> pose_landmark_measurements;
    for (const auto &m : vec) {
      if (std::holds_alternative<RelativePoseLandmarkMeasurement>(m))
        pose_landmark_measurements.emplace_back(
            std::get<RelativePoseLandmarkMeasurement>(m));
    }
    return pose_landmark_measurements;
  }

  std::vector<RangeMeasurement> GetRangeMeasurements() const {
    std::vector<RangeMeasurement> range_measurements;
    for (const auto &m : vec) {
      if (std::holds_alternative<RangeMeasurement>(m))
        range_measurements.emplace_back(std::get<RangeMeasurement>(m));
    }
    return range_measurements;
  }

  // Setters
  void push_back(const RelativeMeasurement &relative_measurement) {
    switch (relative_measurement.measurementType) {
    case MeasurementType::PosePose:
      addPosePoseMeasurement(relative_measurement);
      break;
    case MeasurementType::PoseLandmark:
      addPoseLandmarkMeasurement(relative_measurement);
      break;
    case MeasurementType::Range:
      addRangeMeasurement(relative_measurement);
      break;
    default:
      LOG(WARNING) << "Warning: unknown relative measurement type: "
                   << MeasurementTypeToString(
                          relative_measurement.measurementType)
                   << "!";
    }
  }

  // A utility function for streaming this class to cout
  inline friend std::ostream &
  operator<<(std::ostream &os,
             const RelativeMeasurements &relative_measurements) {
    for (const auto &m : relative_measurements.vec) {
      std::visit([&os](const auto &m) { os << m; }, m);
    }
    return os;
  }

  // Vector of relative measurements
  std::vector<RelativeMeasurementVariant> vec;

private:
  /**
   * @brief Template function to cast a Relative Measurement object to a
   * specific type T, such as RelativePosePoseMeasurement,
   * RelativePoseLandmarkMeasurement, or RangeMeasurement.
   * @tparam T
   * @param relative_measurement
   * @return
   */
  template <typename T>
  T castRelativeMeasurement(const RelativeMeasurement &relative_measurement) {
    if constexpr (std::is_same_v<T, RelativePosePoseMeasurement>)
      return dynamic_cast<const T &>(relative_measurement);
    else if constexpr (std::is_same_v<T, RelativePoseLandmarkMeasurement>)
      return dynamic_cast<const T &>(relative_measurement);
    else if constexpr (std::is_same_v<T, RangeMeasurement>)
      return dynamic_cast<const T &>(relative_measurement);
    else
      LOG(FATAL) << "Error: cannot cast relative measurement: "
                 << relative_measurement << "!";
    return T{};
  }

  /**
   * @brief Add a relative pose-pose measurement.
   * @param relative_measurement
   */
  void addPosePoseMeasurement(const RelativeMeasurement &relative_measurement) {
    const RelativePosePoseMeasurement &pose_pose_measurement =
        castRelativeMeasurement<RelativePosePoseMeasurement>(
            relative_measurement);
    vec.push_back(pose_pose_measurement);
  }

  /**
   * @brief Add a relative pose-landmark measurement.
   * @param relative_measurement
   */
  void
  addPoseLandmarkMeasurement(const RelativeMeasurement &relative_measurement) {
    const RelativePoseLandmarkMeasurement &pose_landmark_measurement =
        castRelativeMeasurement<RelativePoseLandmarkMeasurement>(
            relative_measurement);
    vec.push_back(pose_landmark_measurement);
  }

  /**
   * @brief Add a range measurement.
   * @param relative_measurement
   */
  void addRangeMeasurement(const RelativeMeasurement &relative_measurement) {
    const RangeMeasurement &range_measurement =
        castRelativeMeasurement<RangeMeasurement>(relative_measurement);
    vec.push_back(range_measurement);
  }
};

/**
 * @brief A simple struct that contains all measurement types.
 */
struct Measurements {
  // Measurements
  std::vector<PosePrior> pose_priors;
  std::vector<LandmarkPrior> landmark_priors;
  RelativeMeasurements relative_measurements;

  // Ground truth initialization
  std::shared_ptr<RangeAidedArray> ground_truth_init;

  // Simple default constructor; does nothing
  Measurements() = default;

  // A utility function for streaming this struct to cout
  inline friend std::ostream &operator<<(std::ostream &os,
                                         const Measurements &measurements) {
    os << "Measurements:" << std::endl;
    os << "Priors:" << std::endl;
    for (const auto &pose_prior : measurements.pose_priors) {
      os << pose_prior;
    }
    for (const auto &landmark_prior : measurements.landmark_priors) {
      os << landmark_prior;
    }
    os << "Relative Measurements:" << std::endl;
    os << measurements.relative_measurements << std::endl;
    os << "Ground Truth Initialization Matrix:" << std::endl;
    if (measurements.ground_truth_init) {
      os << std::fixed << std::setprecision(9)
         << measurements.ground_truth_init->getData() << std::endl;
    } else {
      os << "Ground truth initialization matrix not set." << std::endl;
    }
    return os;
  }
};

// Ordered map of robot IDs and associated measurements
typedef std::map<unsigned int, Measurements> RobotMeasurements;

// A utility function for streaming RobotMeasurements to cout
inline std::ostream &operator<<(std::ostream &os,
                                const RobotMeasurements &robot_measurements) {
  for (const auto &[robot_id, measurements] : robot_measurements) {
    os << "Robot " << robot_id << " Measurements" << std::endl;
    os << measurements;
  }
  return os;
}

/**
 * @brief A simple struct that contains ground truth states.
 */
struct GroundTruth {
  PoseDict poses;
  LandmarkDict landmarks;
  UnitSphereDict unit_spheres;

  // Simple default constructor; does nothing
  GroundTruth() = default;

  // A utility function for streaming this struct to cout
  inline friend std::ostream &operator<<(std::ostream &os,
                                         const GroundTruth &ground_truth) {
    os << "Ground Truth:" << std::endl;
    os << "Poses:" << std::endl;
    os << ground_truth.poses;
    os << "Landmarks:" << std::endl;
    os << ground_truth.landmarks;
    os << "Unit Spheres:" << std::endl;
    os << ground_truth.unit_spheres;
    return os;
  }
};

// Ordered map of local and global states
typedef std::map<StateID, StateID, CompareStateID> LocalToGlobalStateDict;

// A utility function for streaming LocalToGlobalStateDict to cout
inline std::ostream &
operator<<(std::ostream &os,
           const LocalToGlobalStateDict &local_to_global_state_dict) {
  for (const auto &[local_id, global_id] : local_to_global_state_dict)
    os << local_id << " --> " << global_id << std::endl;
  return os;
}

/**
 * @brief A simple struct that contains local to global state mappings.
 */
struct LocalToGlobalStateDicts {
  LocalToGlobalStateDict poses;
  LocalToGlobalStateDict landmarks;
  LocalToGlobalStateDict unit_spheres;

  // Simple default constructor; does nothing
  LocalToGlobalStateDicts() = default;

  // A utility function for streaming this struct to cout
  inline friend std::ostream &
  operator<<(std::ostream &os,
             const LocalToGlobalStateDicts &local_to_global_state_dicts) {
    os << "Local to Global State:" << std::endl;
    os << "Poses:" << std::endl;
    os << local_to_global_state_dicts.poses;
    os << "Landmarks:" << std::endl;
    os << local_to_global_state_dicts.landmarks;
    os << "Unit Spheres:" << std::endl;
    os << local_to_global_state_dicts.unit_spheres;
    return os;
  }
};

/**
 * @brief A simple struct that contains the elements of a G2O dataset.
 */
struct G2ODataset {
  unsigned int dim;       // Dimension (2 or 3)
  unsigned int num_poses; // Number of poses

  // Relative pose-pose measurements
  std::vector<RelativePosePoseMeasurement> pose_pose_measurements;

  // Ground truth (assuming g2o initialization represents ground truth)
  PoseDict ground_truth_poses;

  // Simple default constructor; does nothing
  G2ODataset() = default;

  /**
   * @brief Get ground truth poses as a PoseArray
   * @return
   */
  PoseArray getGroundTruthPoseArray() const {
    if (ground_truth_poses.empty()) {
      LOG(WARNING) << "Warning: g2o dataset ground truth poses are empty! "
                      "Returning empty PoseArray.";
      return PoseArray(3, 0);
    }

    CHECK_EQ(ground_truth_poses.size(), num_poses);
    CHECK(dim == 2 || dim == 3);
    PoseArray T(dim, num_poses);
    for (const auto &[pose_id, pose] : ground_truth_poses) {
      T.pose(pose_id.frame_id) = pose.pose();
    }

    return T;
  }

  // A utility function for streaming this struct to cout
  inline friend std::ostream &operator<<(std::ostream &os,
                                         const G2ODataset &g2o_dataset) {
    os << "G2ODataset:" << std::endl;
    os << "Dimension: " << g2o_dataset.dim << std::endl;
    os << "Number of Poses: " << g2o_dataset.num_poses << std::endl;
    os << "Relative Pose-Pose Measurements:" << std::endl;
    for (const auto &m : g2o_dataset.pose_pose_measurements) {
      os << m << std::endl;
    }
    os << "GroundTruth Poses:" << std::endl;
    os << g2o_dataset.ground_truth_poses;
    return os;
  }
};

/**
 * @brief A simple struct that contains the elements of a PyFG dataset.
 */
struct PyFGDataset {
  unsigned int dim;                 // Dimension (2 or 3)
  std::set<unsigned int> robot_IDs; // Robot IDs (includes map ID)

  // Ordered maps of robot id to number of states
  std::map<unsigned int, unsigned int> robot_id_to_num_poses;
  std::map<unsigned int, unsigned int> robot_id_to_num_landmarks;
  std::map<unsigned int, unsigned int> robot_id_to_num_unit_spheres;

  // Measurements
  Measurements measurements;

  // Ground truth
  GroundTruth ground_truth;

  // Simple default constructor; does nothing
  PyFGDataset() = default;

  // A utility function for streaming this struct to cout
  inline friend std::ostream &operator<<(std::ostream &os,
                                         const PyFGDataset &pyfg_dataset) {
    os << "PyFGDataset:" << std::endl;
    os << "Dimension: " << pyfg_dataset.dim << std::endl;
    os << "Number of Robots: " << pyfg_dataset.robot_IDs.size() << std::endl;
    os << "Number of Poses: " << std::endl;
    for (const auto &[robot_id, num_poses] :
         pyfg_dataset.robot_id_to_num_poses) {
      std::cout << "Robot ID: " << robot_id << ", Count: " << num_poses
                << std::endl;
    }
    os << "Number of Landmarks: " << std::endl;
    for (const auto &[robot_id, num_landmarks] :
         pyfg_dataset.robot_id_to_num_landmarks) {
      std::cout << "Robot ID: " << robot_id << ", Count: " << num_landmarks
                << std::endl;
    }
    os << "Number of Unit Spheres: " << std::endl;
    for (const auto &[robot_id, num_unit_spheres] :
         pyfg_dataset.robot_id_to_num_unit_spheres) {
      std::cout << "Robot ID: " << robot_id << ", Count: " << num_unit_spheres
                << std::endl;
    }
    os << "Measurements:" << std::endl;
    os << pyfg_dataset.measurements;
    os << "GroundTruth:" << std::endl;
    os << pyfg_dataset.ground_truth << std::endl;
    return os;
  }
};

} // namespace DCORA
