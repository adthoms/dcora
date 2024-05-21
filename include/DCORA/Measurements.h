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
#include <memory>
#include <set>
#include <tuple>
#include <variant>
#include <vector>

namespace DCORA {

/**
 * @brief A simple struct that contains the elements of a single measurement for
 * (robot, state).
 */
struct Measurement {
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
  Measurement() = default;

  // Virtual destructor to enforce abstract class
  virtual ~Measurement() {}

  // Basic constructor
  Measurement(MeasurementType type, size_t robot, size_t state,
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
                                         const Measurement &measurement) {
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
struct PosePrior : Measurement {
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
      : Measurement(MeasurementType::PosePrior, 0, 0, StateType::Pose, false,
                    1.0) {}

  // Basic constructor
  PosePrior(size_t robot, size_t pose, const Matrix &priorRotation,
            const Vector &priorTranslation, double rotationalPrecision,
            double translationalPrecision, bool fixedWeight = false,
            double weight = 1.0)
      : Measurement(MeasurementType::PosePrior, robot, pose, StateType::Pose,
                    fixedWeight, weight),
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
 * @brief A simple struct that contains the elements of a single point prior for
 * (robot, point).
 */
struct PointPrior : Measurement {
  // Translational measurement
  Vector t;

  // Translational measurement precision
  double tau;

  // Default constructor
  PointPrior()
      : Measurement(MeasurementType::PointPrior, 0, 0, StateType::Point, false,
                    1.0) {}

  // Basic constructor
  PointPrior(size_t robot, size_t point, const Vector &priorTranslation,
             double translationalPrecision, bool fixedWeight = false,
             double weight = 1.0)
      : Measurement(MeasurementType::PointPrior, robot, point, StateType::Point,
                    fixedWeight, weight),
        t(priorTranslation),
        tau(translationalPrecision) {}

  void checkDim(unsigned int d) const override {
    CHECK(d == 2 || d == 3);
    CHECK(t.size() == d);
  }

  StateID getSrcID() const override { return PointID(r, p); }

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

  void print(std::ostream &os) const override {
    os << "R: " << std::endl << R << std::endl;
    os << "t: " << std::endl << t << std::endl;
    os << "Kappa: " << kappa << std::endl;
    os << "Tau: " << tau << std::endl;
  }
};

/**
 * @brief A simple struct that contains the elements of a relative pose-point
 * measurement from (robot, state) pairs: (robot1, pose1) to (robot2, point2).
 */
struct RelativePosePointMeasurement : RelativeMeasurement {
  // Translational measurement
  Vector t;

  // Translational measurement precision
  double tau;

  // Default constructor
  RelativePosePointMeasurement()
      : RelativeMeasurement(MeasurementType::PosePoint, 0, 0, 0, 0,
                            StateType::Pose, StateType::Point, false, 1.0) {}

  // Basic constructor
  RelativePosePointMeasurement(size_t firstRobot, size_t secondRobot,
                               size_t firstPose, size_t secondPoint,
                               const Vector &relativeTranslation,
                               double translationalPrecision,
                               bool fixedWeight = false, double weight = 1.0)
      : RelativeMeasurement(MeasurementType::PosePoint, firstRobot, secondRobot,
                            firstPose, secondPoint, StateType::Pose,
                            StateType::Point, fixedWeight, weight),
        t(relativeTranslation),
        tau(translationalPrecision) {}

  // Copy constructor
  RelativePosePointMeasurement(const RelativePosePointMeasurement &other)
      : RelativeMeasurement(other.measurementType, other.r1, other.r2, other.p1,
                            other.p2, other.stateType1, other.stateType2,
                            other.fixedWeight, other.weight),
        t(other.t),
        tau(other.tau) {}

  // Equality operator
  bool operator==(const RelativePosePointMeasurement &other) const {
    return std::tie(r1, r2, p1, p2, fixedWeight, weight, t, tau) ==
           std::tie(other.r1, other.r2, other.p1, other.p2, other.fixedWeight,
                    other.weight, other.t, other.tau);
  }

  void checkDim(unsigned int d) const override {
    CHECK(d == 2 || d == 3);
    CHECK(t.size() == d);
  }

  StateID getSrcID() const override { return PoseID(r1, p1); }

  StateID getDstID() const override { return PointID(r2, p2); }

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
                   size_t secondState, const double rangeMeasurement,
                   double rangePrecision, StateType stateType1,
                   StateType stateType2, bool fixedWeight = false,
                   double weight = 1.0)
      : RelativeMeasurement(MeasurementType::Range, firstRobot, secondRobot,
                            firstState, secondState, stateType1, stateType2,
                            fixedWeight, weight),
        range(rangeMeasurement),
        precision(rangePrecision) {}

  // Copy constructor
  RangeMeasurement(const RangeMeasurement &other)
      : RelativeMeasurement(other.measurementType, other.r1, other.r2, other.p1,
                            other.p2, other.stateType1, other.stateType2,
                            other.fixedWeight, other.weight),
        range(other.range),
        precision(other.precision) {}

  // Equality operator
  bool operator==(const RangeMeasurement &other) const {
    return std::tie(r1, r2, p1, p2, stateType1, stateType2, fixedWeight, weight,
                    range, precision) ==
           std::tie(other.r1, other.r2, other.p1, other.p2, other.stateType1,
                    other.stateType2, other.fixedWeight, other.weight,
                    other.range, other.precision);
  }

  void checkDim(unsigned int d) const override { CHECK(d == 2 || d == 3); }

  StateID getSrcID() const override {
    if (stateType1 == StateType::Pose)
      return PoseID(r1, p1);
    else if (stateType1 == StateType::Point)
      return PointID(r1, p1);
    else
      LOG(FATAL) << "StateType1 is " << StateTypeToString(stateType1) << " in "
                 << MeasurementTypeToString(measurementType) << "!";
  }

  StateID getDstID() const override {
    if (stateType2 == StateType::Pose)
      return PoseID(r2, p2);
    else if (stateType2 == StateType::Point)
      return PointID(r2, p2);
    else
      LOG(FATAL) << "StateType2 is " << StateTypeToString(stateType2) << " in "
                 << MeasurementTypeToString(measurementType) << "!";
  }

  void print(std::ostream &os) const override {
    os << "range: " << range << std::endl;
    os << "precision: " << precision << std::endl;
  }
};

/**
 * @brief A class that contains all relative measurements within a common
 * vector.
 */
class RelativeMeasurements {
public:
  using RelativeMeasurementVariant =
      std::variant<RelativePosePoseMeasurement, RelativePosePointMeasurement,
                   RangeMeasurement>;

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

  // Validators
  bool isPGOCompatible() const {
    for (const auto &m : vec) {
      if (std::holds_alternative<RelativePosePointMeasurement>(m) ||
          std::holds_alternative<RangeMeasurement>(m))
        return false;
    }
    return true;
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

  std::vector<RelativePosePointMeasurement>
  GetRelativePosePointMeasurements() const {
    std::vector<RelativePosePointMeasurement> pose_point_measurements;
    for (const auto &m : vec) {
      if (std::holds_alternative<RelativePosePointMeasurement>(m))
        pose_point_measurements.emplace_back(
            std::get<RelativePosePointMeasurement>(m));
    }
    return pose_point_measurements;
  }

  std::vector<RangeMeasurement> GetRangeMeasurements() const {
    std::vector<RangeMeasurement> range_measurements;
    for (const auto &m : vec) {
      if (std::holds_alternative<RangeMeasurement>(m))
        range_measurements.emplace_back(std::get<RangeMeasurement>(m));
    }
    return range_measurements;
  }

  void push_back(const RelativeMeasurement &relative_measurement) {
    switch (relative_measurement.measurementType) {
    case MeasurementType::PosePose:
      addPosePoseMeasurement(relative_measurement);
      break;
    case MeasurementType::PosePoint:
      addPosePointMeasurement(relative_measurement);
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
      std::visit([&os](const auto &arg) { os << arg; }, m);
    }
    return os;
  }

  // Vector of relative measurements
  std::vector<RelativeMeasurementVariant> vec;

private:
  /**
   * @brief Template function to cast a Relative Measurement object to a
   * specific type T, such as RelativePosePoseMeasurement,
   * RelativePosePointMeasurement, or RangeMeasurement.
   * @tparam T
   * @param relative_measurement
   * @return
   */
  template <typename T>
  T castRelativeMeasurement(const RelativeMeasurement &relative_measurement) {
    if constexpr (std::is_same_v<T, RelativePosePoseMeasurement>)
      return dynamic_cast<const T &>(relative_measurement);
    else if constexpr (std::is_same_v<T, RelativePosePointMeasurement>)
      return dynamic_cast<const T &>(relative_measurement);
    else if constexpr (std::is_same_v<T, RangeMeasurement>)
      return dynamic_cast<const T &>(relative_measurement);
    else
      LOG(FATAL) << "Error: cannot cast relative measurement: "
                 << relative_measurement << "!";

    return T{};
  }

  /**
   * Add a relative pose pose measurement.
   * @param relative_measurement
   */
  void addPosePoseMeasurement(const RelativeMeasurement &relative_measurement) {
    const RelativePosePoseMeasurement &pose_pose_measurement =
        castRelativeMeasurement<RelativePosePoseMeasurement>(
            relative_measurement);
    vec.push_back(pose_pose_measurement);
  }

  /**
   * Add a relative pose point measurement.
   * @param relative_measurement
   */
  void
  addPosePointMeasurement(const RelativeMeasurement &relative_measurement) {
    const RelativePosePointMeasurement &pose_point_measurement =
        castRelativeMeasurement<RelativePosePointMeasurement>(
            relative_measurement);
    vec.push_back(pose_point_measurement);
  }

  /**
   * Add a range measurement.
   * @param relative_measurement
   */
  void addRangeMeasurement(const RelativeMeasurement &relative_measurement) {
    const RangeMeasurement &range_measurement =
        castRelativeMeasurement<RangeMeasurement>(relative_measurement);
    vec.push_back(range_measurement);
  }
};

/**
 * @brief A simple struct that contains the elements of a PyFG dataset.
 */
struct PyFGDataset {
  unsigned int dim;                 // Problem dimension
  size_t num_poses;                 // Number of poses
  size_t num_points;                // Number of points
  std::set<unsigned int> robot_IDs; // Robot IDs (includes map ID)

  // Measurements
  std::vector<PosePrior> pose_priors;
  std::vector<PointPrior> point_priors;
  RelativeMeasurements relative_measurements;

  // Ground truth poses
  std::vector<unsigned int> ground_truth_pose_robot_ids;
  std::vector<unsigned int> ground_truth_pose_state_ids;
  std::shared_ptr<PoseArray> ground_truth_pose_array;

  // Ground truth points
  std::vector<unsigned int> ground_truth_point_robot_ids;
  std::vector<unsigned int> ground_truth_point_state_ids;
  std::shared_ptr<PointArray> ground_truth_point_array;

  // Simple default constructor; does nothing
  PyFGDataset() = default;

  // A utility function for streaming this struct to cout
  inline friend std::ostream &operator<<(std::ostream &os,
                                         const PyFGDataset &pyfg_dataset) {
    os << "PyFGDataset:" << std::endl;
    os << "Dimension: " << pyfg_dataset.dim << std::endl;
    os << "NumPoses: " << pyfg_dataset.num_poses << std::endl;
    os << "NumPoints: " << pyfg_dataset.num_points << std::endl;
    os << "NumRobots: " << pyfg_dataset.robot_IDs.size() - 1 << std::endl;
    os << "Measurements:" << std::endl;
    os << "Priors:" << std::endl;
    for (const auto &pose_prior : pyfg_dataset.pose_priors) {
      os << pose_prior;
    }
    for (const auto &point_prior : pyfg_dataset.point_priors) {
      os << point_prior;
    }
    os << "Relative Measurements:" << std::endl;
    os << pyfg_dataset.relative_measurements << std::endl;
    os << "GroundTruth:" << std::endl;
    os << "Poses:" << std::endl;
    for (size_t i = 0; i < pyfg_dataset.ground_truth_pose_array->n(); i++) {
      os << "r: " << pyfg_dataset.ground_truth_pose_robot_ids.at(i)
         << std::endl;
      os << "p: " << pyfg_dataset.ground_truth_pose_state_ids.at(i)
         << std::endl;
      os << "T: \n"
         << pyfg_dataset.ground_truth_pose_array->pose(i) << std::endl;
    }
    os << "Points:" << std::endl;
    for (size_t i = 0; i < pyfg_dataset.ground_truth_point_array->n(); i++) {
      os << "r: " << pyfg_dataset.ground_truth_point_robot_ids.at(i)
         << std::endl;
      os << "p: " << pyfg_dataset.ground_truth_point_state_ids.at(i)
         << std::endl;
      os << "t: \n"
         << pyfg_dataset.ground_truth_point_array->translation(i) << std::endl;
    }
    return os;
  }
};

} // namespace DCORA
