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

#include <Eigen/Dense>
#include <glog/logging.h>
#include <iostream>

namespace DCORA {

/**
 * @brief A simple struct that contains the elements of a single measurement for
 * (robot, state).
 */
struct Measurement {
  // measurement type
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
              StateType stateType = StateType::None, bool fixedWeight = false,
              double weight = 1.0)
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
    os << "Fixed weight: " << measurement.fixedWeight << std::endl;
    os << "Weight: " << measurement.weight << std::endl;
    os << "Robot: " << measurement.r << std::endl;
    os << "State: " << measurement.p << " ("
       << StateTypeToString(measurement.stateType) << ")" << std::endl;
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

  // Simple default constructor; does nothing
  PosePrior() = default;

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

  // Simple default constructor; does nothing
  PointPrior() = default;

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
  // measurement type
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
                      StateType stateType1 = StateType::None,
                      StateType stateType2 = StateType::None,
                      bool fixedWeight = false, double weight = 1.0)
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
    os << "Fixed weight: " << measurement.fixedWeight << std::endl;
    os << "Weight: " << measurement.weight << std::endl;
    os << "Robot1: " << measurement.r1 << std::endl;
    os << "State1: " << measurement.p1 << " ("
       << StateTypeToString(measurement.stateType1) << ")" << std::endl;
    os << "Robot2: " << measurement.r2 << std::endl;
    os << "State2: " << measurement.p2 << " ("
       << StateTypeToString(measurement.stateType2) << ")" << std::endl;
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

  // Simple default constructor; does nothing
  RelativePosePoseMeasurement() = default;

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

  // Simple default constructor; does nothing
  RelativePosePointMeasurement() = default;

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

  // Simple default constructor; does nothing
  RangeMeasurement() = default;

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

} // namespace DCORA
