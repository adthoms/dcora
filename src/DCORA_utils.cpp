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

#include <DCORA/DCORA_robust.h>
#include <DCORA/DCORA_utils.h>

#include <Eigen/Geometry>
#include <Eigen/SPQRSupport>
#include <glog/logging.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>

#include <boost/math/distributions/chi_squared.hpp>

namespace DCORA {

std::string
ROptParameters::ROptMethodToString(ROptParameters::ROptMethod method) {
  switch (method) {
  case ROptParameters::ROptMethod::RTR: {
    return "RTR";
  }
  case ROptParameters::ROptMethod::RGD: {
    return "RGD";
  }
  }
  return "";
}

std::string InitializationMethodToString(InitializationMethod method) {
  switch (method) {
  case InitializationMethod::Odometry: {
    return "Odometry";
  }
  case InitializationMethod::Chordal: {
    return "Chordal";
  }
  case InitializationMethod::GNC_TLS: {
    return "GNC_TLS";
  }
  }
  return "";
}

std::string StateTypeToString(const StateType &type) {
  switch (type) {
  case StateType::None: {
    return "None";
  }
  case StateType::Pose: {
    return "Pose";
  }
  case StateType::Point: {
    return "Point";
  }
  case StateType::UnitSphere: {
    return "UnitSphere";
  }
  }
  return "";
}

std::string MeasurementTypeToString(const MeasurementType &type) {
  switch (type) {
  case MeasurementType::PosePrior: {
    return "PosePrior";
  }
  case MeasurementType::PointPrior: {
    return "PointPrior";
  }
  case MeasurementType::PosePose: {
    return "PosePose";
  }
  case MeasurementType::PosePoint: {
    return "PosePoint";
  }
  case MeasurementType::Range: {
    return "Range";
  }
  }
  return "";
}

void SimpleTimer::tic() { t_start = std::chrono::high_resolution_clock::now(); }

double SimpleTimer::toc() {
  t_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> t_elapsed(0);
  t_elapsed = t_end - t_start;
  return t_elapsed.count();
}

HighResClock SimpleTimer::Tic() {
  return std::chrono::high_resolution_clock::now();
}

double SimpleTimer::Toc(const HighResClock &start_time) {
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> t_elapsed(0);
  t_elapsed = end_time - start_time;
  return t_elapsed.count();
}

void writeMatrixToFile(const Matrix &M, const std::string &filename) {
  std::ofstream file;
  file.open(filename);
  if (!file.is_open()) {
    printf("Cannot write to specified file: %s\n", filename.c_str());
    return;
  }
  static const Eigen::IOFormat CSVFormat(Eigen::FullPrecision,
                                         Eigen::DontAlignCols, ", ", "\n");
  file << M.format(CSVFormat);
  file.close();
}

void writeSparseMatrixToFile(const SparseMatrix &M,
                             const std::string &filename) {
  std::ofstream file;
  file.open(filename);
  if (!file.is_open()) {
    printf("Cannot write to specified file: %s\n", filename.c_str());
    return;
  }

  for (int k = 0; k < M.outerSize(); ++k) {
    for (SparseMatrix::InnerIterator it(M, k); it; ++it) {
      file << it.row() << ",";
      file << it.col() << ",";
      file << it.value() << "\n";
    }
  }
  file.close();
}

std::vector<RelativePosePoseMeasurement>
read_g2o_file(const std::string &filename, size_t *num_poses) {
  /*
  The following implementation is adapted from:
  SE-Sync: https://github.com/david-m-rosen/SE-Sync.git
  Cartan-Sync: https://bitbucket.org/jesusbriales/cartan-sync/src
  */

  // Preallocate output vector
  std::vector<DCORA::RelativePosePoseMeasurement> measurements;

  // A single measurement, whose values we will fill in
  DCORA::RelativePosePoseMeasurement measurement;
  measurement.weight = 1.0;

  // A string used to contain the contents of a single line
  std::string line;

  // A string used to extract tokens from each line one-by-one
  std::string token;

  // Preallocate various useful quantities
  double dx, dy, dz, dtheta, dqx, dqy, dqz, dqw, I11, I12, I13, I14, I15, I16,
      I22, I23, I24, I25, I26, I33, I34, I35, I36, I44, I45, I46, I55, I56, I66;

  size_t i, j;

  // Open the file for reading
  std::ifstream infile(filename);

  *num_poses = 0;

  while (std::getline(infile, line)) {
    // Construct a stream from the string
    std::stringstream strstrm(line);

    // Extract the first token from the string
    strstrm >> token;

    if (token == "EDGE_SE2") {
      // This is a 2D pose measurement

      /** The g2o format specifies a 2D relative pose measurement in the
       * following form:
       *
       * EDGE_SE2 id1 id2 dx dy dtheta, I11, I12, I13, I22, I23, I33
       *
       */

      // Extract formatted output
      strstrm >> i >> j >> dx >> dy >> dtheta >> I11 >> I12 >> I13 >> I22 >>
          I23 >> I33;

      // Fill in elements of this measurement

      // Pose ids
      measurement.r1 = 0;
      measurement.r2 = 0;
      measurement.p1 = i;
      measurement.p2 = j;

      // Raw measurements
      measurement.t = Eigen::Matrix<double, 2, 1>(dx, dy);
      measurement.R = Eigen::Rotation2Dd(dtheta).toRotationMatrix();

      Eigen::Matrix2d TranCov;
      TranCov << I11, I12, I12, I22;
      measurement.tau = 2 / TranCov.inverse().trace();

      measurement.kappa = I33;

      if (i + 1 == j) {
        measurement.fixedWeight = true;
      } else {
        measurement.fixedWeight = false;
      }

    } else if (token == "EDGE_SE3:QUAT") {
      // This is a 3D pose measurement

      /** The g2o format specifies a 3D relative pose measurement in the
       * following form:
       *
       * EDGE_SE3:QUAT id1, id2, dx, dy, dz, dqx, dqy, dqz, dqw
       *
       * I11 I12 I13 I14 I15 I16
       *     I22 I23 I24 I25 I26
       *         I33 I34 I35 I36
       *             I44 I45 I46
       *                 I55 I56
       *                     I66
       */

      // Extract formatted output
      strstrm >> i >> j >> dx >> dy >> dz >> dqx >> dqy >> dqz >> dqw >> I11 >>
          I12 >> I13 >> I14 >> I15 >> I16 >> I22 >> I23 >> I24 >> I25 >> I26 >>
          I33 >> I34 >> I35 >> I36 >> I44 >> I45 >> I46 >> I55 >> I56 >> I66;

      // Fill in elements of the measurement

      // Pose ids
      measurement.r1 = 0;
      measurement.r2 = 0;
      measurement.p1 = i;
      measurement.p2 = j;

      // Raw measurements
      measurement.t = Eigen::Matrix<double, 3, 1>(dx, dy, dz);
      measurement.R = Eigen::Quaterniond(dqw, dqx, dqy, dqz).toRotationMatrix();

      // Compute precisions

      // Compute and store the optimal (information-divergence-minimizing) value
      // of the parameter tau
      Eigen::Matrix3d TranCov;
      TranCov << I11, I12, I13, I12, I22, I23, I13, I23, I33;
      measurement.tau = 3 / TranCov.inverse().trace();

      // Compute and store the optimal (information-divergence-minimizing value
      // of the parameter kappa

      Eigen::Matrix3d RotCov;
      RotCov << I44, I45, I46, I45, I55, I56, I46, I56, I66;
      measurement.kappa = 3 / (2 * RotCov.inverse().trace());

      if (i + 1 == j) {
        measurement.fixedWeight = true;
      } else {
        measurement.fixedWeight = false;
      }

    } else if ((token == "VERTEX_SE2") || (token == "VERTEX_SE3:QUAT")) {
      // This is just initialization information, so do nothing
      continue;
    } else {
      LOG(FATAL) << "Error: unrecognized type: " << token << "!";
    }

    // Update maximum value of poses found so far
    size_t max_pair = std::max<double>(measurement.p1, measurement.p2);

    *num_poses = ((max_pair > *num_poses) ? max_pair : *num_poses);
    measurements.push_back(measurement);
  } // while

  infile.close();

  (*num_poses)++; // Account for the use of zero-based indexing

  return measurements;
}

int getDimFromPyfgFirstLine(const std::string &filename) {
  /*
  The following implementation is adapted from:
  CORA: https://github.com/MarineRoboticsGroup/cora
  */

  // Check if the file exists and if it can be read
  std::ifstream in_file(filename);
  if (!in_file.good()) {
    LOG(FATAL) << "Error: could not open file: " << filename << "!";
  }

  // Get the first line and close the file
  std::string line;
  std::getline(in_file, line);
  in_file.close();

  // Get the item type with the first word
  std::istringstream strstrm(line);
  std::string item_type;

  if (!(strstrm >> item_type)) {
    LOG(FATAL) << "Error: could not read item type from line: " << line << "!";
  }
  if (PyFGStringToType.find(item_type) == PyFGStringToType.end()) {
    LOG(FATAL) << "Error: unknown item type: " << item_type << "!";
  }

  switch (PyFGStringToType.find(item_type)->second) {
  case POSE_TYPE_2D:
    return 2;
  case POSE_TYPE_3D:
    return 3;
  case LANDMARK_TYPE_2D:
    return 2;
  case LANDMARK_TYPE_3D:
    return 3;
  case POSE_PRIOR_2D:
    return 2;
  case POSE_PRIOR_3D:
    return 3;
  case LANDMARK_PRIOR_2D:
    return 2;
  case LANDMARK_PRIOR_3D:
    return 3;
  case REL_POSE_POSE_TYPE_2D:
    return 2;
  case REL_POSE_POSE_TYPE_3D:
    return 3;
  case REL_POSE_LANDMARK_TYPE_2D:
    return 2;
  case REL_POSE_LANDMARK_TYPE_3D:
    return 3;
  default:
    LOG(FATAL) << "Error: the first line of the PyFG file is of type: "
               << item_type << "\n"
               << "This PyFG type` does not provide dimension information!";
  }
}

PyFGDataset read_pyfg_file(const std::string &filename) {
  /*
  The following implementation is adapted from:
  CORA: https://github.com/MarineRoboticsGroup/cora
  */

  // Define helper lambda functions

  auto readScalar = [](std::istringstream &strstrm) -> double {
    double result;
    if (strstrm >> result) {
      return result;
    } else {
      LOG(FATAL) << "Error: could not read scalar from string stream: "
                 << strstrm.str() << "!";
    }
  };

  auto readVector = [](std::istringstream &strstrm, int dim) -> Vector {
    Vector result(dim);
    for (int i = 0; i < dim; i++) {
      if (strstrm >> result(i)) {
        continue;
      } else {
        LOG(FATAL) << "Error: could not read vector from string stream: "
                   << strstrm.str() << "!";
      }
    }
    return result;
  };

  auto readThetaAsRotation =
      [readScalar](std::istringstream &strstrm) -> Matrix {
    const double theta = readScalar(strstrm);
    return Eigen::Rotation2Dd(theta).toRotationMatrix();
  };

  auto readQuatAsRotation =
      [readVector](std::istringstream &strstrm) -> Matrix {
    const Vector quat = readVector(strstrm, 4);
    return Eigen::Quaterniond(quat(3), quat(0), quat(1), quat(2))
        .toRotationMatrix();
  };

  auto readScalarAsRange = [readScalar](std::istringstream &strstrm) -> double {
    const double range = readScalar(strstrm);
    CHECK_GT(range, 0.0)
        << "Error: range measurement must be greater than zero: " << range
        << "!";
    return range;
  };

  auto readSymmetric = [](std::istringstream &strstrm, int dim) -> Matrix {
    Matrix cov(dim, dim);
    double val;
    for (int i = 0; i < dim; i++) {
      for (int j = i; j < dim; j++) {
        if (strstrm >> val) {
          cov(i, j) = val;
          cov(j, i) = val;
        } else {
          LOG(WARNING) << "Warning: attempted to parse covariance matrix. i: "
                       << i << " j:" << j << " val:" << val;
          LOG(FATAL)
              << "Error: could not read covariance matrix from string stream: "
              << strstrm.str() << "!";
        }
      }
    }
    return cov;
  };

  auto getTranslationAndRotationCov =
      [](const Matrix &cov) -> std::pair<Matrix, Matrix> {
    CHECK_EQ(cov.rows(), cov.cols())
        << "Error: Covariance matrix is not square: \n"
        << cov << "!";
    Matrix cov_t;
    Matrix cov_R;
    if (cov.rows() == 3) {
      // 2D case
      cov_t = cov.block<2, 2>(0, 0);
      cov_R = cov.block<1, 1>(2, 2);
    } else if (cov.rows() == 6) {
      // 3D case
      cov_t = cov.block<3, 3>(0, 0);
      cov_R = cov.block<3, 3>(3, 3);
    } else {
      LOG(FATAL) << "Error: could not get translation and rotation "
                    "covariance matrices from covariance matrix: \n"
                 << cov << "!";
    }
    return std::make_pair(cov_t, cov_R);
  };

  auto getTau = [](const Matrix &cov) -> double {
    CHECK_EQ(cov.rows(), cov.cols())
        << "Error: Translation covariance matrix is not square: \n"
        << cov << "!";
    double tau;
    if (cov.rows() == 2) {
      // 2D case
      tau = 2 / cov.trace();
    } else if (cov.rows() == 3) {
      // 3D case
      tau = 3 / cov.trace();
    } else {
      LOG(FATAL) << "Error: could not get Tau value from translation "
                    "covariance matrix: \n"
                 << cov << "!";
    }
    return tau;
  };

  auto getKappa = [](const Matrix &cov) -> double {
    CHECK_EQ(cov.rows(), cov.cols())
        << "Error: Rotation covariance matrix is not square: \n"
        << cov << "!";
    double kappa;
    if (cov.rows() == 1) {
      // 2D case
      kappa = 1.0 / cov(0, 0);
    } else if (cov.rows() == 3) {
      // 3D case
      kappa = 3 / (2 * cov.trace());
    } else {
      LOG(FATAL) << "Error: could not get Kappa value from rotation covariance "
                    "matrix: \n"
                 << cov << "!";
    }
    return kappa;
  };

  auto getRobotAndStateIDFromSymbol =
      [](const std::string &sym) -> std::pair<unsigned int, unsigned int> {
    unsigned int robotID;
    unsigned int stateID;
    if (sym[0] == 'L') {
      // Symbol is a point
      if (std::isupper(sym[1])) {
        if (sym[1] == 'M') {
          // Point is associated with the map, though does not obey PyFG
          // formatting.
          LOG(WARNING)
              << "Warning: point symbol 'LM#' is (by default) associated with "
                 "the map. Map point features should be formatted as 'L#'.";
        }
        // Point is associated with a robot according to PyFG formatting
        robotID = static_cast<unsigned int>(sym[1] - 'A');
        stateID = std::stoi(sym.substr(2));
      } else {
        // Point is associated with the map
        robotID = static_cast<unsigned int>('M' - 'A');
        stateID = std::stoi(sym.substr(1));
      }
    } else if (std::isupper(sym[0])) {
      // Symbol is a pose
      robotID = static_cast<unsigned int>(sym[0] - 'A');
      stateID = std::stoi(sym.substr(1));
    } else {
      LOG(FATAL) << "Error: could not read robot and state ID from symbol: "
                 << sym << "!";
    }
    return std::make_pair(robotID, stateID);
  };

  auto getStateTypeFromSymbol = [](const std::string &sym) -> StateType {
    if (sym[0] == 'L')
      return StateType::Point;
    else if (std::isupper(sym[0]))
      return StateType::Pose;
    else
      LOG(FATAL) << "Error: could not read state type from symbol: " << sym
                 << "!";
  };

  auto getTransformFromRotationAndTranslation = [](const Matrix &R,
                                                   const Vector &t) -> Matrix {
    CHECK_EQ(R.rows(), R.cols());
    CHECK_EQ(R.rows(), t.size());

    // Get transform as T = [R | t]
    unsigned int d = R.rows();
    Matrix T = Matrix::Zero(d, d + 1);
    T.block(0, 0, d, d) = R;
    T.block(0, d, d, 1) = t;
    return T;
  };

  // Initialize PyFG dataset
  PyFGDataset pyfg_dataset;

  // Get dimension of PyFG file
  pyfg_dataset.dim = getDimFromPyfgFirstLine(filename);

  // Initialize measurements, whose values we will fill in
  PosePrior pose_prior;
  PointPrior point_prior;
  RelativePosePoseMeasurement pose_pose_measurement;
  RelativePosePointMeasurement pose_point_measurement;
  RangeMeasurement range_measurement;

  // Initialize map for indexing unit spheres according to robot ID
  std::map<unsigned int, unsigned int> robot_id_to_unit_sphere_idx = {};

  // Initialize map to maintain unique range edges
  EdgeIDMap range_edge_id_to_index;
  size_t range_edge_index = 0;

  // A string used to contain the contents of a single line
  std::string line;

  // Open the file for reading
  std::ifstream infile(filename);

  while (std::getline(infile, line)) {
    // Construct a stream from the string
    std::istringstream strstrm(line);

    // Tokens for timestamp and state symbols
    double timestamp;
    std::string sym1, sym2;

    // Get the item type with the first word
    std::string item_type;
    if (!(strstrm >> item_type)) {
      LOG(FATAL) << "Error: could not read item type from line: " << line
                 << "!";
    }
    if (PyFGStringToType.find(item_type) == PyFGStringToType.end()) {
      LOG(FATAL) << "Error: unknown item type: " << item_type << "!";
    }

    switch (PyFGStringToType.find(item_type)->second) {
    case POSE_TYPE_2D:
      // VERTEX_SE2 ts sym x y theta
      if (strstrm >> timestamp >> sym1) {
        // Read string stream
        const Vector t = readVector(strstrm, 2);
        const Matrix R = readThetaAsRotation(strstrm);

        // Parse symbol
        const auto [robotID, stateID] = getRobotAndStateIDFromSymbol(sym1);
        pyfg_dataset.robot_IDs.emplace(robotID);

        // Populate ground truth
        const PoseID pose_id = PoseID(robotID, stateID);
        const Matrix T = getTransformFromRotationAndTranslation(R, t);
        const Pose gt_pose = Pose(T);
        if (pyfg_dataset.ground_truth.poses.find(pose_id) !=
            pyfg_dataset.ground_truth.poses.end()) {
          LOG(FATAL) << "Error: duplicate pose ID: " << pose_id << "!";
        }
        pyfg_dataset.ground_truth.poses[pose_id] = gt_pose;

        // Increment number of poses for this robot
        pyfg_dataset.robot_id_to_num_poses[robotID]++;

      } else {
        LOG(FATAL) << "Error: could not read pose variable from line: " << line
                   << "!";
      }
      break;
    case POSE_TYPE_3D:
      // VERTEX_SE3:QUAT ts sym x y z qx qy qz qw
      if (strstrm >> timestamp >> sym1) {
        // Read string stream
        const Vector t = readVector(strstrm, 3);
        const Matrix R = readQuatAsRotation(strstrm);

        // Parse symbol
        const auto [robotID, stateID] = getRobotAndStateIDFromSymbol(sym1);
        pyfg_dataset.robot_IDs.emplace(robotID);

        // Populate ground truth
        const PoseID pose_id = PoseID(robotID, stateID);
        const Matrix T = getTransformFromRotationAndTranslation(R, t);
        const Pose gt_pose = Pose(T);
        if (pyfg_dataset.ground_truth.poses.find(pose_id) !=
            pyfg_dataset.ground_truth.poses.end()) {
          LOG(FATAL) << "Error: duplicate pose ID: " << pose_id << "!";
        }
        pyfg_dataset.ground_truth.poses[pose_id] = gt_pose;

        // Increment number of poses for this robot
        pyfg_dataset.robot_id_to_num_poses[robotID]++;

      } else {
        LOG(FATAL) << "Error: could not read pose variable from line: " << line
                   << "!";
      }
      break;
    case POSE_PRIOR_2D:
      // VERTEX_SE2:PRIOR ts sym x y theta cov_ij
      // for i in [1,3] and j in [i,3]
      if (strstrm >> timestamp >> sym1) {
        // Read string stream
        const Vector t = readVector(strstrm, 2);
        const Matrix R = readThetaAsRotation(strstrm);
        const Matrix cov = readSymmetric(strstrm, 3);

        // Parse symbol and covariance
        const auto [robotID, stateID] = getRobotAndStateIDFromSymbol(sym1);
        const auto [cov_t, cov_R] = getTranslationAndRotationCov(cov);

        // Fill in measurement
        pose_prior.r = robotID;
        pose_prior.p = stateID;
        pose_prior.R = R;
        pose_prior.t = t;
        pose_prior.kappa = getKappa(cov_R);
        pose_prior.tau = getTau(cov_t);

        // Add measurement
        pyfg_dataset.measurements.pose_priors.push_back(pose_prior);
      } else {
        LOG(FATAL) << "Error: could not read pose prior from line: " << line
                   << "!";
      }
      break;
    case POSE_PRIOR_3D:
      // VERTEX_SE3:QUAT:PRIOR ts sym x y z qx qy qz qw cov_ij
      // for i in [1,6] and j in [i,6]
      if (strstrm >> timestamp >> sym1) {
        // Read string stream
        const Vector t = readVector(strstrm, 3);
        const Matrix R = readQuatAsRotation(strstrm);
        const Matrix cov = readSymmetric(strstrm, 6);

        // Parse symbol and covariance
        const auto [robotID, stateID] = getRobotAndStateIDFromSymbol(sym1);
        const auto [cov_t, cov_R] = getTranslationAndRotationCov(cov);

        // Fill in measurement
        pose_prior.r = robotID;
        pose_prior.p = stateID;
        pose_prior.R = R;
        pose_prior.t = t;
        pose_prior.kappa = getKappa(cov_R);
        pose_prior.tau = getTau(cov_t);

        // Add measurement
        pyfg_dataset.measurements.pose_priors.push_back(pose_prior);
      } else {
        LOG(FATAL) << "Error: could not read pose prior from line: " << line
                   << "!";
      }
      break;
    case LANDMARK_TYPE_2D:
      // VERTEX_XY sym x y
      if (strstrm >> sym1) {
        // Read string stream
        const Vector t = readVector(strstrm, 2);

        // Parse symbol
        const auto [robotID, stateID] = getRobotAndStateIDFromSymbol(sym1);
        pyfg_dataset.robot_IDs.emplace(robotID);

        // Populate ground truth
        const PointID landmark_id = PointID(robotID, stateID);
        const Point gt_landmark = Point(t);
        if (pyfg_dataset.ground_truth.landmarks.find(landmark_id) !=
            pyfg_dataset.ground_truth.landmarks.end()) {
          LOG(FATAL) << "Error: duplicate landmark ID: " << landmark_id << "!";
        }
        pyfg_dataset.ground_truth.landmarks[landmark_id] = gt_landmark;

        // Increment number of landmarks for this robot
        pyfg_dataset.robot_id_to_num_landmarks[robotID]++;

      } else {
        LOG(FATAL) << "Error: could not read point variable from line: " << line
                   << "!";
      }
      break;
    case LANDMARK_TYPE_3D:
      // VERTEX_XYZ sym x y z
      if (strstrm >> sym1) {
        // Read string stream
        const Vector t = readVector(strstrm, 3);

        // Parse symbol
        const auto [robotID, stateID] = getRobotAndStateIDFromSymbol(sym1);
        pyfg_dataset.robot_IDs.emplace(robotID);

        // Populate ground truth
        const PointID landmark_id = PointID(robotID, stateID);
        const Point gt_landmark = Point(t);
        if (pyfg_dataset.ground_truth.landmarks.find(landmark_id) !=
            pyfg_dataset.ground_truth.landmarks.end()) {
          LOG(FATAL) << "Error: duplicate landmark ID: " << landmark_id << "!";
        }
        pyfg_dataset.ground_truth.landmarks[landmark_id] = gt_landmark;

        // Increment number of landmarks for this robot
        pyfg_dataset.robot_id_to_num_landmarks[robotID]++;

      } else {
        LOG(FATAL) << "Error: could not read point variable from line: " << line
                   << "!";
      }
      break;
    case LANDMARK_PRIOR_2D:
      // VERTEX_XY:PRIOR ts sym x y cov_11 cov_12 cov_22
      if (strstrm >> timestamp >> sym1) {
        // Read string stream
        const Vector t = readVector(strstrm, 2);
        const Matrix cov = readSymmetric(strstrm, 2);

        // Parse symbol
        const auto [robotID, stateID] = getRobotAndStateIDFromSymbol(sym1);

        // Fill in measurement
        point_prior.r = robotID;
        point_prior.p = stateID;
        point_prior.t = t;
        point_prior.tau = getTau(cov);

        // Add measurement
        pyfg_dataset.measurements.point_priors.push_back(point_prior);
      } else {
        LOG(FATAL) << "Error: could not read point prior from line: " << line
                   << "!";
      }
      break;
    case LANDMARK_PRIOR_3D:
      // VERTEX_XYZ:PRIOR ts sym x y z cov_11 cov_12 cov_13 cov_22 cov_23 cov_33
      if (strstrm >> timestamp >> sym1) {
        // Read string stream
        const Vector t = readVector(strstrm, 3);
        const Matrix cov = readSymmetric(strstrm, 3);

        // Parse symbol
        const auto [robotID, stateID] = getRobotAndStateIDFromSymbol(sym1);

        // Fill in measurement
        point_prior.r = robotID;
        point_prior.p = stateID;
        point_prior.t = t;
        point_prior.tau = getTau(cov);

        // Add measurement
        pyfg_dataset.measurements.point_priors.push_back(point_prior);
      } else {
        LOG(FATAL) << "Error: could not read point prior from line: " << line
                   << "!";
      }
      break;
    case REL_POSE_POSE_TYPE_2D:
      // EDGE_SE2 ts sym1 sym2 x y theta cov_ij
      // for i in [1,3] and j in [i,3]
      if (strstrm >> timestamp >> sym1 >> sym2) {
        // Read string stream
        const Vector t = readVector(strstrm, 2);
        const Matrix R = readThetaAsRotation(strstrm);
        const Matrix cov = readSymmetric(strstrm, 3);

        // Parse symbols and covariance
        const auto [robot1ID, state1ID] = getRobotAndStateIDFromSymbol(sym1);
        const auto [robot2ID, state2ID] = getRobotAndStateIDFromSymbol(sym2);
        const auto [cov_t, cov_R] = getTranslationAndRotationCov(cov);

        // Fill in measurement
        pose_pose_measurement.r1 = robot1ID;
        pose_pose_measurement.p1 = state1ID;
        pose_pose_measurement.r2 = robot2ID;
        pose_pose_measurement.p2 = state2ID;
        pose_pose_measurement.R = R;
        pose_pose_measurement.t = t;
        pose_pose_measurement.kappa = getKappa(cov_R);
        pose_pose_measurement.tau = getTau(cov_t);

        // Add measurement
        pyfg_dataset.measurements.relative_measurements.vec.push_back(
            pose_pose_measurement);
      } else {
        LOG(FATAL)
            << "Error: could not read relative pose measurement from line: "
            << line << "!";
      }
      break;
    case REL_POSE_POSE_TYPE_3D:
      // EDGE_SE3:QUAT ts sym1 sym2 x y z qx qy qz qw cov_ij
      // for i in [1,6] and j in [i,6]
      if (strstrm >> timestamp >> sym1 >> sym2) {
        // Read string stream
        const Vector t = readVector(strstrm, 3);
        const Matrix R = readQuatAsRotation(strstrm);
        const Matrix cov = readSymmetric(strstrm, 6);

        // Parse symbols and covariance
        const auto [robot1ID, state1ID] = getRobotAndStateIDFromSymbol(sym1);
        const auto [robot2ID, state2ID] = getRobotAndStateIDFromSymbol(sym2);
        const auto [cov_t, cov_R] = getTranslationAndRotationCov(cov);

        // Fill in measurement
        pose_pose_measurement.r1 = robot1ID;
        pose_pose_measurement.p1 = state1ID;
        pose_pose_measurement.r2 = robot2ID;
        pose_pose_measurement.p2 = state2ID;
        pose_pose_measurement.R = R;
        pose_pose_measurement.t = t;
        pose_pose_measurement.kappa = getKappa(cov_R);
        pose_pose_measurement.tau = getTau(cov_t);

        // Add measurement
        pyfg_dataset.measurements.relative_measurements.vec.push_back(
            pose_pose_measurement);
      } else {
        LOG(FATAL)
            << "Error: could not read relative pose measurement from line: "
            << line << "!";
      }
      break;
    case REL_POSE_LANDMARK_TYPE_2D:
      // EDGE_SE2_XY ts sym1 sym2 x y cov_11 cov_12 cov_22
      if (strstrm >> timestamp >> sym1 >> sym2) {
        // Read string stream
        const Vector t = readVector(strstrm, 2);
        const Matrix cov = readSymmetric(strstrm, 2);

        // Parse symbols
        const auto [robot1ID, state1ID] = getRobotAndStateIDFromSymbol(sym1);
        const auto [robot2ID, state2ID] = getRobotAndStateIDFromSymbol(sym2);

        // Fill in measurement
        pose_point_measurement.r1 = robot1ID;
        pose_point_measurement.p1 = state1ID;
        pose_point_measurement.r2 = robot2ID;
        pose_point_measurement.p2 = state2ID;
        pose_point_measurement.t = t;
        pose_point_measurement.tau = getTau(cov);

        // Add measurement
        pyfg_dataset.measurements.relative_measurements.vec.push_back(
            pose_point_measurement);
      } else {
        LOG(FATAL) << "Error: could not read relative pose-point measurement "
                      "from line: "
                   << line << "!";
      }
      break;
    case REL_POSE_LANDMARK_TYPE_3D:
      // EDGE_SE3_XYZ ts sym1 sym2 x y z cov_ij
      // for i in [1,3] and j in [i,3]
      if (strstrm >> timestamp >> sym1 >> sym2) {
        // Read string stream
        const Vector t = readVector(strstrm, 3);
        const Matrix cov = readSymmetric(strstrm, 3);

        // Parse symbols
        const auto [robot1ID, state1ID] = getRobotAndStateIDFromSymbol(sym1);
        const auto [robot2ID, state2ID] = getRobotAndStateIDFromSymbol(sym2);

        // Fill in measurement
        pose_point_measurement.r1 = robot1ID;
        pose_point_measurement.p1 = state1ID;
        pose_point_measurement.r2 = robot2ID;
        pose_point_measurement.p2 = state2ID;
        pose_point_measurement.t = t;
        pose_point_measurement.tau = getTau(cov);

        // Add measurement
        pyfg_dataset.measurements.relative_measurements.vec.push_back(
            pose_point_measurement);
      } else {
        LOG(FATAL) << "Error: could not read relative pose-point measurement "
                      "from line: "
                   << line << "!";
      }
      break;
    case RANGE_MEASURE_TYPE:
      // EDGE_RANGE ts sym1 sym2 range cov
      if (strstrm >> timestamp >> sym1 >> sym2) {
        // Read string stream
        const double range = readScalarAsRange(strstrm);
        const double cov = readScalar(strstrm);

        // Parse symbols
        const auto [robot1ID, state1ID] = getRobotAndStateIDFromSymbol(sym1);
        const auto [robot2ID, state2ID] = getRobotAndStateIDFromSymbol(sym2);

        // Fill in measurement
        range_measurement.r1 = robot1ID;
        range_measurement.p1 = state1ID;
        range_measurement.r2 = robot2ID;
        range_measurement.p2 = state2ID;
        range_measurement.stateType1 = getStateTypeFromSymbol(sym1);
        range_measurement.stateType2 = getStateTypeFromSymbol(sym2);
        range_measurement.range = range;
        range_measurement.precision = 1.0 / cov;

        /**
         * @brief Set unit sphere indexes
         */

        // Ensure unique range measurements for correct unit sphere indexing
        const EdgeID &range_edge_id = range_measurement.getEdgeID();
        if (range_edge_id_to_index.find(range_edge_id) !=
            range_edge_id_to_index.end()) {
          LOG(WARNING) << "Skipping duplicate range measurement from "
                       << range_measurement.getSrcID() << " to "
                       << range_measurement.getDstID();
          continue;
        }

        // Add edge to map
        range_edge_id_to_index.emplace(range_edge_id, range_edge_index);
        range_edge_index++;

        // Update unit sphere index assuming the source robot takes ownership
        range_measurement.l = robot_id_to_unit_sphere_idx[robot1ID];
        robot_id_to_unit_sphere_idx[robot1ID]++;

        // Increment number of unit spheres for this robot
        pyfg_dataset.robot_id_to_num_unit_spheres[robot1ID]++;

        /**
         * @brief Set unit sphere ground truth
         */

        // Get translation of state 1
        Vector state1_translation = Vector::Zero(pyfg_dataset.dim);
        executeStateDependantFunctionals(
            [&]() {
              const PoseID &src_id = PoseID(range_measurement.getSrcID());
              state1_translation.noalias() =
                  pyfg_dataset.ground_truth.poses.at(src_id).translation();
            },
            [&]() {
              const PointID &src_id = PointID(range_measurement.getSrcID());
              state1_translation.noalias() =
                  pyfg_dataset.ground_truth.landmarks.at(src_id).translation();
            },
            range_measurement.stateType1);

        // Get translation of state 2
        Vector state2_translation = Vector::Zero(pyfg_dataset.dim);
        executeStateDependantFunctionals(
            [&]() {
              const PoseID &dst_id = PoseID(range_measurement.getDstID());
              state2_translation.noalias() =
                  pyfg_dataset.ground_truth.poses.at(dst_id).translation();
            },
            [&]() {
              const PointID &dst_id = PointID(range_measurement.getDstID());
              state2_translation.noalias() =
                  pyfg_dataset.ground_truth.landmarks.at(dst_id).translation();
            },
            range_measurement.stateType2);

        // Calculate unit sphere variable
        const Vector unit_vector =
            (state2_translation - state1_translation).normalized();
        const Point unit_sphere_var = Point(unit_vector);

        // Populate ground truth
        const UnitSphereID &unit_sphere_id =
            range_measurement.getUnitSphereID();
        pyfg_dataset.ground_truth.unit_spheres[unit_sphere_id] =
            unit_sphere_var;

        // Add measurement
        pyfg_dataset.measurements.relative_measurements.vec.push_back(
            range_measurement);
      } else {
        LOG(FATAL) << "Error: could not read range measurement from line: "
                   << line << "!";
      }
      break;
    }
  }

  infile.close();

  // Set robot states to zero if they do not exist
  for (const auto &robot_id : pyfg_dataset.robot_IDs) {
    pyfg_dataset.robot_id_to_num_poses[robot_id] += 0;
    pyfg_dataset.robot_id_to_num_landmarks[robot_id] += 0;
    pyfg_dataset.robot_id_to_num_unit_spheres[robot_id] += 0;
  }

  return pyfg_dataset;
}

LocalToGlobalStateDicts
getLocalToGlobalStateMapping(const PyFGDataset &pyfg_dataset) {
  LocalToGlobalStateDicts local_to_global_state_dicts;
  const unsigned int global_robot_id = 0;

  // Get ground truth dictionaries
  const PoseDict &gt_pose_dict = pyfg_dataset.ground_truth.poses;
  const LandmarkDict &gt_landmark_dict = pyfg_dataset.ground_truth.landmarks;
  const UnitSphereDict &gt_unit_sphere_dict =
      pyfg_dataset.ground_truth.unit_spheres;

  // Assign local to global state mapping
  unsigned int global_pose_idx = 0;
  for (const auto &[local_pose_id, pose] : gt_pose_dict) {
    // Populate map
    const PoseID global_pose_id(global_robot_id, global_pose_idx);
    local_to_global_state_dicts.poses[local_pose_id] = global_pose_id;

    // Increment
    global_pose_idx++;
  }
  unsigned int global_landmark_idx = 0;
  for (const auto &[local_landmark_id, landmark] : gt_landmark_dict) {
    // Populate map
    const PointID global_landmark_id(global_robot_id, global_landmark_idx);
    local_to_global_state_dicts.landmarks[local_landmark_id] =
        global_landmark_id;

    // Increment
    global_landmark_idx++;
  }
  unsigned int global_unit_sphere_idx = 0;
  for (const auto &[local_unit_sphere_id, unit_sphere] : gt_unit_sphere_dict) {
    // Populate map
    const UnitSphereID global_unit_sphere(global_robot_id,
                                          global_unit_sphere_idx);
    local_to_global_state_dicts.unit_spheres[local_unit_sphere_id] =
        global_unit_sphere;

    // Increment
    global_unit_sphere_idx++;
  }

  return local_to_global_state_dicts;
}

Measurements getGlobalMeasurements(const PyFGDataset &pyfg_dataset) {
  Measurements global_measurements;
  const LocalToGlobalStateDicts local_to_global_state_dicts =
      getLocalToGlobalStateMapping(pyfg_dataset);

  // Lambda function for globally reindexing relative measurements
  auto globallyReindexRelativeMeasurement = [](RelativeMeasurement &meas,
                                               const StateID &global_src_id,
                                               const StateID &global_dst_id) {
    meas.r1 = global_src_id.robot_id;
    meas.p1 = global_src_id.frame_id;
    meas.r2 = global_dst_id.robot_id;
    meas.p2 = global_dst_id.frame_id;
  };

  // TODO(AT): Add support for priors

  // Copy relative measurements and modify source and destination states
  for (const auto &m : pyfg_dataset.measurements.relative_measurements.vec) {
    if (std::holds_alternative<RelativePosePoseMeasurement>(m)) {
      RelativePosePoseMeasurement meas =
          std::get<RelativePosePoseMeasurement>(m);

      // Get global pose ids
      const StateID &global_pose_src_id =
          local_to_global_state_dicts.poses.at(meas.getSrcID());
      const StateID &global_pose_dst_id =
          local_to_global_state_dicts.poses.at(meas.getDstID());

      // Reindex to global state ids
      globallyReindexRelativeMeasurement(meas, global_pose_src_id,
                                         global_pose_dst_id);

      // Add reindexed measurement
      global_measurements.relative_measurements.vec.push_back(meas);

    } else if (std::holds_alternative<RelativePosePointMeasurement>(m)) {
      RelativePosePointMeasurement meas =
          std::get<RelativePosePointMeasurement>(m);

      // Get global pose id for source and landmark id for destination
      const StateID &global_pose_src_id =
          local_to_global_state_dicts.poses.at(meas.getSrcID());
      const StateID &global_landmark_dst_id =
          local_to_global_state_dicts.landmarks.at(meas.getDstID());

      // Reindex to global state ids
      globallyReindexRelativeMeasurement(meas, global_pose_src_id,
                                         global_landmark_dst_id);

      // Add reindexed measurement
      global_measurements.relative_measurements.vec.push_back(meas);

    } else {
      CHECK(std::holds_alternative<RangeMeasurement>(m));
      RangeMeasurement meas = std::get<RangeMeasurement>(m);

      // Get global ids depending on state type
      StateID global_state_src_id;
      executeStateDependantFunctionals(
          [&]() {
            global_state_src_id =
                local_to_global_state_dicts.poses.at(meas.getSrcID());
          },
          [&]() {
            global_state_src_id =
                local_to_global_state_dicts.landmarks.at(meas.getSrcID());
          },
          meas.stateType1);

      StateID global_state_dst_id;
      executeStateDependantFunctionals(
          [&]() {
            global_state_dst_id =
                local_to_global_state_dicts.poses.at(meas.getDstID());
          },
          [&]() {
            global_state_dst_id =
                local_to_global_state_dicts.landmarks.at(meas.getDstID());
          },
          meas.stateType2);

      // Reindex unit sphere variables
      const UnitSphereID &unit_sphere_id = meas.getUnitSphereID();
      meas.l =
          local_to_global_state_dicts.unit_spheres.at(unit_sphere_id).frame_id;

      // Reindex to global state ids
      globallyReindexRelativeMeasurement(meas, global_state_src_id,
                                         global_state_dst_id);

      // Add reindexed measurement
      global_measurements.relative_measurements.vec.push_back(meas);
    }
  }

  // Note: global measurements will always be consecutively indexed from
  // zero regardless of the sequencing of local measurements

  // Initialize ground truth initialization
  auto sumDictValues = [](const std::map<unsigned int, unsigned int> &map) {
    return std::accumulate(
        map.begin(), map.end(), 0,
        [](int value, const std::pair<unsigned int, unsigned int> &p) {
          return value + p.second;
        });
  };
  unsigned int d = pyfg_dataset.dim;
  unsigned int n = sumDictValues(pyfg_dataset.robot_id_to_num_poses);
  unsigned int l = sumDictValues(pyfg_dataset.robot_id_to_num_unit_spheres);
  unsigned int b = sumDictValues(pyfg_dataset.robot_id_to_num_landmarks);
  RangeAidedArray ground_truth_init(d, n, l, b);

  // Set ground truth initialization
  for (const auto &[local_pose_id, pose] : pyfg_dataset.ground_truth.poses) {
    const StateID &global_pose_id =
        local_to_global_state_dicts.poses.at(local_pose_id);
    ground_truth_init.pose(global_pose_id.frame_id) = pose.getData();
  }
  for (const auto &[local_landmark_id, landmark] :
       pyfg_dataset.ground_truth.landmarks) {
    const StateID &global_landmark_id =
        local_to_global_state_dicts.landmarks.at(local_landmark_id);
    ground_truth_init.landmark(global_landmark_id.frame_id) =
        landmark.getData();
  }
  for (const auto &[local_unit_sphere_id, unit_sphere] :
       pyfg_dataset.ground_truth.unit_spheres) {
    const StateID &global_unit_sphere_id =
        local_to_global_state_dicts.unit_spheres.at(local_unit_sphere_id);
    ground_truth_init.unitSphere(global_unit_sphere_id.frame_id) =
        unit_sphere.getData();
  }

  // Set ground truth initialization in global measurements
  global_measurements.ground_truth_init =
      std::make_shared<RangeAidedArray>(ground_truth_init);

  return global_measurements;
}

RobotMeasurements getRobotMeasurements(const PyFGDataset &pyfg_dataset) {
  RobotMeasurements robot_measurements;
  std::map<unsigned int, unsigned int> robot_first_pose_id;
  std::map<unsigned int, unsigned int> robot_first_point_id;

  // Copy measurements from dataset to robot
  for (const auto &robot_id : pyfg_dataset.robot_IDs) {
    Measurements measurements;
    std::set<unsigned int> pose_ids;
    std::set<unsigned int> point_ids;

    // add priors
    for (const auto &pose_prior : pyfg_dataset.measurements.pose_priors) {
      if (pose_prior.r == robot_id) {
        measurements.pose_priors.push_back(pose_prior);
        pose_ids.insert(pose_prior.p);
      }
    }
    for (const auto &point_prior : pyfg_dataset.measurements.point_priors) {
      if (point_prior.r == robot_id) {
        measurements.point_priors.push_back(point_prior);
        point_ids.insert(point_prior.p);
      }
    }

    // add relative measurements
    for (const auto &m : pyfg_dataset.measurements.relative_measurements.vec) {
      std::visit(
          [&](auto &&m) {
            if (m.r1 == robot_id || m.r2 == robot_id) {
              measurements.relative_measurements.vec.push_back(m);
              if (m.r1 == robot_id) {
                executeStateDependantFunctionals(
                    [&]() { pose_ids.insert(m.p1); },
                    [&]() { point_ids.insert(m.p1); }, m.stateType1);
              }
              if (m.r2 == robot_id) {
                executeStateDependantFunctionals(
                    [&]() { pose_ids.insert(m.p2); },
                    [&]() { point_ids.insert(m.p2); }, m.stateType2);
              }
            }
          },
          m);
    }

    // check for monotonically increasing sets of consecutive IDs
    auto areStateIDsConsecutive = [](const std::set<unsigned int> &ids) {
      auto it = std::adjacent_find(ids.begin(), ids.end(),
                                   [](int a, int b) { return a + 1 != b; });
      return it == ids.end();
    };
    if (!areStateIDsConsecutive(pose_ids))
      LOG(FATAL) << "Error: Pose IDs are not consecutive for robot " << robot_id
                 << "!";
    if (!areStateIDsConsecutive(point_ids))
      LOG(FATAL) << "Error: Point IDs are not consecutive for robot "
                 << robot_id << "!";

    // get first IDs for reindexing
    const unsigned int first_pose_id = *pose_ids.begin();
    const unsigned int first_point_id = *point_ids.begin();
    if (first_pose_id != 0)
      LOG(WARNING) << "WARNING: Pose IDs do not start at 0 for robot "
                   << robot_id << " and will be reindexed.";
    if (first_point_id != 0)
      LOG(WARNING) << "WARNING: Point IDs do not start at 0 for robot "
                   << robot_id << " and will be reindexed.";
    robot_first_pose_id[robot_id] = first_pose_id;
    robot_first_point_id[robot_id] = first_point_id;

    // emplace
    robot_measurements[robot_id] = measurements;
  }

  // reindex state IDs from zero
  for (auto &[robot_id, measurements] : robot_measurements) {
    for (auto &pose_prior : measurements.pose_priors) {
      pose_prior.p -= robot_first_pose_id[robot_id];
    }
    for (auto &point_prior : measurements.point_priors) {
      point_prior.p -= robot_first_point_id[robot_id];
    }
    for (auto &m : measurements.relative_measurements.vec) {
      std::visit(
          [&](auto &&m) {
            executeStateDependantFunctionals(
                [&]() { m.p1 -= robot_first_pose_id[m.r1]; },
                [&]() { m.p1 -= robot_first_point_id[m.r1]; }, m.stateType1);
            executeStateDependantFunctionals(
                [&]() { m.p2 -= robot_first_pose_id[m.r2]; },
                [&]() { m.p2 -= robot_first_point_id[m.r2]; }, m.stateType2);
          },
          m);
    }
  }

  for (const auto &robot_id : pyfg_dataset.robot_IDs) {
    // Initialize ground truth initialization
    unsigned int d = pyfg_dataset.dim;
    unsigned int n = pyfg_dataset.robot_id_to_num_poses.at(robot_id);
    unsigned int l = pyfg_dataset.robot_id_to_num_unit_spheres.at(robot_id);
    unsigned int b = pyfg_dataset.robot_id_to_num_landmarks.at(robot_id);
    RangeAidedArray ground_truth_init(d, n, l, b);

    // Set ground truth initialization
    for (const auto &[local_pose_id, pose] : pyfg_dataset.ground_truth.poses) {
      if (local_pose_id.robot_id != robot_id)
        continue;

      // Reindex id and add to the ground truth initialization
      unsigned int idx =
          local_pose_id.frame_id - robot_first_pose_id.at(robot_id);
      ground_truth_init.pose(idx) = pose.getData();
    }
    for (const auto &[local_landmark_id, landmark] :
         pyfg_dataset.ground_truth.landmarks) {
      if (local_landmark_id.robot_id != robot_id)
        continue;

      // Reindex id and add to the ground truth initialization
      unsigned int idx =
          local_landmark_id.frame_id - robot_first_point_id.at(robot_id);
      ground_truth_init.landmark(idx) = landmark.getData();
    }
    for (const auto &[local_unit_sphere_id, unit_sphere] :
         pyfg_dataset.ground_truth.unit_spheres) {
      if (local_unit_sphere_id.robot_id != robot_id)
        continue;

      // Index id and add to the ground truth initialization
      unsigned int idx = local_unit_sphere_id.frame_id;
      ground_truth_init.unitSphere(idx) = unit_sphere.getData();
    }

    // Set ground truth initialization in robot measurements
    robot_measurements[robot_id].ground_truth_init =
        std::make_shared<RangeAidedArray>(ground_truth_init);
  }

  return robot_measurements;
}

void executeStateDependantFunctionals(std::function<void()> poseFunction,
                                      std::function<void()> pointFunction,
                                      const StateType &state_type) {
  switch (state_type) {
  case StateType::Pose:
    poseFunction();
    break;
  case StateType::Point:
    pointFunction();
    break;
  default:
    LOG(FATAL) << "Invalid StateType: " << StateTypeToString(state_type) << "!";
  }
}

void get_dimension_and_num_poses(
    const std::vector<RelativePosePoseMeasurement> &measurements,
    size_t *dimension, size_t *num_poses) {
  CHECK(!measurements.empty());
  *dimension = measurements[0].t.size();
  CHECK(*dimension == 2 || *dimension == 3);
  *num_poses = 0;
  for (const auto &meas : measurements) {
    *num_poses = std::max(*num_poses, meas.p1 + 1);
    *num_poses = std::max(*num_poses, meas.p2 + 1);
  }
}

void constructBMatrices(
    const std::vector<RelativePosePoseMeasurement> &measurements,
    SparseMatrix *B1, SparseMatrix *B2, SparseMatrix *B3) {
  // Clear input matrices
  B1->setZero();
  B2->setZero();
  B3->setZero();

  size_t num_poses = 0;
  size_t d = (!measurements.empty() ? measurements[0].t.size() : 0);

  std::vector<Eigen::Triplet<double>> triplets;

  // Useful quantities to cache
  size_t d2 = d * d;
  size_t d3 = d * d * d;

  size_t i, j; // Indices for the tail and head of the given measurement
  double sqrttau;
  size_t max_pair;

  /// Construct the matrix B1 from equation (69a) in the tech report
  triplets.reserve(2 * d * measurements.size());

  for (size_t e = 0; e < measurements.size(); e++) {
    i = measurements[e].p1;
    j = measurements[e].p2;
    sqrttau = sqrt(measurements[e].tau);

    // Block corresponding to the tail of the measurement
    for (size_t l = 0; l < d; l++) {
      triplets.emplace_back(e * d + l, i * d + l,
                            -sqrttau); // Diagonal element corresponding to tail
      triplets.emplace_back(e * d + l, j * d + l,
                            sqrttau); // Diagonal element corresponding to head
    }

    // Keep track of the number of poses we've seen
    max_pair = std::max<size_t>(i, j);
    if (max_pair > num_poses)
      num_poses = max_pair;
  }
  num_poses++; // Account for zero-based indexing

  B1->resize(d * measurements.size(), d * num_poses);
  B1->setFromTriplets(triplets.begin(), triplets.end());

  /// Construct matrix B2 from equation (69b) in the tech report
  triplets.clear();
  triplets.reserve(d2 * measurements.size());

  for (size_t e = 0; e < measurements.size(); e++) {
    i = measurements[e].p1;
    sqrttau = sqrt(measurements[e].tau);
    for (size_t k = 0; k < d; k++)
      for (size_t r = 0; r < d; r++)
        triplets.emplace_back(d * e + r, d2 * i + d * k + r,
                              -sqrttau * measurements[e].t(k));
  }

  B2->resize(d * measurements.size(), d2 * num_poses);
  B2->setFromTriplets(triplets.begin(), triplets.end());

  /// Construct matrix B3 from equation (69c) in the tech report
  triplets.clear();
  triplets.reserve((d3 + d2) * measurements.size());

  for (size_t e = 0; e < measurements.size(); e++) {
    double sqrtkappa = std::sqrt(measurements[e].kappa);
    const Matrix &R = measurements[e].R;

    for (size_t r = 0; r < d; r++)
      for (size_t c = 0; c < d; c++) {
        i = measurements[e].p1; // Tail of measurement
        j = measurements[e].p2; // Head of measurement

        // Representation of the -sqrt(kappa) * Rt(i,j) \otimes I_d block
        for (size_t l = 0; l < d; l++)
          triplets.emplace_back(e * d2 + d * r + l, i * d2 + d * c + l,
                                -sqrtkappa * R(c, r));
      }

    for (size_t l = 0; l < d2; l++)
      triplets.emplace_back(e * d2 + l, j * d2 + l, sqrtkappa);
  }

  B3->resize(d2 * measurements.size(), d2 * num_poses);
  B3->setFromTriplets(triplets.begin(), triplets.end());
}

Matrix recoverTranslations(const SparseMatrix &B1, const SparseMatrix &B2,
                           const Matrix &R) {
  unsigned int d = R.rows();
  unsigned int n = R.cols() / d;

  // Vectorization of R matrix
  Eigen::Map<Eigen::VectorXd> rvec(const_cast<double *>(R.data()), d * d * n);

  // Form the matrix comprised of the right (n-1) block columns of B1
  SparseMatrix B1red = B1.rightCols(d * (n - 1));

  Eigen::VectorXd c = B2 * rvec;

  // Solve
  Eigen::SPQR<SparseMatrix> QR(B1red);
  Eigen::VectorXd tred = -QR.solve(c);

  // Reshape this result into a d x (n-1) matrix
  Eigen::Map<Eigen::MatrixXd> tred_mat(tred.data(), d, n - 1);

  // Allocate output matrix
  Eigen::MatrixXd t = Eigen::MatrixXd::Zero(d, n);

  // Set rightmost n-1 columns
  t.rightCols(n - 1) = tred_mat;

  return t;
}

Matrix projectToRotationGroup(const Matrix &M) {
  // Compute the SVD of M
  Eigen::JacobiSVD<Matrix> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);

  double detU = svd.matrixU().determinant();
  double detV = svd.matrixV().determinant();

  if (detU * detV > 0) {
    return svd.matrixU() * svd.matrixV().transpose();
  } else {
    Eigen::MatrixXd Uprime = svd.matrixU();
    Uprime.col(Uprime.cols() - 1) *= -1;
    return Uprime * svd.matrixV().transpose();
  }
}

Matrix projectToStiefelManifold(const Matrix &M) {
  size_t r = M.rows();
  size_t d = M.cols();
  CHECK(r >= d);
  Eigen::JacobiSVD<Matrix> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
  return svd.matrixU() * svd.matrixV().transpose();
}

Matrix projectToObliqueManifold(const Matrix &M) {
  size_t l = M.cols();
  Matrix X = M;
#pragma omp parallel for
  for (size_t i = 0; i < l; ++i) {
    X.col(i) = X.col(i).normalized();
  }
  return X;
}

Matrix symBlockDiagProduct(const Matrix &A, const Matrix &BT, const Matrix &C,
                           unsigned int r, unsigned int d, unsigned int n) {
  /*
  The following implementation is adapted from:
  CORA: https://github.com/MarineRoboticsGroup/cora
  */
  Matrix R(r, d * n);
  Matrix P(d, d);
  Matrix S(d, d);
  for (unsigned int i = 0; i < n; ++i) {
    auto start_col = static_cast<Eigen::Index>(i * d);
    P = BT.block(start_col, 0, d, r) * C.block(0, start_col, r, d);
    S = .5 * (P + P.transpose());
    R.block(0, start_col, r, d) = A.block(0, start_col, r, d) * S;
  }
  return R;
}

Matrix projectToStiefelManifoldTangentSpace(const Matrix &Y, const Matrix &V,
                                            unsigned int r, unsigned int d,
                                            unsigned int n) {
  /*
  The following implementation is adapted from:
  CORA: https://github.com/MarineRoboticsGroup/cora
  */
  return V - symBlockDiagProduct(Y, Y.transpose(), V, r, d, n);
}

Matrix projectToObliqueManifoldTangentSpace(const Matrix &Y, const Matrix &V) {
  /*
  The following implementation is adapted from:
  CORA: https://github.com/MarineRoboticsGroup/cora
  */
  Vector inner_prods = (Y.array() * V.array()).colwise().sum();
  Matrix scaled_cols = Y.array().rowwise() * inner_prods.transpose().array();
  return V - scaled_cols;
}

Matrix fixedStiefelVariable(unsigned r, unsigned d) {
  std::srand(1);
  return randomStiefelVariable(r, d);
}

Matrix fixedEuclideanVariable(unsigned r, unsigned b) {
  std::srand(1);
  return randomEuclideanVariable(r, b);
}

Matrix fixedObliqueVariable(unsigned r, unsigned l) {
  std::srand(1);
  return randomObliqueVariable(r, l);
}

Matrix randomStiefelVariable(unsigned r, unsigned d) {
  if (d == 0) {
    return Matrix::Zero(r, 0);
  }
  ROPTLIB::StieVariable var(r, d);
  var.RandInManifold();
  return Eigen::Map<Matrix>(const_cast<double *>(var.ObtainReadData()), r, d);
}

Matrix randomEuclideanVariable(unsigned r, unsigned b) {
  if (b == 0) {
    return Matrix::Zero(r, 0);
  }
  ROPTLIB::EucVariable var(r, b);
  var.RandInManifold();
  return Eigen::Map<Matrix>(const_cast<double *>(var.ObtainReadData()), r, b);
}

Matrix randomObliqueVariable(unsigned r, unsigned l) {
  if (l == 0) {
    return Matrix::Zero(r, 0);
  }
  ROPTLIB::ObliqueVariable var(r, l);
  var.RandInManifold();
  return Eigen::Map<Matrix>(const_cast<double *>(var.ObtainReadData()), r, l);
}

double computeMeasurementError(const RelativePosePoseMeasurement &m,
                               const Matrix &R1, const Matrix &t1,
                               const Matrix &R2, const Matrix &t2) {
  double rotationErrorSq = (R1 * m.R - R2).squaredNorm();
  double translationErrorSq = (t2 - t1 - R1 * m.t).squaredNorm();
  return m.kappa * rotationErrorSq + m.tau * translationErrorSq;
}

double chi2inv(double quantile, size_t dof) {
  boost::math::chi_squared_distribution<double> chi2(dof);
  return boost::math::quantile(chi2, quantile);
}

double angular2ChordalSO3(double rad) { return 2 * sqrt(2) * sin(rad / 2); }

void checkRotationMatrix(const Matrix &R) {
  const auto d = R.rows();
  CHECK(R.cols() == d);
  double err_det = abs(R.determinant() - 1.0);
  double err_norm = (R.transpose() * R - Matrix::Identity(d, d)).norm();
  if (err_det > 1e-5 || err_norm > 1e-5) {
    LOG(WARNING) << "[checkRotationMatrix] Invalid rotation: err_det="
                 << err_det << ", err_norm=" << err_norm;
  }
}

void checkStiefelMatrix(const Matrix &Y) {
  const auto d = Y.cols();
  double err_norm = (Y.transpose() * Y - Matrix::Identity(d, d)).norm();
  if (err_norm > 1e-5) {
    LOG(WARNING) << "[checkStiefelMatrix] Invalid Stiefel: err_norm="
                 << err_norm;
  }
}

void checkSEMatrixSize(const Matrix &X, unsigned int r, unsigned int d,
                       unsigned int n) {
  CHECK_EQ(X.rows(), (int)r);
  CHECK_EQ(X.cols(), (int)(d + 1) * n);
}

void checkRAMatrixSize(const Matrix &X, unsigned int r, unsigned int d,
                       unsigned int n, unsigned int l, unsigned int b) {
  CHECK_EQ(X.rows(), (int)r);
  CHECK_EQ(X.cols(), (int)(d + 1) * n + l + b);
}

std::tuple<Matrix, Matrix> partitionSEMatrix(const Matrix &X, unsigned int r,
                                             unsigned int d, unsigned int n) {
  checkSEMatrixSize(X, r, d, n);
  Matrix X_SE_R = Matrix::Zero(r, d * n);
  Matrix X_SE_t = Matrix::Zero(r, n);
#pragma omp parallel for
  for (size_t i = 0; i < n; ++i) {
    auto Y = X.block(0, i * (d + 1), r, d + 1);
    X_SE_R.block(0, i * d, r, d) = Y.block(0, 0, r, d);
    X_SE_t.col(i) = Y.col(d);
  }
  return std::make_tuple(X_SE_R, X_SE_t);
}

std::tuple<Matrix, Matrix, Matrix, Matrix>
partitionRAMatrix(const Matrix &X, unsigned int r, unsigned int d,
                  unsigned int n, unsigned int l, unsigned int b) {
  checkRAMatrixSize(X, r, d, n, l, b);
  Matrix X_SE_R = X.block(0, 0, r, d * n);
  Matrix X_OB = X.block(0, d * n, r, l);
  Matrix X_SE_t = X.block(0, d * n + l, r, n);
  Matrix X_E = X.block(0, d * n + l + n, r, b);
  return std::make_tuple(X_SE_R, X_OB, X_SE_t, X_E);
}

Matrix createSEMatrix(const Matrix &X_SE_R, const Matrix &X_SE_t) {
  size_t r = X_SE_R.rows();
  size_t n = X_SE_t.cols();
  if (n == 0) {
    return Matrix::Zero(r, 0);
  }
  size_t d = X_SE_R.cols() / n;
  CHECK_EQ(X_SE_R.rows(), X_SE_t.rows());
  CHECK_EQ(X_SE_R.cols() + X_SE_t.cols(), (d + 1) * n);

  Matrix X(r, (d + 1) * n);
#pragma omp parallel for
  for (size_t i = 0; i < n; ++i) {
    X.block(0, i * (d + 1), r, d) = X_SE_R.block(0, i * d, r, d);
    X.col(i * (d + 1) + d) = X_SE_t.col(i);
  }
  return X;
}

Matrix createRAMatrix(const Matrix &X_SE_R, const Matrix &X_OB,
                      const Matrix &X_SE_t, const Matrix &X_E) {
  Matrix X(X_SE_R.rows(),
           X_SE_R.cols() + X_OB.cols() + X_SE_t.cols() + X_E.cols());
  X << X_SE_R, X_OB, X_SE_t, X_E;
  return X;
}

void copyEigenMatrixToROPTLIBVariable(const Matrix &Y, ROPTLIB::Variable *var,
                                      size_t memSize) {
  const double *matrix_data = Y.data();
  double *prodvar_data = var->ObtainWriteEntireData();
  memcpy(prodvar_data, matrix_data, sizeof(double) * memSize);
}

Matrix projectToSEMatrix(const Matrix &M, unsigned int r, unsigned int d,
                         unsigned int n) {
  checkSEMatrixSize(M, r, d, n);
  Matrix X = M;
#pragma omp parallel for
  for (size_t i = 0; i < n; ++i) {
    X.block(0, i * (d + 1), r, d) =
        projectToStiefelManifold(X.block(0, i * (d + 1), r, d));
  }
  return X;
}

Matrix projectToRAMatrix(const Matrix &M, unsigned int r, unsigned int d,
                         unsigned int n, unsigned int l, unsigned int b) {
  auto [X_SE_R, X_OB, X_SE_t, X_E] = partitionRAMatrix(M, r, d, n, l, b);
  Matrix X_SE_proj = projectToSEMatrix(createSEMatrix(X_SE_R, X_SE_t), r, d, n);
  auto [X_SE_R_proj, X_SE_t_proj] = partitionSEMatrix(X_SE_proj, r, d, n);
  return createRAMatrix(X_SE_R_proj, projectToObliqueManifold(X_OB),
                        X_SE_t_proj, X_E);
}

} // namespace DCORA
