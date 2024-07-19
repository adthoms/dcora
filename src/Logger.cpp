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

#include <DCORA/Logger.h>
#include <Eigen/Geometry>
#include <glog/logging.h>
#include <utility>

namespace DCORA {

Logger::Logger(std::string logDir) : logDirectory(std::move(logDir)) {}

void Logger::logMeasurements(
    std::vector<RelativePosePoseMeasurement> *measurements,
    const std::string &filename) {
  if (measurements->empty())
    return;

  std::ofstream file;
  file.open(logDirectory + filename);
  if (!file.is_open())
    return;

  size_t d = measurements->at(0).R.rows();
  if (d == 2)
    return;

  // Insert header row
  file << "robot_src,pose_src,robot_dst,pose_dst,qx,qy,qz,qw,tx,ty,tz,kappa,"
          "tau,is_known_inlier,weight\n";

  for (RelativePosePoseMeasurement m : *measurements) {
    // Convert rotation matrix to quaternion
    Eigen::Matrix3d R = m.R;
    Eigen::Quaternion<double> quat(R);
    file << m.r1 << ",";
    file << m.p1 << ",";
    file << m.r2 << ",";
    file << m.p2 << ",";
    file << quat.x() << ",";
    file << quat.y() << ",";
    file << quat.z() << ",";
    file << quat.w() << ",";
    file << m.t(0) << ",";
    file << m.t(1) << ",";
    file << m.t(2) << ",";
    file << m.kappa << ",";
    file << m.tau << ",";
    file << m.fixedWeight << ",";
    file << m.weight << "\n";
  }

  file.close();
}

void Logger::logMeasurements(RelativeMeasurements *measurements,
                             const std::string &filename) {
  if (measurements->vec.empty())
    return;

  std::ofstream file;
  file.open(logDirectory + filename);
  if (!file.is_open())
    return;

  const std::vector<RelativePosePoseMeasurement> &pose_pose_measurements =
      measurements->GetRelativePosePoseMeasurements();
  const std::vector<RelativePosePointMeasurement> &pose_point_measurements =
      measurements->GetRelativePosePointMeasurements();
  const std::vector<RangeMeasurement> &range_measurements =
      measurements->GetRangeMeasurements();

  for (const auto &meas : pose_pose_measurements) {
    file << "Pose-Pose Measurements\n";
    file << meas << "\n";
  }
  for (const auto &meas : pose_point_measurements) {
    file << "Pose-Point Measurements\n";
    file << meas << "\n";
  }
  for (const auto &meas : range_measurements) {
    file << "Range Measurements\n";
    file << meas << "\n";
  }

  file.close();
}

void Logger::logTrajectory(unsigned int d, unsigned int n, const Matrix &T,
                           const std::string &filename) {
  CHECK_EQ(T.rows(), d);
  CHECK_EQ(T.cols(), (d + 1) * n);
  std::ofstream file;
  file.open(logDirectory + filename);
  if (!file.is_open())
    return;

  auto getTranslationAndQuaternion =
      [&d](const Matrix &T) -> std::pair<Eigen::Vector3d, Eigen::Quaterniond> {
    Eigen::Matrix3d R = Matrix::Identity(3, 3);
    Eigen::Vector3d t = Matrix::Zero(3, 1);
    if (d == 2) {
      R.topLeftCorner<2, 2>() = T.block<2, 2>(0, 0);
      t.head<2>() = T.block<2, 1>(0, 2);
    } else {
      CHECK_EQ(d, 3);
      R = T.block<3, 3>(0, 0);
      t = T.block<3, 1>(0, 3);
    }
    Eigen::Quaterniond quat(R);
    return {t, quat};
  };

  file << "# pose_index x y z qx qy qz qw\n";
  for (size_t i = 0; i < n; ++i) {
    Matrix Ti = T.block(0, i * (d + 1), d, d + 1);
    auto [t, quat] = getTranslationAndQuaternion(Ti);
    file << i << " " << std::fixed << std::setprecision(9);
    file << t.x() << " " << t.y() << " " << t.z() << " ";
    file << quat.x() << " " << quat.y() << " " << quat.z() << " " << quat.w()
         << "\n";
  }

  file.close();
}

} // namespace DCORA
