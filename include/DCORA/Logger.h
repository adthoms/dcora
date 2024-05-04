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
#include <DCORA/Measurements.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace DCORA {

class PGOLogger {
public:
  /**
   * @brief Constructor
   * @param logDir directory to store all log files
   */
  explicit PGOLogger(std::string logDir);

  /**
   * @brief Destructor
   */
  ~PGOLogger() = default;

  /**
   * @brief Log trajectory to file (currently only work for 3D data)
   * @param d dimension (2 or 3)
   * @param n number of poses
   * @param T d-by-(d+1)*n matrix where each d-by-(d+1) block represents a pose
   * @param filename filename
   */
  void logTrajectory(unsigned d, unsigned n, const Matrix &T,
                     const std::string &filename);

  /**
   * @brief Log measurements to file (currently only work for 3D data)
   * @param measurements a vector of relative pose measurements
   * @param filename
   */
  void logMeasurements(std::vector<RelativeSEMeasurement> *measurements,
                       const std::string &filename);

  /**
   * @brief Load trajectory from file
   * @param filename
   * @return
   */
  Matrix loadTrajectory(const std::string &filename);

  /**
   * @brief Read a list of measurements from file
   * @param filename
   * @param load_weight
   * @return
   */
  static std::vector<RelativeSEMeasurement>
  loadMeasurements(const std::string &filename, bool load_weight = false);

private:
  std::string logDirectory;
};

} // namespace DCORA
