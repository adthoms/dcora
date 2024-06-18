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

#include <DCORA/DCORA_utils.h>
#include <DCORA/Measurements.h>
#include <DCORA/manifold/Elements.h>

#include <iostream>

#include "gtest/gtest.h"
#include <boost/filesystem.hpp>

std::string getDCORADataFilePath(const std::string &file_name) {
  boost::filesystem::path dcora_build_path = boost::filesystem::current_path();
  boost::filesystem::path dcora_path = dcora_build_path.parent_path();
  boost::filesystem::path dcora_data_file_path =
      dcora_path / "data" / file_name;
  return dcora_data_file_path.string();
}

TEST(testDCORA, testReadPyFGFileSE2) {
  const std::string pyfg_file_path_str =
      getDCORADataFilePath("pyfg_se2_test_data.pyfg");
  DCORA::PyFGDataset pyfg_dataset = DCORA::read_pyfg_file(pyfg_file_path_str);
  // TODO(JV): update PyFGDataset to print to yaml and compare to expected pyfg
  // file using yaml-to-pyfg loader
}

TEST(testDCORA, testReadPyFGFileSE3) {
  const std::string pyfg_file_path_str =
      getDCORADataFilePath("pyfg_se3_test_data.pyfg");
  DCORA::PyFGDataset pyfg_dataset = DCORA::read_pyfg_file(pyfg_file_path_str);
  // TODO(JV): update PyFGDataset to print to yaml and compare to expected pyfg
  // file using yaml-to-pyfg loader
}
