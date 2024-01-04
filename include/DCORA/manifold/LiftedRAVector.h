/* ----------------------------------------------------------------------------
 * Copyright 2023, University of California Los Angeles, Los Angeles, CA 90095
 * All Rights Reserved
 * Authors: Alex Thoms and Alan Papalia
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#pragma once

#include <DCORA/manifold/LiftedSEVector.h>

#include "Manifolds/Oblique/Oblique.h"

namespace DCORA {

/**
 * @brief This class acts as container for ROPTLIB::ProductElement vectors for
 * RA-SLAM synchronization
 */
class LiftedRAVector : public LiftedSEVector {
public:
  /**
   * @brief Constructor
   * @param r relaxation rank
   * @param d dimension (2/3)
   * @param n number of poses
   * @param l number of ranges
   * @param b number of landmarks
   */
  LiftedRAVector(unsigned int r, unsigned int d, unsigned int n, unsigned int l,
                 unsigned int b);
  /**
   * @brief Destructor
   */
  ~LiftedRAVector() override;
  /**
   * @brief Get underlying ROPTLIB vector
   * @return
   */
  ROPTLIB::ProductElement *vec() override { return MyRAVector; }
  /**
   * @brief Get data as Eigen matrix
   * @return
   */
  Matrix getData() override;
  /**
   * @brief Set data from Eigen matrix
   * @param X
   */
  void setData(const Matrix &X) override;
  /**
   * @brief Set underlying ROPTLIB vector size
   * @return
   */
  void setSize() override;

private:
  unsigned int l_, b_;
  ROPTLIB::ObliqueVector *ObliqueRangeVector;
  ROPTLIB::EucVector *EuclideanLandmarkVector;
  ROPTLIB::ProductElement *MyOBVector;
  ROPTLIB::ProductElement *MyEVector;
  ROPTLIB::ProductElement *MyRAVector;
};

} // namespace DCORA
