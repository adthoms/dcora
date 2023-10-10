/* ----------------------------------------------------------------------------
 * Copyright 2023, University of California Los Angeles, Los Angeles, CA 90095
 * All Rights Reserved
 * Authors: Alex Thoms and Alan Papalia
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#ifndef LIFTEDRAVECTOR_H
#define LIFTEDRAVECTOR_H

#include <DCORA/manifold/LiftedSEVector.h>

#include "Manifolds/Oblique/Oblique.h"

/*Define the namespace*/
namespace DCORA {

/**
 * @brief This class represents a RA SLAM synchronization vector
 */
class LiftedRAVector : public LiftedSEVector {
 public:
  /**
   * @brief Constructor
   * @param r
   * @param d
   * @param n
   * @param b
   * @param l
   */
  LiftedRAVector(int r, int d, int n, int b, int l);
  /**
   * @brief Destructor
   */
  ~LiftedRAVector() override;
  /**
   * @brief Get underlying ROPTLIB vector
   * @return
   */
  ROPTLIB::ProductElement* vec() override { return MyRAVector; }
  /**
   * @brief Get data as Eigen matrix
   * @return
   */
  Matrix getData() override;
  /**
   * @brief Set data from Eigen matrix
   * @param X
   */
  void setData(const Matrix& X) override;
  /**
   * @brief Set underlying ROPTLIB vector size
   * @return
   */
  void setSize() override;

 private:
  unsigned int b_, l_;
  ROPTLIB::EucVector* EuclideanLandmarkVector;
  ROPTLIB::ObliqueVector* ObliqueRangeVector;
  ROPTLIB::ProductElement* MyEVector;
  ROPTLIB::ProductElement* MyOBVector;
  ROPTLIB::ProductElement* MyRAVector;
};
}  // namespace DCORA

#endif