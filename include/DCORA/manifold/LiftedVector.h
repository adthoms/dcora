/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#pragma once

#include <DCORA/DCORA_types.h>
#include <DCORA/DCORA_utils.h>

#include "Manifolds/Euclidean/Euclidean.h"
#include "Manifolds/ProductManifold.h"
#include "Manifolds/Stiefel/Stiefel.h"

namespace DCORA {

/**
 * @brief This class acts as container for ROPTLIB::ProductElement vectors for
 * SE(n) synchronization
 */
class LiftedSEVector {
public:
  /**
   * @brief Constructor
   * @param r relaxation rank
   * @param d dimension (2/3)
   * @param n number of poses
   */
  LiftedSEVector(unsigned int r, unsigned int d, unsigned int n);
  /**
   * @brief Destructor
   */
  virtual ~LiftedSEVector();
  /**
   * @brief Get underlying ROPTLIB vector
   * @return
   */
  virtual ROPTLIB::ProductElement *vec() { return MySEVector; }
  /**
   * @brief Get data as Eigen matrix
   * @return
   */
  virtual Matrix getData();
  /**
   * @brief Set data from Eigen matrix
   * @param X
   */
  virtual void setData(const Matrix &X);
  /**
   * @brief Set underlying ROPTLIB vector size
   * @return
   */
  virtual void setSize();
  /**
   * @brief Set underlying ROPTLIB vector size from ROPTLIB product element size
   * @param productElement
   * @param row
   * @param col
   * @param num_el
   */
  static void setSizeFromProductElement(ROPTLIB::ProductElement *productElement,
                                        unsigned int *row, unsigned int *col,
                                        unsigned int *num_el);

protected:
  unsigned int r_, d_, n_;
  ROPTLIB::StieVector *StiefelVector;
  ROPTLIB::EucVector *EuclideanVector;
  ROPTLIB::ProductElement *CartanVector;
  ROPTLIB::ProductElement *MySEVector;
};

} // namespace DCORA
