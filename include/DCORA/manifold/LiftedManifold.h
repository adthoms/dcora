/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#pragma once

#include <DCORA/DCORA_types.h>
#include <DCORA/manifold/Elements.h>

#include "Manifolds/Euclidean/Euclidean.h"
#include "Manifolds/ProductManifold.h"
#include "Manifolds/Stiefel/Stiefel.h"

namespace DCORA {

/**
 * @brief This class represents a manifold for the SE(n) synchronization problem
 */
class LiftedSEManifold {
public:
  /**
   * @brief Constructor
   * @param r relaxation rank
   * @param d dimension (2/3)
   * @param n number of poses
   */
  LiftedSEManifold(unsigned int r, unsigned int d, unsigned int n);
  /**
   * @brief Destructor
   */
  virtual ~LiftedSEManifold();
  /**
   * @brief Get the underlying ROPTLIB product manifold
   * @return
   */
  virtual ROPTLIB::ProductManifold *getManifold() { return MySEManifold; }
  /**
   * @brief Utility function to project a given matrix onto this manifold
   * @param M
   * @return orthogonal projection of M onto this manifold
   */
  virtual Matrix project(const Matrix &M) const;

protected:
  unsigned int r_, d_, n_;
  ROPTLIB::Stiefel *StiefelManifold;
  ROPTLIB::Euclidean *EuclideanManifold;
  ROPTLIB::ProductManifold *CartanManifold;
  ROPTLIB::ProductManifold *MySEManifold;
};

} // namespace DCORA
