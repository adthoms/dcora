/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#ifndef LIFTEDSEMANIFOLD_H
#define LIFTEDSEMANIFOLD_H

#include <DPGO/DPGO_types.h>
#include <DPGO/manifold/Poses.h>

#include "Manifolds/Euclidean/Euclidean.h"
#include "Manifolds/ProductManifold.h"
#include "Manifolds/Stiefel/Stiefel.h"

/*Define the namespace*/
namespace DPGO {

/**
 * @brief This class represents a manifold for the SE(n) synchronization problem
 */
class LiftedSEManifold {
 public:
  /**
   * @brief Constructor
   * @param r
   * @param d
   * @param n
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
}  // namespace DPGO

#endif