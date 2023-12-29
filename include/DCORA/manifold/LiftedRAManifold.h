/* ----------------------------------------------------------------------------
 * Copyright 2023, University of California Los Angeles, Los Angeles, CA 90095
 * All Rights Reserved
 * Authors: Alex Thoms and Alan Papalia
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#ifndef LIFTEDRAMANIFOLD_H
#define LIFTEDRAMANIFOLD_H

#include <DCORA/manifold/LiftedSEManifold.h>

#include "Manifolds/Oblique/Oblique.h"

/*Define the namespace*/
namespace DCORA {

/**
 * @brief This class represents a manifold for the RA SLAM synchronization problem
 */
class LiftedRAManifold : public LiftedSEManifold {
 public:
  /**
   * @brief Constructor
   * @param r
   * @param d
   * @param n
   * @param l
   * @param b
   */
  LiftedRAManifold(unsigned int r, unsigned int d, unsigned int n, unsigned int l, unsigned int b);
  /**
   * @brief Destructor
   */
  ~LiftedRAManifold() override;
  /**
   * @brief Get the underlying ROPTLIB product manifold
   * @return
   */
  ROPTLIB::ProductManifold *getManifold() override { return MyRAManifold; }
  /**
   * @brief Utility function to project a given matrix onto this manifold
   * @param M
   * @return orthogonal projection of M onto this manifold
   */
  Matrix project(const Matrix &M) const override;

 private:
  unsigned int l_, b_;
  ROPTLIB::Oblique *ObliqueManifold;
  ROPTLIB::Euclidean *EuclideanLandmarkManifold;
  ROPTLIB::ProductManifold *MyRAManifold;
};
}  // namespace DCORA

#endif