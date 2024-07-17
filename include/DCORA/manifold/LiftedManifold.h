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
#include <DCORA/manifold/Elements.h>

#include "Manifolds/Euclidean/Euclidean.h"
#include "Manifolds/Oblique/Oblique.h"
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
  ~LiftedSEManifold();
  /**
   * @brief Get the underlying ROPTLIB product manifold
   * @return
   */
  ROPTLIB::ProductManifold *getManifold() { return MySEManifold; }
  /**
   * @brief Utility function to project a given matrix onto this manifold
   * @param M
   * @return orthogonal projection of M onto this manifold
   */
  Matrix project(const Matrix &M);
  /**
   * @brief Utility function to project a matrix v onto the tangent space T_Y(M)
   * of the manifold at point x using ROBOTLIB containers.
   * @param x
   * @param v
   * @param result
   */
  void projectToTangentSpace(ROPTLIB::Variable *x, ROPTLIB::Vector *v,
                             ROPTLIB::Vector *result);

private:
  unsigned int r_, d_, n_;
  ROPTLIB::Stiefel *StiefelManifold;
  ROPTLIB::Euclidean *EuclideanManifold;
  ROPTLIB::ProductManifold *CartanManifold;
  ROPTLIB::ProductManifold *MySEManifold;
};

/**
 * @brief This class represents a manifold for the RA-SLAM synchronization
 * problem
 */
class LiftedRAManifold {
public:
  /**
   * @brief Constructor
   * @param r relaxation rank
   * @param d dimension (2/3)
   * @param n number of poses
   * @param l number of unit spheres
   * @param b number of landmarks
   */
  LiftedRAManifold(unsigned int r, unsigned int d, unsigned int n,
                   unsigned int l, unsigned int b);
  /**
   * @brief Destructor
   */
  ~LiftedRAManifold();
  /**
   * @brief Get the underlying ROPTLIB product manifold
   * @return
   */
  ROPTLIB::ProductManifold *getManifold() { return MyRAManifold; }
  /**
   * @brief Utility function to project a given matrix onto this manifold
   * @param M
   * @return orthogonal projection of M onto this manifold
   */
  Matrix project(const Matrix &M);
  /**
   * @brief Utility function to project a matrix v onto the tangent space T_Y(M)
   * of the manifold at point x using ROBOTLIB containers.
   * @param x
   * @param v
   * @param result
   */
  void projectToTangentSpace(ROPTLIB::Variable *x, ROPTLIB::Vector *v,
                             ROPTLIB::Vector *result);

private:
  unsigned int r_, d_, n_, l_, b_;
  ROPTLIB::Stiefel *StiefelPoseManifold;
  ROPTLIB::Oblique *ObliqueUnitSphereManifold;
  ROPTLIB::Euclidean *EuclideanPoseManifold;
  ROPTLIB::Euclidean *EuclideanLandmarkManifold;
  ROPTLIB::ProductManifold *MyRAManifold;
};

} // namespace DCORA
