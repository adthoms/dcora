/* ----------------------------------------------------------------------------
 * Copyright 2023, University of California Los Angeles, Los Angeles, CA 90095
 * All Rights Reserved
 * Authors: Alex Thoms and Alan Papalia
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DCORA/DCORA_utils.h>
#include <DCORA/manifold/LiftedRAManifold.h>
#include <glog/logging.h>

namespace DCORA {

LiftedRAManifold::LiftedRAManifold(unsigned int r, unsigned int d,
                                   unsigned int n, unsigned int l,
                                   unsigned int b)
    : LiftedSEManifold(r, d, n), l_(l), b_(b) {
  ObliqueManifold = new ROPTLIB::Oblique(r, l);
  EuclideanLandmarkManifold = new ROPTLIB::Euclidean(r, b);
  MyRAManifold = new ROPTLIB::ProductManifold(
      3, CartanManifold, n, ObliqueManifold, 1, EuclideanLandmarkManifold, 1);
}

LiftedRAManifold::~LiftedRAManifold() {
  // Avoid memory leak
  delete ObliqueManifold;
  delete EuclideanLandmarkManifold;
  delete MyRAManifold;
}

Matrix LiftedRAManifold::project(const Matrix &M) const {
  auto [X_SE_R, X_OB, X_SE_t, X_E] = partitionRAMatrix(M, r_, d_, n_, l_, b_);
  return createRAMatrix(projectToRotationGroup(X_SE_R),
                        projectToObliqueManifold(X_OB), X_SE_t, X_E);
}

} // namespace DCORA
