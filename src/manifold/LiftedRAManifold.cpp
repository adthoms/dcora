/* ----------------------------------------------------------------------------
 * Copyright 2023, University of California Los Angeles, Los Angeles, CA 90095
 * All Rights Reserved
 * Authors: Alex Thoms and Alan Papalia
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DPGO/DPGO_utils.h>
#include <DPGO/manifold/LiftedRAManifold.h>
#include <glog/logging.h>

namespace DPGO {
LiftedRAManifold::LiftedRAManifold(unsigned int r, unsigned int d, unsigned int n, unsigned int b, unsigned int l)
    : LiftedSEManifold(r, d, n), b_(b), l_(l) {
  EuclideanLandmarkManifold = new ROPTLIB::Euclidean((int) r, (int) b);
  ObliqueManifold = new ROPTLIB::Oblique((int) r, (int) l);
  MyRAManifold = new ROPTLIB::ProductManifold(3, CartanManifold, n, EuclideanLandmarkManifold, 1, ObliqueManifold, 1);
}

LiftedRAManifold::~LiftedRAManifold() {
  // Avoid memory leak
  delete EuclideanLandmarkManifold;
  delete ObliqueManifold;
  delete MyRAManifold;
}

Matrix LiftedRAManifold::project(const Matrix &M) const {
  auto [X_SE, X_E, X_OB] = partitionRAMatrix(M, r_, d_, n_, b_ , l_);

  X_SE = LiftedSEManifold::project(X_SE);
#pragma omp parallel for
  for (size_t i = 0; i < l_; ++i) {
    X_OB.block(0, i, r_, 1) =
        projectToObliqueManifold(X_OB.block(0, i, r_, 1));
  }

  return createRAMatrix(X_SE, X_E, X_OB);
}

}  // namespace DPGO