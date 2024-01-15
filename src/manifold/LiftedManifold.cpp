/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DCORA/DCORA_utils.h>
#include <DCORA/manifold/LiftedManifold.h>
#include <glog/logging.h>

namespace DCORA {

LiftedSEManifold::LiftedSEManifold(unsigned int r, unsigned int d,
                                   unsigned int n)
    : r_(r), d_(d), n_(n) {
  StiefelManifold = new ROPTLIB::Stiefel(r, d);
  StiefelManifold->ChooseStieParamsSet3();
  EuclideanManifold = new ROPTLIB::Euclidean(r);
  CartanManifold =
      new ROPTLIB::ProductManifold(2, StiefelManifold, 1, EuclideanManifold, 1);
  MySEManifold = new ROPTLIB::ProductManifold(1, CartanManifold, n);
}

LiftedSEManifold::~LiftedSEManifold() {
  // Avoid memory leak
  delete StiefelManifold;
  delete EuclideanManifold;
  delete CartanManifold;
  delete MySEManifold;
}

Matrix LiftedSEManifold::project(const Matrix &M) const {
  checkSEMatrixSize(M, r_, d_, n_);
  Matrix X = M;
#pragma omp parallel for
  for (size_t i = 0; i < n_; ++i) {
    X.block(0, i * (d_ + 1), r_, d_) =
        projectToStiefelManifold(X.block(0, i * (d_ + 1), r_, d_));
  }
  return X;
}

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
