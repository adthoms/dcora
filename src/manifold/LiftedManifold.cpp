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

void LiftedSEManifold::projectToTangentSpace(ROPTLIB::Variable *x,
                                             ROPTLIB::Vector *v,
                                             ROPTLIB::Vector *result) {
  MySEManifold->Projection(x, v, result);
}

Matrix LiftedSEManifold::project(const Matrix &M) {
  return projectToSEMatrix(M, r_, d_, n_);
}

LiftedRAManifold::LiftedRAManifold(unsigned int r, unsigned int d,
                                   unsigned int n, unsigned int l,
                                   unsigned int b)
    : r_(r), d_(d), n_(n), l_(l), b_(b) {
  StiefelPoseManifold = new ROPTLIB::Stiefel(r, d);
  StiefelPoseManifold->ChooseStieParamsSet3();
  ObliqueUnitSphereManifold = new ROPTLIB::Oblique(r, l);
  ObliqueUnitSphereManifold->ChooseObliqueParamsSet3();
  EuclideanPoseManifold = new ROPTLIB::Euclidean(r, n);
  EuclideanLandmarkManifold = new ROPTLIB::Euclidean(r, b);
  MyRAManifold = new ROPTLIB::ProductManifold(
      4, StiefelPoseManifold, n, ObliqueUnitSphereManifold, 1,
      EuclideanPoseManifold, 1, EuclideanLandmarkManifold, 1);
}

LiftedRAManifold::~LiftedRAManifold() {
  // Avoid memory leak
  delete StiefelPoseManifold;
  delete ObliqueUnitSphereManifold;
  delete EuclideanPoseManifold;
  delete EuclideanLandmarkManifold;
  delete MyRAManifold;
}

Matrix LiftedRAManifold::project(const Matrix &M) {
  return projectToRAMatrix(M, r_, d_, n_, l_, b_);
}

void LiftedRAManifold::projectToTangentSpace(ROPTLIB::Variable *x,
                                             ROPTLIB::Vector *v,
                                             ROPTLIB::Vector *result) {
  MyRAManifold->Projection(x, v, result);
}

} // namespace DCORA
