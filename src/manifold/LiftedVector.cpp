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

#include <DCORA/manifold/LiftedVector.h>
#include <glog/logging.h>

namespace DCORA {

LiftedSEVector::LiftedSEVector(unsigned int r, unsigned int d, unsigned int n)
    : r_(r), d_(d), n_(n) {
  StiefelVector = new ROPTLIB::StieVector(r, d);
  EuclideanVector = new ROPTLIB::EucVector(r);
  CartanVector =
      new ROPTLIB::ProductElement(2, StiefelVector, 1, EuclideanVector, 1);
  MySEVector = new ROPTLIB::ProductElement(1, CartanVector, n);
}

LiftedSEVector::~LiftedSEVector() {
  // Avoid memory leak
  delete StiefelVector;
  delete EuclideanVector;
  delete CartanVector;
  delete MySEVector;
}

Matrix LiftedSEVector::getData() {
  return Eigen::Map<Matrix>(const_cast<double *>(MySEVector->ObtainReadData()),
                            r_, (d_ + 1) * n_);
}

void LiftedSEVector::setData(const Matrix &X) {
  checkSEMatrixSize(X, r_, d_, n_);
  size_t mem_size = r_ * (d_ + 1) * n_;
  copyEigenMatrixToROPTLIBVariable(X, MySEVector, mem_size);
}

LiftedRAVector::LiftedRAVector(unsigned int r, unsigned int d, unsigned int n,
                               unsigned int l, unsigned int b)
    : r_(r), d_(d), n_(n), l_(l), b_(b) {
  StiefelPoseVector = new ROPTLIB::StieVector(r, d);
  ObliqueUnitSphereVector = new ROPTLIB::ObliqueVector(r, l);
  EuclideanPoseVector = new ROPTLIB::EucVector(r, n);
  EuclideanLandmarkVector = new ROPTLIB::EucVector(r, b);
  MyRAVector = new ROPTLIB::ProductElement(
      4, StiefelPoseVector, n, ObliqueUnitSphereVector, 1, EuclideanPoseVector,
      1, EuclideanLandmarkVector, 1);
}

LiftedRAVector::~LiftedRAVector() {
  // Avoid memory leak
  delete StiefelPoseVector;
  delete ObliqueUnitSphereVector;
  delete EuclideanPoseVector;
  delete EuclideanLandmarkVector;
  delete MyRAVector;
}

Matrix LiftedRAVector::getData() {
  return Eigen::Map<Matrix>(const_cast<double *>(MyRAVector->ObtainReadData()),
                            r_, (d_ + 1) * n_ + l_ + b_);
}

void LiftedRAVector::setData(const Matrix &X) {
  checkRAMatrixSize(X, r_, d_, n_, l_, b_);
  size_t mem_size = r_ * ((d_ + 1) * n_ + l_ + b_);
  copyEigenMatrixToROPTLIBVariable(X, MyRAVector, mem_size);
}

} // namespace DCORA
