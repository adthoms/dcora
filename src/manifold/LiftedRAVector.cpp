/* ----------------------------------------------------------------------------
 * Copyright 2023, University of California Los Angeles, Los Angeles, CA 90095
 * All Rights Reserved
 * Authors: Alex Thoms and Alan Papalia
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DPGO/manifold/LiftedRAVector.h>
#include <glog/logging.h>

namespace DPGO {

LiftedRAVector::LiftedRAVector(int r, int d, int n, int b, int l) 
    : LiftedSEVector(r, d, n) {
  EuclideanLandmarkVector = new ROPTLIB::EucVector(r, b);
  ObliqueRangeVector = new ROPTLIB::ObliqueVector(r, l);
  MyEVector = new ROPTLIB::ProductElement(1, EuclideanLandmarkVector, 1);
  MyOBVector = new ROPTLIB::ProductElement(1, ObliqueRangeVector, 1);
  MyRAVector = new ROPTLIB::ProductElement(3, MySEVector, 1, MyEVector, 1, MyOBVector, 1);
}

LiftedRAVector::~LiftedRAVector() {
  // Avoid memory leak
  delete EuclideanLandmarkVector;
  delete ObliqueRangeVector;
  delete MyEVector;
  delete MyOBVector;
  delete MyRAVector;
}

Matrix LiftedRAVector::getData() {
  setSize();
  Matrix X_SE = Eigen::Map<Matrix>((double *)MySEVector->ObtainReadData(), r_,  n_ * (d_ + 1));
  Matrix X_E = Eigen::Map<Matrix>((double *)MyEVector->ObtainReadData(), r_, b_);
  Matrix X_OB = Eigen::Map<Matrix>((double *)MyOBVector->ObtainReadData(), r_, l_);
  return createRAMatrix(X_SE, X_E, X_OB);
}

void LiftedRAVector::setData(const Matrix &X) {
  setSize();
  auto [X_SE, X_E, X_OB] = partitionRAMatrix(X, r_, d_, n_, b_ , l_);
  copyEigenMatrixToROPTLIBVariable(X_SE, MySEVector, r_ * (d_ + 1) * n_);
  copyEigenMatrixToROPTLIBVariable(X_E, MyEVector, r_ * b_);
  copyEigenMatrixToROPTLIBVariable(X_OB, MyOBVector, r_ * l_);
}

void LiftedRAVector::setSize() {
  LiftedSEVector::setSize();
  unsigned int col_e, col_ob;
  LiftedSEVector::getSize(MyEVector, r_, col_e, b_);
  LiftedSEVector::getSize(MyOBVector, r_, col_ob, l_);
  CHECK_EQ(col_e, 1);
  CHECK_EQ(col_e, col_ob);
}

}  // namespace DPGO