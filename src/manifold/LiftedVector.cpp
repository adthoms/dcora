/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DCORA/manifold/LiftedVector.h>
#include <glog/logging.h>

namespace DCORA {

LiftedSEVector::LiftedSEVector(unsigned int r, unsigned int d, unsigned int n) {
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
  setSize();
  return Eigen::Map<Matrix>(const_cast<double *>(MySEVector->ObtainReadData()),
                            r_, n_ * (d_ + 1));
}
void LiftedSEVector::setData(const Matrix &X) {
  setSize();
  checkSEMatrixSize(X, r_, d_, n_);
  copyEigenMatrixToROPTLIBVariable(X, MySEVector, r_ * (d_ + 1) * n_);
}

void LiftedSEVector::setSize() {
  LiftedSEVector::setSizeFromProductElement(MySEVector, &r_, &d_, &n_);
}

void LiftedSEVector::setSizeFromProductElement(
    ROPTLIB::ProductElement *productElement, unsigned int *row,
    unsigned int *col, unsigned int *num_el) {
  auto *T =
      dynamic_cast<ROPTLIB::ProductElement *>(productElement->GetElement(0));
  const int *sizes = T->GetElement(0)->Getsize();
  *row = static_cast<unsigned int>(sizes[0]);
  *col = static_cast<unsigned int>(sizes[1]);
  *num_el = static_cast<unsigned int>(productElement->GetNumofElement());
}

LiftedRAVector::LiftedRAVector(unsigned int r, unsigned int d, unsigned int n,
                               unsigned int l, unsigned int b)
    : LiftedSEVector(r, d, n) {
  ObliqueRangeVector = new ROPTLIB::ObliqueVector(r, l);
  EuclideanLandmarkVector = new ROPTLIB::EucVector(r, b);
  MyOBVector = new ROPTLIB::ProductElement(1, ObliqueRangeVector, 1);
  MyEVector = new ROPTLIB::ProductElement(1, EuclideanLandmarkVector, 1);
  MyRAVector = new ROPTLIB::ProductElement(3, MySEVector, 1, MyOBVector, 1,
                                           MyEVector, 1);
}

LiftedRAVector::~LiftedRAVector() {
  // Avoid memory leak
  delete ObliqueRangeVector;
  delete EuclideanLandmarkVector;
  delete MyOBVector;
  delete MyEVector;
  delete MyRAVector;
}

Matrix LiftedRAVector::getData() {
  setSize();
  return Eigen::Map<Matrix>(const_cast<double *>(MyRAVector->ObtainReadData()),
                            r_, n_ * (d_ + 1) + l_ + b_);
}

void LiftedRAVector::setData(const Matrix &X) {
  setSize();
  auto [X_SE_R, X_OB, X_SE_t, X_E] = partitionRAMatrix(X, r_, d_, n_, l_, b_);
  Matrix X_SE = createSEMatrix(X_SE_R, X_SE_t);
  copyEigenMatrixToROPTLIBVariable(X_SE, MySEVector, r_ * (d_ + 1) * n_);
  copyEigenMatrixToROPTLIBVariable(X_OB, MyOBVector, r_ * l_);
  copyEigenMatrixToROPTLIBVariable(X_E, MyEVector, r_ * b_);
  copyEigenMatrixToROPTLIBVariable(X, MyRAVector,
                                   r_ * ((d_ + 1) * n_ + l_ + b_));
}

void LiftedRAVector::setSize() {
  // set r, d, n
  LiftedSEVector::setSize();
  // set l and b
  unsigned int row, col;
  LiftedSEVector::setSizeFromProductElement(MyOBVector, &row, &col, &l_);
  LiftedSEVector::setSizeFromProductElement(MyEVector, &row, &col, &b_);
}

} // namespace DCORA
