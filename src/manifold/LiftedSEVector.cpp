/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DCORA/manifold/LiftedSEVector.h>
#include <glog/logging.h>

namespace DCORA {

LiftedSEVector::LiftedSEVector(int r, int d, int n) {
  StiefelVector = new ROPTLIB::StieVector(r, d);
  EuclideanVector = new ROPTLIB::EucVector(r);
  CartanVector = new ROPTLIB::ProductElement(2, StiefelVector, 1, EuclideanVector, 1);
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
  return Eigen::Map<Matrix>((double *)MySEVector->ObtainReadData(), r_,
                            n_ * (d_ + 1));
}
void LiftedSEVector::setData(const Matrix &X) {
  setSize();
  checkSEMatrixSize(X, r_, d_, n_);
  copyEigenMatrixToROPTLIBVariable(X, MySEVector, r_ * (d_ + 1) * n_);
}

void LiftedSEVector::setSize() {
  LiftedSEVector::setSizeFromProductElement(MySEVector, r_, d_, n_);
}

void LiftedSEVector::setSizeFromProductElement(ROPTLIB::ProductElement* productElement, unsigned int &row, unsigned int &col, unsigned int &num_el) {
  auto *T = dynamic_cast<ROPTLIB::ProductElement *>(productElement->GetElement(0));
  const int *sizes = T->GetElement(0)->Getsize();
  row = sizes[0];
  col = sizes[1];
  num_el = productElement->GetNumofElement();
}

}  // namespace DCORA
