/* ----------------------------------------------------------------------------
 * Copyright 2023, University of California Los Angeles, Los Angeles, CA 90095
 * All Rights Reserved
 * Authors: Alex Thoms and Alan Papalia
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DCORA/DCORA_utils.h>
#include <DCORA/manifold/LiftedRAVariable.h>
#include <glog/logging.h>

namespace DCORA {
LiftedRAVariable::LiftedRAVariable(unsigned int r, unsigned int d, unsigned int n, unsigned int l, unsigned int b) :
    LiftedSEVariable(r, d, n), l_(l), b_(b),
    translation_ranges_var_(std::make_unique<ROPTLIB::ObliqueVariable>((int) r, (int) l)),
    translation_landmark_var_(std::make_unique<ROPTLIB::EucVariable>((int) r)),
    varOB_(std::make_unique<ROPTLIB::ProductElement>(1, translation_ranges_var_.get(), 1)),
    varE_(std::make_unique<ROPTLIB::ProductElement>(1, translation_landmark_var_.get(), b)),
    varRA_(std::make_unique<ROPTLIB::ProductElement>(3, varSE_.get(), 1, varOB_.get(), 1, varE_.get(), 1)),
    X_OB_((double *) varOB_->ObtainWriteEntireData(), r, l),
    X_E_((double *) varE_->ObtainWriteEntireData(), r, b),
    X_RA_((double *) varRA_->ObtainWriteEntireData(), r, (d + 1) * n + l + b) {
  for (unsigned int i = 0; i < l_; ++i) {
    rangeUnitSphereVariable(i) = Vector::Zero(r_);
  }
  for (unsigned int i = 0; i < b_; ++i) {
    landmarkTranslation(i) = Vector::Zero(r_);
  }
  auto [X_SE_R, X_SE_t] = partitionSEMatrix(X_SE_, r, d, n);
  X_RA_ = createRAMatrix(X_SE_R, X_OB_, X_SE_t, X_E_);
}

LiftedRAVariable::LiftedRAVariable(const LiftedPoseArray &poses, const LiftedTranslationArray &landmarks, const LiftedTranslationArray &ranges)
    : LiftedRAVariable(poses.r(), poses.d(), poses.n(), landmarks.n(), ranges.n()) /* TODO: update construction */ {
  Matrix X_SE = poses.getData(); // TODO: update for LiftedRangeAidedArray
  auto [X_SE_R, X_SE_t] = partitionSEMatrix(X_SE, r_, d_, n_); // TODO: update for LiftedRangeAidedArray
  setData(createRAMatrix(X_SE_R, ranges.getData(), X_SE_t, landmarks.getData())); // TODO: update for LiftedRangeAidedArray
}

LiftedRAVariable::LiftedRAVariable(const LiftedRAVariable &other) :
    LiftedRAVariable(other.r(), other.d(), other.n(), other.l(), other.b()) {
  setData(other.getData());
}

LiftedRAVariable &LiftedRAVariable::operator=(const LiftedRAVariable &other) {
  LiftedSEVariable::operator=(other);
  l_ = other.l();
  b_ = other.b();
  translation_ranges_var_ = std::make_unique<ROPTLIB::ObliqueVariable>((int) r_, (int) l_);
  translation_landmark_var_ = std::make_unique<ROPTLIB::EucVariable>((int) r_);
  varOB_ = std::make_unique<ROPTLIB::ProductElement>(1, translation_ranges_var_.get(), 1);
  varE_ = std::make_unique<ROPTLIB::ProductElement>(1, translation_landmark_var_.get(), b_);
  varRA_ = std::make_unique<ROPTLIB::ProductElement>(3, varSE_.get(), 1, varOB_.get(), 1, varE_.get(), 1);
  new(&X_OB_) Eigen::Map<Matrix>((double *) varOB_->ObtainWriteEntireData(), r_, l_);
  new(&X_E_) Eigen::Map<Matrix>((double *) varE_->ObtainWriteEntireData(), r_, b_);
  new(&X_RA_) Eigen::Map<Matrix>((double *) varRA_->ObtainWriteEntireData(), r_, (d_ + 1) * n_ + l_ + b_);
  setData(other.getData());
  return *this;
}

Matrix LiftedRAVariable::getData() const {
  return X_RA_;
}

void LiftedRAVariable::setData(const Matrix &X) {
  auto [X_SE_R, X_OB, X_SE_t, X_E] = partitionRAMatrix(X, r_, d_, n_, l_, b_);
  Matrix X_SE = createSEMatrix(X_SE_R, X_SE_t);
  copyEigenMatrixToROPTLIBVariable(X_SE, varSE_.get(), r_ * (d_ + 1) * n_);
  copyEigenMatrixToROPTLIBVariable(X_OB, varOB_.get(), r_ * l_);
  copyEigenMatrixToROPTLIBVariable(X_E, varE_.get(), r_ * b_);
  copyEigenMatrixToROPTLIBVariable(X, varRA_.get(), r_ * ((d_ + 1) * n_ + l_ + b_));
}

Eigen::Ref<Vector> LiftedRAVariable::landmarkTranslation(unsigned int index) {
  CHECK(index < b_);
  auto Xi = X_E_.block(0, index, r_, 1);
  return Xi.col(0);
}

Vector LiftedRAVariable::landmarkTranslation(unsigned int index) const {
  CHECK(index < b_);
  auto Xi = X_E_.block(0, index, r_, 1);
  return Xi.col(0);
}

Eigen::Ref<Vector> LiftedRAVariable::rangeUnitSphereVariable(unsigned int index) {
  CHECK(index < l_);
  auto Xi = X_OB_.block(0, index, r_, 1);
  return Xi.col(0);
}

Vector LiftedRAVariable::rangeUnitSphereVariable(unsigned int index) const {
  CHECK(index < l_);
  auto Xi = X_OB_.block(0, index, r_, 1);
  return Xi.col(0);
}

}  // namespace DCORA