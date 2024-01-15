/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DCORA/DCORA_utils.h>
#include <DCORA/manifold/LiftedVariable.h>
#include <glog/logging.h>

namespace DCORA {

LiftedSEVariable::LiftedSEVariable(unsigned int r, unsigned int d,
                                   unsigned int n)
    : r_(r),
      d_(d),
      n_(n),
      rotation_var_(std::make_unique<ROPTLIB::StieVariable>(r, d)),
      translation_var_(std::make_unique<ROPTLIB::EucVariable>(r)),
      pose_var_(std::make_unique<ROPTLIB::ProductElement>(
          2, rotation_var_.get(), 1, translation_var_.get(), 1)),
      varSE_(std::make_shared<ROPTLIB::ProductElement>(1, pose_var_.get(), n)),
      X_SE_(const_cast<double *>(varSE_->ObtainWriteEntireData()), r,
            (d + 1) * n) {
  Matrix Yinit = Matrix::Zero(r_, d_);
  Yinit.block(0, 0, d_, d_) = Matrix::Identity(d_, d_);
  for (unsigned int i = 0; i < n; ++i) {
    rotation(i) = Yinit;
    translation(i) = Vector::Zero(r_);
  }
}

LiftedSEVariable::LiftedSEVariable(const LiftedPoseArray &poses)
    : LiftedSEVariable(poses.r(), poses.d(), poses.n()) {
  setData(poses.getData());
}

LiftedSEVariable::LiftedSEVariable(const LiftedSEVariable &other)
    : LiftedSEVariable(other.r(), other.d(), other.n()) {
  setData(other.getData());
}

LiftedSEVariable &LiftedSEVariable::operator=(const LiftedSEVariable &other) {
  r_ = other.r();
  d_ = other.d();
  n_ = other.n();
  rotation_var_ = std::make_unique<ROPTLIB::StieVariable>(r_, d_);
  translation_var_ = std::make_unique<ROPTLIB::EucVariable>(r_);
  pose_var_ = std::make_unique<ROPTLIB::ProductElement>(
      2, rotation_var_.get(), 1, translation_var_.get(), 1);
  varSE_ = std::make_shared<ROPTLIB::ProductElement>(1, pose_var_.get(), n_);
  new (&X_SE_) Eigen::Map<Matrix>(
      const_cast<double *>(varSE_->ObtainWriteEntireData()), r_, (d_ + 1) * n_);
  setData(other.getData());
  return *this;
}

Matrix LiftedSEVariable::getData() const { return X_SE_; }

void LiftedSEVariable::setData(const Matrix &X) {
  checkSEMatrixSize(X, r_, d_, n_);
  copyEigenMatrixToROPTLIBVariable(X, varSE_.get(), r_ * (d_ + 1) * n_);
}

Eigen::Ref<Matrix> LiftedSEVariable::pose(unsigned int index) {
  CHECK(index < n_);
  return X_SE_.block(0, index * (d_ + 1), r_, d_ + 1);
}

Matrix LiftedSEVariable::pose(unsigned int index) const {
  CHECK(index < n_);
  return X_SE_.block(0, index * (d_ + 1), r_, d_ + 1);
}

Eigen::Ref<Matrix> LiftedSEVariable::rotation(unsigned int index) {
  CHECK(index < n_);
  auto Xi = X_SE_.block(0, index * (d_ + 1), r_, d_ + 1);
  return Xi.block(0, 0, r_, d_);
}

Matrix LiftedSEVariable::rotation(unsigned int index) const {
  CHECK(index < n_);
  auto Xi = X_SE_.block(0, index * (d_ + 1), r_, d_ + 1);
  return Xi.block(0, 0, r_, d_);
}

Eigen::Ref<Vector> LiftedSEVariable::translation(unsigned int index) {
  CHECK(index < n_);
  auto Xi = X_SE_.block(0, index * (d_ + 1), r_, d_ + 1);
  return Xi.col(d_);
}

Vector LiftedSEVariable::translation(unsigned int index) const {
  CHECK(index < n_);
  auto Xi = X_SE_.block(0, index * (d_ + 1), r_, d_ + 1);
  return Xi.col(d_);
}

LiftedRAVariable::LiftedRAVariable(unsigned int r, unsigned int d,
                                   unsigned int n, unsigned int l,
                                   unsigned int b)
    : LiftedSEVariable(r, d, n),
      l_(l),
      b_(b),
      ranges_var_(std::make_unique<ROPTLIB::ObliqueVariable>(r, l)),
      landmark_var_(std::make_unique<ROPTLIB::EucVariable>(r, b)),
      varOB_(
          std::make_unique<ROPTLIB::ProductElement>(1, ranges_var_.get(), 1)),
      varE_(
          std::make_unique<ROPTLIB::ProductElement>(1, landmark_var_.get(), 1)),
      varRA_(std::make_unique<ROPTLIB::ProductElement>(
          3, varSE_.get(), 1, varOB_.get(), 1, varE_.get(), 1)),
      X_OB_(const_cast<double *>(varOB_->ObtainWriteEntireData()), r, l),
      X_E_(const_cast<double *>(varE_->ObtainWriteEntireData()), r, b),
      X_RA_(const_cast<double *>(varRA_->ObtainWriteEntireData()), r,
            (d + 1) * n + l + b) {
  for (unsigned int i = 0; i < l_; ++i) {
    rangeUnitSphereVariable(i) = Vector::Zero(r_);
  }
  for (unsigned int i = 0; i < b_; ++i) {
    landmarkTranslation(i) = Vector::Zero(r_);
  }
  auto [X_SE_R, X_SE_t] = partitionSEMatrix(X_SE_, r, d, n);
  X_RA_ = createRAMatrix(X_SE_R, X_OB_, X_SE_t, X_E_);
}

LiftedRAVariable::LiftedRAVariable(const LiftedRangeAidedArray &rangeAidedArray)
    : LiftedRAVariable(rangeAidedArray.r(), rangeAidedArray.d(),
                       rangeAidedArray.n(), rangeAidedArray.l(),
                       rangeAidedArray.b()) {
  auto [X_SE_R, X_SE_t] = partitionSEMatrix(
      rangeAidedArray.GetLiftedPoseArray()->getData(), r_, d_, n_);
  Matrix X_OB = rangeAidedArray.GetLiftedRangeArray()->getData();
  Matrix X_E = rangeAidedArray.GetLiftedLandmarkArray()->getData();
  setData(createRAMatrix(X_SE_R, X_OB, X_SE_t, X_E));
}

LiftedRAVariable::LiftedRAVariable(const LiftedRAVariable &other)
    : LiftedRAVariable(other.r(), other.d(), other.n(), other.l(), other.b()) {
  setData(other.getData());
}

LiftedRAVariable &LiftedRAVariable::operator=(const LiftedRAVariable &other) {
  LiftedSEVariable::operator=(other);
  l_ = other.l();
  b_ = other.b();
  ranges_var_ = std::make_unique<ROPTLIB::ObliqueVariable>(r_, l_);
  landmark_var_ = std::make_unique<ROPTLIB::EucVariable>(r_, b_);
  varOB_ = std::make_unique<ROPTLIB::ProductElement>(1, ranges_var_.get(), 1);
  varE_ = std::make_unique<ROPTLIB::ProductElement>(1, landmark_var_.get(), 1);
  varRA_ = std::make_unique<ROPTLIB::ProductElement>(
      3, varSE_.get(), 1, varOB_.get(), 1, varE_.get(), 1);
  new (&X_OB_) Eigen::Map<Matrix>(
      const_cast<double *>(varOB_->ObtainWriteEntireData()), r_, l_);
  new (&X_E_) Eigen::Map<Matrix>(
      const_cast<double *>(varE_->ObtainWriteEntireData()), r_, b_);
  new (&X_RA_)
      Eigen::Map<Matrix>(const_cast<double *>(varRA_->ObtainWriteEntireData()),
                         r_, (d_ + 1) * n_ + l_ + b_);
  setData(other.getData());
  return *this;
}

Matrix LiftedRAVariable::getData() const { return X_RA_; }

void LiftedRAVariable::setData(const Matrix &X) {
  auto [X_SE_R, X_OB, X_SE_t, X_E] = partitionRAMatrix(X, r_, d_, n_, l_, b_);
  Matrix X_SE = createSEMatrix(X_SE_R, X_SE_t);
  copyEigenMatrixToROPTLIBVariable(X_SE, varSE_.get(), r_ * (d_ + 1) * n_);
  copyEigenMatrixToROPTLIBVariable(X_OB, varOB_.get(), r_ * l_);
  copyEigenMatrixToROPTLIBVariable(X_E, varE_.get(), r_ * b_);
  copyEigenMatrixToROPTLIBVariable(X, varRA_.get(),
                                   r_ * ((d_ + 1) * n_ + l_ + b_));
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

Eigen::Ref<Vector>
LiftedRAVariable::rangeUnitSphereVariable(unsigned int index) {
  CHECK(index < l_);
  auto Xi = X_OB_.block(0, index, r_, 1);
  return Xi.col(0);
}

Vector LiftedRAVariable::rangeUnitSphereVariable(unsigned int index) const {
  CHECK(index < l_);
  auto Xi = X_OB_.block(0, index, r_, 1);
  return Xi.col(0);
}

} // namespace DCORA
