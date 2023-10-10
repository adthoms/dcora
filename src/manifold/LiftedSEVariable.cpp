/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DCORA/DCORA_utils.h>
#include <DCORA/manifold/LiftedSEVariable.h>
#include <glog/logging.h>

using namespace std;
using namespace ROPTLIB;

namespace DCORA {
LiftedSEVariable::LiftedSEVariable(unsigned int r, unsigned int d, unsigned int n) :
    r_(r), d_(d), n_(n),
    rotation_var_(std::make_unique<ROPTLIB::StieVariable>((int) r, (int) d)),
    translation_var_(std::make_unique<ROPTLIB::EucVariable>((int) r)),
    pose_var_(std::make_unique<ROPTLIB::ProductElement>(2, rotation_var_.get(), 1, translation_var_.get(), 1)),
    varSE_(std::make_shared<ROPTLIB::ProductElement>(1, pose_var_.get(), n)),
    X_SE_((double *) varSE_->ObtainWriteEntireData(), r, (d + 1) * n) {
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

LiftedSEVariable::LiftedSEVariable(const LiftedSEVariable &other) :
    LiftedSEVariable(other.r(), other.d(), other.n()) {
  setData(other.getData());
}

LiftedSEVariable &LiftedSEVariable::operator=(const LiftedSEVariable &other) {
  r_ = other.r();
  d_ = other.d();
  n_ = other.n();
  rotation_var_ = std::make_unique<ROPTLIB::StieVariable>((int) r_, (int) d_);
  translation_var_ = std::make_unique<ROPTLIB::EucVariable>((int) r_);
  pose_var_ = std::make_unique<ROPTLIB::ProductElement>(2, rotation_var_.get(), 1, translation_var_.get(), 1);
  varSE_ = std::make_shared<ROPTLIB::ProductElement>(1, pose_var_.get(), n_);
  // Update the Eigen::Map object using the "placement new object"
  // Reference: https://eigen.tuxfamily.org/dox/group__TutorialMapClass.html#TutorialMapPlacementNew
  new(&X_SE_) Eigen::Map<Matrix>((double *) varSE_->ObtainWriteEntireData(), r_, (d_ + 1) * n_);
  setData(other.getData());
  return *this;
}

Matrix LiftedSEVariable::getData() const {
  return X_SE_;
}

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

}  // namespace DCORA