/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include "DCORA/manifold/Poses.h"
#include "DCORA/DCORA_utils.h"
#include <glog/logging.h>

namespace DCORA {

LiftedArray::LiftedArray(unsigned int r, unsigned int d, unsigned int n) :
    r_(r), d_(d), n_(n), dim_(d+1) {
  X_ = Matrix::Zero(r_, dim_ * n_);
}

Matrix LiftedArray::getData() const {
  return X_;
}

void LiftedArray::setData(const Matrix &X) {
  CHECK_EQ(X.rows(), r_);
  CHECK_EQ(X.cols(), dim_ * n_);
  X_ = X;
}

Eigen::Ref<Vector> LiftedArray::translation(unsigned int index) {
  CHECK_LT(index, n_);
  auto Xi = X_.block(0, index * dim_, r_, dim_);
  return Xi.col(dim_ - 1);
}

Vector LiftedArray::translation(unsigned int index) const {
  CHECK_LT(index, n_);
  auto Xi = X_.block(0, index * dim_, r_, dim_);
  return Xi.col(dim_ - 1);
}

double LiftedArray::averageTranslationDistance(const LiftedArray &array1, const LiftedArray &array2) {
  CHECK_EQ(array1.d(), array2.d());
  CHECK_EQ(array1.n(), array2.n());
  double average_distance = 0;
  for (unsigned int i = 0; i < array1.n(); ++i) {
    average_distance += (array1.translation(i) - array2.translation(i)).norm();
  }
  average_distance = average_distance / (double) array1.n();
  return average_distance;
}

double LiftedArray::maxTranslationDistance(const LiftedArray &array1, const LiftedArray &array2) {
  CHECK_EQ(array1.d(), array2.d());
  CHECK_EQ(array1.n(), array2.n());
  double max_distance = 0;
  for (unsigned int i = 0; i < array1.n(); ++i) {
    max_distance = std::max(max_distance, (array1.translation(i) - array2.translation(i)).norm());
  }
  return max_distance;
}

LiftedPoseArray::LiftedPoseArray(unsigned int r, unsigned int d, unsigned int n) :
    LiftedArray(r, d, n) {
  Matrix Yinit = Matrix::Zero(r_, d_);
  Yinit.block(0, 0, d_, d_) = Matrix::Identity(d_, d_);
  for (unsigned int i = 0; i < n; ++i) {
    rotation(i) = Yinit;
    translation(i) = Vector::Zero(r_);
  }
}

void LiftedPoseArray::checkData() const {
  for (unsigned i = 0; i < n_; ++i) {
    checkStiefelMatrix(rotation(i));
  }
}

Eigen::Ref<Matrix> LiftedPoseArray::pose(unsigned int index) {
  CHECK_LT(index, n_);
  return X_.block(0, index * dim_, r_, dim_);
}

Matrix LiftedPoseArray::pose(unsigned int index) const {
  CHECK_LT(index, n_);
  return X_.block(0, index * dim_, r_, dim_);
}

Eigen::Ref<Matrix> LiftedPoseArray::rotation(unsigned int index) {
  CHECK_LT(index, n_);
  auto Xi = X_.block(0, index * dim_, r_, dim_);
  return Xi.block(0, 0, r_, d_);
}

Matrix LiftedPoseArray::rotation(unsigned int index) const {
  CHECK_LT(index, n_);
  auto Xi = X_.block(0, index * dim_, r_, dim_);
  return Xi.block(0, 0, r_, d_);
}

LiftedTranslationArray::LiftedTranslationArray(unsigned int r, unsigned int d, unsigned int n) :
    LiftedArray(r, d, n) {
  dim_ = 1;
  X_ = Matrix::Zero(r_, dim_ * n_);
}

LiftedRangeAidedArray::LiftedRangeAidedArray(unsigned int r, unsigned int d, unsigned int n, unsigned int l, unsigned int b) :
    r_(r), d_(d), n_(n), l_(l), b_(b),
    poses_(std::make_unique<LiftedPoseArray>(r, d, n)),
    ranges_(std::make_unique<LiftedUnitSphereAuxiliaryArray>(r, d, l)),
    landmarks_(std::make_unique<LiftedLandmarkArray>(r, d, b)) {}

LiftedRangeAidedArray::LiftedRangeAidedArray(const LiftedRangeAidedArray &other) :
    LiftedRangeAidedArray(other.r(), other.d(), other.n(), other.l(), other.b()) {
  setData(other.getData());
}

LiftedRangeAidedArray &LiftedRangeAidedArray::operator=(const LiftedRangeAidedArray &other) {
  r_ = other.r();
  d_ = other.d();
  n_ = other.n();
  l_ = other.l();
  b_ = other.b();
  poses_ = std::make_unique<LiftedPoseArray>(r_, d_, n_);
  ranges_ = std::make_unique<LiftedUnitSphereAuxiliaryArray>(r_, d_, l_);
  landmarks_ = std::make_unique<LiftedLandmarkArray>(r_, d_, b_);
  poses_->setData(other.GetLiftedPoseArray()->getData());
  ranges_->setData(other.GetLiftedUnitSphereAuxiliaryArray()->getData());
  landmarks_->setData(other.GetLiftedLandmarkArray()->getData());
  return *this;
}

Matrix LiftedRangeAidedArray::getData() const {
  auto [X_SE_R, X_SE_t] = partitionSEMatrix(poses_->getData(), r_, d_, n_);
  return createRAMatrix(X_SE_R, ranges_->getData(), X_SE_t, landmarks_->getData());
}

void LiftedRangeAidedArray::setData(const Matrix &X) {
  CHECK_EQ(X.rows(), r_);
  CHECK_EQ(X.cols(), (d_ + 1) * n_ + l_ + b_);

  auto [X_SE_R, X_OB, X_SE_t, X_E] = partitionRAMatrix(X, r_, d_, n_, l_, b_);
  poses_->setData(createSEMatrix(X_SE_R, X_SE_t));
  ranges_->setData(X_OB);
  landmarks_->setData(X_E);
}

Pose::Pose(const Matrix &T)
    : Pose(T.rows()) {
  CHECK_EQ(T.rows(), d_);
  CHECK_EQ(T.cols(), dim_);
  setData(T);
}

Pose Pose::Identity(unsigned int d) {
  return Pose(d);
}

Pose Pose::identity() const {
  return Pose(d_);
}

Pose Pose::inverse() const {
  Matrix TInv = matrix().inverse();
  return Pose(TInv.block(0, 0, d_, dim_));
}

Pose Pose::operator*(const Pose &other) const {
  CHECK_EQ(d(), other.d());
  Matrix Tr = matrix() * other.matrix();
  return Pose(Tr.block(0, 0, d_, dim_));
}

Matrix Pose::matrix() const {
  Matrix T = Matrix::Identity(dim_, dim_);
  T.block(0, 0, d_, d_) = rotation();
  T.block(0, d_, d_, 1) = translation();
  return T;
}

Translation::Translation(const Vector &P)
    : Translation(P.rows()) {
  setData(P);
}

Translation Translation::ZeroVector(unsigned int d) {
  return Translation(d);
}

Translation Translation::zeroVector() const {
  return Translation(d_);
}

Vector Translation::vector() const {
  return translation();
}

}  // namespace DCORA
