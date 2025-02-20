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

#include "DCORA/manifold/Elements.h"

#include <glog/logging.h>

#include "DCORA/DCORA_utils.h"

namespace DCORA {

LiftedArray::LiftedArray(unsigned int r, unsigned int d, unsigned int n)
    : r_(r), d_(d), n_(n), dim_(d + 1) {
  X_ = Matrix::Zero(r_, dim_ * n_);
}

Matrix LiftedArray::getData() const { return X_; }

void LiftedArray::setData(const Matrix &X) {
  CHECK_EQ(X.rows(), r_);
  CHECK_EQ(X.cols(), dim_ * n_);
  X_ = X;
}

void LiftedArray::setDataToZero() { X_.setZero(); }

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

double LiftedArray::averageTranslationDistance(const LiftedArray &array1,
                                               const LiftedArray &array2) {
  CHECK_EQ(array1.d(), array2.d());
  CHECK_EQ(array1.n(), array2.n());
  double average_distance = 0;
  for (unsigned int i = 0; i < array1.n(); ++i) {
    average_distance += (array1.translation(i) - array2.translation(i)).norm();
  }
  average_distance = average_distance / static_cast<double>(array1.n());
  return average_distance;
}

double LiftedArray::maxTranslationDistance(const LiftedArray &array1,
                                           const LiftedArray &array2) {
  CHECK_EQ(array1.d(), array2.d());
  CHECK_EQ(array1.n(), array2.n());
  double max_distance = 0;
  for (unsigned int i = 0; i < array1.n(); ++i) {
    max_distance = std::max(
        max_distance, (array1.translation(i) - array2.translation(i)).norm());
  }
  return max_distance;
}

LiftedPoseArray::LiftedPoseArray(unsigned int r, unsigned int d, unsigned int n)
    : LiftedArray(r, d, n) {
  Matrix Yinit = Matrix::Zero(r_, d_);
  Yinit.block(0, 0, d_, d_) = Matrix::Identity(d_, d_);
  for (unsigned int i = 0; i < n; ++i) {
    rotation(i) = Yinit;
    translation(i) = Vector::Zero(r_);
  }
}

void LiftedPoseArray::setRandomData() {
  Matrix M = Matrix::Random(r_, (d_ + 1) * n_);
  X_ = projectToSEMatrix(M, r_, d_, n_);
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

LiftedPointArray::LiftedPointArray(unsigned int r, unsigned int d,
                                   unsigned int n)
    : LiftedArray(r, d, n) {
  dim_ = 1;
  X_ = Matrix::Zero(r_, dim_ * n_);
}

LiftedRangeAidedArray::LiftedRangeAidedArray(unsigned int r, unsigned int d,
                                             unsigned int n, unsigned int l,
                                             unsigned int b,
                                             GraphType graphType)
    : r_(r),
      d_(d),
      n_(n),
      l_(l),
      b_(b),
      graph_type_(graphType),
      poses_(std::make_unique<LiftedPoseArray>(r, d, n)),
      unit_spheres_(std::make_unique<LiftedPointArray>(r, d, l)),
      landmarks_(std::make_unique<LiftedPointArray>(r, d, b)) {}

LiftedRangeAidedArray::LiftedRangeAidedArray(const LiftedRangeAidedArray &other)
    : LiftedRangeAidedArray(other.r(), other.d(), other.n(), other.l(),
                            other.b(), other.graphType()) {
  setData(other.getData());
}

LiftedRangeAidedArray &
LiftedRangeAidedArray::operator=(const LiftedRangeAidedArray &other) {
  r_ = other.r();
  d_ = other.d();
  n_ = other.n();
  l_ = other.l();
  b_ = other.b();
  graph_type_ = other.graphType();
  poses_ = std::make_unique<LiftedPoseArray>(r_, d_, n_);
  unit_spheres_ = std::make_unique<LiftedPointArray>(r_, d_, l_);
  landmarks_ = std::make_unique<LiftedPointArray>(r_, d_, b_);
  poses_->setData(other.GetLiftedPoseArray()->getData());
  unit_spheres_->setData(other.GetLiftedUnitSphereArray()->getData());
  landmarks_->setData(other.GetLiftedLandmarkArray()->getData());
  return *this;
}

bool LiftedRangeAidedArray::isPGOCompatible() const {
  if (graph_type_ == GraphType::RangeAidedSLAMGraph)
    return false;

  if (l_ > 0 || b_ > 0)
    LOG(FATAL) << "Error: LiftedRangeAidedArray cannot contain unit spheres "
                  "or landmarks when graph type is set to: "
               << GraphTypeToString(graph_type_) << "!";

  return true;
}

Matrix LiftedRangeAidedArray::getData() const {
  return (isPGOCompatible()) ? getDataSE() : getDataRA();
}

Matrix LiftedRangeAidedArray::getDataSE() const { return poses_->getData(); }

Matrix LiftedRangeAidedArray::getDataRA() const {
  auto [X_SE_R, X_SE_t] = partitionSEMatrix(poses_->getData(), r_, d_, n_);
  return createRAMatrix(X_SE_R, unit_spheres_->getData(), X_SE_t,
                        landmarks_->getData());
}

void LiftedRangeAidedArray::setData(const Matrix &X) {
  if (isPGOCompatible())
    setDataSE(X);
  else
    setDataRA(X);
}

void LiftedRangeAidedArray::setDataSE(const Matrix &X) {
  CHECK_EQ(X.rows(), r_);
  CHECK_EQ(X.cols(), (d_ + 1) * n_);
  poses_->setData(X);
}

void LiftedRangeAidedArray::setDataRA(const Matrix &X) {
  CHECK_EQ(X.rows(), r_);
  CHECK_EQ(X.cols(), (d_ + 1) * n_ + l_ + b_);
  auto [X_SE_R, X_OB, X_SE_t, X_E] = partitionRAMatrix(X, r_, d_, n_, l_, b_);
  poses_->setData(createSEMatrix(X_SE_R, X_SE_t));
  unit_spheres_->setData(X_OB);
  landmarks_->setData(X_E);
}

void LiftedRangeAidedArray::setRandomData() {
  Matrix M = Matrix::Random(r_, (d_ + 1) * n_);
  Matrix X_SE_rand = projectToSEMatrix(M, r_, d_, n_);
  Matrix X_OB_rand = randomObliqueVariable(r_, l_);
  Matrix X_E_rand = randomEuclideanVariable(r_, b_);
  poses_->setData(X_SE_rand);
  unit_spheres_->setData(X_OB_rand);
  landmarks_->setData(X_E_rand);
}

void LiftedRangeAidedArray::setDataToZero() {
  poses_->setDataToZero();
  unit_spheres_->setDataToZero();
  landmarks_->setDataToZero();
}

Eigen::Ref<Matrix> LiftedRangeAidedArray::pose(unsigned int index) {
  CHECK_LT(index, n_);
  return poses_->pose(index);
}

Matrix LiftedRangeAidedArray::pose(unsigned int index) const {
  CHECK_LT(index, n_);
  return poses_->pose(index);
}

Eigen::Ref<Matrix> LiftedRangeAidedArray::rotation(unsigned int index) {
  CHECK_LT(index, n_);
  return poses_->rotation(index);
}

Matrix LiftedRangeAidedArray::rotation(unsigned int index) const {
  CHECK_LT(index, n_);
  return poses_->rotation(index);
}

Eigen::Ref<Vector> LiftedRangeAidedArray::translation(unsigned int index) {
  CHECK_LT(index, n_);
  return poses_->translation(index);
}

Vector LiftedRangeAidedArray::translation(unsigned int index) const {
  CHECK_LT(index, n_);
  return poses_->translation(index);
}

Eigen::Ref<Vector> LiftedRangeAidedArray::unitSphere(unsigned int index) {
  CHECK_LT(index, l_);
  return unit_spheres_->translation(index);
}

Vector LiftedRangeAidedArray::unitSphere(unsigned int index) const {
  CHECK_LT(index, l_);
  return unit_spheres_->translation(index);
}

Eigen::Ref<Vector> LiftedRangeAidedArray::landmark(unsigned int index) {
  CHECK_LT(index, b_);
  return landmarks_->translation(index);
}

Vector LiftedRangeAidedArray::landmark(unsigned int index) const {
  CHECK_LT(index, b_);
  return landmarks_->translation(index);
}

void LiftedRangeAidedArray::setLiftedPoseArray(
    const LiftedPoseArray &liftedPoseArray) {
  CHECK_EQ(liftedPoseArray.r(), r_);
  CHECK_EQ(liftedPoseArray.d(), d_);
  CHECK_EQ(liftedPoseArray.n(), n_);
  poses_->setData(liftedPoseArray.getData());
}

void LiftedRangeAidedArray::setLiftedUnitSphereArray(
    const LiftedPointArray &liftedUnitSphereArray) {
  CHECK_EQ(liftedUnitSphereArray.r(), r_);
  CHECK_EQ(liftedUnitSphereArray.d(), d_);
  CHECK_EQ(liftedUnitSphereArray.n(), l_);
  CHECK(liftedUnitSphereArray.getData().isApprox(
      projectToObliqueManifold(liftedUnitSphereArray.getData())));
  unit_spheres_->setData(liftedUnitSphereArray.getData());
}

void LiftedRangeAidedArray::setLiftedLandmarkArray(
    const LiftedPointArray &liftedLandmarkArray) {
  CHECK_EQ(liftedLandmarkArray.r(), r_);
  CHECK_EQ(liftedLandmarkArray.d(), d_);
  CHECK_EQ(liftedLandmarkArray.n(), b_);
  landmarks_->setData(liftedLandmarkArray.getData());
}

PoseArray RangeAidedArray::getPoseArray() const {
  CHECK_EQ(r_, d_);
  PoseArray poseArray(d_, n_);
  poseArray.setData(poses_->getData());
  return poseArray;
}

PointArray RangeAidedArray::getUnitSphereArray() const {
  CHECK_EQ(r_, d_);
  PointArray unitSphereArray(d_, l_);
  unitSphereArray.setData(unit_spheres_->getData());
  return unitSphereArray;
}

PointArray RangeAidedArray::getLandmarkArray() const {
  CHECK_EQ(r_, d_);
  PointArray landmarkArray(d_, b_);
  landmarkArray.setData(landmarks_->getData());
  return landmarkArray;
}

Pose::Pose(const Matrix &T) : Pose(T.rows()) {
  CHECK_EQ(T.rows(), d_);
  CHECK_EQ(T.cols(), dim_);
  setData(T);
}

Pose Pose::Identity(unsigned int d) { return Pose(d); }

Pose Pose::identity() const { return Pose(d_); }

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

Point::Point(const Vector &P) : Point(P.rows()) { setData(P); }

Point Point::ZeroVector(unsigned int d) { return Point(d); }

Point Point::zeroVector() const { return Point(d_); }

Vector Point::vector() const { return translation(); }

} // namespace DCORA
