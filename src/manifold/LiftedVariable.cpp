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
      varSE_(std::make_unique<ROPTLIB::ProductElement>(1, pose_var_.get(), n)),
      X_SE_(const_cast<double *>(varSE_->ObtainWriteEntireData()), r,
            (d + 1) * n) {
  Matrix Yinit = Matrix::Zero(r_, d_);
  Yinit.block(0, 0, d_, d_) = Matrix::Identity(d_, d_);
  for (unsigned int i = 0; i < n; ++i) {
    rotation(i) = Yinit;
    translation(i) = Vector::Zero(r_);
  }
}

LiftedSEVariable::LiftedSEVariable(const LiftedPoseArray &liftedPoseArray)
    : LiftedSEVariable(liftedPoseArray.r(), liftedPoseArray.d(),
                       liftedPoseArray.n()) {
  setData(liftedPoseArray.getData());
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
  varSE_ = std::make_unique<ROPTLIB::ProductElement>(1, pose_var_.get(), n_);
  new (&X_SE_) Eigen::Map<Matrix>(
      const_cast<double *>(varSE_->ObtainWriteEntireData()), r_, (d_ + 1) * n_);
  setData(other.getData());
  return *this;
}

Matrix LiftedSEVariable::getData() const { return X_SE_; }

void LiftedSEVariable::setData(const Matrix &X) {
  checkSEMatrixSize(X, r_, d_, n_);
  size_t mem_size = r_ * (d_ + 1) * n_;
  copyEigenMatrixToROPTLIBVariable(X, varSE_.get(), mem_size);
}

void LiftedSEVariable::setRandomData() { varSE_.get()->RandInManifold(); }

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
    : r_(r),
      d_(d),
      n_(n),
      l_(l),
      b_(b),
      is_oblique_var_empty_(false),
      is_landmark_var_empty_(false),
      rotation_var_(std::make_unique<ROPTLIB::StieVariable>(r_, d_)),
      translation_var_(std::make_unique<ROPTLIB::EucVariable>(r_, n)),
      unit_sphere_var_([&] {
        if (l > 0) {
          return std::make_unique<ROPTLIB::ObliqueVariable>(r, l);
        } else {
          // Initialize dummy variable and set flag
          is_oblique_var_empty_ = true;
          return std::make_unique<ROPTLIB::ObliqueVariable>(r, 1);
        }
      }()),
      landmark_var_([&] {
        if (b > 0) {
          return std::make_unique<ROPTLIB::EucVariable>(r, b);
        } else {
          // Initialize dummy variable and set flag
          is_landmark_var_empty_ = true;
          return std::make_unique<ROPTLIB::EucVariable>(r, 1);
        }
      }()),
      varRA_([&] {
        if (!is_oblique_var_empty_ && !is_landmark_var_empty_) {
          return std::make_unique<ROPTLIB::ProductElement>(
              4, rotation_var_.get(), n, unit_sphere_var_.get(), 1,
              translation_var_.get(), 1, landmark_var_.get(), 1);
        } else if (!is_oblique_var_empty_ && is_landmark_var_empty_) {
          return std::make_unique<ROPTLIB::ProductElement>(
              3, rotation_var_.get(), n, unit_sphere_var_.get(), 1,
              translation_var_.get(), 1);
        } else if (is_oblique_var_empty_ && !is_landmark_var_empty_) {
          return std::make_unique<ROPTLIB::ProductElement>(
              3, rotation_var_.get(), n, translation_var_.get(), 1,
              landmark_var_.get(), 1);
        } else {
          CHECK(is_oblique_var_empty_);
          CHECK(is_landmark_var_empty_);
          return std::make_unique<ROPTLIB::ProductElement>(
              2, rotation_var_.get(), n, translation_var_.get(), 1);
        }
      }()),
      X_RA_(const_cast<double *>(varRA_->ObtainWriteEntireData()), r,
            (d + 1) * n + l + b) {
  Matrix Yinit = Matrix::Zero(r_, d_);
  Yinit.block(0, 0, d_, d_) = Matrix::Identity(d_, d_);
  for (unsigned int i = 0; i < n; ++i) {
    rotation(i) = Yinit;
    translation(i) = Vector::Zero(r_);
  }
  for (unsigned int i = 0; i < l_; ++i) {
    unitSphere(i) = Vector::Zero(r_);
  }
  for (unsigned int i = 0; i < b_; ++i) {
    landmark(i) = Vector::Zero(r_);
  }
}

LiftedRAVariable::LiftedRAVariable(
    const LiftedRangeAidedArray &liftedRangeAidedArray)
    : LiftedRAVariable(liftedRangeAidedArray.r(), liftedRangeAidedArray.d(),
                       liftedRangeAidedArray.n(), liftedRangeAidedArray.l(),
                       liftedRangeAidedArray.b()) {
  setData(liftedRangeAidedArray.getData());
}

LiftedRAVariable::LiftedRAVariable(const LiftedRAVariable &other)
    : LiftedRAVariable(other.r(), other.d(), other.n(), other.l(), other.b()) {
  setData(other.getData());
}

LiftedRAVariable &LiftedRAVariable::operator=(const LiftedRAVariable &other) {
  r_ = other.r();
  d_ = other.d();
  n_ = other.n();
  l_ = other.l();
  b_ = other.b();
  is_oblique_var_empty_ = other.isObliqueVariableEmpty();
  is_landmark_var_empty_ = other.isLandmarkVariableEmpty();
  rotation_var_ = std::make_unique<ROPTLIB::StieVariable>(r_, d_);
  translation_var_ = std::make_unique<ROPTLIB::EucVariable>(r_, n_);

  // Construct additional variables if not empty
  if (!is_oblique_var_empty_)
    unit_sphere_var_ = std::make_unique<ROPTLIB::ObliqueVariable>(r_, l_);
  if (!is_landmark_var_empty_)
    landmark_var_ = std::make_unique<ROPTLIB::EucVariable>(r_, b_);

  // Construct RA variable
  if (!is_oblique_var_empty_ && !is_landmark_var_empty_) {
    varRA_ = std::make_unique<ROPTLIB::ProductElement>(
        4, rotation_var_.get(), n_, unit_sphere_var_.get(), 1,
        translation_var_.get(), 1, landmark_var_.get(), 1);
  } else if (!is_oblique_var_empty_ && is_landmark_var_empty_) {
    varRA_ = std::make_unique<ROPTLIB::ProductElement>(
        3, rotation_var_.get(), n_, unit_sphere_var_.get(), 1,
        translation_var_.get(), 1);
  } else if (is_oblique_var_empty_ && !is_landmark_var_empty_) {
    varRA_ = std::make_unique<ROPTLIB::ProductElement>(
        3, rotation_var_.get(), n_, translation_var_.get(), 1,
        landmark_var_.get(), 1);
  } else {
    CHECK(is_oblique_var_empty_);
    CHECK(is_landmark_var_empty_);
    varRA_ = std::make_unique<ROPTLIB::ProductElement>(
        2, rotation_var_.get(), n_, translation_var_.get(), 1);
  }

  new (&X_RA_)
      Eigen::Map<Matrix>(const_cast<double *>(varRA_->ObtainWriteEntireData()),
                         r_, (d_ + 1) * n_ + l_ + b_);
  setData(other.getData());
  return *this;
}

Matrix LiftedRAVariable::getData() const { return X_RA_; }

void LiftedRAVariable::setData(const Matrix &X) {
  checkRAMatrixSize(X, r_, d_, n_, l_, b_);
  size_t mem_size = r_ * ((d_ + 1) * n_ + l_ + b_);
  copyEigenMatrixToROPTLIBVariable(X, varRA_.get(), mem_size);
}

void LiftedRAVariable::setRandomData() { varRA_.get()->RandInManifold(); }

Eigen::Ref<Matrix> LiftedRAVariable::pose(unsigned int index) {
  CHECK(index < n_);
  Matrix X_SE(r_, d_ + 1);
  X_SE.block(0, 0, r_, d_) = X_RA_.block(0, index * d_, r_, d_);
  X_SE.block(0, d_, r_, 1) = X_RA_.col(index + (d_ * n_ + l_));
  return Eigen::Ref<Matrix>(X_SE);
}

Matrix LiftedRAVariable::pose(unsigned int index) const {
  CHECK(index < n_);
  Matrix X_SE(r_, d_ + 1);
  X_SE.block(0, 0, r_, d_) = X_RA_.block(0, index * d_, r_, d_);
  X_SE.block(0, d_, r_, 1) = X_RA_.col(index + (d_ * n_ + l_));
  return X_SE;
}

Eigen::Ref<Matrix> LiftedRAVariable::rotation(unsigned int index) {
  CHECK(index < n_);
  return X_RA_.block(0, index * d_, r_, d_);
}

Matrix LiftedRAVariable::rotation(unsigned int index) const {
  CHECK(index < n_);
  return X_RA_.block(0, index * d_, r_, d_);
}

Eigen::Ref<Vector> LiftedRAVariable::translation(unsigned int index) {
  CHECK(index < n_);
  return X_RA_.col(index + (d_ * n_ + l_));
}

Vector LiftedRAVariable::translation(unsigned int index) const {
  CHECK(index < n_);
  return X_RA_.col(index + (d_ * n_ + l_));
}

Eigen::Ref<Vector> LiftedRAVariable::unitSphere(unsigned int index) {
  CHECK(index < l_);
  return X_RA_.col(index + (d_ * n_));
}

Vector LiftedRAVariable::unitSphere(unsigned int index) const {
  CHECK(index < l_);
  return X_RA_.col(index + (d_ * n_));
}

Eigen::Ref<Vector> LiftedRAVariable::landmark(unsigned int index) {
  CHECK(index < b_);
  return X_RA_.col(index + (d_ * n_ + l_ + n_));
}

Vector LiftedRAVariable::landmark(unsigned int index) const {
  CHECK(index < b_);
  return X_RA_.col(index + (d_ * n_ + l_ + n_));
}

} // namespace DCORA
