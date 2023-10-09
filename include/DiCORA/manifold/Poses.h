/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#ifndef DiCORA_INCLUDE_DiCORA_MANIFOLD_POSES_H_
#define DiCORA_INCLUDE_DiCORA_MANIFOLD_POSES_H_

#include <set>
#include "DiCORA/DiCORA_types.h"

namespace DiCORA {
/**
 * @brief A class representing an array of n "lifted" elements of dimension r by dim
 * Elements consist of poses Ti = [Yi pi] and translations pj, where:
 * dim = d + 1 for a pose array, dim = 1 for a translation array
 * Each rotation Yi is a r-by-d matrix representing an element of the Stiefel manifold
 * Each translation pi is a r-dimensional vector
 */
class LiftedArray {
 public:
  /**
   * @brief Constructor
   * @param r relaxation rank
   * @param d dimension of element
   * @param n number of elements
   */
  LiftedArray(unsigned int r, unsigned int d, unsigned int n);
  /**
   * @brief Get relaxation rank
   * @return
   */
  unsigned int r() const { return r_; }
  /**
   * @brief Get dimension
   * @return
   */
  unsigned int d() const { return d_; }
  /**
   * @brief Get number of elements
   * @return
   */
  unsigned int n() const { return n_; }
  /**
   * @brief Return the underlying Eigen matrix
   * @return
   */
  Matrix getData() const;
  /**
   * @brief Set the underlying Eigen matrix
   * @param X
   */
  void setData(const Matrix &X);
  /**
   * @brief Obtain the writable translation at the specified index, expressed as an r dimensional vector
   * @param index
   * @return
   */
  Eigen::Ref<Vector> translation(unsigned int index);
  /**
   * @brief Obtain the read-only translation at the specified index, expressed as an r dimensional vector
   * @param index
   * @return
   */
  Vector translation(unsigned int index) const;
  /**
   * @brief Compute the average translation distance between two lifted arrays
   * Internally check that both arrays should have same dimension and number of elements
   * @param array1
   * @param array2
   * @return
   */
  static double averageTranslationDistance(const LiftedArray &array1, const LiftedArray &array2);
  /**
   * @brief Compute the max translation distance between two lifted arrays
   * Internally check that both arrays should have same dimension and number of elements
   * @param array1
   * @param array2
   * @return
   */
  static double maxTranslationDistance(const LiftedArray &array1, const LiftedArray &array2);
 protected:
  // Dimension constants
  unsigned int r_, d_, n_;
  // default to pose array
  unsigned int dim_;
  // Eigen matrix that stores the pose array
  Matrix X_;
};

/**
 * @brief A class representing an array of "lifted" poses
 * Internally store as r by (d+1)n matrix X = [X1, ... Xn], where each Xi = [Yi pi]
 */
class LiftedPoseArray : public LiftedArray {
 public:
  /**
   * @brief Constructor. The value of the pose array is guaranteed to be valid.
   * @param r
   * @param d
   * @param n
   */
  LiftedPoseArray(unsigned int r, unsigned int d, unsigned int n);
  /**
   * @brief Check that the stored data are valid
   */
  void checkData() const;
  /**
   * @brief Obtain the writable pose at the specified index, expressed as an r-by-(d+1) matrix
   * @param index
   * @return
   */
  Eigen::Ref<Matrix> pose(unsigned int index);
  /**
   * @brief Obtain the read-only pose at the specified index, expressed as an r-by-(d+1) matrix
   * @param index
   * @return
   */
  Matrix pose(unsigned int index) const;
  /**
   * @brief Obtain the writable rotation at the specified index, expressed as an r-by-d matrix
   * @param index
   * @return
   */
  Eigen::Ref<Matrix> rotation(unsigned int index);
  /**
   * @brief Obtain the read-only rotation at the specified index, expressed as an r-by-d matrix
   * @param index
   * @return
   */
  Matrix rotation(unsigned int index) const;
};

/**
 * @brief A class representing an array of "lifted" translations
 * Internally store as r by n matrix X = [pi, ... pn]
 */
class LiftedTranslationArray : public LiftedArray {
 public:
  /**
   * @brief Constructor. The value of the translation array is guaranteed to be valid.
   * @param r
   * @param d
   * @param n
   */
  LiftedTranslationArray(unsigned int r, unsigned int d, unsigned int n);
  /**
   * @brief Set the underlying Eigen matrix
   * @param P
   */
  void setData(const Vector &P);
};

/**
 * @brief A class representing an array of standard poses in SE(d)
 * Internally store as d by (d+1)n matrix X = [X1, ... Xn], where each Xi = [Ri ti]
 * Each rotation Ri is a d-by-d matrix
 * Each translation pi is a d-dimensional vector
 */
class PoseArray : public LiftedPoseArray {
 public:
  PoseArray(unsigned int d, unsigned int n) : LiftedPoseArray(d, d, n) {}
};

/**
 * @brief A class representing an array of standard translations in E(d)
 * Internally store as d by n matrix X = [p1, ... pn]
 * Each translation pi is a d-dimensional vector
 */
class TranslationArray : public LiftedTranslationArray {
 public:
  TranslationArray(unsigned int d, unsigned int n) : LiftedTranslationArray(d, d, n) {}
};

/**
 * @brief A class representing a single "lifted" pose Xi = [Yi pi]
 */
class LiftedPose : public LiftedPoseArray {
 public:
  LiftedPose() : LiftedPose(3, 3) {}
  LiftedPose(unsigned int r, unsigned int d) : LiftedPoseArray(r, d, 1) {}
  /**
   * @brief Constructor from Eigen matrix
   * @param X r by d+1 matrix X = [Y p]
   */
  explicit LiftedPose(const Matrix &X) :
      LiftedPose(X.rows(), X.cols() - 1) { setData(X); }
  /**
   * @brief Return the writable pose
   * @return
   */
  Eigen::Ref<Matrix> pose() { return LiftedPoseArray::pose(0); }
  /**
   * @brief Return the read-only pose
   * @return
   */
  Matrix pose() const { return LiftedPoseArray::pose(0); }
  /**
   * @brief Return the writable rotation
   * @return
   */
  Eigen::Ref<Matrix> rotation() { return LiftedPoseArray::rotation(0); }
  /**
   * @brief Return the read-only rotation
   * @return
   */
  Matrix rotation() const { return LiftedPoseArray::rotation(0); }
  /**
   * @brief Return the writable translation
   * @return
   */
  Eigen::Ref<Vector> translation() { return LiftedPoseArray::translation(0); }
  /**
   * @brief Return the read-only translation
   * @return
   */
  Vector translation() const { return LiftedPoseArray::translation(0); }
};

/**
 * @brief A class representing a single "lifted" translation pi
 */
class LiftedTranslation : public LiftedTranslationArray {
 public:
  LiftedTranslation() : LiftedTranslation(3, 3) {}
  LiftedTranslation(unsigned int r, unsigned int d) : LiftedTranslationArray(r, d, 1) {}
  /**
   * @brief Constructor from Eigen vector
   * @param P r-dimensional vector
   */
  explicit LiftedTranslation(const Vector &P) :
      LiftedTranslation(P.size(), 1) { setData(P); }
  /**
   * @brief Return the writable translation
   * @return
   */
  Eigen::Ref<Vector> translation() { return LiftedTranslationArray::translation(0); }
  /**
   * @brief Return the read-only translation
   * @return
   */
  Vector translation() const { return LiftedTranslationArray::translation(0); }
};

/**
 * @brief Representing a single standard pose in SE(d)
 */
class Pose : public LiftedPose {
 public:
  // Constructor
  Pose() : Pose(3) {}
  explicit Pose(unsigned int d) : LiftedPose(d, d) {}
  /**
   * @brief Constructor from Eigen matrix
   * @param T d by (d+1) matrix T = [R t]
   */
  explicit Pose(const Matrix &T);
  /**
   * @brief Return the identity pose of specified dimension
   * @param d
   * @return
   */
  static Pose Identity(unsigned int d);
  /**
   * @brief Return the identity element
   * @return
   */
  Pose identity() const;
  /**
   * @brief Return the inverse of this pose
   * @return
   */
  Pose inverse() const;
  /**
   * @brief The multiplication operator
   * @param other
   * @return (*this) * other
   */
  Pose operator*(const Pose &other) const;
  /**
   * @brief Return the homogeneous (d+1)-by-(d+1) matrix representing this pose
   * @return
   */
  Matrix matrix() const;
};

class Translation : public LiftedTranslation {
 public:
  // Constructor
  Translation() : Translation(3) {}
  explicit Translation(unsigned int d) : LiftedTranslation(d,d) {}
  /**
   * @brief Constructor from Eigen vector
   * @param P r-dimensional vector
   */
  explicit Translation(const Vector &P);
  /**
   * @brief Return the zero vector of specified dimension
   * @param d
   * @return
   */
  static Translation ZeroVector(unsigned int d);
  /**
   * @brief Return the zero vector element
   * @return
   */
  Translation zeroVector() const;
  /**
   * @brief Return the vector representing this translation
   * @return
   */
  Vector vector() const;
};

// Ordered map of PoseID to LiftedPose object
typedef std::map<PoseID, LiftedPose, ComparePoseID> PoseDict;
// Ordered set of PoseID
typedef std::set<PoseID, ComparePoseID> PoseSet;

}
#endif //DiCORA_INCLUDE_DiCORA_MANIFOLD_POSES_H_
