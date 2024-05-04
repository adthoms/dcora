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

#pragma once

#include <map>
#include <memory>
#include <set>

#include "DCORA/DCORA_types.h"

namespace DCORA {

/**
 * @brief A class representing an array of n "lifted" elements of dimension r by
 * dim. Elements consist of poses Ti = [Yi pi] and translations pj, where:
 * dim = d + 1 for a pose array, dim = 1 for a translation array. Each rotation
 * Yi is a r-by-d matrix representing an element of the Stiefel manifold while
 * each translation pi/pj is a r-dimensional vector representing an element of
 * the Euclidean space. Note that translations can also represent unit-sphere
 * auxiliary variables for ranges, which are r-dimensional vectors
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
   * @brief Obtain the writable translation at the specified index, expressed as
   * an r dimensional vector
   * @param index
   * @return
   */
  Eigen::Ref<Vector> translation(unsigned int index);
  /**
   * @brief Obtain the read-only translation at the specified index, expressed
   * as an r dimensional vector
   * @param index
   * @return
   */
  Vector translation(unsigned int index) const;
  /**
   * @brief Compute the average translation distance between two lifted arrays.
   * Internally check that both arrays should have same dimension and number of
   * elements
   * @param array1
   * @param array2
   * @return
   */
  static double averageTranslationDistance(const LiftedArray &array1,
                                           const LiftedArray &array2);
  /**
   * @brief Compute the max translation distance between two lifted arrays.
   * Internally check that both arrays should have same dimension and number of
   * elements
   * @param array1
   * @param array2
   * @return
   */
  static double maxTranslationDistance(const LiftedArray &array1,
                                       const LiftedArray &array2);

protected:
  // Dimension constants
  unsigned int r_, d_, n_, dim_;
  // Eigen matrix that stores the array
  Matrix X_;
};

/**
 * @brief A class representing an array of "lifted" poses. Internally store as
 * r-by-(d+1)n matrix: X = [X1 ... Xn], where each Xi = [Yi pi]
 */
class LiftedPoseArray : public LiftedArray {
public:
  /**
   * @brief Constructor. The value of the pose array is guaranteed to be valid.
   * @param r relaxation rank
   * @param d dimension (2/3)
   * @param n number of poses
   */
  LiftedPoseArray(unsigned int r, unsigned int d, unsigned int n);
  /**
   * @brief Check that the stored data are valid
   */
  void checkData() const;
  /**
   * @brief Obtain the writable pose at the specified index, expressed as an
   * r-by-(d+1) matrix
   * @param index
   * @return
   */
  Eigen::Ref<Matrix> pose(unsigned int index);
  /**
   * @brief Obtain the read-only pose at the specified index, expressed as an
   * r-by-(d+1) matrix
   * @param index
   * @return
   */
  Matrix pose(unsigned int index) const;
  /**
   * @brief Obtain the writable rotation at the specified index, expressed as an
   * r-by-d matrix
   * @param index
   * @return
   */
  Eigen::Ref<Matrix> rotation(unsigned int index);
  /**
   * @brief Obtain the read-only rotation at the specified index, expressed as
   * an r-by-d matrix
   * @param index
   * @return
   */
  Matrix rotation(unsigned int index) const;
};

/**
 * @brief A class representing an array of "lifted" translations
 * Internally store as r-by-n matrix: X = [pi ... pn]
 */
class LiftedPointArray : public LiftedArray {
public:
  /**
   * @brief Constructor
   * @param r relaxation rank
   * @param d dimension (2/3)
   * @param n number of translations
   */
  LiftedPointArray(unsigned int r, unsigned int d, unsigned int n);
};

typedef LiftedPointArray LiftedRangeArray;
typedef LiftedPointArray LiftedLandmarkArray;

/**
 * @brief A class representing an array of "lifted" poses, unit-sphere auxiliary
 * variables, and translations in RA ordering. Internally store as
 * r-by-(d+1)n+l+b matrix: X = [Y1 ... Yn | r1 ... rn | p1 ... pn | l1 ... ln]
 */
class LiftedRangeAidedArray {
public:
  /**
   * @brief Constructor
   * @param r relaxation rank
   * @param d dimension (2/3)
   * @param n number of poses
   * @param l number of ranges
   * @param b number of landmarks
   */
  LiftedRangeAidedArray(unsigned int r, unsigned int d, unsigned int n,
                        unsigned int l, unsigned int b);
  /**
   * @brief Copy constructor
   * @param other
   */
  LiftedRangeAidedArray(const LiftedRangeAidedArray &other);
  /**
   * @brief Copy assignment operator
   * @param other
   * @return
   */
  LiftedRangeAidedArray &operator=(const LiftedRangeAidedArray &other);
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
   * @brief Get number of poses
   * @return
   */
  unsigned int n() const { return n_; }
  /**
   * @brief Get number of ranges
   * @return
   */
  unsigned int l() const { return l_; }
  /**
   * @brief Get number of landmarks
   * @return
   */
  unsigned int b() const { return b_; }
  /**
   * @brief Return the underlying Eigen matrices of encapsulated arrays in RA
   * ordering
   * @return
   */
  Matrix getData() const;
  /**
   * @brief Set the underlying Eigen matrices of encapsulated arrays in RA
   * ordering
   * @param X
   */
  void setData(const Matrix &X);
  /**
   * @brief Get "lifted" pose array
   * @return
   */
  LiftedPoseArray *GetLiftedPoseArray() const { return poses_.get(); }
  /**
   * @brief Get "lifted" range array
   * @return
   */
  LiftedRangeArray *GetLiftedRangeArray() const { return ranges_.get(); }
  /**
   * @brief Get "lifted" landmark array
   * @return
   */
  LiftedLandmarkArray *GetLiftedLandmarkArray() const {
    return landmarks_.get();
  }

private:
  // Dimension constants
  unsigned int r_, d_, n_, l_, b_;
  // "Lifted" arrays
  std::unique_ptr<LiftedPoseArray> poses_;
  std::unique_ptr<LiftedRangeArray> ranges_;
  std::unique_ptr<LiftedLandmarkArray> landmarks_;
};

/**
 * @brief A class representing an array of standard poses in SE(d)
 * Internally store as d-by-(d+1)n matrix:
 * X = [X1, ... Xn], where each Xi = [Ri ti]
 * Each rotation Ri is a d-by-d matrix while each translation pi is a
 * d-dimensional vector
 */
class PoseArray : public LiftedPoseArray {
public:
  PoseArray(unsigned int d, unsigned int n) : LiftedPoseArray(d, d, n) {}
};

/**
 * @brief A class representing an array of standard translations in E(d).
 * Internally store as d-by-n matrix: X = [p1 ... pn]
 * Each translation pi is a d-dimensional vector
 */
class PointArray : public LiftedPointArray {
public:
  PointArray(unsigned int d, unsigned int n) : LiftedPointArray(d, d, n) {}
};

typedef PointArray RangeArray;
typedef PointArray LandmarkArray;

/**
 * @brief A class representing an array of poses, unit-sphere auxiliary
 * variables, and translations in RA ordering. Internally store as
 * d-by-(d+1)n+l+b matrix: X = [X1, ... Xn | r1 ... rn | p1 ... pn | l1 ... ln]
 * See PoseArray, RangeArray, and PointArray for details
 */
class RangeAidedArray : public LiftedRangeAidedArray {
public:
  RangeAidedArray(unsigned int d, unsigned int n, unsigned int l,
                  unsigned int b)
      : LiftedRangeAidedArray(d, d, n, l, b) {}
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
   * @param X r-by-(d+1) matrix X = [Y p]
   */
  explicit LiftedPose(const Matrix &X) : LiftedPose(X.rows(), X.cols() - 1) {
    setData(X);
  }
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
class LiftedPoint : public LiftedPointArray {
public:
  LiftedPoint() : LiftedPoint(3, 3) {}
  LiftedPoint(unsigned int r, unsigned int d) : LiftedPointArray(r, d, 1) {}
  /**
   * @brief Constructor from Eigen vector
   * @param P r-dimensional vector
   */
  explicit LiftedPoint(const Vector &P) : LiftedPoint(P.rows(), P.rows()) {
    setData(P);
  }
  /**
   * @brief Return the writable translation
   * @return
   */
  Eigen::Ref<Vector> translation() { return LiftedPointArray::translation(0); }
  /**
   * @brief Return the read-only translation
   * @return
   */
  Vector translation() const { return LiftedPointArray::translation(0); }
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

class Point : public LiftedPoint {
public:
  // Constructor
  Point() : Point(3) {}
  explicit Point(unsigned int d) : LiftedPoint(d, d) {}
  /**
   * @brief Constructor from Eigen vector
   * @param P r-dimensional vector
   */
  explicit Point(const Vector &P);
  /**
   * @brief Return the zero vector of specified dimension
   * @param d
   * @return
   */
  static Point ZeroVector(unsigned int d);
  /**
   * @brief Return the zero vector element
   * @return
   */
  Point zeroVector() const;
  /**
   * @brief Return the vector representing this translation
   * @return
   */
  Vector vector() const;
};

// Ordered map of PoseID to LiftedPose object
typedef std::map<PoseID, LiftedPose, CompareStateID> PoseDict;
// Ordered set of PoseID
typedef std::set<PoseID, CompareStateID> PoseSet;

} // namespace DCORA
