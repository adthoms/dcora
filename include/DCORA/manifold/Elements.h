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
 * dim = d + 1 for a pose array, dim = 1 for a point array. Each rotation Yi is
 * a r-by-d matrix representing an element of the Stiefel manifold while each
 * translation pi/pj is a r-dimensional vector representing an element of the
 * Euclidean space. Note that translations can also represent unit-sphere
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
   * @brief Set the underlying Eigen matrix to zero
   */
  void setDataToZero();
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
   * @brief Set the underlying Eigen matrix as random in the SE manifold
   */
  void setRandomData();
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

/**
 * @brief A class representing an array of "lifted" poses, unit-sphere auxiliary
 * variables, and translations in RA ordering. Internally store as
 * r-by-(d+1)n+l+b matrix: X = [Y1 ... Yn | r1 ... rl | p1 ... pn | L1 ... Lb].
 * Optionally, the graph type can be specfied to be PGO compatible, in which
 * case this class acts as a wrapper to the LiftedPoseArray.
 */
class LiftedRangeAidedArray {
public:
  /**
   * @brief Constructor
   * @param r relaxation rank
   * @param d dimension (2/3)
   * @param n number of poses
   * @param l number of unit spheres
   * @param b number of landmarks
   * @param graphType type of graph
   */
  LiftedRangeAidedArray(unsigned int r, unsigned int d, unsigned int n,
                        unsigned int l, unsigned int b,
                        GraphType graphType = GraphType::RangeAidedSLAMGraph);
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
   * @brief Get graph type
   * @return
   */
  GraphType graphType() const { return graph_type_; }
  /**
   * @brief Return the underlying Eigen matrices of encapsulated arrays
   * @return
   */
  Matrix getData() const;
  /**
   * @brief Set the underlying Eigen matrices of encapsulated arrays
   * @param X
   */
  void setData(const Matrix &X);
  /**
   * @brief Set the underlying Eigen matrix as random in the RA manifold
   */
  void setRandomData();
  /**
   * @brief Set the underlying Eigen matrices of encapsulated arrays to zero
   */
  void setDataToZero();
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
   * @brief Obtain the writable pose rotation at the specified index, expressed
   * as an r-by-d matrix
   * @param index
   * @return
   */
  Eigen::Ref<Matrix> rotation(unsigned int index);
  /**
   * @brief Obtain the read-only pose rotation at the specified index, expressed
   * as an r-by-d matrix
   * @param index
   * @return
   */
  Matrix rotation(unsigned int index) const;
  /**
   * @brief Obtain the writable pose translation at the specified index,
   * expressed as an r dimensional vector
   * @param index
   * @return
   */
  Eigen::Ref<Vector> translation(unsigned int index);
  /**
   * @brief Obtain the read-only pose translation at the specified index,
   * expressed as an r dimensional vector
   * @param index
   * @return
   */
  Vector translation(unsigned int index) const;
  /**
   * @brief Obtain the writable unit-sphere auxiliary variable, expressed as an
   * r dimensional vector
   * @param index
   * @return
   */
  Eigen::Ref<Vector> unitSphere(unsigned int index);
  /**
   * @brief Obtain the read-only unit-sphere auxiliary variable, expressed as an
   * r dimensional vector
   * @param index
   * @return
   */
  Vector unitSphere(unsigned int index) const;
  /**
   * @brief Obtain the writable landmark at the specified index,
   * expressed as an r dimensional vector
   * @param index
   * @return
   */
  Eigen::Ref<Vector> landmark(unsigned int index);
  /**
   * @brief Obtain the read-only landmark at the specified index,
   * expressed as an r dimensional vector
   * @param index
   * @return
   */
  Vector landmark(unsigned int index) const;
  /**
   * @brief Set the lifted pose array
   * @param liftedPoseArray
   */
  void setLiftedPoseArray(const LiftedPoseArray &liftedPoseArray);
  /**
   * @brief Set the lifted unit sphere array
   * @param liftedUnitSphereArray
   */
  void setLiftedUnitSphereArray(const LiftedPointArray &liftedUnitSphereArray);
  /**
   * @brief Set the lifted landmark array
   * @param liftedLandmarkArray
   */
  void setLiftedLandmarkArray(const LiftedPointArray &liftedLandmarkArray);
  /**
   * @brief Get "lifted" pose array
   * @return
   */
  LiftedPoseArray *GetLiftedPoseArray() const { return poses_.get(); }
  /**
   * @brief Get "lifted" unit sphere array
   * @return
   */
  LiftedPointArray *GetLiftedUnitSphereArray() const {
    return unit_spheres_.get();
  }
  /**
   * @brief Get "lifted" landmark array
   * @return
   */
  LiftedPointArray *GetLiftedLandmarkArray() const { return landmarks_.get(); }

protected:
  /**
   * @brief Return true if the array is compatible with PGO
   * @return
   */
  bool isPGOCompatible() const;
  /**
   * @brief Helper function to return the underlying Eigen matrix of the
   * encapsulated LiftedPoseArray array in SE ordering
   * @return
   */
  Matrix getDataSE() const;
  /**
   * @brief Helper function to return the underlying Eigen matrices of
   * encapsulated arrays in RA ordering
   * @return
   */
  Matrix getDataRA() const;
  /**
   * @brief Helper function to set the underlying Eigen matrix of the
   * encapsulated LiftedPoseArray array in SE ordering
   * @param X
   */
  void setDataSE(const Matrix &X);
  /**
   * @brief Helper function to set the underlying Eigen matrices of encapsulated
   * arrays in RA ordering
   * @param X
   */
  void setDataRA(const Matrix &X);

  // Dimension constants
  unsigned int r_, d_, n_, l_, b_;
  // Graph type
  GraphType graph_type_;
  // "Lifted" arrays
  std::unique_ptr<LiftedPoseArray> poses_;
  std::unique_ptr<LiftedPointArray> unit_spheres_;
  std::unique_ptr<LiftedPointArray> landmarks_;
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

/**
 * @brief A class representing an array of poses, unit-sphere auxiliary
 * variables, and translations in RA ordering. Internally store as
 * r-by-(d+1)n+l+b matrix: X = [Y1 ... Yn | r1 ... rl | p1 ... pn | L1 ... Lb].
 * Optionally, the graph type can be specfied to be PGO compatible, in which
 * case this class acts as a wrapper to the LiftedPoseArray.
 */
class RangeAidedArray : public LiftedRangeAidedArray {
public:
  RangeAidedArray(unsigned int d, unsigned int n, unsigned int l,
                  unsigned int b,
                  GraphType graphType = GraphType::RangeAidedSLAMGraph)
      : LiftedRangeAidedArray(d, d, n, l, b, graphType) {}
  /**
   * @brief Get pose array
   * @return
   */
  PoseArray getPoseArray() const;
  /**
   * @brief Get unit sphere array
   * @return
   */
  PointArray getUnitSphereArray() const;
  /**
   * @brief Get landmark array
   * @return
   */
  PointArray getLandmarkArray() const;
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

/**
 * @brief Representing a single standard euclidean vector in SE(d)
 */
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

// A utility function for streaming PoseDict to cout
inline std::ostream &operator<<(std::ostream &os, const PoseDict &pose_dict) {
  for (const auto &[pose_id, lifted_pose] : pose_dict) {
    os << "ID: " << pose_id << std::endl;
    os << "Variable:\n" << lifted_pose.pose() << std::endl;
  }
  return os;
}

// A utility function for streaming PoseSet to cout
inline std::ostream &operator<<(std::ostream &os, const PoseSet &pose_set) {
  for (const auto &pose_id : pose_set)
    os << "ID: " << pose_id << std::endl;
  return os;
}

// Ordered map of LandmarkID to LiftedPoint object
typedef std::map<LandmarkID, LiftedPoint, CompareStateID> LandmarkDict;
// Ordered set of LandmarkID
typedef std::set<LandmarkID, CompareStateID> LandmarkSet;

// A utility function for streaming LandmarkDict to cout
inline std::ostream &operator<<(std::ostream &os,
                                const LandmarkDict &landmark_dict) {
  for (const auto &[landmark_id, lifted_landmark] : landmark_dict) {
    os << "ID: " << landmark_id << std::endl;
    os << "Variable:\n" << lifted_landmark.translation() << std::endl;
  }
  return os;
}

// A utility function for streaming LandmarkSet to cout
inline std::ostream &operator<<(std::ostream &os,
                                const LandmarkSet &landmark_set) {
  for (const auto &landmark_id : landmark_set)
    os << "ID: " << landmark_id << std::endl;
  return os;
}

// Ordered map of UnitSphereID to LiftedPoint object
typedef std::map<UnitSphereID, LiftedPoint, CompareStateID> UnitSphereDict;
// Ordered set of UnitSphereID
typedef std::set<UnitSphereID, CompareStateID> UnitSphereSet;

// A utility function for streaming UnitSphereDict to cout
inline std::ostream &operator<<(std::ostream &os,
                                const UnitSphereDict &unit_sphere_dict) {
  for (const auto &[unit_sphere_id, lifted_unit_sphere] : unit_sphere_dict) {
    os << "ID: " << unit_sphere_id << std::endl;
    os << "Variable:\n" << lifted_unit_sphere.translation() << std::endl;
  }
  return os;
}

// A utility function for streaming UnitSphereSet to cout
inline std::ostream &operator<<(std::ostream &os,
                                const UnitSphereSet &unit_sphere_set) {
  for (const auto &unit_sphere_id : unit_sphere_set)
    os << "ID: " << unit_sphere_id << std::endl;
  return os;
}

} // namespace DCORA
