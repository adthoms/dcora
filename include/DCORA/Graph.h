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

#include <DCORA/DCORA_types.h>
#include <DCORA/DCORA_utils.h>
#include <DCORA/Measurements.h>
#include <DCORA/manifold/Elements.h>

#include <limits>
#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace DCORA {

/**
 * @brief A graph class representing the local optimization problem in
 * distributed PGO or RA-SLAM.
 */
class Graph {
public:
  /**
   * @brief Store statistics for the current graph
   */
  class Statistics {
  public:
    Statistics()
        : total_loop_closures(0),
          accept_loop_closures(0),
          reject_loop_closures(0),
          undecided_loop_closures(0) {}
    double total_loop_closures;
    double accept_loop_closures;
    double reject_loop_closures;
    double undecided_loop_closures;
  };
  /**
   * @brief Constructor
   * @param id robot ID
   * @param r relaxation rank
   * @param d dimension (2/3)
   * @param graphType graph type
   */
  Graph(unsigned int id, unsigned int r, unsigned int d,
        GraphType graphType = GraphType::PoseGraph);
  /**
   * @brief Destructor
   */
  ~Graph();
  /**
   * @brief Get dimension
   * @return
   */
  unsigned int d() const { return d_; }
  /**
   * @brief Get relaxation rank
   * @return
   */
  unsigned int r() const { return r_; }
  /**
   * @brief Get number of poses
   * @return
   */
  unsigned int n() const { return n_; }
  /**
   * @brief Get number of unit spheres
   * @return
   */
  unsigned int l() const { return l_; }
  /**
   * @brief Get number of landmarks
   * @return
   */
  unsigned int b() const { return b_; }
  /**
   * @brief Get the underlying problem dimension
   * @return
   */
  unsigned int k() const { return (d_ + 1) * n_ + l_ + b_; }
  /**
   * @brief Return number of odometry edges
   * @return
   */
  unsigned int numOdometry() const { return odometry_.size(); }
  /**
   * @brief Return number of private loop closures
   * @return
   */
  unsigned int numPrivateLoopClosures() const {
    return private_lcs_.vec.size();
  }
  /**
   * @brief Return number of shared loop closures
   * @return
   */
  unsigned int numSharedLoopClosures() const { return shared_lcs_.vec.size(); }
  /**
   * @brief Clear all contents and reset this graph to be empty
   */
  void empty();
  /**
   * @brief Clear all temporary data and only keep measurements
   */
  void reset();
  /**
   * @brief Return true if the graph is compatible with PGO
   * @return
   */
  bool isPGOCompatible() const;
  /**
   * @brief Clear all cached neighbor states
   */
  void clearNeighborStates();
  /**
   * @brief Update the number of poses and landmarks
   * @param stateID
   */
  void updateNumPosesAndLandmarks(const StateID &stateID);
  /**
   * @brief Update the number of unit spheres
   * @param measurement
   */
  void updateNumUnitSpheres(const RelativeMeasurement &measurements);
  /**
   * @brief Set measurements for this graph
   * @param measurements
   */
  void
  setMeasurements(const std::vector<RelativePosePoseMeasurement> &measurements);
  /**
   * @brief Set measurements for this graph
   * @param measurements
   */
  void setMeasurements(const RelativeMeasurements &measurements);
  /**
   * @brief Add a single measurement to this graph. Ignored if the input
   * measurement already exists.
   * @param m
   */
  void addMeasurement(const RelativeMeasurement &m);
  /**
   * @brief Return a copy of the list of odometry edges
   * @return
   */
  std::vector<RelativePosePoseMeasurement> odometry() const {
    return odometry_;
  }
  /**
   * @brief Return a copy of the list of private loop closures
   * @return
   */
  RelativeMeasurements privateLoopClosures() const { return private_lcs_; }
  /**
   * @brief Return a copy of the list of shared loop closures
   * @return
   */
  RelativeMeasurements sharedLoopClosures() const { return shared_lcs_; }
  /**
   * @brief Return a copy of all shared loop closures with the specified
   * neighbor
   * @param neighbor_id
   * @return
   */
  RelativeMeasurements sharedLoopClosuresWithRobot(unsigned neighbor_id) const;
  /**
   * @brief Return a copy of all measurements
   * @return
   */
  RelativeMeasurements allMeasurements() const;
  /**
   * @brief Return a copy of all LOCAL measurements (i.e., without shared loop
   * closures)
   * @return
   */
  RelativeMeasurements localMeasurements() const;
  /**
   * @brief Clear all priors
   */
  void clearPriors();
  /**
   * @brief Add a pose prior term
   * @param index The index of the local variable
   * @param Xi Corresponding pose prior term
   */
  void setPrior(unsigned index, const LiftedPose &Xi);
  /**
   * @brief Add a landmark prior term
   * @param index The index of the local variable
   * @param ti Corresponding landmark prior term
   */
  void setPrior(unsigned index, const LiftedPoint &ti);
  /**
   * @brief Set neighbor states
   * @param pose_dict
   * @param unit_sphere_dict
   * @param landmark_dict
   */
  void setNeighborStates(const PoseDict &pose_dict,
                         const UnitSphereDict &unit_sphere_dict,
                         const LandmarkDict &landmark_dict);
  /**
   * @brief Get quadratic cost matrix
   * @return
   */
  const SparseMatrix &quadraticMatrix();
  /**
   * @brief Clear the quadratic cost matrix
   */
  void clearQuadraticMatrix();
  /**
   * @brief Get linear cost matrix
   * @return
   */
  const Matrix &linearMatrix();
  /**
   * @brief Clear the linear cost matrix
   */
  void clearLinearMatrix();
  /**
   * @brief Construct data matrices that are needed for optimization, if they do
   * not yet exist. Return true if construction is successful
   * @return
   */
  bool constructDataMatrices();
  /**
   * @brief Clear data matrices
   */
  void clearDataMatrices();
  /**
   * @brief Return true if preconditioner is available.
   * @return
   */
  bool hasPreconditioner();
  /**
   * @brief Get preconditioner
   * @return
   */
  const CholmodSolverPtr &preconditioner();
  /**
   * @brief Get the set of my pose IDs that are shared with other robots
   * @return
   */
  PoseSet myPublicPoseIDs() const { return loc_shared_pose_ids_; }
  /**
   * @brief Get the set of my landmark IDs that are shared with other robots
   * @return
   */
  LandmarkSet myPublicLandmarkIDs() const { return loc_shared_landmark_ids_; }
  /**
   * @brief Get the set of my unit sphere IDs that are shared with other robots
   * @return
   */
  UnitSphereSet myPublicUnitSphereIDs() const {
    return loc_shared_unit_sphere_ids_;
  }
  /**
   * @brief Get the set of pose IDs that ALL neighbors need to share with me
   * @return
   */
  PoseSet neighborPublicPoseIDs() const { return nbr_shared_pose_ids_; }
  /**
   * @brief Get the set of landmark IDs that ALL neighbors need to share with me
   * @return
   */
  LandmarkSet neighborPublicLandmarkIDs() const {
    return nbr_shared_landmark_ids_;
  }
  /**
   * @brief Get the set of unit sphere IDs that ALL neighbors need to share with
   * me
   * @return
   */
  UnitSphereSet neighborPublicUnitSphereIDs() const {
    return nbr_shared_unit_sphere_ids_;
  }
  /**
   * @brief Get the set of pose IDs that active neighbors need to share with me.
   * A neighbor is active if it is actively participating in distributed
   * optimization with this robot.
   * @return
   */
  PoseSet activeNeighborPublicPoseIDs() const;
  /**
   * @brief Get the set of landmark IDs that active neighbors need to share with
   * me. A neighbor is active if it is actively participating in distributed
   * optimization with this robot.
   * @return
   */
  LandmarkSet activeNeighborPublicLandmarkIDs() const;
  /**
   * @brief Get the set of unit sphere IDs that active neighbors need to share
   * with me. A neighbor is active if it is actively participating in
   * distributed optimization with this robot.
   * @return
   */
  UnitSphereSet activeNeighborPublicUnitSphereIDs() const;
  /**
   * @brief Get the set of neighbor robot IDs that share shared loop closures
   * with me
   * @return
   */
  std::set<unsigned> neighborIDs() const { return nbr_robot_ids_; }
  /**
   * @brief Return the number of neighbors
   * @return
   */
  size_t numNeighbors() const { return nbr_robot_ids_.size(); }
  /**
   * @brief Return the IDs of active neighbors.
   * A neighbor is active if it is actively participating in distributed
   * optimization with this robot.
   * @return
   */
  std::set<unsigned> activeNeighborIDs() const;
  /**
   * @brief Return the number of active neighbors.
   * A neighbor is active if it is actively participating in distributed
   * optimization with this robot.
   * @return
   */
  size_t numActiveNeighbors() const;
  /**
   * @brief Return true if the input robot is a neighbor (i.e., shared loop
   * closure)
   * @param robot_id
   * @return
   */
  bool hasNeighbor(unsigned int robot_id) const;
  /**
   * @brief Check if the input neighbor is active. Return false if the input
   * robot is not a neighbor or is ignored
   * @param neighbor_id
   * @return
   */
  bool isNeighborActive(unsigned int neighbor_id) const;
  /**
   * @brief Set the input neighbor to be active or inactive.
   * Does nothing if the input is not a neighbor.
   * @param neighbor_id
   * @param active
   */
  void setNeighborActive(unsigned int neighbor_id, bool active);
  /**
   * @brief Return true if the given neighbor pose ID is required by me
   * @param pose_id
   * @return
   */
  bool requireNeighborPose(const PoseID &pose_id) const;
  /**
   * @brief Return true if the given neighbor landmark ID is required by me
   * @param landmark_id
   * @return
   */
  bool requireNeighborLandmark(const LandmarkID &landmark_id) const;
  /**
   * @brief Return true if the given neighbor unit sphere ID is required by me
   * @param unit_sphere_id
   * @return
   */
  bool requireNeighborUnitSphere(const UnitSphereID &unit_sphere_id) const;
  /**
   * @brief Compute number of accepted, rejected, and undecided loop closures
   * Note that loop closures with inactive neighbors are not included
   * @return
   */
  Statistics statistics() const;
  /**
   * @brief Check if a measurement exists in the graph
   * @param edgeID
   * @return
   */
  bool hasMeasurement(const EdgeID &edgeID) const;
  /**
   * @brief Find and return a writable pointer to the specified measurement
   * within this graph. If the measurement does not exist, return nullptr.
   * @param edgeID
   * @return
   */
  RelativeMeasurement *findMeasurement(const EdgeID &edgeID);
  /**
   * @brief Return a vector of writable pointers to all loop closures in the
   * graph (contains both private and inter-robot loop closures)
   * @return
   */
  std::vector<RelativeMeasurementPointerVariant> activeLoopClosures();

protected:
  // ID associated with this agent
  const unsigned int id_;

  // Problem dimensions
  unsigned int r_, d_, n_, l_, b_;

  // Graph Type
  GraphType graph_type;

  // Store odometry measurements of this robot
  std::vector<RelativePosePoseMeasurement> odometry_;

  // Store private loop closures of this robot
  RelativeMeasurements private_lcs_;

  // Store shared loop closure measurements
  RelativeMeasurements shared_lcs_;

  // Store the set of public poses that need to be sent to other robots
  PoseSet loc_shared_pose_ids_;

  // Store the set of public landmarks that need to be sent to other robots
  LandmarkSet loc_shared_landmark_ids_;

  // Store the set of public unit spheres that need to be sent to other robots
  UnitSphereSet loc_shared_unit_sphere_ids_;

  // Store the set of public poses needed from other robots
  PoseSet nbr_shared_pose_ids_;

  // Store the set of public landmarks needed from other robots
  LandmarkSet nbr_shared_landmark_ids_;

  // Store the set of public unit spheres needed from other robots
  UnitSphereSet nbr_shared_unit_sphere_ids_;

  // Store the set of neighboring agents
  std::set<unsigned> nbr_robot_ids_;

  // Store the set of deactivated neighbors
  std::map<unsigned, bool> neighbor_active_;

  // Store public poses from neighbors
  PoseDict neighbor_poses_;

  // Store public landmarks from neighbors
  LandmarkDict neighbor_landmarks_;

  // Store public unit spheres from neighbors
  UnitSphereDict neighbor_unit_spheres_;

  // Quadratic matrix in cost function
  std::optional<SparseMatrix> Q_;

  // Linear matrix in cost function
  std::optional<Matrix> G_;

  // Preconditioner
  std::optional<CholmodSolverPtr> precon_;

  // Invalid index when populating cost terms
  static constexpr size_t IDX_NOT_SET = std::numeric_limits<size_t>::max();

  // Expected number of non-zero elements per column in sparse matrices
  static constexpr unsigned int SPARSE_ENTRIES = 8;

  // Timing
  SimpleTimer timer_;
  double ms_construct_Q_{};
  double ms_construct_G_{};
  double ms_construct_precon_{};
  /**
   * @brief Add odometry edge. Ignored if the input measurement already exists.
   * @param factor
   */
  void addOdometry(const RelativeMeasurement &factor);
  /**
   * @brief Add private loop closure. Ignored if the input measurement already
   * exists.
   * @param factor
   */
  void addPrivateLoopClosure(const RelativeMeasurement &factor);
  /**
   * @brief Add shared loop closure. Ignored if the input measurement already
   * exists.
   * @param factor
   */
  void addSharedLoopClosure(const RelativeMeasurement &factor);
  /**
   * @brief Construct the quadratic cost matrix
   * @return
   */
  bool constructQ();
  /**
   * @brief Construct the linear cost matrix
   * @return
   */
  bool constructG();
  /**
   * @brief Helper function to construct the quadratic cost term for PGO
   * @return
   */
  bool constructQuadraticCostTermPGO();
  /**
   * @brief Helper function to construct the linear cost term for PGO
   * @return
   */
  bool constructLinearCostTermPGO();
  /**
   * @brief Helper function to construct the quadratic cost term for RA-SLAM
   * @return
   */
  bool constructQuadraticCostTermRASLAM();
  /**
   * @brief Helper function to construct the linear cost term for RA-SLAM
   * @return
   */
  bool constructLinearCostTermRASLAM();
  /**
   * @brief Helper function to set indices i and j when constructing cost terms.
   * See Eq (7) of the SE-Sync paper for details
   * @return
   */
  std::optional<bool>
  setIndicesFromStateOwnership(const RelativeMeasurement &measurement,
                               size_t *i, size_t *j);
  /**
   * @brief Helper function to determine if a state (which is either the source
   * or destination state of a relative measurement) is owned by an active
   * neighbor. Return true if the neighbor is active and if the query
   * neighborStateID belongs to the neighbor. If the neighbor is inactive,
   * return false. If a measurement is taken with an inactive neighbors or if
   * the query neighborStateID does not belong to the neighbor, return
   * std::nullopt. Note that, in the context of the Graph class, a neighbor is
   * considered "active" if it shares a loop closure with the agent associated
   * with this graph. "Active" in this context has no notion of whether or not
   * the neighbor is actively participating in distributed optimization.
   * @param neighborStateID
   * @return
   */
  std::optional<bool>
  isStateOwnedByActiveNeighbor(const StateID &neighborStateID);
  /**
   * @brief Helper function to return the lifted data matrix of the fixed public
   * variable (associated with neighborStateID) owned by a neighbor of this
   * agent. Supported fixed variables include poses, landmarks, and unit spheres
   * @param neighborStateID
   * @return
   */
  Matrix getNeighborFixedVariableLiftedData(const StateID &neighborStateID);
  /**
   * @brief Construct the preconditioner for this graph
   * @return
   */
  bool constructPreconditioner();
  /**
   * @brief Helper function for computing the regularization term reg for the
   * regularized Cholesky preconditioner of P so that (P + reg*I)^-1 has a
   * suitable condition number
   * @param P
   * @return
   */
  double computePreconditionerRegularization(const SparseMatrix &P);

private:
  // Mapping Edge ID to the corresponding index in the vector of measurements
  // (either odometry, private loop closures, or shared loop closures)
  EdgeIDMap edge_id_to_index_;

  // Use measurements with inactive neighbors when constructing data matrices
  bool use_inactive_neighbors_;

  // Weights for prior terms (TODO(YT): expose as parameter)
  double prior_kappa_;
  double prior_tau_;

  // Priors
  std::map<unsigned, LiftedPose> pose_priors_;
  std::map<unsigned, LiftedPoint> landmark_priors_;
};

} // namespace DCORA
