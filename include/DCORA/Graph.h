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

#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace DCORA {

/**
 * @brief A graph class representing the local optimization problem in
 * distributed PGO and/or RA-SLAM.
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
   */
  Graph(unsigned int id, unsigned int r, unsigned int d);
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
   * @brief Return number of odometry edges
   * @return
   */
  unsigned int numOdometry() const { return odometry_.size(); }
  /**
   * @brief Return number of private loop closures;
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
   * @brief Return the number of all measurements
   * @return
   */
  unsigned int numMeasurements() const {
    return numOdometry() + numPrivateLoopClosures() + numSharedLoopClosures();
  }
  /**
   * @brief Return true if the graph is compatible with PGO
   * @return
   */
  bool isPGOCompatible() const { return (b_ == 0 && l_ == 0); }
  /**
   * @brief Clear all contents and reset this graph to be empty
   */
  void empty();
  /**
   * @brief Clear all temporary data and only keep measurements
   */
  void reset();
  /**
   * @brief Clear all cached neighbor states
   */
  void clearNeighborStates();
  /**
   * @brief Update the number of poses and landmarks
   * @param stateID
   * @return
   */
  void updateNumStates(const StateID &stateID);
  /**
   * @brief Update the number of unit sphere variables. A flag sets the robot ID
   * to be used for ownership of the associated range measurement.
   * @param measurement
   * @param useSourceIDforOwnership
   * @return
   */
  void updateNumRanges(const RelativeMeasurement &measurement,
                       bool useSourceIDforOwnership = true);
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
   * @brief Return a copy of all inter-robot loop closures with the specified
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
   * @brief Return a copy of all LOCAL measurements (i.e., without inter-robot
   * loop closures)
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
   * @brief Add a point prior term
   * @param index The index of the local variable
   * @param ti Corresponding point prior term
   */
  void setPrior(unsigned index, const LiftedPoint &ti);
  /**
   * @brief Set neighbor state
   * @param pose_dict
   * @param point_dict
   */
  void setNeighborStates(const PoseDict &pose_dict,
                         const PointDict &point_dict);
  /**
   * @brief Set neighbor poses
   * @param pose_dict
   */
  void setNeighborPoses(const PoseDict &pose_dict);
  /**
   * @brief Set neighbor points
   * @param point_dict
   */
  void setNeighborPoints(const PointDict &point_dict);
  /**
   * @brief Get quadratic cost matrix.
   * @return
   */
  const SparseMatrix &quadraticMatrix();
  /**
   * @brief Clear the quadratic cost matrix
   */
  void clearQuadraticMatrix();
  /**
   * @brief Get linear cost matrix.
   * @return
   */
  const Matrix &linearMatrix();
  /**
   * @brief Clear the linear cost matrix
   */
  void clearLinearMatrix();
  /**
   * @brief Construct data matrices that are needed for optimization, if they do
   * not yet exist
   * @return true if construction is successful
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
  PoseSet myPublicPoseIDs() const { return local_shared_pose_ids_; }
  /**
   * @brief Get the set of my point IDs that are shared with other robots
   * @return
   */
  PointSet myPublicPointIDs() const { return local_shared_point_ids_; }
  /**
   * @brief Get the set of Pose IDs that ALL neighbors need to share with me
   * @return
   */
  PoseSet neighborPublicPoseIDs() const { return nbr_shared_pose_ids_; }
  /**
   * @brief Get the set of Point IDs that ALL neighbors need to share with me
   * @return
   */
  PointSet neighborPublicPointIDs() const { return nbr_shared_point_ids_; }
  /**
   * @brief Get the set of Pose IDs that active neighbors need to share with me.
   * A neighbor is active if it is actively participating in distributed
   * optimization with this robot.
   */
  PoseSet activeNeighborPublicPoseIDs() const;
  /**
   * @brief Get the set of Point IDs that active neighbors need to share with
   * me. A neighbor is active if it is actively participating in distributed
   * optimization with this robot.
   */
  PointSet activeNeighborPublicPointIDs() const;
  /**
   * @brief Get the set of neighbor robot IDs that share inter-robot loop
   * closures with me
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
   * @brief Return true if the input robot is a neighbor (i.e., share
   * inter-robot loop closure)
   * @param robot_id
   * @return
   */
  bool hasNeighbor(unsigned int robot_id) const;
  /**
   * @brief Check if the input neighbor is active
   * @return false if the input robot is not a neighbor or is ignored
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
   * @brief Return true if the given neighbor point ID is required by me
   * @param point_id
   * @return
   */
  bool requireNeighborPoint(const PointID &point_id) const;
  /**
   * @brief Compute number of accepted, rejected, and undecided loop closures
   * Note that loop closures with inactive neighbors are not included
   * @return
   */
  Statistics statistics() const;
  /**
   * @brief Check if a measurement exists in the graph
   * @param srcID
   * @param dstID
   * @return
   */
  bool hasMeasurement(const StateID &srcID, const StateID &dstID) const;
  /**
   * @brief Find and return a writable pointer to the specified measurement
   * within this graph
   * @param measurements
   * @param srcID
   * @param dstID
   * @return writable pointer to the desired measurement (nullptr if measurement
   * does not exists)
   */
  RelativeMeasurement *findMeasurement(const StateID &srcID,
                                       const StateID &dstID);
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

  // Store odometry measurements of this robot
  std::vector<RelativePosePoseMeasurement> odometry_;

  // Store private loop closures of this robot
  RelativeMeasurements private_lcs_;

  // Store shared loop closure measurements
  RelativeMeasurements shared_lcs_;

  // Store the set of public poses that need to be sent to other robots
  PoseSet local_shared_pose_ids_;

  // Store the set of public points that need to be sent to other robots
  PointSet local_shared_point_ids_;

  // Store the set of public poses needed from other robots
  PoseSet nbr_shared_pose_ids_;

  // Store the set of public points needed from other robots
  PointSet nbr_shared_point_ids_;

  // Store the set of neighboring agents
  std::set<unsigned> nbr_robot_ids_;

  // Store the set of deactivated neighbors
  std::map<unsigned, bool> neighbor_active_;

  // Store public poses from neighbors
  PoseDict neighbor_poses_;

  // Store public points from neighbors
  PointDict neighbor_points_;

  // Quadratic matrix in cost function
  std::optional<SparseMatrix> Q_;

  // Linear matrix in cost function
  std::optional<Matrix> G_;

  // Preconditioner
  std::optional<CholmodSolverPtr> precon_;

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
   * @brief Construct the preconditioner for this graph
   * @return
   */
  bool constructPreconditioner();

private:
  // Mapping Edge ID to the corresponding index in the vector of measurements
  // (either odometry, private loop closures, or public loop closures)
  std::unordered_map<EdgeID, size_t, HashEdgeID> edge_id_to_index_;

  // Use measurements with inactive neighbors when constructing data matrices
  bool use_inactive_neighbors_;

  // Weights for prior terms (TODO(YT): expose as parameter)
  double prior_kappa_;
  double prior_tau_;

  // Priors
  std::map<unsigned, LiftedPose> pose_priors_;
  std::map<unsigned, LiftedPoint> point_priors_;
};

} // namespace DCORA
