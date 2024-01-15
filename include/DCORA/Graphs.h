/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#pragma once

#include <DCORA/DCORA_types.h>
#include <DCORA/DCORA_utils.h>
#include <DCORA/RelativeMeasurement.h>
#include <DCORA/manifold/Elements.h>

#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace DCORA {

/**
 * @brief A pose graph class representing the local optimization problem in
 * distributed PGO.
 */
class PoseGraph {
public:
  /**
   * @brief Store statistics for the current pose graph
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
   * @param id
   * @param r
   * @param d
   */
  PoseGraph(unsigned int id, unsigned int r, unsigned int d);
  /**
   * @brief Destructor
   */
  ~PoseGraph();
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
   * @brief Return number of odometry edges
   * @return
   */
  unsigned int numOdometry() const { return odometry_.size(); }
  /**
   * @brief Return number of private loop closures;
   * @return
   */
  unsigned int numPrivateLoopClosures() const { return private_lcs_.size(); }
  /**
   * @brief Return number of shared loop closures
   * @return
   */
  unsigned int numSharedLoopClosures() const { return shared_lcs_.size(); }
  /**
   * @brief Return the number of all measurements
   * @return
   */
  unsigned int numMeasurements() const;
  /**
   * @brief Clear all contents and reset this pose graph to be empty
   */
  void empty();
  /**
   * @brief Clear all temporary data and only keep measurements
   */
  void reset();
  /**
   * @brief Clear all cached neighbor poses
   */
  void clearNeighborPoses();
  /**
   * @brief Set measurements for this pose graph
   * @param measurements
   */
  void setMeasurements(const std::vector<RelativeSEMeasurement> &measurements);
  /**
   * @brief Add a single measurement to this pose graph. Ignored if the input
   * measurement already exists.
   * @param m
   */
  void addMeasurement(const RelativeSEMeasurement &m);
  /**
   * @brief Return a copy of the list of odometry edges
   * @return
   */
  std::vector<RelativeSEMeasurement> odometry() const { return odometry_; }
  /**
   * @brief Return a copy of the list of private loop closures
   * @return
   */
  std::vector<RelativeSEMeasurement> privateLoopClosures() const {
    return private_lcs_;
  }
  /**
   * @brief Return a copy of the list of shared loop closures
   * @return
   */
  std::vector<RelativeSEMeasurement> sharedLoopClosures() const {
    return shared_lcs_;
  }
  /**
   * @brief Return a copy of all inter-robot loop closures with the specified
   * neighbor
   * @param neighbor_id
   * @return
   */
  std::vector<RelativeSEMeasurement>
  sharedLoopClosuresWithRobot(unsigned neighbor_id) const;
  /**
   * @brief Return a copy of all measurements
   * @return
   */
  std::vector<RelativeSEMeasurement> measurements() const;
  /**
   * @brief Return a copy of all LOCAL measurements (i.e., without inter-robot
   * loop closures)
   * @return
   */
  std::vector<RelativeSEMeasurement> localMeasurements() const;
  /**
   * @brief Clear all priors
   */
  void clearPriors();
  /**
   * @brief Add a prior term
   * @param index The index of the local variable
   * @param Xi Corresponding prior term
   */
  void setPrior(unsigned index, const LiftedPose &Xi);
  /**
   * @brief Set neighbor poses
   * @param pose_dict
   */
  void setNeighborPoses(const PoseDict &pose_dict);
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
   * @brief Get the set of Pose IDs that ALL neighbors need to share with me
   * @return
   */
  PoseSet neighborPublicPoseIDs() const { return nbr_shared_pose_ids_; }
  /**
   * @brief Get the set of Pose IDs that active neighbors need to share with me.
   * A neighbor is active if it is actively participating in distributed
   * optimization with this robot.
   */
  PoseSet activeNeighborPublicPoseIDs() const;
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
   */
  std::set<unsigned> activeNeighborIDs() const;
  /**
   * @brief Return the number of active neighbors.
   * A neighbor is active if it is actively participating in distributed
   * optimization with this robot.
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
   */
  bool isNeighborActive(unsigned int neighbor_id) const;
  /**
   * @brief Set the input neighbor to be active or inactive.
   * Does nothing if the input is not a neighbor.
   */
  void setNeighborActive(unsigned int neighbor_id, bool active);
  /**
   * @brief Return true if the given neighbor pose ID is required by me
   * @param pose_id
   * @return
   */
  bool requireNeighborPose(const PoseID &pose_id) const;
  /**
   * @brief Compute number of accepted, rejected, and undecided loop closures
   * Note that loop closures with inactive neighbors are not included
   * @return
   */
  Statistics statistics() const;
  /**
   * @brief Check if a measurement exists in the pose graph
   */
  bool hasMeasurement(const PoseID &srcID, const PoseID &dstID) const;
  /**
   * @brief Find and return a writable pointer to the specified measurement
   * within this pose graph
   * @param measurements
   * @param srcID
   * @param dstID
   * @return writable pointer to the desired measurement (nullptr if measurement
   * does not exists)
   */
  RelativeSEMeasurement *findMeasurement(const PoseID &srcID,
                                         const PoseID &dstID);
  /**
   * @brief Return a vector of writable pointers to all loop closures in the
   * pose graph (contains both private and inter-robot loop closures)
   * @return Vector of pointers to all loop closures
   */
  std::vector<RelativeSEMeasurement *> allLoopClosures();
  /**
   * @brief Return a vector of pointers to all active loop closures
   */
  std::vector<RelativeSEMeasurement *> activeLoopClosures();
  /**
   * @brief Return a vector of pointers to all inactive loop closures
   */
  std::vector<RelativeSEMeasurement *> inactiveLoopClosures();
  /**
   * @brief Set to true to use measurements with inactive neighbors
   */
  void useInactiveNeighbors(bool use = true);

protected:
  // ID associated with this agent
  const unsigned int id_;

  // Problem dimensions
  unsigned int r_, d_, n_;

  // Store odometry measurement of this robot
  std::vector<RelativeSEMeasurement> odometry_;

  // Store private loop closures of this robot
  std::vector<RelativeSEMeasurement> private_lcs_;

  // Store shared loop closure measurements
  std::vector<RelativeSEMeasurement> shared_lcs_;

  // Store the set of public poses that need to be sent to other robots
  PoseSet local_shared_pose_ids_;

  // Store the set of public poses needed from other robots
  PoseSet nbr_shared_pose_ids_;

  // Store the set of neighboring agents
  std::set<unsigned> nbr_robot_ids_;

  // Store the set of deactivated neighbors
  std::map<unsigned, bool> neighbor_active_;

  // Store public poses from neighbors
  PoseDict neighbor_poses_;

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
  void addOdometry(const RelativeSEMeasurement &factor);
  /**
   * @brief Add private loop closure. Ignored if the input measurement already
   * exists.
   * @param factor
   */
  void addPrivateLoopClosure(const RelativeSEMeasurement &factor);
  /**
   * @brief Add shared loop closure. Ignored if the input measurement already
   * exists.
   * @param factor
   */
  void addSharedLoopClosure(const RelativeSEMeasurement &factor);
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
  /**
   * @brief Based on the current list of inter-robot (shared)
   * loop closures, update the list of public pose IDs of myself
   * and my neighbors.
   */
  void updatePublicPoseIDs();

private:
  // Mapping Edge ID to the corresponding index in the vector of measurements
  // (either odometry, private loop closures, or public loop closures)
  std::unordered_map<EdgeID, size_t, HashEdgeID> edge_id_to_index_;

  // Use measurements with inactive neighbors when constructing data matrices
  bool use_inactive_neighbors_;

  // Weights for prior terms (TODO: expose as parameter)
  double prior_kappa_;
  double prior_tau_;

  // Priors
  std::map<unsigned, LiftedPose> priors_;
};

} // namespace DCORA
