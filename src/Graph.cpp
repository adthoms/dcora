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

#include <DCORA/Graph.h>
#include <glog/logging.h>

namespace DCORA {

Graph::Graph(unsigned int id, unsigned int r, unsigned int d)
    : id_(id),
      r_(r),
      d_(d),
      n_(0),
      l_(0),
      b_(0),
      use_inactive_neighbors_(false),
      prior_kappa_(10000),
      prior_tau_(100) {
  CHECK(r >= d);
  empty();
}

Graph::~Graph() { empty(); }

void Graph::empty() {
  // Reset this graph to be empty
  n_ = 0;
  l_ = 0;
  b_ = 0;
  edge_id_to_index_.clear();
  odometry_.clear();
  private_lcs_.vec.clear();
  shared_lcs_.vec.clear();
  local_shared_pose_ids_.clear();
  local_shared_point_ids_.clear();
  nbr_shared_pose_ids_.clear();
  nbr_shared_point_ids_.clear();
  nbr_robot_ids_.clear();
  neighbor_active_.clear();
  clearNeighborStates();
  clearDataMatrices();
  clearPriors();
}

void Graph::reset() {
  clearNeighborStates();
  clearDataMatrices();
  clearPriors();
  for (const auto neighbor_id : nbr_robot_ids_) {
    neighbor_active_[neighbor_id] = true;
  }
}

void Graph::clearNeighborStates() {
  neighbor_poses_.clear();
  neighbor_points_.clear();
  G_.reset(); // Clearing neighbor poses requires re-computing linear matrix
}

void Graph::updateNumStates(const StateID &stateID) {
  // Update num poses
  if (stateID.isPose())
    n_ = std::max(n_, static_cast<unsigned int>(stateID.frame_id + 1));
  // Update num landmarks
  if (stateID.isPoint())
    b_ = std::max(b_, static_cast<unsigned int>(stateID.frame_id + 1));
}

void Graph::setMeasurements(
    const std::vector<RelativePosePoseMeasurement> &measurements) {
  // Reset this graph to be empty
  empty();
  for (const auto &m : measurements)
    addMeasurement(m);
}

void Graph::setMeasurements(const RelativeMeasurements &measurements) {
  // Reset this graph to be empty
  empty();
  for (const auto &m : measurements.vec)
    std::visit([this](auto &&m) { this->addMeasurement(m); }, m);
}

void Graph::addMeasurement(const RelativeMeasurement &m) {
  if (m.r1 != id_ && m.r2 != id_) {
    LOG(WARNING) << "Input contains irrelevant edges! \n" << m;
    return;
  }
  if (m.r1 == id_ && m.r2 == id_) {
    if (m.measurementType == MeasurementType::PosePose && m.p1 + 1 == m.p2)
      addOdometry(m);
    else
      addPrivateLoopClosure(m);
  } else {
    addSharedLoopClosure(m);
  }
}

void Graph::addOdometry(const RelativeMeasurement &factor) {
  // Check for duplicate odometry
  const StateID src_id = factor.getSrcID();
  const StateID dst_id = factor.getDstID();
  if (hasMeasurement(src_id, dst_id))
    return;

  // Check that this is a valid measurement
  factor.checkDim(d_);

  // Check that this is an odometry measurement
  CHECK(factor.measurementType == MeasurementType::PosePose);
  CHECK(factor.r1 == id_);
  CHECK(factor.r2 == id_);
  CHECK(factor.p1 + 1 == factor.p2);

  // Dynamically cast to odometry measurement
  const RelativePosePoseMeasurement &odom_factor =
      dynamic_cast<const RelativePosePoseMeasurement &>(factor);

  // Update states
  updateNumStates(dst_id); // dst_id > src_id

  // Add relative measurement factor to odometry
  odometry_.push_back(odom_factor);

  // Update edges
  const EdgeID edge_id(src_id, dst_id);
  edge_id_to_index_.emplace(edge_id, odometry_.size() - 1);
}

void Graph::addPrivateLoopClosure(const RelativeMeasurement &factor) {
  // Check for duplicate private loop closure
  const StateID src_id = factor.getSrcID();
  const StateID dst_id = factor.getDstID();
  if (hasMeasurement(src_id, dst_id))
    return;

  // Check that this is a valid measurement
  factor.checkDim(d_);

  // Check that this is a private loop closure
  CHECK(factor.r1 == id_);
  CHECK(factor.r2 == id_);

  // Update states
  updateNumStates(src_id);
  updateNumStates(dst_id);

  // Add relative measurement factor to private loop closures
  private_lcs_.push_back(factor);

  // Update edges
  const EdgeID edge_id(src_id, dst_id);
  edge_id_to_index_.emplace(edge_id, private_lcs_.vec.size() - 1);
}

void Graph::addSharedLoopClosure(const RelativeMeasurement &factor) {
  // Check for duplicate shared loop closure
  const StateID src_id = factor.getSrcID();
  const StateID dst_id = factor.getDstID();
  if (hasMeasurement(src_id, dst_id))
    return;

  // Check that this is a valid measurement
  factor.checkDim(d_);

  // Update local and neighbor shared state IDs. Set active neighbor.
  if (factor.r1 == id_) {
    CHECK(factor.r2 != id_);

    // Update states
    updateNumStates(src_id);

    // Add local shared state to graph
    switch (factor.stateType1) {
    case StateType::Pose:
      local_shared_pose_ids_.emplace(factor.r1, factor.p1);
      break;
    case StateType::Point:
      local_shared_point_ids_.emplace(factor.r1, factor.p1);
      break;
    default:
      LOG(FATAL) << "Invalid StateType1: "
                 << StateTypeToString(factor.stateType1) << "!";
    }

    // Add neighbor shared state to graph
    switch (factor.stateType2) {
    case StateType::Pose:
      nbr_shared_pose_ids_.emplace(factor.r2, factor.p2);
      break;
    case StateType::Point:
      nbr_shared_point_ids_.emplace(factor.r2, factor.p2);
      break;
    default:
      LOG(FATAL) << "Invalid StateType2: "
                 << StateTypeToString(factor.stateType2) << "!";
    }

    // Update neighbor robot IDs
    nbr_robot_ids_.insert(factor.r2);

    // Set active neighbor
    neighbor_active_[factor.r2] = true;
  } else {
    CHECK(factor.r2 == id_);

    // Update states
    updateNumStates(dst_id);

    // Add local shared state to graph
    switch (factor.stateType2) {
    case StateType::Pose:
      local_shared_pose_ids_.emplace(factor.r2, factor.p2);
      break;
    case StateType::Point:
      local_shared_point_ids_.emplace(factor.r2, factor.p2);
      break;
    default:
      LOG(FATAL) << "Invalid StateType2: "
                 << StateTypeToString(factor.stateType2) << "!";
    }

    // Add neighbor shared state to graph
    switch (factor.stateType1) {
    case StateType::Pose:
      nbr_shared_pose_ids_.emplace(factor.r1, factor.p1);
      break;
    case StateType::Point:
      nbr_shared_point_ids_.emplace(factor.r1, factor.p1);
      break;
    default:
      LOG(FATAL) << "Invalid StateType1: "
                 << StateTypeToString(factor.stateType1) << "!";
    }

    // Update neighbor robot IDs
    nbr_robot_ids_.insert(factor.r1);

    // Set active neighbor
    neighbor_active_[factor.r1] = true;
  }

  // Add relative measurement factor to shared loop closures
  shared_lcs_.push_back(factor);

  // Update edges
  const EdgeID edge_id(src_id, dst_id);
  edge_id_to_index_.emplace(edge_id, shared_lcs_.vec.size() - 1);
}

RelativeMeasurements
Graph::sharedLoopClosuresWithRobot(unsigned int neighbor_id) const {
  RelativeMeasurements result;
  // TODO(AT): update for all measurements
  for (const auto &m : shared_lcs_.GetRelativePosePoseMeasurements()) {
    if (m.r1 == neighbor_id || m.r2 == neighbor_id)
      result.vec.emplace_back(m);
  }
  return result;
}

RelativeMeasurements Graph::measurements() const {
  RelativeMeasurements measurements(localMeasurements());
  measurements.vec.insert(measurements.vec.end(), shared_lcs_.vec.begin(),
                          shared_lcs_.vec.end());
  return measurements;
}

RelativeMeasurements Graph::localMeasurements() const {
  RelativeMeasurements measurements;
  measurements.vec.reserve(odometry_.size() + private_lcs_.vec.size());
  measurements.vec.insert(measurements.vec.end(), odometry_.begin(),
                          odometry_.end());
  measurements.vec.insert(measurements.vec.end(), private_lcs_.vec.begin(),
                          private_lcs_.vec.end());
  return measurements;
}

void Graph::clearPriors() {
  pose_priors_.clear();
  point_priors_.clear();
}

void Graph::setPrior(unsigned index, const LiftedPose &Xi) {
  CHECK_LT(index, n());
  CHECK_EQ(d(), Xi.d());
  CHECK_EQ(r(), Xi.r());
  pose_priors_[index] = Xi;
}

void Graph::setPrior(unsigned index, const LiftedPoint &ti) {
  CHECK_LT(index, b());
  CHECK_EQ(d(), ti.d());
  CHECK_EQ(r(), ti.r());
  point_priors_[index] = ti;
}

void Graph::setNeighborStates(const PoseDict &pose_dict,
                              const PointDict &point_dict) {
  neighbor_poses_ = pose_dict;
  neighbor_points_ = point_dict;
  G_.reset(); // Setting neighbor states requires re-computing linear matrix
}

void Graph::setNeighborPoses(const PoseDict &pose_dict) {
  neighbor_poses_ = pose_dict;
  G_.reset(); // Setting neighbor poses requires re-computing linear matrix
}

void Graph::setNeighborPoints(const PointDict &point_dict) {
  neighbor_points_ = point_dict;
  G_.reset(); // Setting neighbor points requires re-computing linear matrix
}

bool Graph::hasNeighbor(unsigned int robot_id) const {
  return nbr_robot_ids_.find(robot_id) != nbr_robot_ids_.end();
}

bool Graph::isNeighborActive(unsigned int neighbor_id) const {
  if (!hasNeighbor(neighbor_id)) {
    return false;
  }
  return neighbor_active_.at(neighbor_id);
}

void Graph::setNeighborActive(unsigned int neighbor_id, bool active) {
  if (!hasNeighbor(neighbor_id)) {
    return;
  }
  if (neighbor_active_.at(neighbor_id) != active) {
    clearDataMatrices();
  }
  neighbor_active_[neighbor_id] = active;
}

bool Graph::requireNeighborPose(const PoseID &pose_id) const {
  return nbr_shared_pose_ids_.find(pose_id) != nbr_shared_pose_ids_.end();
}

bool Graph::requireNeighborPoint(const PointID &point_id) const {
  return nbr_shared_point_ids_.find(point_id) != nbr_shared_point_ids_.end();
}

bool Graph::hasMeasurement(const StateID &srcID, const StateID &dstID) const {
  const EdgeID edge_id(srcID, dstID);
  return edge_id_to_index_.find(edge_id) != edge_id_to_index_.end();
}

RelativeMeasurement *Graph::findMeasurement(const StateID &srcID,
                                            const StateID &dstID) {
  RelativeMeasurement *edge = nullptr;
  if (hasMeasurement(srcID, dstID)) {
    const EdgeID edge_id(srcID, dstID);
    size_t index = edge_id_to_index_.at(edge_id);
    if (edge_id.isOdometry()) {
      edge = &odometry_[index];
    } else if (edge_id.isPrivateLoopClosure()) {
      // TODO(AT): update for all measurements
      auto m = private_lcs_.GetRelativePosePoseMeasurements();
      edge = &m[index];
    } else {
      // TODO(AT): update for all measurements
      auto m = shared_lcs_.GetRelativePosePoseMeasurements();
      edge = &m[index];
    }
  }
  if (edge) {
    // Sanity check
    CHECK(edge->stateType1 == srcID.state_type);
    CHECK(edge->stateType2 == dstID.state_type);
    CHECK_EQ(edge->r1, srcID.robot_id);
    CHECK_EQ(edge->p1, srcID.frame_id);
    CHECK_EQ(edge->r2, dstID.robot_id);
    CHECK_EQ(edge->p2, dstID.frame_id);
  }
  return edge;
}

std::vector<RelativePosePoseMeasurement *> Graph::allLoopClosures() {
  std::vector<RelativePosePoseMeasurement *> output;
  // TODO(AT): Function is unused. Deprecate.
  for (auto &m : private_lcs_.GetRelativePosePoseMeasurements()) {
    output.push_back(&m);
  }
  for (auto &m : shared_lcs_.GetRelativePosePoseMeasurements()) {
    output.push_back(&m);
  }
  return output;
}

std::set<unsigned> Graph::activeNeighborIDs() const {
  std::set<unsigned> output;
  for (unsigned neighbor_id : nbr_robot_ids_) {
    if (isNeighborActive(neighbor_id)) {
      output.emplace(neighbor_id);
    }
  }
  return output;
}

size_t Graph::numActiveNeighbors() const { return activeNeighborIDs().size(); }

PoseSet Graph::activeNeighborPublicPoseIDs() const {
  PoseSet output;
  for (const auto &pose_id : nbr_shared_pose_ids_) {
    if (isNeighborActive(pose_id.robot_id)) {
      output.emplace(pose_id);
    }
  }
  return output;
}

PointSet Graph::activeNeighborPublicPointIDs() const {
  PointSet output;
  for (const auto &point_id : nbr_shared_point_ids_) {
    if (isNeighborActive(point_id.robot_id)) {
      output.emplace(point_id);
    }
  }
  return output;
}

std::vector<RelativePosePoseMeasurement *> Graph::activeLoopClosures() {
  std::vector<RelativePosePoseMeasurement *> output;
  // TODO(AT): update for all measurements
  for (auto &m : private_lcs_.GetRelativePosePoseMeasurements()) {
    output.push_back(&m);
  }
  for (auto &m : shared_lcs_.GetRelativePosePoseMeasurements()) {
    if (m.r1 == id_ && isNeighborActive(m.r2)) {
      output.push_back(&m);
    } else if (m.r2 == id_ && isNeighborActive(m.r1)) {
      output.push_back(&m);
    }
  }
  return output;
}

std::vector<RelativePosePoseMeasurement *> Graph::inactiveLoopClosures() {
  std::vector<RelativePosePoseMeasurement *> output;
  // TODO(AT): Function is unused. Deprecate.
  for (auto &m : shared_lcs_.GetRelativePosePoseMeasurements()) {
    if (m.r1 == id_ && !isNeighborActive(m.r2)) {
      output.push_back(&m);
    } else if (m.r2 == id_ && !isNeighborActive(m.r1)) {
      output.push_back(&m);
    }
  }
  return output;
}

Graph::Statistics Graph::statistics() const {
  // Currently, this function is only meaningful for GNC_TLS
  double totalCount = 0;
  double acceptCount = 0;
  double rejectCount = 0;
  // TODO(YT): specify tolerance for rejected and accepted loop closures
  // TODO(AT): update for all measurements
  for (const auto &m : private_lcs_.GetRelativePosePoseMeasurements()) {
    // if (m.fixedWeight) continue;
    if (m.weight == 1) {
      acceptCount += 1;
    } else if (m.weight == 0) {
      rejectCount += 1;
    }
    totalCount += 1;
  }
  // TODO(AT): update for all measurements
  for (const auto &m : shared_lcs_.GetRelativePosePoseMeasurements()) {
    // Skip loop closures with inactive neighbors
    if (m.r1 == id_ && !isNeighborActive(m.r2)) {
      continue;
    }
    if (m.r2 == id_ && !isNeighborActive(m.r1)) {
      continue;
    }
    if (m.weight == 1) {
      acceptCount += 1;
    } else if (m.weight == 0) {
      rejectCount += 1;
    }
    totalCount += 1;
  }

  Graph::Statistics statistics;
  statistics.total_loop_closures = totalCount;
  statistics.accept_loop_closures = acceptCount;
  statistics.reject_loop_closures = rejectCount;
  statistics.undecided_loop_closures = totalCount - acceptCount - rejectCount;

  return statistics;
}

const SparseMatrix &Graph::quadraticMatrix() {
  if (!Q_.has_value())
    constructQ();
  CHECK(Q_.has_value());
  return Q_.value();
}

void Graph::clearQuadraticMatrix() {
  Q_.reset();
  precon_.reset(); // Also clear the preconditioner since it depends on Q
}

const Matrix &Graph::linearMatrix() {
  if (!G_.has_value())
    constructG();
  CHECK(G_.has_value());
  return G_.value();
}

void Graph::clearLinearMatrix() { G_.reset(); }

bool Graph::constructDataMatrices() {
  if (!Q_.has_value() && !constructQ())
    return false;
  if (!G_.has_value() && !constructG())
    return false;
  return true;
}

void Graph::clearDataMatrices() {
  clearQuadraticMatrix();
  clearLinearMatrix();
}

bool Graph::constructQ() {
  timer_.tic();
  if (!private_lcs_.isPGOCompatible() || !shared_lcs_.isPGOCompatible())
    LOG(FATAL) << "Error: Loop closures are not PGO compatible!";
  std::vector<RelativePosePoseMeasurement> privateMeasurements = odometry_;
  std::vector<RelativePosePoseMeasurement> private_lcs_pose_pose =
      private_lcs_.GetRelativePosePoseMeasurements();
  privateMeasurements.insert(privateMeasurements.end(),
                             private_lcs_pose_pose.begin(),
                             private_lcs_pose_pose.end());

  // Initialize Q with private measurements
  SparseMatrix QLocal = constructConnectionLaplacianSE(privateMeasurements);

  // Initialize relative SE matrix in homogeneous form
  Matrix T = Matrix::Zero(d_ + 1, d_ + 1);

  // Initialize aggregate weight matrix
  Matrix Omega = Matrix::Zero(d_ + 1, d_ + 1);

  // Shared (inter-robot) measurements only affect the diagonal blocks
  Matrix QDiagRow(d_ + 1, (d_ + 1) * n_);
  QDiagRow.setZero();

  // Go through shared loop closures
  for (const auto &m : shared_lcs_.GetRelativePosePoseMeasurements()) {
    // Set relative SE matrix (homogeneous form)
    T.block(0, 0, d_, d_) = m.R;
    T.block(0, d_, d_, 1) = m.t;
    T(d_, d_) = 1;

    // Set aggregate weight matrix
    for (unsigned row = 0; row < d_; ++row) {
      Omega(row, row) = m.weight * m.kappa;
    }
    Omega(d_, d_) = m.weight * m.tau;

    if (m.r1 == id_) {
      // First pose belongs to this robot
      // Hence, this is an outgoing edge in the pose graph
      CHECK(m.r2 != id_);
      const PoseID nID(m.r2, m.p2);
      bool has_neighbor_pose =
          (neighbor_poses_.find(nID) != neighbor_poses_.end());
      if (isNeighborActive(m.r2)) {
        // Measurement with active neighbor
        if (!has_neighbor_pose) {
          LOG(WARNING) << "Missing active neighbor pose " << nID.robot_id
                       << ", " << nID.frame_id;
          return false;
        }
      } else {
        // Measurement with inactive neighbor
        if (!use_inactive_neighbors_ || !has_neighbor_pose) {
          continue;
        }
      }
      // Modify quadratic cost
      int idx = static_cast<int>(m.p1);
      Matrix W = T * Omega * T.transpose();
      QDiagRow.block(0, idx * (d_ + 1), d_ + 1, d_ + 1) += W;

    } else {
      // Second pose belongs to this robot
      // Hence, this is an incoming edge in the pose graph
      CHECK(m.r2 == id_);
      const PoseID nID(m.r1, m.p1);
      bool has_neighbor_pose =
          (neighbor_poses_.find(nID) != neighbor_poses_.end());
      if (isNeighborActive(m.r1)) {
        // Measurement with active neighbor
        if (!has_neighbor_pose) {
          LOG(WARNING) << "Missing active neighbor pose " << nID.robot_id
                       << ", " << nID.frame_id;
          return false;
        }
      } else {
        // Measurement with inactive neighbor
        if (!use_inactive_neighbors_ || !has_neighbor_pose) {
          continue;
        }
      }
      // Modify quadratic cost
      int idx = static_cast<int>(m.p2);
      QDiagRow.block(0, idx * (d_ + 1), d_ + 1, d_ + 1) += Omega;
    }
  }

  // Go through priors
  for (const auto &it : pose_priors_) {
    unsigned idx = it.first;
    for (unsigned row = 0; row < d_; ++row) {
      Omega(row, row) = prior_kappa_;
    }
    Omega(d_, d_) = prior_tau_;
    QDiagRow.block(0, idx * (d_ + 1), d_ + 1, d_ + 1) += Omega;
  }

  // Convert to a sparse matrix
  std::vector<Eigen::Triplet<double>> tripletList;
  tripletList.reserve((d_ + 1) * (d_ + 1) * n_);
  for (unsigned idx = 0; idx < n_; ++idx) {
    unsigned row_base = idx * (d_ + 1);
    unsigned col_base = row_base;
    for (unsigned r = 0; r < d_ + 1; ++r) {
      for (unsigned c = 0; c < d_ + 1; ++c) {
        double val = QDiagRow(r, col_base + c);
        tripletList.emplace_back(row_base + r, col_base + c, val);
      }
    }
  }
  SparseMatrix QDiag(QLocal.rows(), QLocal.cols());
  QDiag.setFromTriplets(tripletList.begin(), tripletList.end());

  Q_.emplace(QLocal + QDiag);
  ms_construct_Q_ = timer_.toc();
  // LOG(INFO) << "Robot " << id_ << " construct Q ms: " << ms_construct_Q_;
  return true;
}

bool Graph::constructG() {
  timer_.tic();
  unsigned d = d_;
  Matrix G(r_, (d_ + 1) * n_);
  G.setZero();
  Matrix T = Matrix::Zero(d + 1, d + 1);
  Matrix Omega = Matrix::Zero(d + 1, d + 1);
  // Go through shared measurements
  for (const auto &m : shared_lcs_.GetRelativePosePoseMeasurements()) {
    // Construct relative SE matrix in homogeneous form
    T.block(0, 0, d, d) = m.R;
    T.block(0, d, d, 1) = m.t;
    T(d, d) = 1;

    // Construct aggregate weight matrix
    for (unsigned row = 0; row < d; ++row) {
      Omega(row, row) = m.weight * m.kappa;
    }
    Omega(d, d) = m.weight * m.tau;

    if (m.r1 == id_) {
      // First pose belongs to this robot
      // Hence, this is an outgoing edge in the pose graph
      CHECK(m.r2 != id_);
      const PoseID nID(m.r2, m.p2);
      auto pair = neighbor_poses_.find(nID);
      bool has_neighbor_pose = (pair != neighbor_poses_.end());
      if (isNeighborActive(m.r2)) {
        // Measurement with active neighbor
        if (!has_neighbor_pose) {
          LOG(WARNING) << "Missing active neighbor pose " << nID.robot_id
                       << ", " << nID.frame_id;
          return false;
        }
      } else {
        // Measurement with inactive neighbor
        if (!use_inactive_neighbors_ || !has_neighbor_pose) {
          continue;
        }
      }
      Matrix Xj = pair->second.pose();
      int idx = static_cast<int>(m.p1);
      // Modify linear cost
      Matrix L = -Xj * Omega * T.transpose();
      G.block(0, idx * (d_ + 1), r_, d_ + 1) += L;
    } else {
      // Second pose belongs to this robot
      // Hence, this is an incoming edge in the pose graph
      CHECK(m.r2 == id_);
      const PoseID nID(m.r1, m.p1);
      auto pair = neighbor_poses_.find(nID);
      bool has_neighbor_pose = (pair != neighbor_poses_.end());
      if (isNeighborActive(m.r1)) {
        // Measurement with active neighbor
        if (!has_neighbor_pose) {
          LOG(WARNING) << "Missing active neighbor pose " << nID.robot_id
                       << ", " << nID.frame_id;
          return false;
        }
      } else {
        // Measurement with inactive neighbor
        if (!use_inactive_neighbors_ || !has_neighbor_pose) {
          continue;
        }
      }
      Matrix Xi = pair->second.pose();
      int idx = static_cast<int>(m.p2);
      // Modify linear cost
      Matrix L = -Xi * T * Omega;
      G.block(0, idx * (d_ + 1), r_, d_ + 1) += L;
    }
  }
  // Go through priors
  for (const auto &it : pose_priors_) {
    unsigned idx = it.first;
    const Matrix &P = it.second.getData();
    for (unsigned row = 0; row < d_; ++row) {
      Omega(row, row) = prior_kappa_;
    }
    Omega(d_, d_) = prior_tau_;
    Matrix L = -P * Omega;
    G.block(0, idx * (d_ + 1), r_, d_ + 1) += L;
  }
  G_.emplace(G);
  ms_construct_G_ = timer_.toc();
  // LOG(INFO) << "Robot " << id_ << " construct G ms: " << ms_construct_G_;
  return true;
}

bool Graph::hasPreconditioner() {
  if (!precon_.has_value())
    constructPreconditioner();
  return precon_.has_value();
}

const CholmodSolverPtr &Graph::preconditioner() {
  if (!precon_.has_value())
    constructPreconditioner();
  CHECK(precon_.has_value());
  return precon_.value();
}

bool Graph::constructPreconditioner() {
  timer_.tic();
  // Update preconditioner
  SparseMatrix P = quadraticMatrix();
  for (int i = 0; i < P.rows(); ++i) {
    P.coeffRef(i, i) += 1e-1;
  }
  auto solver = std::make_shared<CholmodSolver>();
  solver->compute(P);
  if (solver->info() != Eigen::ComputationInfo::Success)
    return false;
  precon_.emplace(solver);
  ms_construct_precon_ = timer_.toc();
  // LOG(INFO) << "Construct precon ms: " << ms_construct_precon_;
  return true;
}

void Graph::updatePublicPoseIDs() {
  // TODO(AT): Function is unused. Deprecate.
  local_shared_pose_ids_.clear();
  nbr_shared_pose_ids_.clear();
  for (const auto &m : shared_lcs_.GetRelativePosePoseMeasurements()) {
    if (m.r1 == id_) {
      CHECK(m.r2 != id_);
      local_shared_pose_ids_.emplace(m.r1, m.p1);
      nbr_shared_pose_ids_.emplace(m.r2, m.p2);
    } else {
      CHECK(m.r2 == id_);
      local_shared_pose_ids_.emplace(m.r2, m.p2);
      nbr_shared_pose_ids_.emplace(m.r1, m.p1);
    }
  }
}

void Graph::useInactiveNeighbors(bool use) {
  // TODO(AT): Function is unused. Deprecate.
  use_inactive_neighbors_ = use;
  clearDataMatrices();
}

} // namespace DCORA
