/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DCORA/PoseGraph.h>
#include <glog/logging.h>

namespace DCORA {

PoseGraph::PoseGraph(unsigned int id, unsigned int r, unsigned int d)
    : id_(id),
      r_(r),
      d_(d),
      n_(0),
      use_inactive_neighbors_(false),
      prior_kappa_(10000),
      prior_tau_(100) {
  CHECK(r >= d);
  empty();
}

PoseGraph::~PoseGraph() { empty(); }

void PoseGraph::empty() {
  // Reset this pose graph to be empty
  n_ = 0;
  edge_id_to_index_.clear();
  odometry_.clear();
  private_lcs_.clear();
  shared_lcs_.clear();
  local_shared_pose_ids_.clear();
  nbr_shared_pose_ids_.clear();
  nbr_robot_ids_.clear();
  neighbor_active_.clear();
  clearNeighborPoses();
  clearDataMatrices();
  clearPriors();
}

void PoseGraph::reset() {
  clearNeighborPoses();
  clearDataMatrices();
  clearPriors();
  for (const auto neighbor_id : nbr_robot_ids_) {
    neighbor_active_[neighbor_id] = true;
  }
}

void PoseGraph::clearNeighborPoses() {
  neighbor_poses_.clear();
  G_.reset(); // Clearing neighbor poses requires re-computing linear matrix
}

unsigned int PoseGraph::numMeasurements() const {
  return numOdometry() + numPrivateLoopClosures() + numSharedLoopClosures();
}

void PoseGraph::setMeasurements(
    const std::vector<RelativeSEMeasurement> &measurements) {
  // Reset this pose graph to be empty
  empty();
  for (const auto &m : measurements)
    addMeasurement(m);
}

void PoseGraph::addMeasurement(const RelativeSEMeasurement &m) {
  if (m.r1 != id_ && m.r2 != id_) {
    LOG(WARNING) << "Input contains irrelevant edges! \n" << m;
    return;
  }
  if (m.r1 == id_ && m.r2 == id_) {
    if (m.p1 + 1 == m.p2)
      addOdometry(m);
    else
      addPrivateLoopClosure(m);
  } else {
    addSharedLoopClosure(m);
  }
}

void PoseGraph::addOdometry(const RelativeSEMeasurement &factor) {
  // Check for duplicate inter-robot loop closure
  const PoseID src_id(factor.r1, factor.p1);
  const PoseID dst_id(factor.r2, factor.p2);
  if (hasMeasurement(src_id, dst_id))
    return;

  // Check that this is an odometry measurement
  CHECK(factor.r1 == id_);
  CHECK(factor.r2 == id_);
  CHECK(factor.p1 + 1 == factor.p2);
  CHECK(factor.R.rows() == d_ && factor.R.cols() == d_);
  CHECK(factor.t.rows() == d_ && factor.t.cols() == 1);
  n_ = std::max(n_, (unsigned int)factor.p2 + 1);
  odometry_.push_back(factor);
  const EdgeID edge_id(src_id, dst_id);
  edge_id_to_index_.emplace(edge_id, odometry_.size() - 1);
}

void PoseGraph::addPrivateLoopClosure(const RelativeSEMeasurement &factor) {
  // Check for duplicate inter-robot loop closure
  const PoseID src_id(factor.r1, factor.p1);
  const PoseID dst_id(factor.r2, factor.p2);
  if (hasMeasurement(src_id, dst_id))
    return;

  CHECK(factor.r1 == id_);
  CHECK(factor.r2 == id_);
  CHECK(factor.R.rows() == d_ && factor.R.cols() == d_);
  CHECK(factor.t.rows() == d_ && factor.t.cols() == 1);
  // update number of poses
  n_ = std::max(n_, (unsigned int)std::max(factor.p1 + 1, factor.p2 + 1));
  private_lcs_.push_back(factor);
  const EdgeID edge_id(src_id, dst_id);
  edge_id_to_index_.emplace(edge_id, private_lcs_.size() - 1);
}

void PoseGraph::addSharedLoopClosure(const RelativeSEMeasurement &factor) {
  // Check for duplicate inter-robot loop closure
  const PoseID src_id(factor.r1, factor.p1);
  const PoseID dst_id(factor.r2, factor.p2);
  if (hasMeasurement(src_id, dst_id))
    return;

  CHECK(factor.R.rows() == d_ && factor.R.cols() == d_);
  CHECK(factor.t.rows() == d_ && factor.t.cols() == 1);
  if (factor.r1 == id_) {
    CHECK(factor.r2 != id_);
    n_ = std::max(n_, (unsigned int)factor.p1 + 1);
    local_shared_pose_ids_.emplace(factor.r1, factor.p1);
    nbr_shared_pose_ids_.emplace(factor.r2, factor.p2);
    nbr_robot_ids_.insert(factor.r2);
    neighbor_active_[factor.r2] = true;
  } else {
    CHECK(factor.r2 == id_);
    n_ = std::max(n_, (unsigned int)factor.p2 + 1);
    local_shared_pose_ids_.emplace(factor.r2, factor.p2);
    nbr_shared_pose_ids_.emplace(factor.r1, factor.p1);
    nbr_robot_ids_.insert(factor.r1);
    neighbor_active_[factor.r1] = true;
  }

  shared_lcs_.push_back(factor);
  const EdgeID edge_id(src_id, dst_id);
  edge_id_to_index_.emplace(edge_id, shared_lcs_.size() - 1);
}

std::vector<RelativeSEMeasurement>
PoseGraph::sharedLoopClosuresWithRobot(unsigned int neighbor_id) const {
  std::vector<RelativeSEMeasurement> result;
  for (const auto &m : shared_lcs_) {
    if (m.r1 == neighbor_id || m.r2 == neighbor_id)
      result.emplace_back(m);
  }
  return result;
}

std::vector<RelativeSEMeasurement> PoseGraph::measurements() const {
  std::vector<RelativeSEMeasurement> measurements = odometry_;
  measurements.insert(measurements.end(), private_lcs_.begin(),
                      private_lcs_.end());
  measurements.insert(measurements.end(), shared_lcs_.begin(),
                      shared_lcs_.end());
  return measurements;
}

std::vector<RelativeSEMeasurement> PoseGraph::localMeasurements() const {
  std::vector<RelativeSEMeasurement> measurements = odometry_;
  measurements.insert(measurements.end(), private_lcs_.begin(),
                      private_lcs_.end());
  return measurements;
}

void PoseGraph::clearPriors() { priors_.clear(); }

void PoseGraph::setPrior(unsigned index, const LiftedPose &Xi) {
  CHECK_LT(index, n());
  CHECK_EQ(d(), Xi.d());
  CHECK_EQ(r(), Xi.r());
  priors_[index] = Xi;
}

void PoseGraph::setNeighborPoses(const PoseDict &pose_dict) {
  neighbor_poses_ = pose_dict;
  G_.reset(); // Setting neighbor poses requires re-computing linear matrix
}

bool PoseGraph::hasNeighbor(unsigned int robot_id) const {
  return nbr_robot_ids_.find(robot_id) != nbr_robot_ids_.end();
}

bool PoseGraph::isNeighborActive(unsigned int neighbor_id) const {
  if (!hasNeighbor(neighbor_id)) {
    return false;
  }
  return neighbor_active_.at(neighbor_id);
}

void PoseGraph::setNeighborActive(unsigned int neighbor_id, bool active) {
  if (!hasNeighbor(neighbor_id)) {
    return;
  }
  if (neighbor_active_.at(neighbor_id) != active) {
    clearDataMatrices();
  }
  neighbor_active_[neighbor_id] = active;
}

bool PoseGraph::requireNeighborPose(const PoseID &pose_id) const {
  return nbr_shared_pose_ids_.find(pose_id) != nbr_shared_pose_ids_.end();
}

bool PoseGraph::hasMeasurement(const PoseID &srcID, const PoseID &dstID) const {
  const EdgeID edge_id(srcID, dstID);
  return edge_id_to_index_.find(edge_id) != edge_id_to_index_.end();
}

RelativeSEMeasurement *PoseGraph::findMeasurement(const PoseID &srcID,
                                                  const PoseID &dstID) {
  RelativeSEMeasurement *edge = nullptr;
  if (hasMeasurement(srcID, dstID)) {
    const EdgeID edge_id(srcID, dstID);
    size_t index = edge_id_to_index_.at(edge_id);
    if (edge_id.isOdometry()) {
      edge = &odometry_[index];
    } else if (edge_id.isPrivateLoopClosure()) {
      edge = &private_lcs_[index];
    } else {
      edge = &shared_lcs_[index];
    }
  }
  if (edge) {
    // Sanity check
    CHECK_EQ(edge->r1, srcID.robot_id);
    CHECK_EQ(edge->p1, srcID.frame_id);
    CHECK_EQ(edge->r2, dstID.robot_id);
    CHECK_EQ(edge->p2, dstID.frame_id);
  }
  return edge;
}

std::vector<RelativeSEMeasurement *> PoseGraph::allLoopClosures() {
  std::vector<RelativeSEMeasurement *> output;
  for (auto &m : private_lcs_) {
    output.push_back(&m);
  }
  for (auto &m : shared_lcs_) {
    output.push_back(&m);
  }
  return output;
}

std::set<unsigned> PoseGraph::activeNeighborIDs() const {
  std::set<unsigned> output;
  for (unsigned neighbor_id : nbr_robot_ids_) {
    if (isNeighborActive(neighbor_id)) {
      output.emplace(neighbor_id);
    }
  }
  return output;
}

size_t PoseGraph::numActiveNeighbors() const {
  return activeNeighborIDs().size();
}

PoseSet PoseGraph::activeNeighborPublicPoseIDs() const {
  PoseSet output;
  for (const auto &pose_id : nbr_shared_pose_ids_) {
    if (isNeighborActive(pose_id.robot_id)) {
      output.emplace(pose_id);
    }
  }
  return output;
}

std::vector<RelativeSEMeasurement *> PoseGraph::activeLoopClosures() {
  std::vector<RelativeSEMeasurement *> output;
  for (auto &m : private_lcs_) {
    output.push_back(&m);
  }
  for (auto &m : shared_lcs_) {
    if (m.r1 == id_ && isNeighborActive(m.r2)) {
      output.push_back(&m);
    } else if (m.r2 == id_ && isNeighborActive(m.r1)) {
      output.push_back(&m);
    }
  }
  return output;
}

std::vector<RelativeSEMeasurement *> PoseGraph::inactiveLoopClosures() {
  std::vector<RelativeSEMeasurement *> output;
  for (auto &m : shared_lcs_) {
    if (m.r1 == id_ && !isNeighborActive(m.r2)) {
      output.push_back(&m);
    } else if (m.r2 == id_ && !isNeighborActive(m.r1)) {
      output.push_back(&m);
    }
  }
  return output;
}

PoseGraph::Statistics PoseGraph::statistics() const {
  // Currently, this function is only meaningful for GNC_TLS
  double totalCount = 0;
  double acceptCount = 0;
  double rejectCount = 0;
  // TODO(adthoms): specify tolerance for rejected and accepted loop closures
  for (const auto &m : private_lcs_) {
    // if (m.fixedWeight) continue;
    if (m.weight == 1) {
      acceptCount += 1;
    } else if (m.weight == 0) {
      rejectCount += 1;
    }
    totalCount += 1;
  }
  for (const auto &m : shared_lcs_) {
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

  PoseGraph::Statistics statistics;
  statistics.total_loop_closures = totalCount;
  statistics.accept_loop_closures = acceptCount;
  statistics.reject_loop_closures = rejectCount;
  statistics.undecided_loop_closures = totalCount - acceptCount - rejectCount;

  return statistics;
}

const SparseMatrix &PoseGraph::quadraticMatrix() {
  if (!Q_.has_value())
    constructQ();
  CHECK(Q_.has_value());
  return Q_.value();
}

void PoseGraph::clearQuadraticMatrix() {
  Q_.reset();
  precon_.reset(); // Also clear the preconditioner since it depends on Q
}

const Matrix &PoseGraph::linearMatrix() {
  if (!G_.has_value())
    constructG();
  CHECK(G_.has_value());
  return G_.value();
}

void PoseGraph::clearLinearMatrix() { G_.reset(); }

bool PoseGraph::constructDataMatrices() {
  if (!Q_.has_value() && !constructQ())
    return false;
  if (!G_.has_value() && !constructG())
    return false;
  return true;
}

void PoseGraph::clearDataMatrices() {
  clearQuadraticMatrix();
  clearLinearMatrix();
}

bool PoseGraph::constructQ() {
  timer_.tic();
  std::vector<RelativeSEMeasurement> privateMeasurements = odometry_;
  privateMeasurements.insert(privateMeasurements.end(), private_lcs_.begin(),
                             private_lcs_.end());

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
  for (const auto &m : shared_lcs_) {
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
  for (const auto &it : priors_) {
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

bool PoseGraph::constructG() {
  timer_.tic();
  unsigned d = d_;
  Matrix G(r_, (d_ + 1) * n_);
  G.setZero();
  Matrix T = Matrix::Zero(d + 1, d + 1);
  Matrix Omega = Matrix::Zero(d + 1, d + 1);
  // Go through shared measurements
  for (const auto &m : shared_lcs_) {
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
  for (const auto &it : priors_) {
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

bool PoseGraph::hasPreconditioner() {
  if (!precon_.has_value())
    constructPreconditioner();
  return precon_.has_value();
}
/**
 * @brief Get preconditioner
 * @return
 */
const CholmodSolverPtr &PoseGraph::preconditioner() {
  if (!precon_.has_value())
    constructPreconditioner();
  CHECK(precon_.has_value());
  return precon_.value();
}

bool PoseGraph::constructPreconditioner() {
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

void PoseGraph::updatePublicPoseIDs() {
  local_shared_pose_ids_.clear();
  nbr_shared_pose_ids_.clear();

  for (const auto &m : shared_lcs_) {
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

void PoseGraph::useInactiveNeighbors(bool use) {
  use_inactive_neighbors_ = use;
  clearDataMatrices();
}

} // namespace DCORA
