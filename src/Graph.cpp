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
  loc_shared_pose_ids_.clear();
  loc_shared_landmark_ids_.clear();
  nbr_shared_pose_ids_.clear();
  nbr_shared_landmark_ids_.clear();
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
  neighbor_landmarks_.clear();
  G_.reset(); // Clearing neighbor poses requires re-computing linear matrix
}

void Graph::updateNumStates(const StateID &stateID) {
  CHECK_EQ(stateID.robot_id, id_);
  // Update num poses
  if (stateID.isPose())
    n_ = std::max(n_, static_cast<unsigned int>(stateID.frame_id + 1));
  // Update num landmarks
  if (stateID.isPoint())
    b_ = std::max(b_, static_cast<unsigned int>(stateID.frame_id + 1));
}

void Graph::updateNumRanges(const RelativeMeasurement &measurement) {
  if (measurement.measurementType != MeasurementType::Range)
    return;

  if (measurement.r1 == id_) {
    // range measurement's unit sphere variable belongs to this agent
    const RangeMeasurement &range_measurement =
        dynamic_cast<const RangeMeasurement &>(measurement);
    l_ = std::max(l_, static_cast<unsigned int>(range_measurement.l + 1));
  }
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
    std::visit([this](auto &&m) { addMeasurement(m); }, m);
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
  const StateID &src_id = factor.getSrcID();
  const StateID &dst_id = factor.getDstID();
  const MeasurementType &meas_type = factor.measurementType;
  if (hasMeasurement(src_id, dst_id, meas_type))
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
  const EdgeID edge_id(src_id, dst_id, meas_type);
  edge_id_to_index_.emplace(edge_id, odometry_.size() - 1);
}

void Graph::addPrivateLoopClosure(const RelativeMeasurement &factor) {
  // Check for duplicate private loop closure
  const StateID &src_id = factor.getSrcID();
  const StateID &dst_id = factor.getDstID();
  const MeasurementType &meas_type = factor.measurementType;
  if (hasMeasurement(src_id, dst_id, meas_type))
    return;

  // Check that this is a valid measurement
  factor.checkDim(d_);

  // Check that this is a private loop closure
  CHECK(factor.r1 == id_);
  CHECK(factor.r2 == id_);

  // Update number of poses and landmarks
  updateNumStates(src_id);
  updateNumStates(dst_id);
  // Update number of unit sphere variables
  updateNumRanges(factor);

  // Add relative measurement factor to private loop closures
  private_lcs_.push_back(factor);

  // Update edges
  const EdgeID edge_id(src_id, dst_id, meas_type);
  edge_id_to_index_.emplace(edge_id, private_lcs_.vec.size() - 1);
}

void Graph::addSharedLoopClosure(const RelativeMeasurement &factor) {
  // Check for duplicate shared loop closure
  const StateID &src_id = factor.getSrcID();
  const StateID &dst_id = factor.getDstID();
  const MeasurementType &meas_type = factor.measurementType;
  if (hasMeasurement(src_id, dst_id, meas_type))
    return;

  // Check that this is a valid measurement
  factor.checkDim(d_);

  // Update number of unit sphere variables
  updateNumRanges(factor);

  // Update local and neighbor shared state IDs. Set active neighbor.
  if (factor.r1 == id_) {
    CHECK(factor.r2 != id_);

    // Update number of poses and landmarks
    updateNumStates(src_id);

    // Add local shared state to graph
    executeStateDependantFunctionals(
        [&, this]() { loc_shared_pose_ids_.emplace(factor.r1, factor.p1); },
        [&, this]() { loc_shared_landmark_ids_.emplace(factor.r1, factor.p1); },
        factor.stateType1);

    // Add neighbor shared state to graph
    executeStateDependantFunctionals(
        [&, this]() { nbr_shared_pose_ids_.emplace(factor.r2, factor.p2); },
        [&, this]() { nbr_shared_landmark_ids_.emplace(factor.r2, factor.p2); },
        factor.stateType2);

    // Update neighbor robot IDs
    nbr_robot_ids_.insert(factor.r2);

    // Set active neighbor
    neighbor_active_[factor.r2] = true;
  } else {
    CHECK(factor.r2 == id_);

    // Update number of poses and landmarks
    updateNumStates(dst_id);

    // Add local shared state to graph
    executeStateDependantFunctionals(
        [&, this]() { loc_shared_pose_ids_.emplace(factor.r2, factor.p2); },
        [&, this]() { loc_shared_landmark_ids_.emplace(factor.r2, factor.p2); },
        factor.stateType2);

    // Add neighbor shared state to graph
    executeStateDependantFunctionals(
        [&, this]() { nbr_shared_pose_ids_.emplace(factor.r1, factor.p1); },
        [&, this]() { nbr_shared_landmark_ids_.emplace(factor.r1, factor.p1); },
        factor.stateType1);

    // Update neighbor robot IDs
    nbr_robot_ids_.insert(factor.r1);

    // Set active neighbor
    neighbor_active_[factor.r1] = true;
  }

  // Add relative measurement factor to shared loop closures
  shared_lcs_.push_back(factor);

  // Update edges
  const EdgeID edge_id(src_id, dst_id, meas_type);
  edge_id_to_index_.emplace(edge_id, shared_lcs_.vec.size() - 1);
}

RelativeMeasurements
Graph::sharedLoopClosuresWithRobot(unsigned int neighbor_id) const {
  RelativeMeasurements result;
  for (const auto &m : shared_lcs_.vec) {
    std::visit(
        [&result, neighbor_id](auto &&m) {
          if (m.r1 == neighbor_id || m.r2 == neighbor_id)
            result.vec.emplace_back(m);
        },
        m);
  }
  return result;
}

RelativeMeasurements Graph::allMeasurements() const {
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
  landmark_priors_.clear();
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
  landmark_priors_[index] = ti;
}

void Graph::setNeighborStates(const PoseDict &pose_dict,
                              const PointDict &landmark_dict) {
  neighbor_poses_ = pose_dict;
  neighbor_landmarks_ = landmark_dict;
  G_.reset(); // Setting neighbor states requires re-computing linear matrix
}

void Graph::setNeighborPoses(const PoseDict &pose_dict) {
  neighbor_poses_ = pose_dict;
  G_.reset(); // Setting neighbor poses requires re-computing linear matrix
}

void Graph::setNeighborLandmarks(const PointDict &landmark_dict) {
  neighbor_landmarks_ = landmark_dict;
  G_.reset(); // Setting neighbor landmarks requires re-computing linear matrix
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

bool Graph::requireNeighborLandmark(const PointID &landmark_id) const {
  return nbr_shared_landmark_ids_.find(landmark_id) !=
         nbr_shared_landmark_ids_.end();
}

bool Graph::hasMeasurement(const StateID &srcID, const StateID &dstID,
                           const MeasurementType &measType) const {
  const EdgeID edge_id(srcID, dstID, measType);
  return edge_id_to_index_.find(edge_id) != edge_id_to_index_.end();
}

RelativeMeasurement *Graph::findMeasurement(const StateID &srcID,
                                            const StateID &dstID,
                                            const MeasurementType &measType) {
  RelativeMeasurement *edge = nullptr;
  auto getEdgePointerFromRelativeMeasurementVariant = [](auto &&arg) {
    using T = std::decay_t<decltype(arg)>;
    if constexpr (std::is_base_of_v<RelativeMeasurement, T>)
      return dynamic_cast<RelativeMeasurement *>(&arg);
    else
      LOG(FATAL) << "Error: cannot dynamically cast RelativeMeasurement!";
    return static_cast<RelativeMeasurement *>(nullptr);
  };
  if (hasMeasurement(srcID, dstID, measType)) {
    const EdgeID edge_id(srcID, dstID, measType);
    size_t index = edge_id_to_index_.at(edge_id);
    if (edge_id.isOdometry()) {
      edge = &odometry_[index];
    } else if (edge_id.isPrivateLoopClosure()) {
      edge = std::visit(getEdgePointerFromRelativeMeasurementVariant,
                        private_lcs_.vec[index]);
    } else {
      edge = std::visit(getEdgePointerFromRelativeMeasurementVariant,
                        shared_lcs_.vec[index]);
    }
  }
  if (edge) {
    // Sanity check
    CHECK(edge->measurementType == measType);
    CHECK(edge->stateType1 == srcID.state_type);
    CHECK(edge->stateType2 == dstID.state_type);
    CHECK_EQ(edge->r1, srcID.robot_id);
    CHECK_EQ(edge->p1, srcID.frame_id);
    CHECK_EQ(edge->r2, dstID.robot_id);
    CHECK_EQ(edge->p2, dstID.frame_id);
  }
  return edge;
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

PointSet Graph::activeNeighborPublicLandmarkIDs() const {
  PointSet output;
  for (const auto &point_id : nbr_shared_landmark_ids_) {
    if (isNeighborActive(point_id.robot_id)) {
      output.emplace(point_id);
    }
  }
  return output;
}

std::vector<RelativeMeasurementPointerVariant> Graph::activeLoopClosures() {
  std::vector<RelativeMeasurementPointerVariant> output;
  for (auto &m : private_lcs_.vec) {
    std::visit([&output](auto &&m) { output.push_back(&m); }, m);
  }
  for (auto &m : shared_lcs_.vec) {
    std::visit(
        [&output, this](auto &&m) {
          if (m.r1 == id_ && isNeighborActive(m.r2)) {
            output.push_back(&m);
          } else if (m.r2 == id_ && isNeighborActive(m.r1)) {
            output.push_back(&m);
          }
        },
        m);
  }
  return output;
}

Graph::Statistics Graph::statistics() const {
  // Currently, this function is only meaningful for GNC_TLS
  double totalCount = 0;
  double acceptCount = 0;
  double rejectCount = 0;
  // TODO(YT): specify tolerance for rejected and accepted loop closures
  for (const auto &m : private_lcs_.vec) {
    std::visit(
        [&](auto &&m) {
          if (m.weight == 1) {
            acceptCount += 1;
          } else if (m.weight == 0) {
            rejectCount += 1;
          }
          totalCount += 1;
        },
        m);
  }
  for (const auto &m : shared_lcs_.vec) {
    std::visit(
        [&](auto &&m) {
          // Skip loop closures with inactive neighbors
          bool skip = false;
          if (m.r1 == id_ && !isNeighborActive(m.r2))
            skip = true;
          if (m.r2 == id_ && !isNeighborActive(m.r1))
            skip = true;
          if (!skip) {
            if (m.weight == 1) {
              acceptCount += 1;
            } else if (m.weight == 0) {
              rejectCount += 1;
            }
            totalCount += 1;
          }
        },
        m);
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
  precon_.reset(); // Clear preconditioner since it depends on Q
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
  bool Q_constructed = false;
  if (isPGOCompatible())
    Q_constructed = constructQuadraticCostTermPGO();
  else
    Q_constructed = constructQuadraticCostTermRASLAM();
  ms_construct_Q_ = timer_.toc();
  return Q_constructed;
}

bool Graph::constructG() {
  timer_.tic();
  bool G_constructed = false;
  if (isPGOCompatible())
    G_constructed = constructLinearCostTermPGO();
  else
    G_constructed = constructLinearCostTermRASLAM();
  ms_construct_G_ = timer_.toc();
  return G_constructed;
}

bool Graph::constructQuadraticCostTermPGO() {
  // Set measurements
  const RelativeMeasurements &measurements = allMeasurements();

  // Set dimensions
  const size_t m = measurements.vec.size();
  const unsigned int dh = d_ + 1;

  /**
   * @brief Constructing the quadratic cost term for PGO
   *
   * The quadratic cost term Q is a [(d + 1) × (d + 1)](n_b × n_b) matrix of the
   * form:
   *
   *   Q = Ab^T × Omega × Ab
   *     = AbT × Omega × AbT^T
   *
   * where:
   *   n_b is the number of poses owned by this agent (i.e. agent b)
   *   AbT is the incidence matrix of this agent
   *   Omega is a block diagonal matrix of measurement weights
   *
   * Dimensions: [rows × cols]
   *   Incidence Matrix:
   *     AbT: [n_b × m][(d + 1) × (d + 1)] - block matrix
   *   Weight Matrix:
   *     Omega: [m × m][(d + 1) × (d + 1)] - block diagonal matrix
   *
   * Indexing: The following table illustrates edge direction e=(i,j) based on
   * local and shared measurements:
   *
   *     |  local   | shared (r1=id) | shared (r2=id) |
   *   ------------------------------------------------
   *   i | leaving  |     leaving    |       ***      |
   *   j | entering |       ***      |     leaving    |
   *
   * For book keeping, we we use a matrix-centric approach and update Q for all
   * measurements, indexing the contribution of each measurement using agent b's
   * posed IDs.
   */
  const size_t rowsAbT = dh * n_;
  const size_t colsAbT = dh * m;

  // Initialize incidence matrix
  SparseMatrix AbT(rowsAbT, colsAbT);
  AbT.reserve(Eigen::VectorXi::Constant(colsAbT, SPARSE_ENTRIES));

  // Initialize weight matrix
  DiagonalMatrix Omega(colsAbT); // One block per measurement
  DiagonalMatrix::DiagonalVectorType &diagonalOmega = Omega.diagonal();

  // Populate AbT and Omega
  for (size_t k = 0; k < m; k++) {
    const RelativeMeasurementVariant &measVariant = measurements.vec.at(k);
    CHECK(!std::holds_alternative<RelativePosePointMeasurement>(measVariant));
    CHECK(!std::holds_alternative<RangeMeasurement>(measVariant));
    const RelativePosePoseMeasurement &meas =
        std::get<RelativePosePoseMeasurement>(measVariant);
    size_t i = IDX_NOT_SET;
    size_t j = IDX_NOT_SET;

    // Assign isotropic weights in diagonal matrix
    for (size_t r = 0; r < d_; r++)
      diagonalOmega[k * dh + r] = meas.weight * meas.kappa;

    diagonalOmega[k * dh + d_] = meas.weight * meas.tau;

    // Set indices according to pose ownership
    std::optional<bool> are_indices_set =
        setIndicesFromStateOwnership(meas, &i, &j);
    if (are_indices_set == false)
      return false;
    else if (are_indices_set == std::nullopt)
      continue;

    // Populate incidence matrix
    if (i != IDX_NOT_SET) {
      // AT(i,k) = -Tij (NOTE: NEGATIVE)
      for (size_t c = 0; c < d_; c++)
        for (size_t r = 0; r < d_; r++)
          AbT.insert(i * dh + r, k * dh + c) = -meas.R(r, c);

      for (size_t r = 0; r < d_; r++)
        AbT.insert(i * dh + r, k * dh + d_) = -meas.t(r);

      AbT.insert(i * dh + d_, k * dh + d_) = -1;
    }
    if (j != IDX_NOT_SET) {
      // AT(j,k) = +I (NOTE: POSITIVE)
      for (size_t r = 0; r < dh; r++)
        AbT.insert(j * dh + r, k * dh + r) = +1;
    }
  }

  // Compress sparse matrix
  AbT.makeCompressed();

  // Set quadratic cost matrix
  const SparseMatrix Q = AbT * Omega * AbT.transpose();
  Q_.emplace(Q);

  return true;
}

bool Graph::constructLinearCostTermPGO() {
  // Set measurements
  const RelativeMeasurements &measurements = sharedLoopClosures();

  // Set dimensions
  const size_t m = measurements.vec.size();
  const unsigned int dh = d_ + 1;

  /**
   * @brief Constructing the linear cost term for PGO
   *
   * The linear cost term G is a [r × ((d+1) × n_b)] matrix of the form:
   *
   *   G = Xc^T × Ac^T × Omega × Ab
   *     = XcT × AcT × Omega × AbT^T
   *
   * where:
   *   n_b is the number of poses owned by this agent (i.e. agent b)
   *   XcT is a matrix of fixed public poses of the neighbor agent (i.e agent c)
   *   AcT is the incidence matrix of the neighbor agent
   *   AbT is the incidence matrix of this agent
   *   Omega is a block diagonal matrix of measurement weights
   *
   * Note: neighbor agent c is viewed as a meta agent including all agents that
   * are not agent b such that:
   *
   *   c:= [N]/{b}; where N is the total number of agents
   *
   * Dimensions: [rows × cols]
   *   XcT: [r × ((d + 1) × n_c)]
   *   AcT: [n_c × m][(d + 1) × (d + 1)] - block matrix
   *   AbT: [n_b × m][(d + 1) × (d + 1)] - block matrix
   *   Omega: [m × m][(d + 1) × (d + 1)] - block diagonal matrix
   *
   * For book keeping, we look at the contribution of each measurement and
   * update G via the addition of this contribution:
   *
   *   G(1:r, idx:idx+d) += L_i; for all m_i in the set of shared loop closures
   *                     += Xc_i^T × Ac_i^T × Omega_i × Ab_i
   *                     += XcT_i × AcT_i × Omega_i × AbT_i^T
   *
   * where:
   *   L_i is the linear cost associated with measurement m_i
   *   idx is the index of the pose associated with agent b in measurement m_i
   *
   * Dimensions: [rows × cols]
   *   L_i : [r × (d + 1)]
   *   XcT_i: [r × (d + 1)]
   *   AcT_i: [(d + 1) × (d + 1)]
   *   AbT_i: [(d + 1) × (d + 1)]
   *   Omega_i: [(d + 1) × (d + 1)]
   *
   * For brevity, we drop subscript i when constructing L_i and its submatrices
   */
  Matrix XcT = Matrix::Zero(r_, dh);
  Matrix AcT = Matrix::Zero(dh, dh);
  Matrix AbT = Matrix::Zero(dh, dh);
  Matrix Omega = Matrix::Zero(dh, dh);

  // Initialize entries of incidence matrices
  Matrix T = Matrix::Identity(dh, dh);
  Matrix I = Matrix::Identity(dh, dh);

  // Initialize linear cost
  LiftedPoseArray G(r_, d_, n_);
  G.setDataToZero();

  // Iterate over all shared pose-pose loop closures
  for (size_t k = 0; k < m; k++) {
    const RelativeMeasurementVariant &measVariant = measurements.vec.at(k);
    CHECK(!std::holds_alternative<RelativePosePointMeasurement>(measVariant));
    CHECK(!std::holds_alternative<RangeMeasurement>(measVariant));
    const RelativePosePoseMeasurement &meas =
        std::get<RelativePosePoseMeasurement>(measVariant);
    size_t i = IDX_NOT_SET;
    size_t j = IDX_NOT_SET;

    // Update measurement transformation matrix
    T.block(0, 0, d_, d_) = meas.R;
    T.block(0, d_, d_, 1) = meas.t;

    // Update measurement weight matrix
    for (unsigned i = 0; i < d_; ++i)
      Omega(i, i) = meas.weight * meas.kappa;

    Omega(d_, d_) = meas.weight * meas.tau;

    // Set indices according to pose ownership
    std::optional<bool> are_indices_set =
        setIndicesFromStateOwnership(meas, &i, &j);
    if (are_indices_set == false)
      return false;
    else if (are_indices_set == std::nullopt)
      continue;

    // Update linear cost
    if (i != IDX_NOT_SET) {
      AbT = -T; // Leaving node i of agent b
      AcT = I;  // Entering node j of agent c
      const StateID &neighborDstStateID = meas.getDstID();
      XcT = getNeighborFixedVariableLiftedData(neighborDstStateID);

      // Add measurement contribution to linear cost
      G.pose(i) += XcT * AcT * Omega * AbT.transpose();
    }
    if (j != IDX_NOT_SET) {
      AbT = I;  // Entering node j of agent b
      AcT = -T; // Leaving node i of agent c
      const StateID &neighborSrcStateID = meas.getSrcID();
      XcT = getNeighborFixedVariableLiftedData(neighborSrcStateID);

      // Add measurement contribution to linear cost
      G.pose(j) += XcT * AcT * Omega * AbT.transpose();
    }
  }

  // Maintain legacy support for pose priors
  // TODO(AT): Treat priors as relative measurements
  for (const auto &it : pose_priors_) {
    unsigned idx = it.first;
    const Matrix &P = it.second.getData();
    for (unsigned row = 0; row < d_; ++row) {
      Omega(row, row) = prior_kappa_;
    }
    Omega(d_, d_) = prior_tau_;
    Matrix L = -P * Omega;
    G.pose(idx) += L;
  }

  // Set linear cost matrix
  G_.emplace(G.getData());

  return true;
}

bool Graph::constructQuadraticCostTermRASLAM() {
  // Set measurements
  const RelativeMeasurements &measurements = allMeasurements();
  const std::vector<RelativePosePoseMeasurement> &pose_pose_measurements =
      measurements.GetRelativePosePoseMeasurements();
  const std::vector<RelativePosePointMeasurement> &pose_point_measurements =
      measurements.GetRelativePosePointMeasurements();
  const std::vector<RangeMeasurement> &range_measurements =
      measurements.GetRangeMeasurements();

  // Set dimensions
  const size_t mPosePose = pose_pose_measurements.size();
  const size_t mPosePoint = pose_point_measurements.size();
  const size_t mRange = range_measurements.size();
  const size_t mPose = mPosePose + mPosePoint;
  CHECK_LE(l_, mRange);

  /**
   * @brief Constructing the quadratic cost term for RA-SLAM
   *
   * The quadratic cost term Q is a [k × k] block symmetric matrix of the form:
   *
   *   Q = Q_p + Q_r; k = (d + 1) × n_b + l_b + b_b
   *
   *   <-----------col------------>
   *
   *      dn_b     l_b    n_b + b_b
   *   ----------------------------             ^
   *   |  Q_11  |   0    |  Q_13  |  dn_b       |
   *   |  ****  |  Q_22  |  Q_23  |  l_b       row
   *   |  ****  |  ****  |  Q_33  |  n_b + b_b  |
   *   ----------------------------             v
   *
   * where:
   *   n_b is the number of poses owned by this agent (i.e. agent b)
   *   l_b is the number of unit sphere variables owned by this agent
   *   b_b is the number of landmarks owned by this agent
   *
   * Indexing: The following table illustrates edge direction e=(i,j) based on
   * local and shared measurements:
   *
   *     |  local   | shared (r1=id) | shared (r2=id) |
   *   ------------------------------------------------
   *   i | leaving  |     leaving    |       ***      |
   *   j | entering |       ***      |     leaving    |
   *
   * In our implementation, we calculate the submatrices of Q_p and Q_r
   * separately and then combine them to form Q. For book keeping, we we use a
   * matrix-centric approach and update Q_p and Q_r for all measurements,
   * indexing the contribution of each measurement using agent b's state and
   * unit sphere IDs. For convenience, we drop subscript b for remaining
   * formalisms.
   */

  /**
   * @brief Constructing Q_p submatrices
   *
   * Data matrix Q_p is a block symmetric matrix of the form:
   *
   *   <-----------col------------>
   *
   *       dn       l      n + b
   *   ----------------------------           ^
   *   | Q_p_11 |   0    | Q_p_13 |  dn       |
   *   |  ****  |   0    |   0    |  l       row
   *   |  ****  |  ****  | Q_p_33 |  n + b    |
   *   ----------------------------           v
   *
   *   Q_p_11 = L(G^rho) + Sigma
   *          = ARho^T × OmegaRho × ARho + T^T × OmegaTau × T
   *          = ARhoT × OmegaRho × ARhoT^T + TT × OmegaTau × TT^T
   *
   *   Q_p_13 = V
   *          = T^T × OmegaTau × ATau
   *          = TT × OmegaTau × ATauT^T
   *
   *   Q_p_33 = L(G^tau)
   *          = ATau^T × OmegaTau × ATau
   *          = ATauT × OmegaTau × ATauT^T
   *
   *  Dimensions: [rows × cols]
   *    Incidence Matrices:
   *      ARhoT: [n × mPosePose](d × d) - block matrix
   *      ATauT: [(n + b) × mPose] matrix
   *    Weight Matrices:
   *      OmegaRhoT: [mPosePose × mPosePose] (d × d) - block diagonal matrix
   *      OmegaTauT: [mPose × mPose] diagonal matrix
   *    Data Matrix:
   *      TT: [dn × mPose] matrix
   */
  const size_t rowsARhoT = d_ * n_;
  const size_t colsARhoT = d_ * mPosePose;
  const size_t rowsATauT = n_ + b_;
  const size_t colsATauT = mPose;
  const size_t rowsTT = d_ * n_;
  const size_t colsTT = mPose;

  // Initialize incidence matrices
  SparseMatrix ARhoT(rowsARhoT, colsARhoT);
  ARhoT.reserve(Eigen::VectorXi::Constant(colsARhoT, SPARSE_ENTRIES));
  SparseMatrix ATauT(rowsATauT, colsATauT);
  ATauT.reserve(Eigen::VectorXi::Constant(colsATauT, SPARSE_ENTRIES));

  // Initialize weight matrices
  DiagonalMatrix OmegaRho(colsARhoT); // One block per measurement
  DiagonalMatrix::DiagonalVectorType &diagonalOmegaRho = OmegaRho.diagonal();
  DiagonalMatrix OmegaTau(colsATauT); // One entry per measurement
  DiagonalMatrix::DiagonalVectorType &diagonalOmegaTau = OmegaTau.diagonal();

  // Initialize data matrix
  SparseMatrix TT(rowsTT, colsTT);
  TT.reserve(Eigen::VectorXi::Constant(colsTT, SPARSE_ENTRIES));

  // Populate ARhoT, OmegaRho, ATauT, OmegaTau, and TT
  for (size_t k = 0; k < mPosePose; k++) {
    const RelativePosePoseMeasurement &meas = pose_pose_measurements.at(k);
    size_t i = IDX_NOT_SET;
    size_t j = IDX_NOT_SET;

    // Assign isotropic weights in diagonal matrices
    for (size_t r = 0; r < d_; r++)
      diagonalOmegaRho[k * d_ + r] = meas.weight * meas.kappa;

    diagonalOmegaTau[k] = meas.weight * meas.tau;

    // Set indices according to pose ownership
    std::optional<bool> are_indices_set =
        setIndicesFromStateOwnership(meas, &i, &j);
    if (are_indices_set == false)
      return false;
    else if (are_indices_set == std::nullopt)
      continue;

    // Populate incidence and data matrices
    if (i != IDX_NOT_SET) {
      // AT(i,k) = -Rij (NOTE: NEGATIVE)
      for (size_t c = 0; c < d_; c++)
        for (size_t r = 0; r < d_; r++)
          ARhoT.insert(i * d_ + r, k * d_ + c) = -meas.R(r, c);

      // Populate with pose translation data
      for (size_t r = 0; r < d_; r++)
        TT.insert(i * d_ + r, k) = -meas.t(r);

      // Populate with pose translation incidences
      ATauT.insert(i, k) = -1;
    }
    if (j != IDX_NOT_SET) {
      // AT(j,k) = +I (NOTE: POSITIVE)
      for (size_t r = 0; r < d_; r++)
        ARhoT.insert(j * d_ + r, k * d_ + r) = +1;

      // Populate with pose translation incidences
      ATauT.insert(j, k) = +1;
    }
  }
  for (size_t k = mPosePose; k < mPose; k++) {
    const RelativePosePointMeasurement &meas =
        pose_point_measurements.at(k - mPosePose);
    size_t i = IDX_NOT_SET;
    size_t j = IDX_NOT_SET;

    // Assign isotropic weights in diagonal matrix
    diagonalOmegaTau[k] = meas.weight * meas.tau;

    // Set indices according to pose/landmark ownership
    std::optional<bool> are_indices_set =
        setIndicesFromStateOwnership(meas, &i, &j);
    if (are_indices_set == false)
      return false;
    else if (are_indices_set == std::nullopt)
      continue;

    // Populate incidence and data matrices
    if (i != IDX_NOT_SET) {
      // Populate with landmark translation data
      for (size_t r = 0; r < d_; r++)
        TT.insert(i * d_ + r, k) = -meas.t(r);

      // Populate with landmark translation incidences
      ATauT.insert(i, k) = -1;
    }
    if (j != IDX_NOT_SET) {
      // Offset landmark indices by the number of poses
      j += n_;

      // Populate with landmark translation incidences
      ATauT.insert(j, k) = +1;
    }
  }

  /**
   * @brief Constructing Q_r submatrices
   *
   * Data matrix Q_r is a block symmetric matrix of the form:
   *
   *   <-----------col------------>
   *
   *       dn       l      n + b
   *   ----------------------------           ^
   *   |   0    |   0    |   0    |  dn       |
   *   |  ****  | Q_r_22 | Q_r_23 |  l       row
   *   |  ****  |  ****  | Q_r_33 |  n + b    |
   *   ----------------------------           v
   *
   *   where:
   *     Q_r_22 = P^T × OmegaRange × D^2 × P
   *            = PT × OmegaRange × DT^T × DT^T × PT^T
   *
   *     Q_r_23 = P^T × D × OmegaRange × C
   *            = PT × DT^T × OmegaRange × CT^T
   *
   *     Q_r_33 = C^T × OmegaRange × C
   *            = CT × OmegaRange × CT^T
   *
   *  Dimensions: [rows × cols]
   *    Incidence Matrix:
   *      CT: [(n + b) × mRange] matrix
   *    Weight Matrix:
   *      OmegaRange: [mRange × mRange] diagonal matrix
   *    Data Matrix:
   *      DT: [mRange × mRange] matrix
   *    Selection Matrix:
   *      PT: [l × mRange]
   */
  const size_t rowsCT = n_ + b_;
  const size_t colsCT = mRange;
  const size_t rowsDT = mRange;
  const size_t colsDT = mRange;
  const size_t rowsPT = l_;
  const size_t colsPT = mRange;

  // Initialize incidence matrix
  SparseMatrix CT(rowsCT, colsCT);
  CT.reserve(Eigen::VectorXi::Constant(colsCT, SPARSE_ENTRIES));

  // Initialize weight matrix
  DiagonalMatrix OmegaRange(colsCT); // One entry per measurement
  DiagonalMatrix::DiagonalVectorType &diagonalOmegaRange =
      OmegaRange.diagonal();

  // Initialize data matrix
  SparseMatrix DT(rowsDT, colsDT);
  DT.reserve(Eigen::VectorXi::Constant(colsDT, SPARSE_ENTRIES));

  // Initialize selection matrix
  SparseMatrix PT(rowsPT, colsPT);
  PT.setZero();

  // Populate CT, OmegaRange, DT, and PT
  for (size_t k = 0; k < mRange; k++) {
    const RangeMeasurement &meas = range_measurements.at(k);
    size_t i = IDX_NOT_SET;
    size_t j = IDX_NOT_SET;

    // Assign isotropic weights in diagonal matrix
    diagonalOmegaRange[k] = meas.weight * meas.precision;

    // Populate data matrix with range data
    DT.insert(k, k) = meas.range;

    // Populate selection matrix based on unit sphere variable ownership
    if (meas.r1 == id_)
      PT.insert(meas.l, k) = 1;

    // Set indices according to pose/landmark ownership
    std::optional<bool> are_indices_set =
        setIndicesFromStateOwnership(meas, &i, &j);
    if (are_indices_set == false)
      return false;
    else if (are_indices_set == std::nullopt)
      continue;

    // Populate incidence matrix with range incidences that connect to pose
    // and/or landmark translations
    if (i != IDX_NOT_SET) {
      // Offset landmark indices by the number of poses
      executeStateDependantFunctionals([&]() { /*No offset for pose indices*/ },
                                       [&]() { i += n_; }, meas.stateType1);
      CT.insert(i, k) = -1;
    }
    if (j != IDX_NOT_SET) {
      // Offset landmark indices by the number of poses
      executeStateDependantFunctionals([&]() { /*No offset for pose indices*/ },
                                       [&]() { j += n_; }, meas.stateType2);
      CT.insert(j, k) = +1;
    }
  }

  // Compress sparse matrices
  ARhoT.makeCompressed();
  ATauT.makeCompressed();
  TT.makeCompressed();
  CT.makeCompressed();
  DT.makeCompressed();
  PT.makeCompressed();

  // Set Q_p and Q_r submatrices
  const SparseMatrix &ARho = ARhoT.transpose();
  const SparseMatrix &ATau = ATauT.transpose();
  const SparseMatrix &T = TT.transpose();
  const SparseMatrix &C = CT.transpose();
  const SparseMatrix &D = DT.transpose();
  const SparseMatrix &P = PT.transpose();
  SparseMatrix Q11 = ARhoT * OmegaRho * ARho + TT * OmegaTau * T;
  SparseMatrix Q13 = TT * OmegaTau * ATau;
  SparseMatrix Q22 = PT * OmegaRange * D * D * P;
  SparseMatrix Q23 = PT * D * OmegaRange * C;
  SparseMatrix Q33 = ATauT * OmegaTau * ATau + CT * OmegaRange * C;

  /**
   * @brief Constructing Q from Q_p and Q_r
   *
   * The following implementation is adapted from:
   * CORA: https://github.com/MarineRoboticsGroup/cora
   */

  // Combine block matrices
  std::vector<Eigen::Triplet<double>> combinedTriplets;
  combinedTriplets.reserve(Q11.nonZeros() + Q13.nonZeros() + Q22.nonZeros() +
                           Q23.nonZeros() + Q33.nonZeros());

  // Lambda function to add triplets to the combined triplets vector
  auto addTriplets = [&combinedTriplets](const SparseMatrix &matrix,
                                         size_t rowOffset, size_t colOffset) {
    for (int k = 0; k < matrix.outerSize(); ++k) {
      for (SparseMatrix::InnerIterator it(matrix, k); it; ++it) {
        combinedTriplets.emplace_back(it.row() + rowOffset,
                                      it.col() + colOffset, it.value());
      }
    }
  };

  // Set matrix dimensions
  const size_t rotMatSize = d_ * n_;
  const size_t rotRangeMatSize = rotMatSize + l_;
  const size_t dataMatSize = rotRangeMatSize + n_ + b_;

  // Q11, Q13, Q22, Q23, Q33
  addTriplets(Q11, 0, 0);
  addTriplets(Q13, 0, rotRangeMatSize);
  addTriplets(Q22, rotMatSize, rotMatSize);
  addTriplets(Q23, rotMatSize, rotRangeMatSize);
  addTriplets(Q33, rotRangeMatSize, rotRangeMatSize);

  // Add Q13 and Q23 transposed to the triplets
  addTriplets(Q13.transpose(), rotRangeMatSize, 0);
  addTriplets(Q23.transpose(), rotRangeMatSize, rotMatSize);

  // Construct the data matrix
  SparseMatrix Q(dataMatSize, dataMatSize);
  Q.setFromTriplets(combinedTriplets.begin(), combinedTriplets.end());
  Q_.emplace(Q);

  return true;
}

bool Graph::constructLinearCostTermRASLAM() {
  // TODO(AT): implement
  LOG(FATAL) << "Error: constructLinearCostTermRASLAM() not implemented yet!";
  return true;
}

std::optional<bool>
Graph::setIndicesFromStateOwnership(const RelativeMeasurement &measurement,
                                    size_t *i, size_t *j) {
  std::optional<bool> is_state_owned_by_inactive_neighbor;
  if (measurement.r1 == id_ && measurement.r2 != id_) {
    // Measurement is an outgoing shared loop closure. Check if the
    // measurement destination state belongs to this agent's neighbor and is
    // inactive
    const StateID &neighborDstStateID = measurement.getDstID();
    is_state_owned_by_inactive_neighbor =
        isStateOwnedByInactiveNeighbor(neighborDstStateID);
    if (is_state_owned_by_inactive_neighbor == true)
      *i = measurement.p1;
    else if (is_state_owned_by_inactive_neighbor == false)
      return false;
    else
      return std::nullopt;

  } else if (measurement.r1 != id_ && measurement.r2 == id_) {
    // Measurement is an incoming shared loop closure. Check if the
    // measurement source state belongs to this agent's neighbor and is
    // inactive
    const StateID &neighborSrcStateID = measurement.getSrcID();
    is_state_owned_by_inactive_neighbor =
        isStateOwnedByInactiveNeighbor(neighborSrcStateID);
    if (is_state_owned_by_inactive_neighbor == true)
      *j = measurement.p2;
    else if (is_state_owned_by_inactive_neighbor == false)
      return false;
    else
      return std::nullopt;

  } else {
    // Measurement is local to the agent's graph
    CHECK(measurement.r1 == id_ && measurement.r2 == id_);
    *i = measurement.p1;
    *j = measurement.p2;
  }

  return true;
}

std::optional<bool>
Graph::isStateOwnedByInactiveNeighbor(const StateID &neighborStateID) {
  // Check for neighbor state
  bool has_neighbor_state;
  executeStateDependantFunctionals(
      [&, this]() {
        const PoseID neighborPoseID(neighborStateID);
        has_neighbor_state =
            (neighbor_poses_.find(neighborPoseID) != neighbor_poses_.end());
      },
      [&, this]() {
        const PointID neighborLandmarkID(neighborStateID);
        has_neighbor_state = (neighbor_landmarks_.find(neighborLandmarkID) !=
                              neighbor_landmarks_.end());
      },
      neighborStateID.state_type);

  // Check if neighbor is inactive
  if (isNeighborActive(neighborStateID.robot_id)) {
    // Measurement with active neighbor
    if (!has_neighbor_state) {
      LOG(WARNING) << "Missing active neighbor state "
                   << neighborStateID.robot_id << ", "
                   << neighborStateID.frame_id;
      return false;
    }
  } else {
    // Measurement with inactive neighbor
    if (!use_inactive_neighbors_ || !has_neighbor_state)
      return std::nullopt;
  }

  return true;
}

Matrix
Graph::getNeighborFixedVariableLiftedData(const StateID &neighborStateID) {
  // Set neighbor state fixed variable to contain its lifted data
  Matrix X;
  executeStateDependantFunctionals(
      [&, this]() {
        const PoseID neighborPoseID(neighborStateID);
        const auto neighborPoseItr = neighbor_poses_.find(neighborPoseID);
        CHECK(neighborPoseItr != neighbor_poses_.end())
            << "Fixed pose variable of agent's neighbor "
            << neighborStateID.robot_id << " not found!";

        X = neighborPoseItr->second.pose();
      },
      [&, this]() {
        const PointID neighborLandmarkID(neighborStateID);
        const auto neighborPointItr =
            neighbor_landmarks_.find(neighborLandmarkID);
        CHECK(neighborPointItr != neighbor_landmarks_.end())
            << "Fixed landmark variable of agent's neighbor "
            << neighborStateID.robot_id << " not found!";

        X = neighborPointItr->second.translation();
      },
      neighborStateID.state_type);

  return X;
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

} // namespace DCORA
