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

Graph::Graph(unsigned int id, unsigned int r, unsigned int d,
             GraphType graphType)
    : id_(id),
      r_(r),
      d_(d),
      n_(0),
      l_(0),
      b_(0),
      graph_type(graphType),
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
  loc_shared_unit_sphere_ids_.clear();
  nbr_shared_pose_ids_.clear();
  nbr_shared_landmark_ids_.clear();
  nbr_shared_unit_sphere_ids_.clear();
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

bool Graph::isPGOCompatible() const {
  if (graph_type == GraphType::RangeAidedSLAMGraph)
    return false;

  if (l_ > 0 || b_ > 0)
    LOG(FATAL)
        << "Error: Pose graph cannot contain unit spheres and landmarks!";

  return true;
}

void Graph::clearNeighborStates() {
  neighbor_poses_.clear();
  neighbor_landmarks_.clear();
  neighbor_unit_spheres_.clear();
  G_.reset(); // Clearing neighbor states requires re-computing linear matrix
}

void Graph::updateNumPosesAndLandmarks(const StateID &stateID) {
  CHECK_EQ(stateID.robot_id, id_);
  // Update num poses
  if (stateID.isPose())
    n_ = std::max(n_, static_cast<unsigned int>(stateID.frame_id + 1));
  // Update num landmarks
  if (stateID.isPoint())
    b_ = std::max(b_, static_cast<unsigned int>(stateID.frame_id + 1));
}

void Graph::updateNumUnitSpheres(const RelativeMeasurement &measurement) {
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
  // Check that this is an odometry measurement
  CHECK(factor.measurementType == MeasurementType::PosePose);
  CHECK(factor.r1 == id_);
  CHECK(factor.r2 == id_);
  CHECK(factor.p1 + 1 == factor.p2);

  // Check that this is a valid measurement
  factor.checkDim(d_);

  // Check for duplicate odometry
  const EdgeID &edge_id = factor.getEdgeID();
  if (hasMeasurement(edge_id))
    return;

  // Update number of poses (landmarks remain the same)
  updateNumPosesAndLandmarks(factor.getDstID()); // dst_id > src_id

  // Dynamically cast to odometry measurement
  const RelativePosePoseMeasurement &odom_factor =
      dynamic_cast<const RelativePosePoseMeasurement &>(factor);

  // Add relative measurement factor to odometry
  odometry_.push_back(odom_factor);

  // Update edges
  edge_id_to_index_.emplace(edge_id, odometry_.size() - 1);
}

void Graph::addPrivateLoopClosure(const RelativeMeasurement &factor) {
  // Check that this is a private loop closure
  CHECK(factor.r1 == id_);
  CHECK(factor.r2 == id_);

  // Check that this is a valid measurement
  factor.checkDim(d_);

  // Check for duplicate private loop closure
  const EdgeID &edge_id = factor.getEdgeID();
  if (hasMeasurement(edge_id)) {
    if (factor.measurementType != MeasurementType::Range)
      return;
    LOG(FATAL) << "Error: Range measurements must be unique to ensure "
                  "correct unit sphere indexing! \n"
               << factor;
  }

  // Update number of poses and landmarks
  updateNumPosesAndLandmarks(factor.getSrcID());
  updateNumPosesAndLandmarks(factor.getDstID());
  // Update number of unit spheres
  updateNumUnitSpheres(factor);

  // Add relative measurement factor to private loop closures
  private_lcs_.push_back(factor);

  // Update edges
  edge_id_to_index_.emplace(edge_id, private_lcs_.vec.size() - 1);
}

void Graph::addSharedLoopClosure(const RelativeMeasurement &factor) {
  // Check that this is a valid measurement
  factor.checkDim(d_);

  // Check for duplicate shared loop closure
  const EdgeID &edge_id = factor.getEdgeID();
  if (hasMeasurement(edge_id)) {
    if (factor.measurementType != MeasurementType::Range)
      return;
    LOG(FATAL) << "Error: Range measurements must be unique to ensure "
                  "correct unit sphere indexing! \n"
               << factor;
  }

  // Update number of unit spheres
  updateNumUnitSpheres(factor);

  // Update local and neighbor shared state and edge IDs. Set active neighbor.
  if (factor.r1 == id_) {
    CHECK(factor.r2 != id_);

    // Update number of poses and landmarks
    updateNumPosesAndLandmarks(factor.getSrcID());

    // Add local shared pose/landmark to graph
    executeStateDependantFunctionals(
        [&, this]() { loc_shared_pose_ids_.emplace(factor.r1, factor.p1); },
        [&, this]() { loc_shared_landmark_ids_.emplace(factor.r1, factor.p1); },
        factor.stateType1);

    // Add neighbor shared pose/landmark to graph
    executeStateDependantFunctionals(
        [&, this]() { nbr_shared_pose_ids_.emplace(factor.r2, factor.p2); },
        [&, this]() { nbr_shared_landmark_ids_.emplace(factor.r2, factor.p2); },
        factor.stateType2);

    // add local shared unit-sphere to graph
    if (factor.measurementType == MeasurementType::Range) {
      const RangeMeasurement &range_factor =
          dynamic_cast<const RangeMeasurement &>(factor);
      loc_shared_unit_sphere_ids_.emplace(range_factor.getUnitSphereID());
    }

    // Update neighbor robot IDs
    nbr_robot_ids_.insert(factor.r2);

    // Set active neighbor
    neighbor_active_[factor.r2] = true;
  } else {
    CHECK(factor.r2 == id_);

    // Update number of poses and landmarks
    updateNumPosesAndLandmarks(factor.getDstID());

    // Add local shared pose/landmark to graph
    executeStateDependantFunctionals(
        [&, this]() { loc_shared_pose_ids_.emplace(factor.r2, factor.p2); },
        [&, this]() { loc_shared_landmark_ids_.emplace(factor.r2, factor.p2); },
        factor.stateType2);

    // Add neighbor shared pose/landmark to graph
    executeStateDependantFunctionals(
        [&, this]() { nbr_shared_pose_ids_.emplace(factor.r1, factor.p1); },
        [&, this]() { nbr_shared_landmark_ids_.emplace(factor.r1, factor.p1); },
        factor.stateType1);

    // Add neighbor shared unit-sphere to graph
    if (factor.measurementType == MeasurementType::Range) {
      const RangeMeasurement &range_factor =
          dynamic_cast<const RangeMeasurement &>(factor);
      nbr_shared_unit_sphere_ids_.emplace(range_factor.getUnitSphereID());
    }

    // Update neighbor robot IDs
    nbr_robot_ids_.insert(factor.r1);

    // Set active neighbor
    neighbor_active_[factor.r1] = true;
  }

  // Add relative measurement factor to shared loop closures
  shared_lcs_.push_back(factor);

  // Update edges
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
                              const LandmarkDict &landmark_dict,
                              const UnitSphereDict &unit_sphere_dict) {
  neighbor_poses_ = pose_dict;
  neighbor_landmarks_ = landmark_dict;
  neighbor_unit_spheres_ = unit_sphere_dict;
  G_.reset(); // Setting neighbor states requires re-computing linear matrix
}

void Graph::setNeighborPoses(const PoseDict &pose_dict) {
  neighbor_poses_ = pose_dict;
  G_.reset();
}

void Graph::setNeighborLandmarks(const LandmarkDict &landmark_dict) {
  neighbor_landmarks_ = landmark_dict;
  G_.reset();
}

void Graph::setNeighborUnitSpheres(const UnitSphereDict &unit_sphere_dict) {
  neighbor_unit_spheres_ = unit_sphere_dict;
  G_.reset();
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

bool Graph::requireNeighborUnitSphere(
    const UnitSphereID &unit_sphere_id) const {
  return nbr_shared_unit_sphere_ids_.find(unit_sphere_id) !=
         nbr_shared_unit_sphere_ids_.end();
}

bool Graph::hasMeasurement(const EdgeID &edgeID) const {
  return edge_id_to_index_.find(edgeID) != edge_id_to_index_.end();
}

RelativeMeasurement *Graph::findMeasurement(const EdgeID &edgeID) {
  RelativeMeasurement *edge = nullptr;
  auto getEdgePointerFromRelativeMeasurementVariant = [](auto &&arg) {
    using T = std::decay_t<decltype(arg)>;
    if constexpr (std::is_base_of_v<RelativeMeasurement, T>)
      return dynamic_cast<RelativeMeasurement *>(&arg);
    else
      LOG(FATAL) << "Error: cannot dynamically cast RelativeMeasurement!";
    return static_cast<RelativeMeasurement *>(nullptr);
  };
  if (hasMeasurement(edgeID)) {
    size_t index = edge_id_to_index_.at(edgeID);
    if (edgeID.isOdometry()) {
      edge = &odometry_[index];
    } else if (edgeID.isPrivateLoopClosure()) {
      edge = std::visit(getEdgePointerFromRelativeMeasurementVariant,
                        private_lcs_.vec[index]);
    } else {
      edge = std::visit(getEdgePointerFromRelativeMeasurementVariant,
                        shared_lcs_.vec[index]);
    }
  }
  if (edge) {
    // Sanity check
    CHECK(edge->measurementType == edgeID.measurement_type);
    CHECK(edge->stateType1 == edgeID.src_state_id.state_type);
    CHECK(edge->stateType2 == edgeID.dst_state_id.state_type);
    CHECK_EQ(edge->r1, edgeID.src_state_id.robot_id);
    CHECK_EQ(edge->p1, edgeID.src_state_id.frame_id);
    CHECK_EQ(edge->r2, edgeID.dst_state_id.robot_id);
    CHECK_EQ(edge->p2, edgeID.dst_state_id.frame_id);
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

LandmarkSet Graph::activeNeighborPublicLandmarkIDs() const {
  LandmarkSet output;
  for (const auto &point_id : nbr_shared_landmark_ids_) {
    if (isNeighborActive(point_id.robot_id)) {
      output.emplace(point_id);
    }
  }
  return output;
}

UnitSphereSet Graph::activeNeighborPublicUnitSphereIDs() const {
  UnitSphereSet output;
  for (const auto &unit_sphere_id : nbr_shared_unit_sphere_ids_) {
    if (isNeighborActive(unit_sphere_id.robot_id)) {
      output.emplace(unit_sphere_id);
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
  const unsigned int dh = d_ + 1;
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
  for (const auto &measVariant : measurements.vec) {
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
      // Get neighbor's fixed lifted pose
      const StateID &neighborDstStateID = meas.getDstID();
      XcT.noalias() = getNeighborFixedVariableLiftedData(neighborDstStateID);

      // Set incidence matrices
      AbT.noalias() = -T; // Leaving node i of agent b
      AcT.noalias() = I;  // Entering node j of agent c

      // Add measurement contribution to linear cost
      G.pose(i) += XcT * AcT * Omega * AbT.transpose();
    } else {
      CHECK(j != IDX_NOT_SET);
      // Get neighbor's fixed lifted pose
      const StateID &neighborSrcStateID = meas.getSrcID();
      XcT.noalias() = getNeighborFixedVariableLiftedData(neighborSrcStateID);

      // Set incidence matrices
      AbT.noalias() = I;  // Entering node j of agent b
      AcT.noalias() = -T; // Leaving node i of agent c

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
   *   l_b is the number of unit spheres owned by this agent
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
  // Set measurements
  const RelativeMeasurements &measurements = sharedLoopClosures();
  const std::vector<RelativePosePoseMeasurement> &pose_pose_measurements =
      measurements.GetRelativePosePoseMeasurements();
  const std::vector<RelativePosePointMeasurement> &pose_point_measurements =
      measurements.GetRelativePosePointMeasurements();
  const std::vector<RangeMeasurement> &range_measurements =
      measurements.GetRangeMeasurements();

  /**
   * @brief Constructing the linear cost term for RA-SLAM
   *
   * The linear cost term G is a [r × ((d+1) × n_b + l_b + b_b)] matrix of the
   * form:
   *
   *   G = Xc^T × Qcb
   *     = XcT × Qcb
   *
   * where:
   *   n_b is the number of poses owned by this agent (i.e. agent b)
   *   l_b is the number of unit sphere variables owned by this agent
   *   b_b is the number of landmarks owned by this agent
   *   XcT is a matrix of fixed public states (poses, unit spheres, and
   *       landmarks) of the neighbor agent (i.e agent c)
   *   Qcb is the bottom left block of Q after block decomposition
   *
   * Note: neighbor agent c is viewed as a meta agent including all agents that
   * are not agent b such that:
   *
   *   c:= [N]/{b}; where N is the total number of agents
   *
   * When calculating the contribution to the linear cost term for each fixed
   * neighbor variable, we consider the block structure of Qcb:
   *
   *   <-----------col------------>
   *
   *      dn_b       l_b    n_b + b_b
   *   -------------------------------             ^
   *   | Q_cb_11 |    0    | Q_cb_13 |  dn_c       |
   *   |    0    | Q_cb_22 | Q_cb_23 |  l_c       row
   *   | Q_cb_31 | Q_cb_32 | Q_cb_33 |  n_c + b_c  |
   *   -------------------------------             v
   *
   * where:
   *   Q_cb_11 = AcRho^T × OmegaRho × AbRho + Tc^T × OmegaTau × Tb
   *           = AcRhoT × OmegaRho × AbRhoT^T + TcT × OmegaTau × TbT^T
   *   Q_cb_12 = 0
   *   Q_cb_13 = Tc^T × OmegaTau × AbTau
   *           = TcT × OmegaTau × AbTauT^T

   *   Q_cb_21 = 0
   *   Q_cb_22 = Pc^T × OmegaRange × D^2 × Pb
   *           = PcT × OmegaRange × DT^T × DT^T × PbT^T
   *   Q_cb_23 = Pc^T × D × OmegaRange × Cb
   *           = PcT × DT^T × OmegaRange × CbT^T
   *
   *   Q_cb_31 = AcTau^T × OmegaTau × Tb
   *           = AcTauT × OmegaTau × TbT^T
   *   Q_cb_32 = Cc^T × OmegaRange × D × Pb
   *           = CcT × OmegaRange × DT^T × PbT^T
   *   Q_cb_33 = AcTau^T × OmegaTau × AbTau + Cc^T × OmegaRange × Cb
   *           = AcTauT × OmegaTau × AbTauT^T + CcT × OmegaRange × CbT^T
   *
   * Considering the blocks associated with a single lifted state XcT, we have:
   *
   * Dimensions: [rows × cols]
   *   AcRhoT: [d × d]
   *   AbRhoT: [d × d]
   *   OmegaRho: [d × d] - diagonal matrix
   *   TcT: [d × 1]
   *   TbT: [d × 1]
   *   AcTauT: [1 × 1]
   *   AbTauT: [1 × 1]
   *   OmegaTau: [1 × 1]
   *   PcT: [1 × 1]
   *   PbT: [1 × 1]
   *   DT: [1 × 1]
   *   CcT: [1 × 1]
   *   CbT: [1 × 1]
   *   OmegaRange: [1 × 1]
   *
   * where XcT is either a lifted pose, lifted unit sphere, or a lifted
   * landmark:
   *
   * Dimensions: [rows × cols]
   *   XcT_pose: [r × (d + 1)]
   *   XcT_unit_sphere: [r × 1]
   *   XcT_landmark: [r × 1]
   *
   * Explicit functions have been created to generate the following mappings
   * from the fixed neighbor state to the lifted linear cost depending on the
   * local state:
   *
   *   ------------------------------------------------
   *   | Fixed Neighbor |    Local    |    Lifted     |
   *   |     State      |    State    |  Linear Cost  |
   *   ------------------------------------------------
   *   | Pose           | Pose        | L_pose        |
   *   | Pose           | Unit Sphere | L_unit_sphere |
   *   | Pose           | Landmark    | L_landmark    |
   *   | Landmark       | Pose        | L_pose        |
   *   | Landmark       | Unit Sphere | L_unit_sphere |
   *   | Landmark       | Landmark    | L_landmark    |
   *   | Unit Sphere    | Pose        | L_pose        |
   *   | Unit Sphere    | Landmark    | L_landmark    |
   *   ------------------------------------------------
   *
   * For book keeping, we look at the contribution of each measurement and
   * update G via the addition of these contributions. See Section "Lambda
   * functions for calculating linear cost contributions for details
   */
  Matrix XcT_pose = Matrix::Zero(r_, d_ + 1);
  Matrix XcT_unit_sphere = Matrix::Zero(r_, 1);
  Matrix XcT_landmark = Matrix::Zero(r_, 1);
  Matrix L_pose = Matrix::Zero(r_, d_ + 1);
  Matrix L_unit_sphere = Matrix::Zero(r_, 1);
  Matrix L_landmark = Matrix::Zero(r_, 1);
  Matrix AcRhoT = Matrix::Zero(d_, d_);
  Matrix AbRhoT = Matrix::Zero(d_, d_);
  Matrix OmegaRho = Matrix::Zero(d_, d_);
  Matrix TcT = Matrix::Zero(d_, 1);
  Matrix TbT = Matrix::Zero(d_, 1);
  Matrix AcTauT = Matrix::Zero(1, 1);
  Matrix AbTauT = Matrix::Zero(1, 1);
  Matrix OmegaTau = Matrix::Zero(1, 1);
  Matrix PcT = Matrix::Zero(1, 1);
  Matrix PbT = Matrix::Zero(1, 1);
  Matrix DT = Matrix::Zero(1, 1);
  Matrix CcT = Matrix::Zero(1, 1);
  Matrix CbT = Matrix::Zero(1, 1);
  Matrix OmegaRange = Matrix::Zero(1, 1);

  // Initialize entries of incidence matrices
  Matrix R = Matrix::Identity(d_, d_);
  Matrix I_dxd = Matrix::Identity(d_, d_);
  Matrix t = Matrix::Zero(d_, 1);
  Matrix Zero_dx1 = Matrix::Zero(d_, 1);

  // Initialize Q_cb block matrices
  Matrix Q_cb_11 = Matrix::Zero(d_, d_);
  Matrix Q_cb_12 = Matrix::Zero(d_, 1);
  Matrix Q_cb_13 = Matrix::Zero(d_, 1);

  Matrix Q_cb_21 = Matrix::Zero(1, d_);
  Matrix Q_cb_22 = Matrix::Zero(1, 1);
  Matrix Q_cb_23 = Matrix::Zero(1, 1);

  Matrix Q_cb_31 = Matrix::Zero(1, d_);
  Matrix Q_cb_32 = Matrix::Zero(1, 1);
  Matrix Q_cb_33 = Matrix::Zero(1, 1);

  // Initialize linear cost
  LiftedRangeAidedArray G(r_, d_, n_, l_, b_);
  G.setDataToZero();

  /**
   * @brief Lambda functions for calculating linear cost contributions
   *
   * The following lambda functions are used to calculate the linear cost
   * contributions as described in the previous section.
   */

  auto updateQuadraticCostSubmatrices = [&]() -> void {
    // Calculate transposes
    // Note: The transpose of a 1 × 1 matrix is the matrix itself
    const Matrix &AcRho = AcRhoT.transpose();
    const Matrix &AbRho = AbRhoT.transpose();
    const Matrix &Tc = TcT.transpose();
    const Matrix &Tb = TbT.transpose();
    const Matrix &AcTau = AcTauT;
    const Matrix &AbTau = AbTauT;
    const Matrix &Pc = PcT;
    const Matrix &Pb = PbT;
    const Matrix &D = DT;
    const Matrix &Cc = CcT;
    const Matrix &Cb = CbT;

    // Update Q_cb block matrices for current measurement
    Q_cb_11.noalias() = AcRhoT * OmegaRho * AbRho + TcT * OmegaTau * Tb;
    Q_cb_13.noalias() = TcT * OmegaTau * AbTau;

    Q_cb_22.noalias() = PcT * OmegaRange * D * D * Pb;
    Q_cb_23.noalias() = PcT * D * OmegaRange * Cb;

    Q_cb_31.noalias() = AcTauT * OmegaTau * Tb;
    Q_cb_32.noalias() = CcT * OmegaRange * D * Pb;
    Q_cb_33.noalias() = AcTauT * OmegaTau * AbTau + CcT * OmegaRange * Cb;
  };

  auto updateLinearCostFromFixedNeighborPoseToLocalPose = [&]() -> void {
    // Partition rotation and translation components
    auto [XcT_pose_rot, XcT_pose_trans] =
        partitionSEMatrix(XcT_pose, r_, d_, 1);

    // Update Q_cb block matrices for current measurement
    updateQuadraticCostSubmatrices();

    // Assign linear cost
    L_pose.topLeftCorner(r_, d_) =
        XcT_pose_rot * Q_cb_11 + XcT_pose_trans * Q_cb_31;
    L_pose.topRightCorner(r_, 1) =
        XcT_pose_rot * Q_cb_13 + XcT_pose_trans * Q_cb_33;
  };

  auto updateLinearCostFromFixedNeighborPoseToLocalLandmark = [&]() -> void {
    // Partition rotation and translation components
    auto [XcT_pose_rot, XcT_pose_trans] =
        partitionSEMatrix(XcT_pose, r_, d_, 1);

    // Update Q_cb block matrices for current measurement
    updateQuadraticCostSubmatrices();

    // Assign linear cost
    L_landmark.noalias() = XcT_pose_rot * Q_cb_13 + XcT_pose_trans * Q_cb_33;
  };

  auto updateLinearCostFromFixedNeighborLandmarkToLocalPose = [&]() -> void {
    // Update Q_cb block matrices for current measurement
    updateQuadraticCostSubmatrices();

    // Assign linear cost
    L_pose.topLeftCorner(r_, d_) = XcT_landmark * Q_cb_31;
    L_pose.topRightCorner(r_, 1) = XcT_landmark * Q_cb_33;
  };

  auto updateLinearCostFromFixedNeighborLandmarkToLocalLandmark =
      [&]() -> void {
    // Update Q_cb block matrices for current measurement
    updateQuadraticCostSubmatrices();

    // Assign linear cost
    L_landmark.noalias() = XcT_landmark * Q_cb_33;
  };

  auto updateLinearCostFromFixedNeighborUnitSphereToLocalPose = [&]() -> void {
    // Update Q_cb block matrices for current measurement
    updateQuadraticCostSubmatrices();

    // Assign linear cost
    L_pose.topLeftCorner(r_, d_) = Matrix::Zero(r_, d_);
    L_pose.topRightCorner(r_, 1) = XcT_unit_sphere * Q_cb_23;
  };

  auto updateLinearCostFromFixedNeighborUnitSphereToLocalLandmark =
      [&]() -> void {
    // Update Q_cb block matrices for current measurement
    updateQuadraticCostSubmatrices();

    // Assign linear cost
    L_landmark.noalias() = XcT_unit_sphere * Q_cb_23;
  };

  auto updateLinearCostFromFixedNeighborPoseToLocalUnitSphere = [&]() -> void {
    // Partition rotation and translation components
    auto [XcT_pose_rot, XcT_pose_trans] =
        partitionSEMatrix(XcT_pose, r_, d_, 1);

    // Update Q_cb block matrices for current measurement
    updateQuadraticCostSubmatrices();

    // Assign linear cost
    L_unit_sphere.noalias() = XcT_pose_trans * Q_cb_32;
  };

  auto updateLinearCostFromFixedNeighborLandmarkToLocalUnitSphere =
      [&]() -> void {
    // Update Q_cb block matrices for current measurement
    updateQuadraticCostSubmatrices();

    // Assign linear cost
    L_unit_sphere.noalias() = XcT_landmark * Q_cb_32;
  };

  // Iterate over all shared pose-pose loop closures
  for (const auto &meas : pose_pose_measurements) {
    size_t i = IDX_NOT_SET;
    size_t j = IDX_NOT_SET;

    // Update measurement rotation and translation matrix
    R.noalias() = meas.R;
    t.noalias() = meas.t;

    // Update measurement weight matrix
    for (unsigned i = 0; i < d_; ++i)
      OmegaRho(i, i) = meas.weight * meas.kappa;

    OmegaTau(1, 1) = meas.weight * meas.tau;

    // Set indices according to pose ownership
    std::optional<bool> are_indices_set =
        setIndicesFromStateOwnership(meas, &i, &j);
    if (are_indices_set == false)
      return false;
    else if (are_indices_set == std::nullopt)
      continue;

    // Update linear cost
    if (i != IDX_NOT_SET) {
      // Get neighbor's fixed lifted pose
      const StateID &neighborDstStateID = meas.getDstID();
      XcT_pose.noalias() =
          getNeighborFixedVariableLiftedData(neighborDstStateID);

      // Set incidence and data matrices
      AbRhoT.noalias() = -R; // Leaving node i of agent b
      TbT.noalias() = -t;
      AbTauT(1, 1) = -1;
      AcRhoT.noalias() = I_dxd; // Entering node j of agent c
      TcT.noalias() = Zero_dx1;
      AcTauT(1, 1) = +1;

      // Add measurement contribution to linear cost
      updateLinearCostFromFixedNeighborPoseToLocalPose();
      G.pose(i) += L_pose;
    } else {
      CHECK(j != IDX_NOT_SET);
      // Get neighbor's fixed lifted pose
      const StateID &neighborSrcStateID = meas.getSrcID();
      XcT_pose.noalias() =
          getNeighborFixedVariableLiftedData(neighborSrcStateID);

      // Set incidence and data matrices
      AbRhoT.noalias() = I_dxd; // Entering node j of agent b
      TbT.noalias() = Zero_dx1;
      AbTauT(1, 1) = +1;
      AcRhoT.noalias() = -R; // Leaving node i of agent c
      TcT.noalias() = -t;
      AcTauT(1, 1) = -1;

      // Add measurement contribution to linear cost
      updateLinearCostFromFixedNeighborPoseToLocalPose();
      G.pose(j) += L_pose;
    }
  }

  // Reset rotation weight matrix to zero
  OmegaRho.setZero();

  // Iterate over all shared pose-point loop closures
  for (const auto &meas : pose_point_measurements) {
    size_t i = IDX_NOT_SET;
    size_t j = IDX_NOT_SET;

    // Update measurement translation matrix
    t.noalias() = meas.t;

    // Update measurement weight matrix
    OmegaTau(1, 1) = meas.weight * meas.tau;

    // Set indices according to pose/landmark ownership
    std::optional<bool> are_indices_set =
        setIndicesFromStateOwnership(meas, &i, &j);
    if (are_indices_set == false)
      return false;
    else if (are_indices_set == std::nullopt)
      continue;

    // Update linear cost
    if (i != IDX_NOT_SET) {
      // Get neighbor's fixed lifted landmark
      const StateID &neighborDstStateID = meas.getDstID();
      XcT_landmark.noalias() =
          getNeighborFixedVariableLiftedData(neighborDstStateID);

      // Set incidence and data matrices
      TbT.noalias() = -t; // Leaving node i of agent b
      AbTauT(1, 1) = -1;
      TcT.noalias() = Zero_dx1; // Entering node j of agent c
      AcTauT(1, 1) = +1;

      // Add measurement contribution to linear cost
      updateLinearCostFromFixedNeighborLandmarkToLocalPose();
      G.pose(i) += L_pose;
    } else {
      CHECK(j != IDX_NOT_SET);
      // Get neighbor's fixed lifted pose
      const StateID &neighborSrcStateID = meas.getSrcID();
      XcT_pose.noalias() =
          getNeighborFixedVariableLiftedData(neighborSrcStateID);

      // Set incidence and data matrices
      TbT.noalias() = Zero_dx1; // Entering node j of agent b
      AbTauT(1, 1) = +1;
      TcT.noalias() = -t; // Leaving node i of agent c
      AcTauT(1, 1) = -1;

      // Add measurement contribution to linear cost
      updateLinearCostFromFixedNeighborPoseToLocalLandmark();
      G.landmark(j) += L_landmark;
    }
  }

  // Reset translation weight matrix to zero
  OmegaTau.setZero();

  // Iterate over all shared range loop closures
  for (const auto &meas : range_measurements) {
    size_t i = IDX_NOT_SET;
    size_t j = IDX_NOT_SET;

    // Update measurement range matrix
    DT(1, 1) = meas.range;

    // Update measurement weight matrix
    OmegaRange(1, 1) = meas.weight * meas.precision;

    // Set indices according to pose/landmark ownership
    std::optional<bool> are_indices_set =
        setIndicesFromStateOwnership(meas, &i, &j);
    if (are_indices_set == false)
      return false;
    else if (are_indices_set == std::nullopt)
      continue;

    // Update linear cost
    if (i != IDX_NOT_SET) {
      // get neighbor's fixed lifted pose/landmark
      const StateID &neighborDstStateID = meas.getDstID();
      const Matrix XcT = getNeighborFixedVariableLiftedData(neighborDstStateID);

      // Set incidence and data matrices
      CbT(1, 1) = -1; // Leaving node i of agent b
      CcT(1, 1) = +1; // Entering node j of agent c

      // Update selection matrices
      PbT(1, 1) = 1;
      PcT(1, 1) = 0;

      // Set unit sphere variable index
      const unsigned int l = meas.l;

      // Add measurement contribution to linear cost
      const StateID &localSrcStateID = meas.getSrcID();
      if (neighborDstStateID.isPose()) {
        // Neighbor's fixed state is a lifted pose
        XcT_pose.noalias() = XcT;

        // Assign linear cost to local pose or landmark
        if (localSrcStateID.isPose()) {
          updateLinearCostFromFixedNeighborPoseToLocalPose();
          G.pose(i) += L_pose;
        } else {
          CHECK(localSrcStateID.isPoint());
          updateLinearCostFromFixedNeighborPoseToLocalLandmark();
          G.landmark(i) += L_landmark;
        }

        // Assign linear cost to local unit sphere variable
        updateLinearCostFromFixedNeighborPoseToLocalUnitSphere();
        G.unitSphere(l) += L_unit_sphere;

      } else {
        CHECK(neighborDstStateID.isPoint());

        // Neighbor's fixed state is a lifted landmark
        XcT_landmark.noalias() = XcT;

        // Assign linear cost to local pose or landmark
        if (localSrcStateID.isPose()) {
          updateLinearCostFromFixedNeighborLandmarkToLocalPose();
          G.pose(i) += L_pose;
        } else {
          CHECK(localSrcStateID.isPoint());
          updateLinearCostFromFixedNeighborLandmarkToLocalLandmark();
          G.landmark(i) += L_landmark;
        }

        // Assign linear cost to local unit sphere variable
        updateLinearCostFromFixedNeighborLandmarkToLocalUnitSphere();
        G.unitSphere(l) += L_unit_sphere;
      }

      /**
       * @brief Unit sphere variables belonging to this agent
       *
       * For shared range loop closures that leave node i of agent b and enter
       * node j of agent c, the unit sphere variables associated with these
       * edges belong to this agent. As such, there are no fixed neighbor unit
       * sphere variables
       */
      continue;

    } else {
      CHECK(j != IDX_NOT_SET);
      // Get neighbor's fixed lifted pose/landmark
      const StateID &neighborSrcStateID = meas.getSrcID();
      const Matrix XcT = getNeighborFixedVariableLiftedData(neighborSrcStateID);

      // set incidence and data matrices
      CbT(1, 1) = +1; // Entering node j of agent b
      CcT(1, 1) = -1; // Leaving node i of agent c

      // Update selection matrices
      PbT(1, 1) = 0;
      PcT(1, 1) = 1;

      // Add measurement contribution to linear cost
      const StateID &localDstStateID = meas.getDstID();
      if (neighborSrcStateID.isPose()) {
        // Neighbor's fixed state is a lifted pose
        XcT_pose.noalias() = XcT;

        // Assign linear cost to local pose or landmark
        if (localDstStateID.isPose()) {
          updateLinearCostFromFixedNeighborPoseToLocalPose();
          G.pose(j) += L_pose;
        } else {
          CHECK(localDstStateID.isPoint());
          updateLinearCostFromFixedNeighborPoseToLocalLandmark();
          G.landmark(j) += L_landmark;
        }

      } else {
        CHECK(neighborSrcStateID.isPoint());

        // Neighbor's fixed state is a lifted landmark
        XcT_landmark.noalias() = XcT;

        // Assign linear cost to local pose or landmark
        if (localDstStateID.isPose()) {
          updateLinearCostFromFixedNeighborLandmarkToLocalPose();
          G.pose(j) += L_pose;
        } else {
          CHECK(localDstStateID.isPoint());
          updateLinearCostFromFixedNeighborLandmarkToLocalLandmark();
          G.landmark(j) += L_landmark;
        }
      }

      /**
       * @brief Fixed neighbor poses/landmarks do not contributed to the linear
       * cost associated with local unit sphere variables
       *
       * Note: Since Pb^T= 0, Q_32 = 0 and we have no contributions from fixed
       * neighbor poses/landmarks to this agent's unit sphere variables.
       * Intuitively, this makes sense as the neighbor poses/landmarks are
       * connected with local poses/landmarks via an edge with a unit sphere
       * variables associated with the neighbor
       */

      /**
       * @brief Unit sphere variables belonging to neighbor agent
       *
       * For shared range loop closures that enter node j of agent b and
       * leave node i of agent c, the unit sphere variables associated with
       * these edges belong to the neighbors of this agent. As such, we get the
       * fixed unit sphere variable of the neighbor
       */
      const StateID &neighborUnitSphereID = meas.getUnitSphereID();
      XcT_unit_sphere.noalias() =
          getNeighborFixedVariableLiftedData(neighborUnitSphereID);

      // Add measurement contribution to linear cost
      if (localDstStateID.isPose()) {
        updateLinearCostFromFixedNeighborUnitSphereToLocalPose();
        G.pose(j) += L_pose;
      } else {
        CHECK(localDstStateID.isPoint());
        updateLinearCostFromFixedNeighborUnitSphereToLocalLandmark();
        G.landmark(j) += L_landmark;
      }
    }
  }

  // Set linear cost matrix
  G_.emplace(G.getData());

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
  Matrix X;
  switch (neighborStateID.state_type) {
  case StateType::Pose: {
    const PoseID neighborPoseID(neighborStateID);
    const auto neighborPoseItr = neighbor_poses_.find(neighborPoseID);
    CHECK(neighborPoseItr != neighbor_poses_.end())
        << "Error: Fixed pose variable of agent's neighbor "
        << neighborStateID.robot_id << " not found!";
    X = neighborPoseItr->second.pose();
  } break;
  case StateType::Point: {
    const PointID neighborLandmarkID(neighborStateID);
    const auto neighborPointItr = neighbor_landmarks_.find(neighborLandmarkID);
    CHECK(neighborPointItr != neighbor_landmarks_.end())
        << "Error: Fixed landmark variable of agent's neighbor "
        << neighborStateID.robot_id << " not found!";
    X = neighborPointItr->second.translation();
  } break;
  case StateType::UnitSphere: {
    const UnitSphereID neighborUnitSphereID(neighborStateID);
    const auto neighborUnitSphereItr =
        neighbor_unit_spheres_.find(neighborUnitSphereID);
    CHECK(neighborUnitSphereItr != neighbor_unit_spheres_.end())
        << "Error: Fixed unit sphere variable of agent's neighbor "
        << neighborStateID.robot_id << " not found!";
    X = neighborUnitSphereItr->second.translation();
    CHECK_LE(X.norm() - 1, 1e-6) << "Error: Unit sphere is not normalized!";
  } break;
  default:
    LOG(FATAL) << "Invalid StateType: "
               << StateTypeToString(neighborStateID.state_type) << "!";
  }

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
