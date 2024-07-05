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

#include <RTRNewton.h>
#include <SolversTR.h>

#include <Eigen/CholmodSupport>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <glog/logging.h>

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

#include <boost/functional/hash.hpp>

namespace DCORA {

typedef Eigen::VectorXd Vector;
typedef Eigen::MatrixXd Matrix;
typedef Eigen::DiagonalMatrix<double, Eigen::Dynamic> DiagonalMatrix;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SparseMatrix;
typedef Eigen::CholmodDecomposition<SparseMatrix> CholmodSolver;
typedef std::shared_ptr<CholmodSolver> CholmodSolverPtr;

/**
 * @brief Algorithms for initializing PGO
 */
enum class InitializationMethod { Odometry, Chordal, GNC_TLS };

/**
 * @brief State types
 */
enum class StateType { None, Pose, Point };

/**
 * @brief Measurement types
 */
enum class MeasurementType {
  PosePrior,
  PointPrior,
  PosePose,
  PosePoint,
  Range
};

/**
 * @brief PyFg types
 */
enum PyFGType {
  POSE_TYPE_2D,
  POSE_TYPE_3D,
  POSE_PRIOR_2D,
  POSE_PRIOR_3D,
  LANDMARK_TYPE_2D,
  LANDMARK_TYPE_3D,
  LANDMARK_PRIOR_2D,
  LANDMARK_PRIOR_3D,
  REL_POSE_POSE_TYPE_2D,
  REL_POSE_POSE_TYPE_3D,
  REL_POSE_LANDMARK_TYPE_2D,
  REL_POSE_LANDMARK_TYPE_3D,
  RANGE_MEASURE_TYPE,
};

/**
 * @brief PyFG string to type dictionary
 */
const std::map<std::string, PyFGType> PyFGStringToType{
    {"VERTEX_SE2", POSE_TYPE_2D},
    {"VERTEX_SE3:QUAT", POSE_TYPE_3D},
    {"VERTEX_SE2:PRIOR", POSE_PRIOR_2D},
    {"VERTEX_SE3:QUAT:PRIOR", POSE_PRIOR_3D},
    {"VERTEX_XY", LANDMARK_TYPE_2D},
    {"VERTEX_XYZ", LANDMARK_TYPE_3D},
    {"VERTEX_XY:PRIOR", LANDMARK_PRIOR_2D},
    {"VERTEX_XYZ:PRIOR", LANDMARK_PRIOR_3D},
    {"EDGE_SE2", REL_POSE_POSE_TYPE_2D},
    {"EDGE_SE3:QUAT", REL_POSE_POSE_TYPE_3D},
    {"EDGE_SE2_XY", REL_POSE_LANDMARK_TYPE_2D},
    {"EDGE_SE3_XYZ", REL_POSE_LANDMARK_TYPE_3D},
    {"EDGE_RANGE", RANGE_MEASURE_TYPE}};

/**
 * @brief Convert initialization method to string
 * @param method
 * @return
 */
std::string InitializationMethodToString(InitializationMethod method);

/**
 * @brief Convert state type to string
 * @param type
 * @return
 */
std::string StateTypeToString(const StateType &type);

/**
 * @brief Convert measurement type to string
 * @param type
 * @return
 */
std::string MeasurementTypeToString(const MeasurementType &type);

/**
 * @brief Parameter settings for Riemannian optimization
 */
class ROptParameters {
public:
  enum class ROptMethod {
    // Riemannian Trust-Region (RTRNewton in ROPTLIB)
    RTR,
    // Riemannian gradient descent (RSD in ROPTLIB)
    RGD
  };
  ROptParameters()
      : method(ROptMethod::RTR),
        verbose(false),
        gradnorm_tol(1e-2),
        RGD_stepsize(1e-3),
        RGD_use_preconditioner(true),
        RTR_iterations(3),
        RTR_tCG_iterations(50),
        RTR_initial_radius(100) {}

  ROptMethod method;
  bool verbose;
  double gradnorm_tol;
  double RGD_stepsize;
  bool RGD_use_preconditioner;
  int RTR_iterations;
  int RTR_tCG_iterations; // Maximum number of tCG iterations
  double RTR_initial_radius;

  /**
   * @brief Convert Riemannian optimization method to string
   * @param method
   * @return
   */
  static std::string ROptMethodToString(ROptMethod method);

  inline friend std::ostream &operator<<(std::ostream &os,
                                         const ROptParameters &params) {
    // clang-format off
    os << "Riemannian optimization parameters: " << std::endl;
    os << "Method: " << ROptMethodToString(params.method) << std::endl;
    os << "Gradient norm tol: " << params.gradnorm_tol << std::endl;
    os << "RGD stepsize: " << params.RGD_stepsize << std::endl;
    os << "RGD use preconditioner: " << params.RGD_use_preconditioner << std::endl; // NOLINT
    os << "RTR iterations: " << params.RTR_iterations << std::endl;
    os << "RTR tCG iterations: " << params.RTR_tCG_iterations << std::endl;
    os << "RTR initial radius: " << params.RTR_initial_radius << std::endl;
    // clang-format on
    return os;
  }
};

// Output statistics of Riemannian optimization
struct ROPTResult {
  ROPTResult(bool suc = false, double f0 = 0, double gn0 = 0, double fStar = 0,
             double gnStar = 0, double ms = 0)
      : success(suc),
        fInit(f0),
        gradNormInit(gn0),
        fOpt(fStar),
        gradNormOpt(gnStar),
        elapsedMs(ms) {}

  bool success;                    // Is the optimization successful
  double fInit;                    // Objective value before optimization
  double gradNormInit;             // Gradient norm before optimization
  double fOpt;                     // Objective value after optimization
  double gradNormOpt;              // Gradient norm after optimization
  double elapsedMs;                // elapsed time in milliseconds
  ROPTLIB::tCGstatusSet tCGStatus; // status of truncated conjugate gradient
                                   // (only used by trust region solver)
};

/**
 * @brief Submatrices used to construct RA-SLAM data matrix Q, where:
 *        Q = Q_p + Q_r
 *        See helper function constructRASLAMDataSubmatrices for details
 */
struct RASLAMDataSubmatrices {
  // Q_p
  // Incidence matrices
  SparseMatrix ARho;
  SparseMatrix ATau;
  // Weight matrices
  DiagonalMatrix OmegaRho;
  DiagonalMatrix OmegaTau;
  // Data matrices
  SparseMatrix T;

  // Q_r
  SparseMatrix C;            // incidence matrix
  DiagonalMatrix OmegaRange; // weight matrix
  SparseMatrix D;            // data matrix

  // Problem dimensions
  unsigned int d, n, l, b;

  RASLAMDataSubmatrices() = default;
};

// Each state is uniquely determined by the robot ID and frame ID
class StateID {
public:
  StateType state_type;  // state type
  unsigned int robot_id; // robot ID
  unsigned int frame_id; // frame ID
  explicit StateID(const StateType &type = StateType::None,
                   unsigned int rid = 0, unsigned int fid = 0)
      : state_type(type), robot_id(rid), frame_id(fid) {}
  StateID(const StateID &other)
      : state_type(other.state_type),
        robot_id(other.robot_id),
        frame_id(other.frame_id) {}
  bool operator==(const StateID &other) const {
    return (state_type == other.state_type && robot_id == other.robot_id &&
            frame_id == other.frame_id);
  }
  bool isPose() const { return state_type == StateType::Pose; }
  bool isPoint() const { return state_type == StateType::Point; }
};

class PoseID : public StateID {
public:
  explicit PoseID(unsigned int rid = 0, unsigned int fid = 0)
      : StateID(StateType::Pose, rid, fid) {}

  explicit PoseID(const StateID &state) {
    if (!state.isPose())
      LOG(FATAL) << "Error: Cannot construct PoseID from StateID: State is not "
                    "of type Pose!";
    state_type = state.state_type;
    robot_id = state.robot_id;
    frame_id = state.frame_id;
  }
};

class PointID : public StateID {
public:
  explicit PointID(unsigned int rid = 0, unsigned int fid = 0)
      : StateID(StateType::Point, rid, fid) {}

  explicit PointID(const StateID &state) {
    if (!state.isPoint())
      LOG(FATAL) << "Error: Cannot construct PointID from StateID: State is "
                    "not of type Point!";
    state_type = state.state_type;
    robot_id = state.robot_id;
    frame_id = state.frame_id;
  }
};

// Comparator for StateID
struct CompareStateID {
  bool operator()(const StateID &a, const StateID &b) const {
    auto pa = std::make_tuple(a.state_type, a.robot_id, a.frame_id);
    auto pb = std::make_tuple(b.state_type, b.robot_id, b.frame_id);
    return pa < pb;
  }
};

// Edge measurement (edge) is uniquely determined by an ordered pair of states
class EdgeID {
public:
  StateID src_state_id;
  StateID dst_state_id;
  EdgeID(const StateID &srcId, const StateID &dstId)
      : src_state_id(srcId), dst_state_id(dstId) {}
  bool operator==(const EdgeID &other) const {
    return (src_state_id == other.src_state_id &&
            dst_state_id == other.dst_state_id);
  }
  bool isOdometry() const {
    return (src_state_id.state_type == StateType::Pose &&
            dst_state_id.state_type == StateType::Pose &&
            src_state_id.robot_id == dst_state_id.robot_id &&
            src_state_id.frame_id + 1 == dst_state_id.frame_id);
  }
  bool isPrivateLoopClosure() const {
    return (src_state_id.robot_id == dst_state_id.robot_id && !isOdometry());
  }
  bool isSharedLoopClosure() const {
    return src_state_id.robot_id != dst_state_id.robot_id;
  }
};

// Comparator for EdgeID
struct CompareEdgeID {
  bool operator()(const EdgeID &a, const EdgeID &b) const {
    // Treat edge ID as an ordered tuple
    const auto ta =
        std::make_tuple(a.src_state_id.state_type, a.dst_state_id.state_type,
                        a.src_state_id.robot_id, a.dst_state_id.robot_id,
                        a.src_state_id.frame_id, a.dst_state_id.frame_id);
    const auto tb =
        std::make_tuple(b.src_state_id.state_type, b.dst_state_id.state_type,
                        b.src_state_id.robot_id, b.dst_state_id.robot_id,
                        b.src_state_id.frame_id, b.dst_state_id.frame_id);
    return ta < tb;
  }
};

// Hasher for EdgeID
struct HashEdgeID {
  std::size_t operator()(const EdgeID &edge_id) const {
    // Reference:
    // https://stackoverflow.com/questions/17016175/c-unordered-map-using-a-custom-class-type-as-the-key
    using boost::hash_combine;
    using boost::hash_value;

    // Start with a hash value of 0
    std::size_t seed = 0;

    // Modify 'seed' by XORing and bit-shifting in
    // one member of 'Key' after the other:
    hash_combine(seed, hash_value(edge_id.src_state_id.state_type));
    hash_combine(seed, hash_value(edge_id.dst_state_id.state_type));
    hash_combine(seed, hash_value(edge_id.src_state_id.robot_id));
    hash_combine(seed, hash_value(edge_id.dst_state_id.robot_id));
    hash_combine(seed, hash_value(edge_id.src_state_id.frame_id));
    hash_combine(seed, hash_value(edge_id.dst_state_id.frame_id));

    // Return the result.
    return seed;
  }
};

// Map from edge ID to edge index
typedef std::unordered_map<EdgeID, size_t, HashEdgeID> EdgeIDMap;

} // namespace DCORA
