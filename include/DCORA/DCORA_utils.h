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
#include <DCORA/Measurements.h>

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <chrono> // NOLINT(build/c++11)
#include <functional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "Manifolds/Euclidean/Euclidean.h"
#include "Manifolds/Oblique/Oblique.h"
#include "Manifolds/Stiefel/Stiefel.h"

using HighResClock =
    std::chrono::time_point<std::chrono::high_resolution_clock>;

namespace DCORA {

class SimpleTimer {
public:
  /**
   * @brief Start timer
   */
  void tic();
  /**
   * @brief Return elapsed time since last tic in ms
   * @return
   */
  double toc();
  /**
   * @brief Start timer and return starting time
   * @return
   */
  static HighResClock Tic();
  /**
   * @brief Return time elapsed since input start_time
   * @param start_time
   * @return elapsed time in ms
   */
  static double Toc(const HighResClock &start_time);

private:
  HighResClock t_start, t_end;
};

/**
 * @brief Write a dense Eigen matrix to file
 * @param M
 * @param filename
 */
void writeMatrixToFile(const Matrix &M, const std::string &filename);

/**
 * @brief Write a sparse matrix to file
 * @param M
 * @param filename
 */
void writeSparseMatrixToFile(const SparseMatrix &M,
                             const std::string &filename);

/**
 * @brief Helper function to read a dataset in .g2o format
 * @param filename
 * @param num_poses
 * @return
 */
std::vector<RelativePosePoseMeasurement>
read_g2o_file(const std::string &filename, size_t *num_poses);

/**
 * @brief Helper function to get the dimension of the RA-SLAM problem encoded in
 * a PyFG file.
 * @param filename
 * @return
 */
int getDimFromPyfgFirstLine(const std::string &filename);

/**
 * @brief Helper function to read a dataset in .pyfg format
 * @param filename
 * @return
 */
PyFGDataset read_pyfg_file(const std::string &filename);

/**
 * @brief Helper function to get the local-to-global state indexes from a PyFG
 * dataset
 * @param pyfg_dataset
 * @return
 */
LocalToGlobalStateDicts
getLocalToGlobalStateMapping(const PyFGDataset &pyfg_dataset);

/**
 * @brief Helper function to globally reindex a PyFG dataset's measurements
 * @param pyfg_dataset
 * @return
 */
Measurements getGlobalMeasurements(const PyFGDataset &pyfg_dataset);

/**
 * @brief Helper function to partition a PyFG dataset's measurements among its
 * robots
 * @param pyfg_dataset
 * @return
 */
RobotMeasurements getRobotMeasurements(const PyFGDataset &pyfg_dataset);

/**
 * @brief Execute functionals based on source and destination state type
 * @param poseFunction
 * @param pointFunction
 * @param state_type
 */
void executeStateDependantFunctionals(std::function<void()> poseFunction,
                                      std::function<void()> pointFunction,
                                      const StateType &state_type);

/**
 * @brief Get the dimension and number of poses from a vector of relative pose
 * measurements
 * @param measurements
 * @param dimension
 * @param num_poses
 */
void get_dimension_and_num_poses(
    const std::vector<RelativePosePoseMeasurement> &measurements,
    size_t *dimension, size_t *num_poses);

/**
 * @brief Given a vector of relative pose measurements, this function computes
 * and returns the B matrices defined in equation (69) of the tech report
 * @param measurements
 * @param B1
 * @param B2
 * @param B3
 */
void constructBMatrices(
    const std::vector<RelativePosePoseMeasurement> &measurements,
    SparseMatrix *B1, SparseMatrix *B2, SparseMatrix *B3);

/**
 * @brief Given the measurement matrices B1 and B2 and a matrix R of rotational
 * state estimates, this function computes and returns the corresponding optimal
 * translation estimates
 * @param B1
 * @param B2
 * @param R
 * @return
 */
Matrix recoverTranslations(const SparseMatrix &B1, const SparseMatrix &B2,
                           const Matrix &R);

/**
 * @brief Project a given matrix to the Rotation Group
 * @param M
 * @return Orthogonal projection of M to the Rotation Group
 */
Matrix projectToRotationGroup(const Matrix &M);

/**
 * @brief Project an input matrix M to the Stiefel manifold
 * @param M
 * @return Orthogonal projection of M to the Stiefel manifold
 */
Matrix projectToStiefelManifold(const Matrix &M);

/**
 * @brief Project an input matrix M to the Oblique manifold
 * @param M
 * @return Orthogonal projection of M to the Oblique manifold
 */
Matrix projectToObliqueManifold(const Matrix &M);

/**
 * @brief Helper function to compute and return the product
 *
 *   P = A * SymBlockDiag(B^T * C)
 *
 * where A, B, and C are r × kn matrices (cf. eq. (5) in the SE-Sync tech
 * report).
 * @param A
 * @param B
 * @param C
 * @param r
 * @param k
 * @param n
 */
Matrix symBlockDiagProduct(const Matrix &A, const Matrix &BT, const Matrix &C,
                           unsigned int r, unsigned int k, unsigned int n);

/**
 * @brief Given a symmetric sparse matrix S, this function returns a Boolean
 * value indicating whether the regularized matrix M := S + eta * I is
 * positive-semidefinite. In the event that M is *not* PSD, this function
 * additionally computes a direction of negative curvature x of S, and its
 * associated Rayleight quotient theta := x'Sx < 0 using a shift-and-invert mode
 * eigen solver. See the original implementation in SE-Sync for details, noting
 * that the shift-and-invert mode eigen solver is unique to DCORA.
 * @param S
 * @param eta
 * @param shift
 * @param theta
 * @param x
 * @return
 */
bool fastVerification(const SparseMatrix &S, double eta, double shift,
                      double *theta, Vector *x);

/**
 * @brief Helper function to determine if a sparse symmetric matrix S is
 * positive-semidefinite (PSD). Return true if the matrix is PSD, false
 * otherwise.
 * @param S
 * @return
 */
bool isSparseSymmetricMatrixPSD(const SparseMatrix &S);

/**
 * @brief Helper function to calculate the minimum eigen pair {λ, v} of a
 * sparse symmetric matrix S using the shift-and-invert mode, where σ is the
 * shift. This function uses a heuristic where by σ is halved if the solver is
 * unsuccessful. This reduction occurs for a set number of iterations, where the
 * minimum allowable shift is then set to -2 × eta.
 * @param S
 * @param sigma
 * @param eta
 * @return
 */
std::pair<double, Vector> computeMinimumEigenPair(const SparseMatrix &S,
                                                  double sigma, double eta);

/**
 * @brief Helper function to construct the dual certificate matrix S(X) for PGO.
 * @param X
 * @param Q
 * @param d
 * @param n
 * @return
 */
SparseMatrix constructDualCertificateMatrixPGO(const Matrix &X,
                                               const SparseMatrix &Q,
                                               unsigned int d, unsigned int n);

/**
 * @brief Helper function to construct the dual certificate matrix S(X) for
 * RA-SLAM.
 * @param X
 * @param Q
 * @param d
 * @param n
 * @param l
 * @param b
 * @return
 */
SparseMatrix
constructDualCertificateMatrixRASLAM(const Matrix &X, const SparseMatrix &Q,
                                     unsigned int d, unsigned int n,
                                     unsigned int l, unsigned int b);
/**
 * @brief Helper function to project a rank-r RA-SLAM solution to a rank-d
 * solution within the feasible set of the MAP RA-SLAM formulation. See
 * Algorithm 3 in the CORA arxiv paper.
 * @param X
 * @param r
 * @param d
 * @param n
 * @param l
 * @param b
 * @return
 */
Matrix projectSolutionRASLAM(const Matrix &X, unsigned int r, unsigned int d,
                             unsigned int n, unsigned int l, unsigned int b);

/**
 * @brief Given an element Y in M and a matrix V in T_X(R^{r × dn}) (that is, a
 * (r × dn)-dimensional matrix V considered as an element of the tangent space
 * to the *entire* ambient Euclidean space at X), this function computes and
 * returns the projection of V onto T_X(M), the tangent space of M at X (cf.
 * eq. (42) in the SE-Sync tech report). Matrices Y and V are r × dn
 * @param Y
 * @param V
 * @param r
 * @param d
 * @param n
 */
Matrix projectToStiefelManifoldTangentSpace(const Matrix &Y, const Matrix &V,
                                            unsigned int r, unsigned int d,
                                            unsigned int n);

/**
 * @brief Projects a matrix V in R^{r × l} onto the tangent space T_Y(M) of
 * the oblique manifold at Y in R^{r × l}.
 * @param V
 * @param Y
 * @return
 */
Matrix projectToObliqueManifoldTangentSpace(const Matrix &Y, const Matrix &V);

/**
 * @brief Generate a fixed element of the Stiefel element
 * The returned value is guaranteed to be the same for each r and d
 * @param r
 * @param b
 * @return
 */
Matrix fixedStiefelVariable(unsigned r, unsigned d);

/**
 * @brief Generate a fixed element of the Euclidean element
 * The returned value is guaranteed to be the same for each r and b
 * @param r
 * @param b
 * @return
 */
Matrix fixedEuclideanVariable(unsigned r, unsigned b = 1);

/**
 * @brief Generate a fixed element of the Oblique element
 * The returned value is guaranteed to be the same for each r and l
 * @param r
 * @param l
 * @return
 */
Matrix fixedObliqueVariable(unsigned r, unsigned l = 1);

/**
 * @brief Generate a random element of the Stiefel manifold
 * @param r
 * @param d
 * @return
 */
Matrix randomStiefelVariable(unsigned r, unsigned d);

/**
 * @brief Generate a random element of the Euclidean manifold
 * @param r
 * @param b
 * @return
 */
Matrix randomEuclideanVariable(unsigned r, unsigned b = 1);

/**
 * @brief Generate a random element of the Oblique manifold
 * @param r
 * @param l
 * @return
 */
Matrix randomObliqueVariable(unsigned r, unsigned l = 1);

/**
 * @brief Compute the error term (weighted squared residual)
 * @param m measurement
 * @param R1 rotation of first pose
 * @param t1 translation of first pose
 * @param R2 rotation of second pose
 * @param t2 translation of second pose
 * @return
 */
double computeMeasurementError(const RelativePosePoseMeasurement &m,
                               const Matrix &R1, const Matrix &t1,
                               const Matrix &R2, const Matrix &t2);

/**
 * @brief Quantile of chi-squared distribution with given degrees of freedom at
 * probability alpha. Equivalent to chi2inv in Matlab.
 * @param quantile
 * @param dof
 * @return
 */
double chi2inv(double quantile, size_t dof);

/**
 * @brief For SO(3), convert angular distance in radian to chordal distance
 * @param rad input angular distance in radian
 * @return
 */
double angular2ChordalSO3(double rad);

/**
 * @brief Verify that the input matrix is a valid rotation
 * @param R
 */
void checkRotationMatrix(const Matrix &R);

/**
 * @brief Check that the input matrix of dimension r-by-d is a valid element of
 * the Stiefel manifold
 * @param Y
 */
void checkStiefelMatrix(const Matrix &Y);

/**
 * @brief Check that the SE input matrix is of dimension r-by-(d+1)*n
 * @param X
 */
void checkSEMatrixSize(const Matrix &X, unsigned int r, unsigned int d,
                       unsigned int n);

/**
 * @brief Check that the RA input matrix is of dimension r-by-(d+1)*n+l+b
 * @param X
 */
void checkRAMatrixSize(const Matrix &X, unsigned int r, unsigned int d,
                       unsigned int n, unsigned int l, unsigned int b);

/**
 * @brief Partition the SE input matrix into SE_R and SE_t matrices,
 * respectively
 * @param X
 * @return
 */
std::tuple<Matrix, Matrix> partitionSEMatrix(const Matrix &X, unsigned int r,
                                             unsigned int d, unsigned int n);

/**
 * @brief Partition the RA input matrix into SE_R, OB, SE_t, and E matrices,
 * respectively
 * @param X
 * @return
 */
std::tuple<Matrix, Matrix, Matrix, Matrix>
partitionRAMatrix(const Matrix &X, unsigned int r, unsigned int d,
                  unsigned int n, unsigned int l, unsigned int b);

/**
 * @brief Create SE matrix from X_SE_R and X_SE_t matrices
 * @param X_SE_R
 * @param X_SE_t
 * @return
 */
Matrix createSEMatrix(const Matrix &X_SE_R, const Matrix &X_SE_t);

/**
 * @brief Create RA matrix from SE_R, OB, X_SE_t, and E matrices
 * @param X_SE_R
 * @param X_OB
 * @param X_SE_t
 * @param X_E
 * @return
 */
Matrix createRAMatrix(const Matrix &X_SE_R, const Matrix &X_OB,
                      const Matrix &X_SE_t, const Matrix &X_E);

/**
 * @brief Copy array data from Eigen matrix to ROPTLIB element
 * @param Y
 * @param var
 * @param mem_size
 */
void copyEigenMatrixToROPTLIBVariable(const Matrix &Y, ROPTLIB::Variable *var,
                                      size_t mem_size);

/**
 * @brief Project matrix M to the SE manifold
 * @param M
 * @param r
 * @param d
 * @param n
 * @return
 */
Matrix projectToSEMatrix(const Matrix &M, unsigned int r, unsigned int d,
                         unsigned int n);

/**
 * @brief Project matrix M to the RA manifold
 * @param M
 * @param r
 * @param d
 * @param n
 * @param l
 * @param b
 * @return
 */
Matrix projectToRAMatrix(const Matrix &M, unsigned int r, unsigned int d,
                         unsigned int n, unsigned int l, unsigned int b);

} // namespace DCORA
