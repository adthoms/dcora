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
#include <DCORA/Graph.h>
#include <DCORA/manifold/LiftedManifold.h>
#include <DCORA/manifold/LiftedVariable.h>
#include <DCORA/manifold/LiftedVector.h>

#include <Eigen/CholmodSupport>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <vector>

#include "Problems/Problem.h"

namespace DCORA {

/**
 * @brief This class implements a ROPTLIB problem with the cost function:
 *
 *   f(X) = 0.5 × <Q, X^T × X> + <X,G>
 *
 * where Q is the quadratic part with dimension k-by-k and G is the
 * linear part with dimension r-by-k. The dimension of k depends on the problem
 * instance, where:
 *   k = (d + 1) × n; for pose graph optimization and
 *   k = (d + 1) × n + l + b; for range-aided SLAM
 */
class QuadraticProblem : public ROPTLIB::Problem {
public:
  /**
   * @brief Construct a quadratic optimization problem from a graph
   * @param graph
   */
  explicit QuadraticProblem(const std::shared_ptr<Graph> &graph);

  /**
   * @brief Deconstructor
   */
  ~QuadraticProblem() override;

  /**
   * @brief Dimension (2 or 3) of estimation problem
   * @return
   */
  unsigned int dimension() const { return graph_->d(); }

  /**
   * @brief Relaxation rank in Riemannian optimization problem
   * @return
   */
  unsigned int relaxation_rank() const { return graph_->r(); }

  /**
   * @brief Number of pose variables
   * @return
   */
  unsigned int num_poses() const { return graph_->n(); }

  /**
   * @brief Number of unit sphere variables
   * @return
   */
  unsigned int num_unit_spheres() const { return graph_->l(); }

  /**
   * @brief Number of landmark variables
   * @return
   */
  unsigned int num_landmarks() const { return graph_->b(); }

  /**
   * @brief Riemannian optimization problem dimension
   * @return
   */
  unsigned int problem_dimension() const { return graph_->k(); }

  /**
   * @brief Return true if SE manifold is used
   * @return
   */
  bool useSEManifold() const { return use_se_manifold_; }

  /**
   * @brief Evaluate objective function
   * @param Y
   * @return
   */
  double f(const Matrix &Y) const;

  /**
   * @brief Evaluate objective function
   * @param x
   * @return
   */
  double f(ROPTLIB::Variable *x) const override;

  /**
   * @brief Evaluate Euclidean gradient
   * @param x
   * @param g
   */
  void EucGrad(ROPTLIB::Variable *x, ROPTLIB::Vector *g) const override;

  /**
   * @brief Evaluate Hessian-vector product
   * @param x
   * @param v
   * @param Hv
   */
  void EucHessianEta(ROPTLIB::Variable *x, ROPTLIB::Vector *v,
                     ROPTLIB::Vector *Hv) const override;

  /**
   * @brief Evaluate preconditioner
   * @param x
   * @param inVec
   * @param outVec
   */
  void PreConditioner(ROPTLIB::Variable *x, ROPTLIB::Vector *inVec,
                      ROPTLIB::Vector *outVec) const override;

  /**
   * @brief Compute the Riemannian gradient at Y (represented in matrix form)
   * @param Y current point on the manifold (matrix form)
   * @return Riemannian gradient at Y as a matrix
   */
  Matrix RieGrad(const Matrix &Y) const;

  /**
   * @brief Compute Riemannian gradient norm at Y
   * @param Y current point on the manifold (matrix form)
   * @return Norm of the Riemannian gradient
   */
  double RieGradNorm(const Matrix &Y) const;

  /**
   * @brief Compute the retraction of tangent vector V at Y
   * @param Y point on the manifold (matrix form)
   * @param V tangent vector at point Y in the manifold's tangent space (matrix
   * form)
   */
  Matrix Retract(const Matrix &Y, const Matrix &V) const;

private:
  // The graph that represents the optimization problem
  std::shared_ptr<Graph> graph_;

  // Bool for using SE manifold or RA manifold
  bool use_se_manifold_;

  // Underlying manifolds
  std::unique_ptr<LiftedSEManifold> M_SE;
  std::unique_ptr<LiftedRAManifold> M_RA;

  /**
   * @brief Helper function to compute the Riemannian gradient at Y in the SE
   * domain. See RieGrad for more details.
   * @param Y
   * @param r
   * @param d
   * @param n
   * @return
   */
  Matrix RieGradSE(const Matrix &Y, unsigned int r, unsigned int d,
                   unsigned int n) const;

  /**
   * @brief Helper function to compute the Riemannian gradient at Y in the RA
   * domain. See RieGrad for more details.
   * @param Y
   * @param r
   * @param d
   * @param n
   * @param l
   * @param b
   * @return
   */
  Matrix RieGradRA(const Matrix &Y, unsigned int r, unsigned int d,
                   unsigned int n, unsigned int l, unsigned int b) const;

  /**
   * @brief Helper function to compute the retraction of tangent vector V at Y
   * in the SE domain. See Retract for more details.
   * @param Y
   * @param V
   * @param r
   * @param d
   * @param n
   * @return
   */
  Matrix RetractSE(const Matrix &Y, const Matrix &V, unsigned int r,
                   unsigned int d, unsigned int n) const;

  /**
   * @brief Helper function to compute the retraction of tangent vector V at Y
   * in the RA domain. See Retract for more details.
   * @param Y
   * @param V
   * @param r
   * @param d
   * @param n
   * @param l
   * @param b
   * @return
   */
  Matrix RetractRA(const Matrix &Y, const Matrix &V, unsigned int r,
                   unsigned int d, unsigned int n, unsigned int l,
                   unsigned int b) const;

  /**
   * @brief Helper function to compute the preconditioner.
   * @param Y
   * @param V
   * @return
   */
  Matrix PreCondition(const Matrix &Y, const Matrix &V) const;

  /**
   * @brief Helper function to compute the preconditioner in the SE domain. See
   * PreCondition for details.
   * @param Y
   * @param V
   * @param r
   * @param d
   * @param n
   * @return
   */
  Matrix PreConditionSE(const Matrix &Y, const Matrix &V, unsigned int r,
                        unsigned int d, unsigned int n) const;

  /**
   * @brief Helper function to compute the preconditioner in the RA domain. See
   * PreCondition for details.
   * @param Y
   * @param V
   * @param r
   * @param d
   * @param n
   * @param l
   * @param b
   * @return
   */
  Matrix PreConditionRA(const Matrix &Y, const Matrix &V, unsigned int r,
                        unsigned int d, unsigned int n, unsigned int l,
                        unsigned int b) const;

  /**
   * @brief Helper function to project vector inVec onto the tangent space
   * of the manifold at x, yielding outVec
   * @param x
   * @param inVec
   * @param outVec
   */
  void projectToTangentSpace(ROPTLIB::Variable *x, ROPTLIB::Vector *inVec,
                             ROPTLIB::Vector *outVec) const;
};

} // namespace DCORA
