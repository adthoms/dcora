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
 * @brief This class implements a ROPTLIB problem with the following cost
 * function: f(X) = 0.5*<Q, XtX> + <X,G>; Q is the quadratic part with dimension
 * (d+1)n-by-(d+1)n; G is the linear part with dimension r-by-(d+1)n
 */
class QuadraticProblem : public ROPTLIB::Problem {
public:
  /**
   * @brief Construct a quadratic optimization problem from a pose graph
   * @param pose_graph input pose graph must be initialized (or can be
   * initialized) otherwise throw a runtime error
   */
  explicit QuadraticProblem(const std::shared_ptr<PoseGraph> &pose_graph);

  /**
   * @brief Deconstructor
   */
  ~QuadraticProblem() override;

  /**
   * @brief Number of pose variables
   * @return
   */
  unsigned int num_poses() const { return pose_graph_->n(); }

  /**
   * @brief Dimension (2 or 3) of estimation problem
   * @return
   */
  unsigned int dimension() const { return pose_graph_->d(); }

  /**
   * @brief Relaxation rank in Riemannian optimization problem
   * @return
   */
  unsigned int relaxation_rank() const { return pose_graph_->r(); }

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

private:
  // The pose graph that represents the optimization problem
  std::shared_ptr<PoseGraph> pose_graph_;

  // Underlying manifold
  LiftedSEManifold *M;

  // Helper functions to convert between ROPTLIB::Element and Eigen Matrix
  Matrix readElement(const ROPTLIB::Element *element) const;
  void setElement(ROPTLIB::Element *element, const Matrix *matrix) const;
};

} // namespace DCORA
