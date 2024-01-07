/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#pragma once

#include <DCORA/CORAAgent.h>
#include <DCORA/DCORA_types.h>
#include <DCORA/QuadraticProblem.h>
#include <DCORA/manifold/LiftedSEManifold.h>
#include <DCORA/manifold/LiftedSEVariable.h>
#include <DCORA/manifold/LiftedSEVector.h>

namespace DCORA {

class QuadraticOptimizer {
public:
  /**
   * @brief Constructor
   * @param p
   * @param params
   */
  QuadraticOptimizer(QuadraticProblem *p,
                     ROptParameters params = ROptParameters());

  /**
   * @brief Destructor
   */
  ~QuadraticOptimizer();

  /**
   * @brief Optimize from the given initial guess
   * @param Y
   * @return
   */
  Matrix optimize(const Matrix &Y);

  /**
   * @brief Set optimization problem
   * @param p
   */
  void setProblem(QuadraticProblem *p) { problem_ = p; }

  /**
   * @brief Turn on/off verbose output
   * @param v
   */
  void setVerbose(bool v) { params_.verbose = v; }

  /**
   * @brief Set optimization algorithm
   * @param alg
   */
  void setAlgorithm(ROptParameters::ROptMethod alg) { params_.method = alg; }

  /**
   * @brief Set maximum step size
   * @param s
   */
  void setRGDStepsize(double s) { params_.RGD_stepsize = s; }

  /**
   * @brief Set number of trust region iterations
   * @param iter
   */
  void setRTRIterations(int iter) { params_.RTR_iterations = iter; }

  /**
   * @brief Set tolerance of trust region
   * @param tol
   */
  void setGradientNormTolerance(double tol) { params_.gradnorm_tol = tol; }

  /**
   * @brief Set the initial trust region radius (default 1e1)
   * @param radius
   */
  void setRTRInitialRadius(double radius) {
    params_.RTR_initial_radius = radius;
  }

  /**
   * @brief Set the maximum number of inner tCG iterations
   * @param iter
   */
  void setRTRtCGIterations(int iter) { params_.RTR_tCG_iterations = iter; }

  /**
   * @brief Return optimization result
   * @return
   */
  ROPTResult getOptResult() const { return result_; }

private:
  // Underlying Riemannian Optimization Problem
  QuadraticProblem *problem_;

  // Optimization algorithm to be used
  ROptParameters params_;

  // Optimization result
  ROPTResult result_;

  // Timing
  SimpleTimer timer_;

  // Apply RTR
  Matrix trustRegion(const Matrix &Yinit);

  // Apply a single RGD iteration with constant step size
  Matrix gradientDescent(const Matrix &Yinit);

  // Apply gradient descent with line search
  Matrix gradientDescentLS(const Matrix &Yinit);
};

} // namespace DCORA
