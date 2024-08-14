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

#include <DCORA/Agent.h>
#include <DCORA/DCORA_types.h>
#include <DCORA/QuadraticProblem.h>
#include <DCORA/manifold/LiftedManifold.h>
#include <DCORA/manifold/LiftedVariable.h>
#include <DCORA/manifold/LiftedVector.h>

#include "RSD.h"
#include "RTRNewton.h"

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

  // Bool for using SE manifold or RA manifold
  bool use_se_manifold_;

  // Optimization result
  ROPTResult result_;

  // Timing
  SimpleTimer timer_;

  // Apply RTR
  Matrix trustRegion(const Matrix &Yinit);
  Matrix trustRegionSE(const Matrix &Yinit, unsigned int r, unsigned int d,
                       unsigned int n);
  Matrix trustRegionRA(const Matrix &Yinit, unsigned int r, unsigned int d,
                       unsigned int n, unsigned int l, unsigned int b);

  // Apply a single RGD iteration with constant step size
  Matrix gradientDescent(const Matrix &Yinit);
  Matrix gradientDescentSE(const Matrix &Yinit, unsigned int r, unsigned int d,
                           unsigned int n);
  Matrix gradientDescentRA(const Matrix &Yinit, unsigned int r, unsigned int d,
                           unsigned int n, unsigned int l, unsigned int b);

  // Apply gradient descent with line search
  Matrix gradientDescentLineSearch(const Matrix &Yinit);
  Matrix gradientDescentLineSearchSE(const Matrix &Yinit, unsigned int r,
                                     unsigned int d, unsigned int n);
  Matrix gradientDescentLineSearchRA(const Matrix &Yinit, unsigned int r,
                                     unsigned int d, unsigned int n,
                                     unsigned int l, unsigned int b);

  // Helper functions for configuring and running the solvers
  bool configureAndRunTrustRegionSolver(ROPTLIB::RTRNewton *Solver);
  void configureAndRunLineSearchSolver(ROPTLIB::RSD *Solver);
};

} // namespace DCORA
