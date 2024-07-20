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

#include <DCORA/QuadraticOptimizer.h>
#include <glog/logging.h>
#include <iostream>

#include "SolversLS.h"

namespace DCORA {

QuadraticOptimizer::QuadraticOptimizer(QuadraticProblem *p,
                                       ROptParameters params)
    : problem_(p), params_(params), use_se_manifold_(p->useSEManifold()) {
  result_.success = false;
}

QuadraticOptimizer::~QuadraticOptimizer() = default;

Matrix QuadraticOptimizer::optimize(const Matrix &Y) {
  // Compute statistics before optimization
  result_.fInit = problem_->f(Y);
  result_.gradNormInit = problem_->RieGradNorm(Y);
  timer_.tic();

  // Optimize!
  Matrix YOpt;
  if (params_.method == ROptParameters::ROptMethod::RTR) {
    YOpt = trustRegion(Y);
  } else {
    YOpt = gradientDescent(Y);
  }

  // Compute statistics after optimization
  result_.elapsedMs = timer_.toc();
  result_.fOpt = problem_->f(YOpt);
  result_.gradNormOpt = problem_->RieGradNorm(YOpt);
  result_.success = true;
  // CHECK_LE(result_.fOpt, result_.fInit + 1e-5);

  return YOpt;
}

Matrix QuadraticOptimizer::trustRegion(const Matrix &Yinit) {
  // Return if the gradient norm is below tolerance
  if (problem_->RieGradNorm(Yinit) < params_.gradnorm_tol)
    return Yinit;

  // Get problem dimensions
  unsigned int r = problem_->relaxation_rank();
  unsigned int d = problem_->dimension();
  unsigned int n = problem_->num_poses();
  unsigned int l = problem_->num_unit_spheres();
  unsigned int b = problem_->num_landmarks();

  // Delegate to specific trust region method based on manifold type
  return use_se_manifold_ ? trustRegionSE(Yinit, r, d, n)
                          : trustRegionRA(Yinit, r, d, n, l, b);
}

Matrix QuadraticOptimizer::trustRegionSE(const Matrix &Yinit, unsigned int r,
                                         unsigned int d, unsigned int n) {
  // Initialize SE variable
  LiftedSEVariable VarInit(r, d, n);
  VarInit.setData(Yinit);
  VarInit.var()->NewMemoryOnWrite();

  // Configure and run solver
  ROPTLIB::RTRNewton Solver(problem_, VarInit.var());
  if (!configureAndRunTrustRegionSolver(&Solver))
    return Yinit;

  // Get solver result
  const auto *Yopt =
      dynamic_cast<const ROPTLIB::ProductElement *>(Solver.GetXopt());
  LiftedSEVariable VarOpt(r, d, n);
  Yopt->CopyTo(VarOpt.var());
  return VarOpt.getData();
}

Matrix QuadraticOptimizer::trustRegionRA(const Matrix &Yinit, unsigned int r,
                                         unsigned int d, unsigned int n,
                                         unsigned int l, unsigned int b) {
  // Initialize RA variable
  LiftedRAVariable VarInit(r, d, n, l, b);
  VarInit.setData(Yinit);
  VarInit.var()->NewMemoryOnWrite();

  // Configure and run solver
  ROPTLIB::RTRNewton Solver(problem_, VarInit.var());
  if (!configureAndRunTrustRegionSolver(&Solver))
    return Yinit;

  // Get solver result
  const auto *Yopt =
      dynamic_cast<const ROPTLIB::ProductElement *>(Solver.GetXopt());
  LiftedRAVariable VarOpt(r, d, n, l, b);
  Yopt->CopyTo(VarOpt.var());
  return VarOpt.getData();
}

Matrix QuadraticOptimizer::gradientDescent(const Matrix &Yinit) {
  // Get problem dimensions
  unsigned int r = problem_->relaxation_rank();
  unsigned int d = problem_->dimension();
  unsigned int n = problem_->num_poses();
  unsigned int l = problem_->num_unit_spheres();
  unsigned int b = problem_->num_landmarks();

  // Delegate to specific gradient descent method based on manifold type
  return use_se_manifold_ ? gradientDescentSE(Yinit, r, d, n)
                          : gradientDescentRA(Yinit, r, d, n, l, b);
}

Matrix QuadraticOptimizer::gradientDescentSE(const Matrix &Yinit,
                                             unsigned int r, unsigned int d,
                                             unsigned int n) {
  // Initialize SE manifold and variables
  LiftedSEManifold M(r, d, n);
  LiftedSEVariable VarInit(r, d, n);
  LiftedSEVariable VarNext(r, d, n);
  LiftedSEVector RGrad(r, d, n);
  VarInit.setData(Yinit);

  // Euclidean gradient
  problem_->EucGrad(VarInit.var(), RGrad.vec());

  // Riemannian gradient
  M.projectToTangentSpace(VarInit.var(), RGrad.vec(), RGrad.vec());

  // Preconditioning
  if (params_.RGD_use_preconditioner) {
    problem_->PreConditioner(VarInit.var(), RGrad.vec(), RGrad.vec());
  }

  // Update
  M.getManifold()->ScaleTimesVector(VarInit.var(), -params_.RGD_stepsize,
                                    RGrad.vec(), RGrad.vec());
  M.getManifold()->Retraction(VarInit.var(), RGrad.vec(), VarNext.var());

  return VarNext.getData();
}

Matrix QuadraticOptimizer::gradientDescentRA(const Matrix &Yinit,
                                             unsigned int r, unsigned int d,
                                             unsigned int n, unsigned int l,
                                             unsigned int b) {
  // Initialize RA manifold and variables
  LiftedRAManifold M(r, d, n, l, b);
  LiftedRAVariable VarInit(r, d, n, l, b);
  LiftedRAVariable VarNext(r, d, n, l, b);
  LiftedRAVector RGrad(r, d, n, l, b);
  VarInit.setData(Yinit);

  // Euclidean gradient
  problem_->EucGrad(VarInit.var(), RGrad.vec());

  // Riemannian gradient
  M.projectToTangentSpace(VarInit.var(), RGrad.vec(), RGrad.vec());

  // Preconditioning
  if (params_.RGD_use_preconditioner) {
    problem_->PreConditioner(VarInit.var(), RGrad.vec(), RGrad.vec());
  }

  // Update
  M.getManifold()->ScaleTimesVector(VarInit.var(), -params_.RGD_stepsize,
                                    RGrad.vec(), RGrad.vec());
  M.getManifold()->Retraction(VarInit.var(), RGrad.vec(), VarNext.var());

  return VarNext.getData();
}

Matrix QuadraticOptimizer::gradientDescentLineSearch(const Matrix &Yinit) {
  // Get problem dimensions
  unsigned int r = problem_->relaxation_rank();
  unsigned int d = problem_->dimension();
  unsigned int n = problem_->num_poses();
  unsigned int l = problem_->num_unit_spheres();
  unsigned int b = problem_->num_landmarks();

  // Delegate to specific gradient descent LS method based on manifold type
  return use_se_manifold_ ? gradientDescentLineSearchSE(Yinit, r, d, n)
                          : gradientDescentLineSearchRA(Yinit, r, d, n, l, b);
}

Matrix QuadraticOptimizer::gradientDescentLineSearchSE(const Matrix &Yinit,
                                                       unsigned int r,
                                                       unsigned int d,
                                                       unsigned int n) {
  // Initialize SE variable
  LiftedSEVariable VarInit(r, d, n);
  VarInit.setData(Yinit);

  // Configure and run solver
  ROPTLIB::RSD Solver(problem_, VarInit.var());
  configureAndRunLineSearchSolver(&Solver);

  // Get solver result
  const auto *Yopt =
      dynamic_cast<const ROPTLIB::ProductElement *>(Solver.GetXopt());
  LiftedSEVariable VarOpt(r, d, n);
  Yopt->CopyTo(VarOpt.var());
  return VarOpt.getData();
}

Matrix QuadraticOptimizer::gradientDescentLineSearchRA(
    const Matrix &Yinit, unsigned int r, unsigned int d, unsigned int n,
    unsigned int l, unsigned int b) {
  // Initialize RA variable
  LiftedRAVariable VarInit(r, d, n, l, b);
  VarInit.setData(Yinit);

  // Configure and run solver
  ROPTLIB::RSD Solver(problem_, VarInit.var());
  configureAndRunLineSearchSolver(&Solver);

  // Get solver result
  const auto *Yopt =
      dynamic_cast<const ROPTLIB::ProductElement *>(Solver.GetXopt());
  LiftedRAVariable VarOpt(r, d, n, l, b);
  Yopt->CopyTo(VarOpt.var());
  return VarOpt.getData();
}

bool QuadraticOptimizer::configureAndRunTrustRegionSolver(
    ROPTLIB::RTRNewton *Solver) {
  Solver->Stop_Criterion =
      ROPTLIB::StopCrit::GRAD_F; // Stopping criterion based on absolute
                                 // gradient norm
  Solver->Tolerance =
      params_.gradnorm_tol; // Tolerance associated with stopping criterion
  Solver->initial_Delta = params_.RTR_initial_radius; // Trust-region radius
  Solver->maximum_Delta =
      5 * Solver->initial_Delta; // Maximum trust-region radius
  if (params_.verbose) {
    Solver->Debug = ROPTLIB::DEBUGINFO::ITERRESULT;
  } else {
    Solver->Debug = ROPTLIB::DEBUGINFO::NOOUTPUT;
  }
  Solver->Max_Iteration = params_.RTR_iterations;
  Solver->Min_Inner_Iter = 0;
  Solver->Max_Inner_Iter = params_.RTR_tCG_iterations;
  Solver->TimeBound = 5.0;

  if (Solver->Max_Iteration == 1) {
    // Shrinking trust-region radius until step is accepted
    double radius = Solver->initial_Delta;
    int total_steps = 0;
    while (true) {
      Solver->initial_Delta = radius;
      Solver->maximum_Delta = radius;
      Solver->Run();
      if (Solver->latestStepAccepted()) {
        break;
      } else if (total_steps > 10) {
        LOG(WARNING) << "Too many RTR rejections. Returning initial guess.";
        return false;
      } else {
        radius = radius / 4;
        total_steps++;
        LOG(WARNING) << "RTR step rejected. Shrinking trust-region radius to: "
                     << radius;
      }
    }
  } else {
    Solver->Run();
  }
  // record tCG status
  result_.tCGStatus = Solver->gettCGStatus();
  return true;
}

void QuadraticOptimizer::configureAndRunLineSearchSolver(ROPTLIB::RSD *Solver) {
  Solver->Stop_Criterion = ROPTLIB::StopCrit::GRAD_F;
  Solver->Tolerance = 1e-2;
  Solver->Max_Iteration = 10;
  Solver->Debug = (params_.verbose ? ROPTLIB::DEBUGINFO::DETAILED
                                   : ROPTLIB::DEBUGINFO::NOOUTPUT);
  Solver->Run();
}

} // namespace DCORA
