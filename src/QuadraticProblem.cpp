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

#include <DCORA/QuadraticProblem.h>

#include <glog/logging.h>
#include <iostream>

namespace DCORA {

QuadraticProblem::QuadraticProblem(const std::shared_ptr<Graph> &graph)
    : graph_(graph), use_se_manifold_(graph_->isPGOCompatible()) {
  ROPTLIB::Problem::SetUseGrad(true);
  ROPTLIB::Problem::SetUseHess(true);

  // Set manifold according to graph type
  if (use_se_manifold_) {
    M_SE = std::make_unique<LiftedSEManifold>(graph_->r(), graph_->d(),
                                              graph_->n());
    ROPTLIB::Problem::SetDomain(M_SE->getManifold());
  } else {
    M_RA = std::make_unique<LiftedRAManifold>(
        graph_->r(), graph_->d(), graph_->n(), graph_->l(), graph_->b());
    ROPTLIB::Problem::SetDomain(M_RA->getManifold());
  }
}

QuadraticProblem::~QuadraticProblem() = default;

double QuadraticProblem::f(const Matrix &Y) const {
  CHECK_EQ((unsigned)Y.rows(), relaxation_rank());
  CHECK_EQ((unsigned)Y.cols(), problem_dimension());
  // returns 0.5 * (Y * Q * Y.transpose()).trace() + (Y * G.transpose()).trace()
  return 0.5 * ((Y * graph_->quadraticMatrix()).cwiseProduct(Y)).sum() +
         (Y.cwiseProduct(graph_->linearMatrix())).sum();
}

double QuadraticProblem::f(ROPTLIB::Variable *x) const {
  Eigen::Map<const Matrix> X(const_cast<double *>(x->ObtainReadData()),
                             relaxation_rank(), problem_dimension());
  return 0.5 * ((X * graph_->quadraticMatrix()).cwiseProduct(X)).sum() +
         (X.cwiseProduct(graph_->linearMatrix())).sum();
}

void QuadraticProblem::EucGrad(ROPTLIB::Variable *x, ROPTLIB::Vector *g) const {
  Eigen::Map<const Matrix> X(const_cast<double *>(x->ObtainReadData()),
                             relaxation_rank(), problem_dimension());
  Eigen::Map<Matrix> EG(const_cast<double *>(g->ObtainWriteEntireData()),
                        relaxation_rank(), problem_dimension());
  EG = X * graph_->quadraticMatrix() + graph_->linearMatrix();
}

void QuadraticProblem::EucHessianEta(ROPTLIB::Variable *x, ROPTLIB::Vector *v,
                                     ROPTLIB::Vector *Hv) const {
  Eigen::Map<const Matrix> V(const_cast<double *>(v->ObtainReadData()),
                             relaxation_rank(), problem_dimension());
  Eigen::Map<Matrix> HV(const_cast<double *>(Hv->ObtainWriteEntireData()),
                        relaxation_rank(), problem_dimension());
  HV = V * graph_->quadraticMatrix();
}

void QuadraticProblem::PreConditioner(ROPTLIB::Variable *x,
                                      ROPTLIB::Vector *inVec,
                                      ROPTLIB::Vector *outVec) const {
  Eigen::Map<const Matrix> INVEC(const_cast<double *>(inVec->ObtainReadData()),
                                 relaxation_rank(), problem_dimension());
  Eigen::Map<Matrix> OUTVEC(
      const_cast<double *>(outVec->ObtainWriteEntireData()), relaxation_rank(),
      problem_dimension());
  if (graph_->hasPreconditioner()) {
    OUTVEC = graph_->preconditioner()->solve(INVEC.transpose()).transpose();
  } else {
    LOG(WARNING) << "Failed to compute preconditioner.";
  }
  projectToTangentSpace(x, outVec, outVec);
}

Matrix QuadraticProblem::RieGrad(const Matrix &Y) const {
  // Get problem dimensions
  unsigned int r = relaxation_rank();
  unsigned int d = dimension();
  unsigned int n = num_poses();
  unsigned int l = num_unit_spheres();
  unsigned int b = num_landmarks();

  // Delegate to specific Riemannian gradient based on manifold type
  return use_se_manifold_ ? RieGradSE(Y, r, d, n) : RieGradRA(Y, r, d, n, l, b);
}

Matrix QuadraticProblem::RieGradSE(const Matrix &Y, unsigned int r,
                                   unsigned int d, unsigned int n) const {
  LiftedSEVariable Var(r, d, n);
  Var.setData(Y);
  LiftedSEVector EGrad(r, d, n);
  EucGrad(Var.var(), EGrad.vec());
  LiftedSEVector RGrad(r, d, n);
  projectToTangentSpace(Var.var(), EGrad.vec(), RGrad.vec());
  return RGrad.getData();
}

Matrix QuadraticProblem::RieGradRA(const Matrix &Y, unsigned int r,
                                   unsigned int d, unsigned int n,
                                   unsigned int l, unsigned int b) const {
  LiftedRAVariable Var(r, d, n, l, b);
  Var.setData(Y);
  LiftedRAVector EGrad(r, d, n, l, b);
  EucGrad(Var.var(), EGrad.vec());
  LiftedRAVector RGrad(r, d, n, l, b);
  projectToTangentSpace(Var.var(), EGrad.vec(), RGrad.vec());
  return RGrad.getData();
}

double QuadraticProblem::RieGradNorm(const Matrix &Y) const {
  return RieGrad(Y).norm();
}

Matrix QuadraticProblem::Retract(const Matrix &Y, const Matrix &V) const {
  // Get problem dimensions
  unsigned int r = relaxation_rank();
  unsigned int d = dimension();
  unsigned int n = num_poses();
  unsigned int l = num_unit_spheres();
  unsigned int b = num_landmarks();

  // Delegate to specific Riemannian gradient based on manifold type
  return use_se_manifold_ ? RetractSE(Y, V, r, d, n)
                          : RetractRA(Y, V, r, d, n, l, b);
}

bool QuadraticProblem::escapeSaddle(const Matrix &Xopt, double theta,
                                    const Vector &v, double gradient_tolerance,
                                    double preconditioned_gradient_tolerance,
                                    Matrix *X) {
  // Get problem dimensions
  unsigned int r = relaxation_rank();
  unsigned int k = problem_dimension();
  CHECK_EQ(Xopt.rows(), r - 1);
  CHECK_EQ(Xopt.cols(), k);

  // Lift first-order critical point to next rank
  Matrix X_plus = DCORA::Matrix::Zero(r, k);
  X_plus.topRows(r - 1) = Xopt;

  // Construct second-order decent direction
  Matrix X_dot_plus = DCORA::Matrix::Zero(r, k);
  X_dot_plus.bottomRows<1>() = v.transpose();

  /*
  The following implementation is adapted from:
  SE-Sync: https://github.com/david-m-rosen/SE-Sync.git
  */

  // Set the initial step length according to Algorithm 7 in DC2-PGO tech
  // report. Retain minimum step length from SE-Sync.
  double alpha = 1.0;
  double alpha_min = 1e-6;

  // Vectors of trial stepsizes and corresponding function values
  std::vector<double> alphas;
  std::vector<double> fvals;

  // Function value at current iterate (saddle point)
  double FX_plus = f(X_plus);

  // Backtracking line search
  Matrix Xtest;
  while (alpha >= alpha_min) {
    // Retract along the given tangent vector using the given stepsize
    Xtest = Retract(X_plus, alpha * X_dot_plus);

    // Ensure that the trial point Xtest has a lower function value than
    // the current iterate X, and that the gradient at Xtest is
    // sufficiently large that we will not automatically trigger the
    // gradient tolerance stopping criterion at the next iteration
    double FXtest = f(Xtest);
    Matrix grad_FXtest = RieGrad(Xtest);
    double grad_FXtest_norm = grad_FXtest.norm();
    double preconditioned_grad_FXtest_norm =
        PreCondition(Xtest, grad_FXtest).norm();

    // Record trial stepsize and function value
    alphas.push_back(alpha);
    fvals.push_back(FXtest);

    if ((FXtest < FX_plus) && (grad_FXtest_norm > gradient_tolerance) &&
        (preconditioned_grad_FXtest_norm > preconditioned_gradient_tolerance)) {
      // Accept this trial point and return success
      *X = Xtest;
      return true;
    }
    alpha /= 2;
  }

  // If control reaches here, we failed to find a trial point that satisfied
  // *both* the function decrease *and* gradient bounds.  In order to make
  // forward progress, we will fall back to accepting the trial point that
  // simply minimized the objective value, provided that it strictly *decreased*
  // the objective from the current (saddle) point

  // Find minimum function value from among the trial points
  auto fmin_iter = std::min_element(fvals.begin(), fvals.end());
  auto min_idx = std::distance(fvals.begin(), fmin_iter);

  double f_min = fvals.at(min_idx);
  double a_min = alphas.at(min_idx);

  if (f_min < FX_plus) {
    // If this trial point strictly decreased the objective value, accept it and
    // return success
    *X = Retract(X_plus, a_min * X_dot_plus);
    return true;
  } else {
    // NO trial point decreased the objective value: we were unable to escape
    // the saddle point!
    return false;
  }
}

Matrix QuadraticProblem::RetractSE(const Matrix &Y, const Matrix &V,
                                   unsigned int r, unsigned int d,
                                   unsigned int n) const {
  LiftedSEVariable inVar(r, d, n);
  inVar.setData(Y);
  LiftedSEVector inVec(r, d, n);
  inVec.setData(V);
  LiftedSEVariable outVar(r, d, n);
  M_SE->getManifold()->Retraction(inVar.var(), inVec.vec(), outVar.var());
  return outVar.getData();
}

Matrix QuadraticProblem::RetractRA(const Matrix &Y, const Matrix &V,
                                   unsigned int r, unsigned int d,
                                   unsigned int n, unsigned int l,
                                   unsigned int b) const {
  LiftedRAVariable inVar(r, d, n, l, b);
  inVar.setData(Y);
  LiftedRAVector inVec(r, d, n, l, b);
  inVec.setData(V);
  LiftedRAVariable outVar(r, d, n, l, b);
  M_RA->getManifold()->Retraction(inVar.var(), inVec.vec(), outVar.var());
  return outVar.getData();
}

Matrix QuadraticProblem::PreCondition(const Matrix &Y, const Matrix &V) const {
  // Get problem dimensions
  unsigned int r = relaxation_rank();
  unsigned int d = dimension();
  unsigned int n = num_poses();
  unsigned int l = num_unit_spheres();
  unsigned int b = num_landmarks();

  // Delegate to specific preconditioner based on manifold type
  return use_se_manifold_ ? PreConditionSE(Y, V, r, d, n)
                          : PreConditionRA(Y, V, r, d, n, l, b);
}

Matrix QuadraticProblem::PreConditionSE(const Matrix &Y, const Matrix &V,
                                        unsigned int r, unsigned int d,
                                        unsigned int n) const {
  LiftedSEVariable inVar(r, d, n);
  inVar.setData(Y);
  LiftedSEVector inVec(r, d, n);
  inVec.setData(V);
  LiftedSEVector outVec(r, d, n);
  PreConditioner(inVar.var(), inVec.vec(), outVec.vec());
  return outVec.getData();
}

Matrix QuadraticProblem::PreConditionRA(const Matrix &Y, const Matrix &V,
                                        unsigned int r, unsigned int d,
                                        unsigned int n, unsigned int l,
                                        unsigned int b) const {
  LiftedRAVariable inVar(r, d, n, l, b);
  inVar.setData(Y);
  LiftedRAVector inVec(r, d, n, l, b);
  inVec.setData(V);
  LiftedRAVector outVec(r, d, n, l, b);
  PreConditioner(inVar.var(), inVec.vec(), outVec.vec());
  return outVec.getData();
}

void QuadraticProblem::projectToTangentSpace(ROPTLIB::Variable *x,
                                             ROPTLIB::Vector *inVec,
                                             ROPTLIB::Vector *outVec) const {
  if (use_se_manifold_) {
    M_SE->projectToTangentSpace(x, inVec, outVec);
  } else {
    M_RA->projectToTangentSpace(x, inVec, outVec);
  }
}

} // namespace DCORA
