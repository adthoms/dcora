/* ----------------------------------------------------------------------------
 * Copyright 2023, University of California Los Angeles, Los Angeles, CA 90095
 * All Rights Reserved
 * Authors: Alex Thoms and Alan Papalia
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#pragma once

#include <DCORA/manifold/LiftedVariable.h>

#include <memory>

#include "Manifolds/Oblique/Oblique.h"

namespace DCORA {

/**
 * @brief This class represents a collection of "lifted" poses, unit sphere
 * variables, and landmarks:
 * X = [Y1 ... Yn | r1 ... rn | p1 ... pn | l1 ... ln]
 * that can be used by ROPTLIB to perform Riemannian optimization
 */
class LiftedRAVariable : public LiftedSEVariable {
public:
  /**
   * @brief Constructor
   * @param r relaxation rank
   * @param d dimension (2/3)
   * @param n number of poses
   * @param l number of ranges
   * @param b number of landmarks
   */
  LiftedRAVariable(unsigned int r, unsigned int d, unsigned int n,
                   unsigned int l, unsigned int b);
  /**
   * @brief Constructor from a lifted range-aided array object
   * @param rangeAidedArray
   */
  explicit LiftedRAVariable(const LiftedRangeAidedArray &rangeAidedArray);
  /**
   * @brief Copy constructor
   * @param other
   */
  LiftedRAVariable(const LiftedRAVariable &other);
  /**
   * @brief Copy assignment operator
   * @param other
   * @return
   */
  LiftedRAVariable &operator=(const LiftedRAVariable &other);
  /**
   * @brief Get number of ranges
   * @return
   */
  unsigned int l() const { return l_; }
  /**
   * @brief Get number of landmarks
   * @return
   */
  unsigned int b() const { return b_; }
  /**
   * @brief Obtain the variable as an ROPTLIB::ProductElement
   * @return
   */
  ROPTLIB::ProductElement *var() override { return varRA_.get(); }
  /**
   * @brief Obtain the variable as an Eigen matrix
   * @return r-by-(d+1)n+l+b matrix of the form:
   * X = [Y1 ... Yn | r1 ... rn | p1 ... pn | l1 ... ln]
   */
  Matrix getData() const override;
  /**
   * @brief Set this variable from an Eigen matrix
   * @param X r-by-(d+1)n+l+b matrix of the form:
   * X = [Y1 ... Yn | r1 ... rn | p1 ... pn | l1 ... ln]
   */
  void setData(const Matrix &X) override;
  /**
   * @brief Obtain the writable landmark translation at the specified index,
   * expressed as an r dimensional vector
   * @param index
   * @return
   */
  Eigen::Ref<Vector> landmarkTranslation(unsigned int index);
  /**
   * @brief Obtain the read-only landmark translation at the specified index,
   * expressed as an r dimensional vector
   * @param index
   * @return
   */
  Vector landmarkTranslation(unsigned int index) const;
  /**
   * @brief Obtain the writable unit-sphere auxiliary variable for a range
   * measurement at the specified index, expressed as an r dimensional vector
   * @param index
   * @return
   */
  Eigen::Ref<Vector> rangeUnitSphereVariable(unsigned int index);
  /**
   * @brief Obtain the read-only unit-sphere auxiliary variable at the specified
   * index, expressed as an r dimensional vector
   * @param index
   * @return
   */
  Vector rangeUnitSphereVariable(unsigned int index) const;

private:
  // const dimensions
  unsigned int l_, b_;
  // The actual variable content is stored inside a ROPTLIB::ProductElement
  std::unique_ptr<ROPTLIB::ObliqueVariable> ranges_var_;
  std::unique_ptr<ROPTLIB::EucVariable> landmark_var_;
  std::unique_ptr<ROPTLIB::ProductElement> varOB_;
  std::unique_ptr<ROPTLIB::ProductElement> varE_;
  std::unique_ptr<ROPTLIB::ProductElement> varRA_;
  // Internal view of the variable:
  // ranges as an eigen matrix of dimension r-by-l
  Eigen::Map<Matrix> X_OB_;
  // landmarks as an eigen matrix of dimension r-by-b
  Eigen::Map<Matrix> X_E_;
  // RA-SLAM domain as an eigen matrix of dimension r-by-(d+1)n+l+b
  Eigen::Map<Matrix> X_RA_;
};

} // namespace DCORA
