/* ----------------------------------------------------------------------------
 * Copyright 2023, University of California Los Angeles, Los Angeles, CA 90095
 * All Rights Reserved
 * Authors: Alex Thoms and Alan Papalia
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#ifndef LIFTEDRAVARIABLE_H
#define LIFTEDRAVARIABLE_H

#include <DCORA/manifold/LiftedSEVariable.h>

#include "Manifolds/Oblique/Oblique.h"

/*Define the namespace*/
namespace DCORA {

/**
 * @brief This object represents a collection of "lifted" poses and translations X = [Y1 p1 ... Yn pn | l1 ... ln | r1 ... rn]
 * that can be used by ROPTLIB to perform Riemannian optimization
 */
class LiftedRAVariable : public LiftedSEVariable {
 public:
  /**
   * @brief Construct a default object
   * @param r relaxation rank
   * @param d dimension (2/3)
   * @param n number of poses
   * @param b number of landmarks
   * @param l number of ranges
   */
  LiftedRAVariable(unsigned int r, unsigned int d, unsigned int n, unsigned int b, unsigned int l);
  /**
   * @brief Constructor from lifted pose and translation array objects
   * @param poses
   */
  LiftedRAVariable(const LiftedPoseArray &poses, const LiftedTranslationArray &landmarks, const LiftedTranslationArray &ranges);
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
   * @brief Get number of landmarks
   * @return
   */
  unsigned int b() const { return b_; }
  /**
   * @brief Get number of ranges
   * @return
   */
  unsigned int l() const { return l_; }
  /**
   * @brief Obtain the variable as an ROPTLIB::ProductElement
   * @return
   */
  ROPTLIB::ProductElement *var() override { return varRA_.get(); }
  /**
   * @brief Obtain the variable as an Eigen matrix
   * @return r by (d+1)n+b+l matrix [Y1 p1 ... Yn pn | l1 ... ln | r1 ... rn]
   */
  Matrix getData() const override;
  /**
   * @brief Set this variable from an Eigen matrix
   * @param X r by (d+1)n+b+l matrix [Y1 p1 ... Yn pn | l1 ... ln | r1 ... rn]
   */
  void setData(const Matrix &X) override;
  /**
   * @brief Obtain the writable landmark translation at the specified index, expressed as an r dimensional vector
   * @param index
   * @return
   */
  Eigen::Ref<Vector> translationLandmark(unsigned int index);
  /**
   * @brief Obtain the read-only landmark translation at the specified index, expressed as an r dimensional vector
   * @param index
   * @return
   */
  Vector translationLandmark(unsigned int index) const;
  /**
   * @brief Obtain the writable range translation at the specified index, expressed as an r dimensional vector
   * @param index
   * @return
   */
  Eigen::Ref<Vector> translationRange(unsigned int index);
  /**
   * @brief Obtain the read-only range translation at the specified index, expressed as an r dimensional vector
   * @param index
   * @return
   */
  Vector translationRange(unsigned int index) const;

 private:
  // const dimensions
  unsigned int b_, l_;
  // The actual content of this variable is stored inside a ROPTLIB::ProductElement
  std::unique_ptr<ROPTLIB::EucVariable> translation_landmark_var_;
  std::unique_ptr<ROPTLIB::ObliqueVariable> translation_ranges_var_;
  std::unique_ptr<ROPTLIB::ProductElement> varE_;
  std::unique_ptr<ROPTLIB::ProductElement> varOB_;
  std::unique_ptr<ROPTLIB::ProductElement> varRA_;
  // Internal view of the landmark variable as an eigen matrix of dimension r-by-b
  Eigen::Map<Matrix> X_E_;
  // Internal view of the range variable as an eigen matrix of dimension r-by-l
  Eigen::Map<Matrix> X_OB_;
  // Internal view of the RA SLAM domain variable as an eigen matrix of dimension r-by-(d+1)n+b+l
  Eigen::Map<Matrix> X_RA_;
};

}  // namespace DCORA

#endif