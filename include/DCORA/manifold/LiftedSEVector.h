/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#ifndef LIFTEDSEVECTOR_H
#define LIFTEDSEVECTOR_H

#include <DCORA/DCORA_types.h>
#include <DCORA/DCORA_utils.h>

#include "Manifolds/Euclidean/Euclidean.h"
#include "Manifolds/ProductManifold.h"
#include "Manifolds/Stiefel/Stiefel.h"

/*Define the namespace*/
namespace DCORA {

/**
 * @brief This class represents a SE(n) synchronization vector
 */
class LiftedSEVector {
 public:
  /**
   * @brief Constructor
   * @param r
   * @param d
   * @param n
   */
  LiftedSEVector(int r, int d, int n);
  /**
   * @brief Destructor
   */
  virtual ~LiftedSEVector();
  /**
   * @brief Get underlying ROPTLIB vector
   * @return
   */
  virtual ROPTLIB::ProductElement* vec() { return MySEVector; }
  /**
   * @brief Get data as Eigen matrix
   * @return
   */
  virtual Matrix getData();
  /**
   * @brief Set data from Eigen matrix
   * @param X
   */
  virtual void setData(const Matrix& X);
  /**
   * @brief Get underlying ROPTLIB vector size as rows, columns, and number of elements, respectively
   * @param productElement
   * @param row
   * @param col
   * @param num_el
   */
  static void getSize(ROPTLIB::ProductElement* productElement, unsigned int &row, unsigned int &col, unsigned int &num_el);
  /**
   * @brief Set underlying ROPTLIB vector size
   * @return
   */
  virtual void setSize();

 protected:
  unsigned int r_, d_, n_;
  ROPTLIB::StieVector* StiefelVector;
  ROPTLIB::EucVector* EuclideanVector;
  ROPTLIB::ProductElement* CartanVector;
  ROPTLIB::ProductElement* MySEVector;
};
}  // namespace DCORA

#endif