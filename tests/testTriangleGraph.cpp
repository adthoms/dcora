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

#include <DCORA/Agent.h>

#include "gtest/gtest.h"

TEST(testDCORA, TriangleGraph) {
  unsigned int id = 0;
  unsigned int d, r;
  d = 3;
  r = 3;
  DCORA::PGOAgentParameters options(d, r, 1);
  DCORA::PGOAgent agent(id, options);

  DCORA::Matrix Tw0 = DCORA::Matrix::Identity(d + 1, d + 1);

  DCORA::Matrix Tw1(d + 1, d + 1);
  // clang-format off
  Tw1 << 0.1436,  0.7406, 0.6564, 1.0000,
        -0.8179, -0.2845, 0.5000, 1.0000,
         0.5571, -0.6087, 0.5649, 1.0000,
              0,       0,      0, 1;
  // clang-format on

  DCORA::Matrix Tw2(d + 1, d + 1);
  // clang-format off
  Tw2 << -0.4069, -0.4150, -0.8138, 2.0000,
          0.4049,  0.7166, -0.5679, 2.0000,
          0.8188, -0.5606, -0.1236, 2.0000,
               0,       0,       0, 1;
  // clang-format on

  DCORA::Matrix Ttrue(d, 3 * (d + 1));
  Ttrue << Tw0.block(0, 0, d, (d + 1)), Tw1.block(0, 0, d, (d + 1)),
      Tw2.block(0, 0, d, (d + 1));

  std::vector<DCORA::RelativePosePoseMeasurement> odometry;
  std::vector<DCORA::RelativePosePoseMeasurement> private_loop_closures;
  std::vector<DCORA::RelativePosePoseMeasurement> shared_loop_closures;

  DCORA::Matrix dT;
  dT = Tw0.inverse() * Tw1;
  DCORA::RelativePosePoseMeasurement m01(id, id, 0, 1, dT.block(0, 0, d, d),
                                         dT.block(0, d, d, 1), 1.0, 1.0);
  odometry.push_back(m01);

  dT = Tw1.inverse() * Tw2;
  DCORA::RelativePosePoseMeasurement m12(id, id, 1, 2, dT.block(0, 0, d, d),
                                         dT.block(0, d, d, 1), 1.0, 1.0);
  odometry.push_back(m12);

  dT = Tw0.inverse() * Tw2;
  DCORA::RelativePosePoseMeasurement m02(id, id, 0, 2, dT.block(0, 0, d, d),
                                         dT.block(0, d, d, 1), 1.0, 1.0);
  private_loop_closures.push_back(m02);

  agent.setMeasurements(odometry, private_loop_closures, shared_loop_closures);
  agent.initialize();

  DCORA::Matrix TLocal = agent.localPoseGraphOptimization();
  ASSERT_LE((Ttrue - TLocal).norm(), 1e-4);

  DCORA::Matrix T;
  agent.getTrajectoryInLocalFrame(&T);
  ASSERT_LE((Ttrue - T).norm(), 1e-4);

  agent.iterate();

  ASSERT_EQ(agent.getID(), id);
  ASSERT_EQ(agent.num_poses(), 3);
  ASSERT_EQ(agent.dimension(), d);
  ASSERT_EQ(agent.relaxation_rank(), r);

  agent.getTrajectoryInLocalFrame(&T);
  ASSERT_LE((Ttrue - T).norm(), 1e-4);
}
