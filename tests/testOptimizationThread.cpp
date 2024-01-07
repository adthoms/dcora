#include <DCORA/CORAAgent.h>
#include <DCORA/DCORA_types.h>
#include <DCORA/DCORA_utils.h>
#include <DCORA/QuadraticProblem.h>

#include "gtest/gtest.h"

TEST(testDCORA, OptimizationThreadBasic) {
  unsigned int d, r;
  d = 3;
  r = 3;
  DCORA::PGOAgentParameters options(d, r, 1);

  DCORA::PGOAgent agent(0, options);

  ASSERT_FALSE(agent.isOptimizationRunning());

  for (unsigned trial = 0; trial < 3; ++trial) {
    agent.startOptimizationLoop();
    sleep(1);
    ASSERT_TRUE(agent.isOptimizationRunning());
    agent.endOptimizationLoop();
    ASSERT_FALSE(agent.isOptimizationRunning());
  }
}

TEST(testDCORA, OptimizationThreadTriangleGraph) {
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
  Ttrue << Tw0, Tw1, Tw2;

  std::vector<DCORA::RelativeSEMeasurement> odometry;
  std::vector<DCORA::RelativeSEMeasurement> private_loop_closures;
  std::vector<DCORA::RelativeSEMeasurement> shared_loop_closures;

  DCORA::Matrix dT;
  dT = Tw0.inverse() * Tw1;
  DCORA::RelativeSEMeasurement m01(id, id, 0, 1, dT.block(0, 0, d, d),
                                   dT.block(0, d, d, 1), 1.0, 1.0);
  odometry.push_back(m01);

  dT = Tw1.inverse() * Tw2;
  DCORA::RelativeSEMeasurement m12(id, id, 1, 2, dT.block(0, 0, d, d),
                                   dT.block(0, d, d, 1), 1.0, 1.0);
  odometry.push_back(m12);

  dT = Tw0.inverse() * Tw2;
  DCORA::RelativeSEMeasurement m02(id, id, 0, 2, dT.block(0, 0, d, d),
                                   dT.block(0, d, d, 1), 1.0, 1.0);
  private_loop_closures.push_back(m02);

  agent.setMeasurements(odometry, private_loop_closures, shared_loop_closures);
  agent.initialize();

  DCORA::Matrix Testimated;
  agent.getTrajectoryInLocalFrame(&Testimated);
  ASSERT_LE((Ttrue - Testimated).norm(), 1e-4);

  agent.startOptimizationLoop();
  sleep(3);
  agent.endOptimizationLoop();

  ASSERT_EQ(agent.getID(), id);
  ASSERT_EQ(agent.num_poses(), 3);
  ASSERT_EQ(agent.dimension(), d);
  ASSERT_EQ(agent.relaxation_rank(), r);

  agent.getTrajectoryInLocalFrame(&Testimated);
  ASSERT_LE((Ttrue - Testimated).norm(), 1e-4);
}
