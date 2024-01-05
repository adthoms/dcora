#include <DCORA/CORAAgent.h>

#include "gtest/gtest.h"

TEST(testDCORA, LineGraph) {
  unsigned int id = 0;
  unsigned int d, r;
  d = 3;
  r = 3;
  DCORA::PGOAgentParameters options(d, r, 1);

  DCORA::Matrix R = DCORA::Matrix::Identity(d, d);
  DCORA::Matrix t = DCORA::Matrix::Random(d, 1);

  std::vector<DCORA::RelativeSEMeasurement> odometry;
  std::vector<DCORA::RelativeSEMeasurement> private_loop_closures;
  std::vector<DCORA::RelativeSEMeasurement> shared_loop_closures;
  DCORA::PGOAgent agent(id, options);
  for (unsigned int i = 0; i < 4; ++i) {
    DCORA::RelativeSEMeasurement m(id, id, i, i + 1, R, t, 1.0, 1.0);
    odometry.push_back(m);
  }
  agent.setMeasurements(odometry, private_loop_closures, shared_loop_closures);
  agent.initialize();
  agent.iterate();

  ASSERT_EQ(agent.getID(), id);
  ASSERT_EQ(agent.num_poses(), 5);
  ASSERT_EQ(agent.dimension(), d);
  ASSERT_EQ(agent.relaxation_rank(), r);
}
