#include <DCORA/Agent.h>

#include "gtest/gtest.h"

TEST(testDCORA, Construction) {
  unsigned int id = 1;
  unsigned int d, r;
  d = 3;
  r = 3;
  DCORA::PGOAgentParameters options(d, r, 1);

  DCORA::PGOAgent agent(id, options);

  ASSERT_EQ(agent.getID(), id);
  ASSERT_EQ(agent.num_poses(), 0);
  ASSERT_EQ(agent.dimension(), d);
  ASSERT_EQ(agent.relaxation_rank(), r);
}
