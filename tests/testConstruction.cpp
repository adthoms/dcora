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

TEST(testDCORA, Construction) {
  unsigned int id = 1;
  unsigned int d, r;
  d = 3;
  r = 3;
  DCORA::AgentParameters options(d, r, 1);

  DCORA::Agent agent(id, options);

  ASSERT_EQ(agent.getID(), id);
  ASSERT_EQ(agent.num_poses(), 0);
  ASSERT_EQ(agent.dimension(), d);
  ASSERT_EQ(agent.relaxation_rank(), r);
}
