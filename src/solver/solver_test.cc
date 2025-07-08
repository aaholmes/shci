#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <cmath>

// Test the epsilon schedule generation logic independently
class AutomatedEpsilonScheduleTest : public ::testing::Test {
 protected:
  // Simulate the automated epsilon schedule generation logic
  std::vector<double> generate_epsilon_schedule(double max_H_same, double max_H_opp, double eps_final) {
    // Calculate eps0 as minimum of the two maximum values scaled by factor
    const double eps0 = std::min(max_H_same, max_H_opp) * 0.1;
    
    // Calculate intermediate epsilon as geometric mean
    const double eps_intermediate = std::sqrt(eps0 * eps_final);
    
    // Create the three-stage schedule
    std::vector<double> schedule;
    
    // Only add intermediate stage if it's meaningfully different from endpoints  
    if (eps_intermediate > eps_final * 2.0 && eps_intermediate < eps0 * 0.5) {
      schedule = {eps0, eps_intermediate, eps_final};
    } else {
      schedule = {eps0, eps_final};
    }
    
    return schedule;
  }
};

TEST_F(AutomatedEpsilonScheduleTest, ThreeStageSchedule) {
  // Test with well-separated Hamiltonian values
  const double max_H_same = 0.05;
  const double max_H_opp = 0.08;
  const double eps_final = 1e-4;
  
  std::vector<double> schedule = generate_epsilon_schedule(max_H_same, max_H_opp, eps_final);
  
  // Should generate three-stage schedule
  EXPECT_EQ(schedule.size(), 3);
  EXPECT_EQ(schedule[2], eps_final);  // Last value should be eps_final
  
  // Schedule should be decreasing
  for (size_t i = 1; i < schedule.size(); i++) {
    EXPECT_LT(schedule[i], schedule[i-1]);
  }
  
  // eps0 should be 10% of minimum Hamiltonian value
  double expected_eps0 = std::min(max_H_same, max_H_opp) * 0.1;
  EXPECT_NEAR(schedule[0], expected_eps0, 1e-10);
  
  // Intermediate should be geometric mean
  double expected_intermediate = std::sqrt(expected_eps0 * eps_final);
  EXPECT_NEAR(schedule[1], expected_intermediate, 1e-10);
}

TEST_F(AutomatedEpsilonScheduleTest, TwoStageSchedule) {
  // Test with values that don't warrant intermediate stage
  const double max_H_same = 0.001;  // Smaller values
  const double max_H_opp = 0.002;
  const double eps_final = 1e-4;
  
  std::vector<double> schedule = generate_epsilon_schedule(max_H_same, max_H_opp, eps_final);
  
  // Should fall back to two-stage schedule
  EXPECT_EQ(schedule.size(), 2);
  EXPECT_EQ(schedule[1], eps_final);
  
  // eps0 should be 10% of minimum Hamiltonian value  
  double expected_eps0 = std::min(max_H_same, max_H_opp) * 0.1;
  EXPECT_NEAR(schedule[0], expected_eps0, 1e-10);
}

TEST_F(AutomatedEpsilonScheduleTest, DecreasingProperty) {
  // Test that schedule is always decreasing
  const double max_H_same = 0.1;
  const double max_H_opp = 0.15;
  const double eps_final = 1e-5;
  
  std::vector<double> schedule = generate_epsilon_schedule(max_H_same, max_H_opp, eps_final);
  
  // Verify strictly decreasing property
  for (size_t i = 1; i < schedule.size(); i++) {
    EXPECT_LT(schedule[i], schedule[i-1]) << "Schedule not decreasing at index " << i;
  }
}

TEST_F(AutomatedEpsilonScheduleTest, PhysicalRanges) {
  // Test with physically realistic Hamiltonian values
  const double max_H_same = 0.02;   // Typical same-spin value
  const double max_H_opp = 0.08;    // Typical opposite-spin value  
  const double eps_final = 1e-4;
  
  std::vector<double> schedule = generate_epsilon_schedule(max_H_same, max_H_opp, eps_final);
  
  // All values should be positive
  for (double eps : schedule) {
    EXPECT_GT(eps, 0.0);
  }
  
  // eps0 should be reasonable fraction of Hamiltonian values
  EXPECT_GT(schedule[0], eps_final);
  EXPECT_LT(schedule[0], std::max(max_H_same, max_H_opp));
}