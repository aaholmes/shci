/**
 * @file timer_test.cc
 * @brief Unit tests for Timer class
 */

#include <gtest/gtest.h>
#include <chrono>
#include <thread>
#include "../timer.h"

/**
 * @class TimerTest
 * @brief Test fixture for Timer class
 */
class TimerTest : public ::testing::Test {
 protected:
  Timer timer;
  
  void SetUp() override {
    timer.reset();
  }
};

/**
 * @brief Test basic timer functionality
 */
TEST_F(TimerTest, BasicTiming) {
  timer.start("test_operation");
  
  // Simulate some work
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  
  timer.end("test_operation");
  
  double elapsed = timer.get("test_operation");
  EXPECT_GT(elapsed, 0.005);  // Should be at least 5ms
  EXPECT_LT(elapsed, 0.100);  // Should be less than 100ms
}

/**
 * @brief Test multiple timers
 */
TEST_F(TimerTest, MultipleTimers) {
  timer.start("operation1");
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  timer.end("operation1");
  
  timer.start("operation2");
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  timer.end("operation2");
  
  double time1 = timer.get("operation1");
  double time2 = timer.get("operation2");
  
  EXPECT_GT(time1, 0.0);
  EXPECT_GT(time2, 0.0);
  EXPECT_GT(time2, time1);  // operation2 should take longer
}

/**
 * @brief Test nested timers
 */
TEST_F(TimerTest, NestedTimers) {
  timer.start("outer");
  
  timer.start("inner1");
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  timer.end("inner1");
  
  timer.start("inner2");
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  timer.end("inner2");
  
  timer.end("outer");
  
  double outer_time = timer.get("outer");
  double inner1_time = timer.get("inner1");
  double inner2_time = timer.get("inner2");
  
  EXPECT_GT(outer_time, inner1_time);
  EXPECT_GT(outer_time, inner2_time);
  EXPECT_GT(outer_time, inner1_time + inner2_time * 0.8);  // Accounting for overhead
}

/**
 * @brief Test timer accumulation
 */
TEST_F(TimerTest, TimerAccumulation) {
  // Run the same operation multiple times
  for (int i = 0; i < 3; ++i) {
    timer.start("repeated_operation");
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    timer.end("repeated_operation");
  }
  
  double total_time = timer.get("repeated_operation");
  EXPECT_GT(total_time, 0.010);  // Should be at least 10ms total
  
  int count = timer.get_count("repeated_operation");
  EXPECT_EQ(count, 3);
  
  double avg_time = timer.get_average("repeated_operation");
  EXPECT_NEAR(avg_time, total_time / 3.0, 0.001);
}

/**
 * @brief Test timer statistics
 */
TEST_F(TimerTest, TimerStatistics) {
  // Create timers with different durations
  timer.start("fast_op");
  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  timer.end("fast_op");
  
  timer.start("slow_op");
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  timer.end("slow_op");
  
  timer.start("medium_op");
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  timer.end("medium_op");
  
  // Test percentage calculations
  double total_time = timer.get_total_time();
  double slow_percentage = timer.get_percentage("slow_op");
  
  EXPECT_GT(total_time, 0.025);  // At least 25ms total
  EXPECT_GT(slow_percentage, 50.0);  // slow_op should be >50% of total
  EXPECT_LT(slow_percentage, 80.0);  // but <80% due to overhead
}

/**
 * @brief Test timer reset functionality
 */
TEST_F(TimerTest, TimerReset) {
  timer.start("test_op");
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  timer.end("test_op");
  
  EXPECT_GT(timer.get("test_op"), 0.0);
  
  timer.reset();
  
  // After reset, timer should return 0 or throw
  EXPECT_EQ(timer.get("test_op"), 0.0);
  EXPECT_EQ(timer.get_count("test_op"), 0);
}

/**
 * @brief Test timer reporting
 */
TEST_F(TimerTest, TimerReporting) {
  timer.start("operation_a");
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  timer.end("operation_a");
  
  timer.start("operation_b");
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  timer.end("operation_b");
  
  // Test string report generation
  std::string report = timer.get_report();
  EXPECT_TRUE(report.find("operation_a") != std::string::npos);
  EXPECT_TRUE(report.find("operation_b") != std::string::npos);
  
  // Test summary report
  std::string summary = timer.get_summary();
  EXPECT_TRUE(summary.find("Total") != std::string::npos);
}

/**
 * @brief Test timer error handling
 */
TEST_F(TimerTest, ErrorHandling) {
  // Test getting time for non-existent timer
  EXPECT_EQ(timer.get("nonexistent"), 0.0);
  
  // Test ending timer that wasn't started
  EXPECT_NO_THROW(timer.end("not_started"));
  
  // Test starting already running timer
  timer.start("test_timer");
  EXPECT_NO_THROW(timer.start("test_timer"));  // Should handle gracefully
  timer.end("test_timer");
}

/**
 * @brief Test high-resolution timing
 */
TEST_F(TimerTest, HighResolutionTiming) {
  timer.start("microsecond_test");
  
  // Very short operation
  volatile int sum = 0;
  for (int i = 0; i < 1000; ++i) {
    sum += i;
  }
  
  timer.end("microsecond_test");
  
  double elapsed = timer.get("microsecond_test");
  EXPECT_GT(elapsed, 0.0);  // Should detect even very short operations
  EXPECT_LT(elapsed, 0.01); // But should be less than 10ms
}

/**
 * @brief Test timer thread safety (basic test)
 */
TEST_F(TimerTest, ThreadSafety) {
  // Simple test with concurrent timers
  std::thread t1([&]() {
    timer.start("thread1");
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    timer.end("thread1");
  });
  
  std::thread t2([&]() {
    timer.start("thread2");
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    timer.end("thread2");
  });
  
  t1.join();
  t2.join();
  
  // Both timers should have recorded time
  EXPECT_GT(timer.get("thread1"), 0.0);
  EXPECT_GT(timer.get("thread2"), 0.0);
}

/**
 * @brief Test timer serialization/output
 */
TEST_F(TimerTest, Serialization) {
  timer.start("serialize_test");
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  timer.end("serialize_test");
  
  // Test JSON output if available
  std::string json_output = timer.to_json();
  EXPECT_TRUE(json_output.find("serialize_test") != std::string::npos);
  
  // Test CSV output if available
  std::string csv_output = timer.to_csv();
  EXPECT_TRUE(csv_output.find("serialize_test") != std::string::npos);
}