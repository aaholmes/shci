/**
 * @file util_test.cc
 * @brief Unit tests for utility functions
 */

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include "../util.h"

/**
 * @class UtilTest
 * @brief Test fixture for utility functions
 */
class UtilTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Setup test data if needed
  }
};

/**
 * @brief Test string utilities
 */
TEST_F(UtilTest, StringUtilities) {
  // Test string formatting utilities
  std::string result = Util::format("Test %d %s", 42, "string");
  EXPECT_TRUE(result.find("42") != std::string::npos);
  EXPECT_TRUE(result.find("string") != std::string::npos);
  
  // Test string trimming
  std::string trimmed = Util::trim("  hello world  ");
  EXPECT_EQ(trimmed, "hello world");
  
  // Test empty string trimming
  std::string empty_trimmed = Util::trim("   ");
  EXPECT_EQ(empty_trimmed, "");
}

/**
 * @brief Test file I/O utilities
 */
TEST_F(UtilTest, FileIOUtilities) {
  // Create temporary test file
  std::string test_content = "line1\nline2\nline3\n";
  EXPECT_NO_THROW(Util::write_file("test_util_temp.txt", test_content));
  
  // Test file reading
  std::string read_content;
  EXPECT_NO_THROW(read_content = Util::read_file("test_util_temp.txt"));
  EXPECT_EQ(read_content, test_content);
  
  // Test file existence check
  EXPECT_TRUE(Util::file_exists("test_util_temp.txt"));
  EXPECT_FALSE(Util::file_exists("nonexistent_file.txt"));
  
  // Clean up
  std::remove("test_util_temp.txt");
}

/**
 * @brief Test numerical utilities
 */
TEST_F(UtilTest, NumericalUtilities) {
  // Test floating point comparison
  EXPECT_TRUE(Util::approx_equal(1.0, 1.0000001, 1e-5));
  EXPECT_FALSE(Util::approx_equal(1.0, 1.001, 1e-5));
  
  // Test range checking
  EXPECT_TRUE(Util::in_range(5, 1, 10));
  EXPECT_FALSE(Util::in_range(15, 1, 10));
  EXPECT_FALSE(Util::in_range(-5, 1, 10));
  
  // Test statistical functions
  std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
  EXPECT_DOUBLE_EQ(Util::mean(data), 3.0);
  EXPECT_DOUBLE_EQ(Util::variance(data), 2.0);
  EXPECT_NEAR(Util::std_dev(data), sqrt(2.0), 1e-10);
}

/**
 * @brief Test time utilities
 */
TEST_F(UtilTest, TimeUtilities) {
  auto start = Util::get_wall_time();
  
  // Small delay
  for (volatile int i = 0; i < 100000; ++i) {}
  
  auto end = Util::get_wall_time();
  auto elapsed = Util::elapsed_seconds(start, end);
  
  // Should have taken some time, but not too much
  EXPECT_GT(elapsed, 0.0);
  EXPECT_LT(elapsed, 1.0);  // Should complete in less than 1 second
}

/**
 * @brief Test memory utilities
 */
TEST_F(UtilTest, MemoryUtilities) {
  // Test memory usage reporting
  size_t memory_usage = Util::get_memory_usage();
  EXPECT_GT(memory_usage, 0);  // Should report some memory usage
  
  // Test memory formatting
  std::string formatted = Util::format_memory(1024 * 1024);  // 1 MB
  EXPECT_TRUE(formatted.find("MB") != std::string::npos);
  
  formatted = Util::format_memory(1024 * 1024 * 1024);  // 1 GB
  EXPECT_TRUE(formatted.find("GB") != std::string::npos);
}

/**
 * @brief Test array utilities
 */
TEST_F(UtilTest, ArrayUtilities) {
  std::vector<int> vec = {3, 1, 4, 1, 5, 9, 2, 6};
  
  // Test sorting
  auto sorted = Util::sort_indices(vec);
  EXPECT_EQ(vec[sorted[0]], 1);  // First element should be 1
  EXPECT_EQ(vec[sorted.back()], 9);  // Last element should be 9
  
  // Test finding maximum
  auto max_it = Util::max_element(vec);
  EXPECT_EQ(*max_it, 9);
  
  // Test finding minimum
  auto min_it = Util::min_element(vec);
  EXPECT_EQ(*min_it, 1);
}

/**
 * @brief Test random number utilities
 */
TEST_F(UtilTest, RandomUtilities) {
  // Test random number generation
  Util::set_random_seed(12345);
  
  double rand1 = Util::random_double(0.0, 1.0);
  double rand2 = Util::random_double(0.0, 1.0);
  
  EXPECT_GE(rand1, 0.0);
  EXPECT_LE(rand1, 1.0);
  EXPECT_GE(rand2, 0.0);
  EXPECT_LE(rand2, 1.0);
  EXPECT_NE(rand1, rand2);  // Should be different
  
  // Test random integers
  int rand_int = Util::random_int(1, 100);
  EXPECT_GE(rand_int, 1);
  EXPECT_LE(rand_int, 100);
}

/**
 * @brief Test error handling utilities
 */
TEST_F(UtilTest, ErrorHandling) {
  // Test assertion utilities
  EXPECT_NO_THROW(Util::ensure(true, "This should not throw"));
  EXPECT_THROW(Util::ensure(false, "This should throw"), std::runtime_error);
  
  // Test warning utilities
  EXPECT_NO_THROW(Util::warn("Test warning message"));
  
  // Test error formatting
  std::string error_msg = Util::format_error("Test error", "test_function", 123);
  EXPECT_TRUE(error_msg.find("Test error") != std::string::npos);
  EXPECT_TRUE(error_msg.find("test_function") != std::string::npos);
  EXPECT_TRUE(error_msg.find("123") != std::string::npos);
}

/**
 * @brief Test command line utilities
 */
TEST_F(UtilTest, CommandLineUtilities) {
  // Test argument parsing
  std::vector<std::string> args = {"program", "--verbose", "--output=file.txt", "input.txt"};
  
  EXPECT_TRUE(Util::has_flag(args, "--verbose"));
  EXPECT_FALSE(Util::has_flag(args, "--quiet"));
  
  std::string output_file = Util::get_option(args, "--output");
  EXPECT_EQ(output_file, "file.txt");
  
  std::string missing_option = Util::get_option(args, "--missing", "default");
  EXPECT_EQ(missing_option, "default");
}

/**
 * @brief Test platform-specific utilities
 */
TEST_F(UtilTest, PlatformUtilities) {
  // Test environment variable access
  Util::set_env_var("TEST_VAR", "test_value");
  std::string env_value = Util::get_env_var("TEST_VAR", "default");
  EXPECT_EQ(env_value, "test_value");
  
  std::string missing_env = Util::get_env_var("MISSING_VAR", "default");
  EXPECT_EQ(missing_env, "default");
  
  // Test CPU information
  int num_cores = Util::get_num_cores();
  EXPECT_GT(num_cores, 0);
  EXPECT_LE(num_cores, 256);  // Reasonable upper bound
}