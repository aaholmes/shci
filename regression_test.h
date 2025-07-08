/**
 * @file regression_test.h
 * @brief Regression testing framework for SHCI calculations
 * 
 * This framework compares SHCI results against reference calculations
 * to ensure accuracy and detect regressions in the algorithm.
 */

#pragma once

#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include "../config.h"
#include "../result.h"

/**
 * @class RegressionTest
 * @brief Framework for regression testing against reference data
 */
class RegressionTest : public ::testing::Test {
 protected:
  void SetUp() override;
  void TearDown() override;
  
 public:
  /**
   * @struct ReferenceData
   * @brief Structure to hold reference calculation data
   */
  struct ReferenceData {
    std::string system_name;
    std::map<std::string, double> energy_var;  // eps_var -> energy
    double energy_hf;
    int n_elecs;
    Config config;
    
    // Tolerance settings
    double energy_tolerance = 1e-6;   // Hartree
    double relative_tolerance = 1e-8; // Relative error
  };
  
  /**
   * @struct TestResult
   * @brief Results from a regression test
   */
  struct TestResult {
    bool passed = true;
    std::string system_name;
    std::vector<std::string> errors;
    std::vector<std::string> warnings;
    std::map<std::string, double> energy_differences;
    double max_energy_difference = 0.0;
    double max_relative_difference = 0.0;
  };
  
  /**
   * @brief Load reference data from electronic-structure-tests directory
   */
  static std::vector<ReferenceData> load_reference_data(const std::string& test_dir);
  
  /**
   * @brief Load single reference calculation
   */
  static ReferenceData load_single_reference(const std::string& system_dir);
  
  /**
   * @brief Run SHCI calculation for comparison
   */
  static TestResult run_regression_test(const ReferenceData& reference);
  
  /**
   * @brief Compare calculated energies with reference
   */
  static TestResult compare_energies(const ReferenceData& reference, 
                                   const Result& calculated);
  
  /**
   * @brief Generate detailed test report
   */
  static std::string generate_test_report(const std::vector<TestResult>& results);
  
  /**
   * @brief Run full regression test suite
   */
  static std::vector<TestResult> run_full_suite(const std::string& test_dir);
  
  /**
   * @brief Run subset of tests (fast mode)
   */
  static std::vector<TestResult> run_fast_suite(const std::string& test_dir);
  
 private:
  /**
   * @brief Helper to parse JSON result files
   */
  static ReferenceData parse_reference_json(const std::string& json_file);
  
  /**
   * @brief Helper to determine appropriate tolerances
   */
  static void set_tolerances(ReferenceData& reference);
  
  /**
   * @brief Check if system should be included in fast suite
   */
  static bool is_fast_system(const std::string& system_name);
};

/**
 * @class PerformanceRegressionTest
 * @brief Test for performance regressions
 */
class PerformanceRegressionTest {
 public:
  /**
   * @struct PerformanceBenchmark
   * @brief Performance benchmark data
   */
  struct PerformanceBenchmark {
    std::string system_name;
    double reference_time;      // seconds
    size_t reference_memory;    // bytes
    int reference_iterations;
    size_t reference_determinants;
    
    // Tolerance for performance regression
    double time_tolerance = 1.5;      // Allow 50% slowdown
    double memory_tolerance = 1.5;    // Allow 50% more memory
  };
  
  /**
   * @brief Run performance regression tests
   */
  static std::vector<TestResult> run_performance_tests(const std::string& test_dir);
  
  /**
   * @brief Compare performance metrics
   */
  static TestResult compare_performance(const PerformanceBenchmark& benchmark,
                                      const Result& calculated);
};

/**
 * @class RobustnessTest
 * @brief Test robustness against parameter variations
 */
class RobustnessTest {
 public:
  /**
   * @brief Test parameter sensitivity
   */
  static std::vector<TestResult> test_parameter_sensitivity(const ReferenceData& reference);
  
  /**
   * @brief Test numerical stability
   */
  static TestResult test_numerical_stability(const ReferenceData& reference);
  
  /**
   * @brief Test convergence behavior
   */
  static TestResult test_convergence_behavior(const ReferenceData& reference);
};

// Utility macros for regression testing

/**
 * @brief Assert energy matches reference within tolerance
 */
#define ASSERT_ENERGY_NEAR(calculated, reference, tolerance) \
  ASSERT_NEAR(calculated, reference, tolerance) \
    << "Energy difference: " << std::abs(calculated - reference) \
    << " Ha exceeds tolerance: " << tolerance << " Ha"

/**
 * @brief Assert relative energy error is within tolerance
 */
#define ASSERT_RELATIVE_ENERGY_NEAR(calculated, reference, tolerance) \
  do { \
    double rel_error = std::abs((calculated - reference) / reference); \
    ASSERT_LT(rel_error, tolerance) \
      << "Relative energy error: " << rel_error \
      << " exceeds tolerance: " << tolerance; \
  } while(0)

/**
 * @brief Expect energy to be within tolerance (non-fatal)
 */
#define EXPECT_ENERGY_NEAR(calculated, reference, tolerance) \
  EXPECT_NEAR(calculated, reference, tolerance) \
    << "Energy difference: " << std::abs(calculated - reference) \
    << " Ha exceeds tolerance: " << tolerance << " Ha"