/**
 * @file regression_test.cc
 * @brief Implementation of regression testing framework
 */

#include "regression_test.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <json/json.h>
#include "../solver/solver.h"
#include "../chem/chem_system.h"
#include "../timer.h"

void RegressionTest::SetUp() {
  // Initialize MPI if needed
  int mpi_initialized;
  MPI_Initialized(&mpi_initialized);
  if (!mpi_initialized) {
    int argc = 0;
    char** argv = nullptr;
    MPI_Init(&argc, &argv);
  }
  
  // Set up test environment
  std::filesystem::create_directories("regression_test_temp");
}

void RegressionTest::TearDown() {
  // Clean up test files
  std::filesystem::remove_all("regression_test_temp");
}

std::vector<RegressionTest::ReferenceData> RegressionTest::load_reference_data(const std::string& test_dir) {
  std::vector<ReferenceData> references;
  
  // Get list of system directories
  for (const auto& entry : std::filesystem::directory_iterator(test_dir)) {
    if (entry.is_directory()) {
      std::string system_name = entry.path().filename().string();
      
      // Skip certain directories
      if (system_name == "proton_transfer" || system_name == "results" || 
          system_name.find(".") == 0) {
        continue;
      }
      
      try {
        ReferenceData ref = load_single_reference(entry.path().string());
        ref.system_name = system_name;
        set_tolerances(ref);
        references.push_back(ref);
      } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to load reference data for " << system_name 
                  << ": " << e.what() << std::endl;
      }
    }
  }
  
  // Also load proton transfer systems
  std::string proton_dir = test_dir + "/proton_transfer";
  if (std::filesystem::exists(proton_dir)) {
    for (const auto& entry : std::filesystem::directory_iterator(proton_dir)) {
      if (entry.is_directory()) {
        std::string system_name = "pt_" + entry.path().filename().string();
        
        try {
          ReferenceData ref = load_single_reference(entry.path().string());
          ref.system_name = system_name;
          set_tolerances(ref);
          references.push_back(ref);
        } catch (const std::exception& e) {
          std::cerr << "Warning: Failed to load reference data for " << system_name 
                    << ": " << e.what() << std::endl;
        }
      }
    }
  }
  
  return references;
}

RegressionTest::ReferenceData RegressionTest::load_single_reference(const std::string& system_dir) {
  ReferenceData reference;
  
  // Load result.json
  std::string result_file = system_dir + "/result.json";
  if (!std::filesystem::exists(result_file)) {
    throw std::runtime_error("No result.json found in " + system_dir);
  }
  
  reference = parse_reference_json(result_file);
  
  // Load config.json
  std::string config_file = system_dir + "/config.json";
  if (std::filesystem::exists(config_file)) {
    reference.config.load(config_file);
  }
  
  return reference;
}

RegressionTest::ReferenceData RegressionTest::parse_reference_json(const std::string& json_file) {
  ReferenceData reference;
  
  std::ifstream file(json_file);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open " + json_file);
  }
  
  Json::Value root;
  Json::CharReaderBuilder builder;
  std::string errs;
  
  if (!Json::parseFromStream(builder, file, &root, &errs)) {
    throw std::runtime_error("JSON parse error in " + json_file + ": " + errs);
  }
  
  // Extract energy data
  if (root.isMember("energy_hf")) {
    reference.energy_hf = root["energy_hf"].asDouble();
  }
  
  if (root.isMember("n_elecs")) {
    reference.n_elecs = root["n_elecs"].asInt();
  }
  
  if (root.isMember("energy_var")) {
    const Json::Value& energy_var = root["energy_var"];
    for (const auto& key : energy_var.getMemberNames()) {
      reference.energy_var[key] = energy_var[key].asDouble();
    }
  }
  
  // Extract configuration if present
  if (root.isMember("config")) {
    const Json::Value& config = root["config"];
    
    if (config.isMember("n_up")) {
      reference.config.set("n_up", config["n_up"].asInt());
    }
    if (config.isMember("n_dn")) {
      reference.config.set("n_dn", config["n_dn"].asInt());
    }
    if (config.isMember("system")) {
      reference.config.set("system", config["system"].asString());
    }
    if (config.isMember("eps_vars")) {
      std::vector<double> eps_vars;
      for (const auto& eps : config["eps_vars"]) {
        eps_vars.push_back(eps.asDouble());
      }
      reference.config.set("eps_vars", eps_vars);
    }
  }
  
  return reference;
}

void RegressionTest::set_tolerances(ReferenceData& reference) {
  // Set tolerances based on system size and reference energy
  int n_elecs = reference.n_elecs;
  
  if (n_elecs <= 4) {
    // Small systems: very tight tolerances
    reference.energy_tolerance = 1e-8;
    reference.relative_tolerance = 1e-10;
  } else if (n_elecs <= 10) {
    // Medium systems: standard tolerances
    reference.energy_tolerance = 1e-6;
    reference.relative_tolerance = 1e-8;
  } else {
    // Large systems: relaxed tolerances
    reference.energy_tolerance = 1e-5;
    reference.relative_tolerance = 1e-7;
  }
  
  // Adjust for system-specific challenges
  if (reference.system_name.find("he2") != std::string::npos) {
    // He2 is very weakly bound, relax tolerances
    reference.energy_tolerance *= 10;
    reference.relative_tolerance *= 10;
  }
  
  if (reference.system_name.find("f2") != std::string::npos ||
      reference.system_name.find("o2") != std::string::npos) {
    // Open-shell systems may have looser convergence
    reference.energy_tolerance *= 5;
    reference.relative_tolerance *= 5;
  }
}

bool RegressionTest::is_fast_system(const std::string& system_name) {
  // Define which systems are fast enough for routine testing
  std::vector<std::string> fast_systems = {
    "be", "be2", "he2", "li2", "lih", "ch_radical"
  };
  
  return std::find(fast_systems.begin(), fast_systems.end(), system_name) != fast_systems.end();
}

RegressionTest::TestResult RegressionTest::run_regression_test(const ReferenceData& reference) {
  TestResult result;
  result.system_name = reference.system_name;
  
  try {
    // Create test directory
    std::string test_dir = "regression_test_temp/" + reference.system_name;
    std::filesystem::create_directories(test_dir);
    
    // Copy FCIDUMP file
    std::string original_test_dir = "../electronic-structure-tests";
    if (reference.system_name.find("pt_") == 0) {
      original_test_dir += "/proton_transfer/" + reference.system_name.substr(3);
    } else {
      original_test_dir += "/" + reference.system_name;
    }
    
    std::filesystem::copy_file(original_test_dir + "/FCIDUMP", 
                              test_dir + "/FCIDUMP");
    
    // Create configuration for test
    Config test_config = reference.config;
    
    // Use a subset of eps_vars for faster testing
    auto eps_vars = reference.config.get<std::vector<double>>("eps_vars", {1e-4});
    if (eps_vars.size() > 2) {
      eps_vars.resize(2);  // Only use first 2 for regression testing
      test_config.set("eps_vars", eps_vars);
    }
    
    // Save test configuration
    test_config.save(test_dir + "/config.json");
    
    // Change to test directory and run calculation
    std::filesystem::current_path(test_dir);
    
    Timer timer;
    timer.start("total_calculation");
    
    // Run SHCI calculation
    if (reference.config.get<std::string>("system", "chem") == "chem") {
      Solver<ChemSystem> solver;
      solver.run();
    } else {
      result.errors.push_back("HEG system testing not yet implemented");
      result.passed = false;
      return result;
    }
    
    timer.end("total_calculation");
    
    // Load and compare results
    if (std::filesystem::exists("result.json")) {
      Result calculated_result;
      calculated_result.load("result.json");
      result = compare_energies(reference, calculated_result);
    } else {
      result.errors.push_back("No result.json generated");
      result.passed = false;
    }
    
    // Add timing information
    double calc_time = timer.get("total_calculation");
    if (calc_time > 300.0) {  // 5 minutes
      result.warnings.push_back("Calculation took longer than expected: " + 
                               std::to_string(calc_time) + " seconds");
    }
    
    // Return to original directory
    std::filesystem::current_path("../..");
    
  } catch (const std::exception& e) {
    result.passed = false;
    result.errors.push_back("Calculation failed: " + std::string(e.what()));
  }
  
  return result;
}

RegressionTest::TestResult RegressionTest::compare_energies(const ReferenceData& reference, 
                                                          const Result& calculated) {
  TestResult result;
  result.system_name = reference.system_name;
  result.passed = true;
  
  // Compare HF energy if available
  if (reference.energy_hf != 0.0 && calculated.energy_hf != 0.0) {
    double hf_diff = std::abs(calculated.energy_hf - reference.energy_hf);
    if (hf_diff > reference.energy_tolerance) {
      result.errors.push_back("HF energy difference: " + std::to_string(hf_diff) + 
                             " Ha exceeds tolerance: " + std::to_string(reference.energy_tolerance));
      result.passed = false;
    }
  }
  
  // Compare variational energies
  for (const auto& ref_pair : reference.energy_var) {
    std::string eps_str = ref_pair.first;
    double ref_energy = ref_pair.second;
    
    // Find corresponding calculated energy
    auto calc_it = calculated.energy_var.find(eps_str);
    if (calc_it == calculated.energy_var.end()) {
      result.warnings.push_back("No calculated energy for eps_var = " + eps_str);
      continue;
    }
    
    double calc_energy = calc_it->second;
    double energy_diff = std::abs(calc_energy - ref_energy);
    double rel_diff = energy_diff / std::abs(ref_energy);
    
    result.energy_differences[eps_str] = energy_diff;
    result.max_energy_difference = std::max(result.max_energy_difference, energy_diff);
    result.max_relative_difference = std::max(result.max_relative_difference, rel_diff);
    
    // Check tolerances
    if (energy_diff > reference.energy_tolerance) {
      result.errors.push_back("Energy difference for " + eps_str + ": " + 
                             std::to_string(energy_diff) + " Ha exceeds tolerance: " + 
                             std::to_string(reference.energy_tolerance) + " Ha");
      result.passed = false;
    }
    
    if (rel_diff > reference.relative_tolerance) {
      result.errors.push_back("Relative energy error for " + eps_str + ": " + 
                             std::to_string(rel_diff) + " exceeds tolerance: " + 
                             std::to_string(reference.relative_tolerance));
      result.passed = false;
    }
  }
  
  return result;
}

std::vector<RegressionTest::TestResult> RegressionTest::run_full_suite(const std::string& test_dir) {
  std::vector<TestResult> results;
  
  // Load all reference data
  auto references = load_reference_data(test_dir);
  
  std::cout << "Running full regression test suite (" << references.size() << " systems)..." << std::endl;
  
  // Run tests for each system
  for (const auto& reference : references) {
    std::cout << "Testing " << reference.system_name << "..." << std::flush;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    TestResult result = run_regression_test(reference);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    
    if (result.passed) {
      std::cout << " PASSED (" << duration.count() << "s)" << std::endl;
    } else {
      std::cout << " FAILED (" << duration.count() << "s)" << std::endl;
      for (const auto& error : result.errors) {
        std::cout << "  ERROR: " << error << std::endl;
      }
    }
    
    results.push_back(result);
  }
  
  return results;
}

std::vector<RegressionTest::TestResult> RegressionTest::run_fast_suite(const std::string& test_dir) {
  std::vector<TestResult> results;
  
  // Load reference data
  auto all_references = load_reference_data(test_dir);
  
  // Filter to fast systems only
  std::vector<ReferenceData> fast_references;
  for (const auto& ref : all_references) {
    if (is_fast_system(ref.system_name)) {
      fast_references.push_back(ref);
    }
  }
  
  std::cout << "Running fast regression test suite (" << fast_references.size() 
            << " systems)..." << std::endl;
  
  // Run tests
  for (const auto& reference : fast_references) {
    std::cout << "Testing " << reference.system_name << "..." << std::flush;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    TestResult result = run_regression_test(reference);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    
    if (result.passed) {
      std::cout << " PASSED (" << duration.count() << "s)" << std::endl;
    } else {
      std::cout << " FAILED (" << duration.count() << "s)" << std::endl;
    }
    
    results.push_back(result);
  }
  
  return results;
}

std::string RegressionTest::generate_test_report(const std::vector<TestResult>& results) {
  std::ostringstream report;
  
  // Header
  report << "=== SHCI Regression Test Report ===" << std::endl;
  report << "Date: " << std::chrono::system_clock::now() << std::endl;
  report << "Total systems tested: " << results.size() << std::endl;
  
  // Summary statistics
  int passed = 0, failed = 0;
  double max_energy_error = 0.0;
  double max_relative_error = 0.0;
  
  for (const auto& result : results) {
    if (result.passed) {
      passed++;
    } else {
      failed++;
    }
    max_energy_error = std::max(max_energy_error, result.max_energy_difference);
    max_relative_error = std::max(max_relative_error, result.max_relative_difference);
  }
  
  report << "Passed: " << passed << ", Failed: " << failed << std::endl;
  report << "Maximum energy error: " << max_energy_error << " Ha" << std::endl;
  report << "Maximum relative error: " << max_relative_error << std::endl;
  report << std::endl;
  
  // Failed tests details
  if (failed > 0) {
    report << "=== Failed Tests ===" << std::endl;
    for (const auto& result : results) {
      if (!result.passed) {
        report << "System: " << result.system_name << std::endl;
        for (const auto& error : result.errors) {
          report << "  ERROR: " << error << std::endl;
        }
        for (const auto& warning : result.warnings) {
          report << "  WARNING: " << warning << std::endl;
        }
        report << std::endl;
      }
    }
  }
  
  // Energy accuracy summary
  report << "=== Energy Accuracy Summary ===" << std::endl;
  report << std::setw(15) << "System" << std::setw(15) << "Max Error (Ha)" 
         << std::setw(15) << "Max Rel Error" << std::setw(10) << "Status" << std::endl;
  report << std::string(55, '-') << std::endl;
  
  for (const auto& result : results) {
    report << std::setw(15) << result.system_name 
           << std::setw(15) << std::scientific << std::setprecision(2) 
           << result.max_energy_difference
           << std::setw(15) << result.max_relative_difference
           << std::setw(10) << (result.passed ? "PASS" : "FAIL") << std::endl;
  }
  
  return report.str();
}

// Google Test integration

/**
 * @brief Individual test for each reference system
 */
class RegressionTestParameterized : public RegressionTest, 
                                   public ::testing::WithParamInterface<RegressionTest::ReferenceData> {
};

TEST_P(RegressionTestParameterized, SystemRegression) {
  ReferenceData reference = GetParam();
  TestResult result = run_regression_test(reference);
  
  EXPECT_TRUE(result.passed) << "Regression test failed for " << reference.system_name;
  
  if (!result.passed) {
    for (const auto& error : result.errors) {
      ADD_FAILURE() << error;
    }
  }
  
  // Check specific energy tolerances
  for (const auto& pair : result.energy_differences) {
    EXPECT_LT(pair.second, reference.energy_tolerance) 
      << "Energy error for " << pair.first << " exceeds tolerance";
  }
  
  EXPECT_LT(result.max_relative_difference, reference.relative_tolerance)
    << "Relative error exceeds tolerance";
}

// Instantiate parameterized tests
INSTANTIATE_TEST_SUITE_P(
  RegressionTests,
  RegressionTestParameterized,
  ::testing::ValuesIn([]() {
    std::string test_dir = "../electronic-structure-tests";
    return RegressionTest::load_reference_data(test_dir);
  }()),
  [](const ::testing::TestParamInfo<RegressionTest::ReferenceData>& info) {
    return info.param.system_name;
  }
);

/**
 * @brief Fast regression test suite
 */
TEST_F(RegressionTest, FastRegressionSuite) {
  std::string test_dir = "../electronic-structure-tests";
  auto results = run_fast_suite(test_dir);
  
  int failed = 0;
  for (const auto& result : results) {
    if (!result.passed) {
      failed++;
    }
  }
  
  EXPECT_EQ(failed, 0) << failed << " systems failed regression tests";
  
  // Generate and print report
  std::string report = generate_test_report(results);
  std::cout << report << std::endl;
}

/**
 * @brief Full regression test suite (may be slow)
 */
TEST_F(RegressionTest, DISABLED_FullRegressionSuite) {
  std::string test_dir = "../electronic-structure-tests";
  auto results = run_full_suite(test_dir);
  
  int failed = 0;
  for (const auto& result : results) {
    if (!result.passed) {
      failed++;
    }
  }
  
  EXPECT_EQ(failed, 0) << failed << " systems failed regression tests";
  
  // Generate and save detailed report
  std::string report = generate_test_report(results);
  std::ofstream report_file("regression_test_report.txt");
  report_file << report;
  report_file.close();
  
  std::cout << "Full regression test report saved to regression_test_report.txt" << std::endl;
}