/**
 * @file hamiltonian_performance_test.cc
 * @brief Performance regression tests and benchmarks for Hamiltonian algorithms
 */

#include <gtest/gtest.h>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <fstream>
#include "hamiltonian.h"
#include "../det/det.h"
#include "../det/half_det.h"
#include "../config.h"

// Performance test system with configurable complexity
class PerformanceTestSystem {
public:
  std::vector<Det> dets;
  bool time_sym = false;
  mutable size_t hamiltonian_calls = 0;
  mutable std::vector<double> call_times;
  
  size_t get_n_dets() const { return dets.size(); }
  
  double get_hamiltonian_elem(const Det& det_i, const Det& det_j, int n_excite) const {
    auto start = std::chrono::high_resolution_clock::now();
    
    hamiltonian_calls++;
    
    // Simulate realistic computation cost
    double result = performComplexCalculation(det_i, det_j);
    
    auto end = std::chrono::high_resolution_clock::now();
    double call_time = std::chrono::duration<double>(end - start).count() * 1000.0; // ms
    call_times.push_back(call_time);
    
    return result;
  }
  
  double get_hamiltonian_elem_time_sym(const Det& det_i, const Det& det_j, int n_excite) const {
    return get_hamiltonian_elem(det_i, det_j, n_excite);
  }
  
  void resetCounters() const {
    hamiltonian_calls = 0;
    call_times.clear();
  }
  
  struct PerformanceStats {
    double total_time_ms;
    double avg_call_time_ms;
    double min_call_time_ms;
    double max_call_time_ms;
    size_t total_calls;
    double calls_per_second;
  };
  
  PerformanceStats getStats() const {
    PerformanceStats stats;
    stats.total_calls = hamiltonian_calls;
    
    if (call_times.empty()) {
      stats.total_time_ms = 0.0;
      stats.avg_call_time_ms = 0.0;
      stats.min_call_time_ms = 0.0;
      stats.max_call_time_ms = 0.0;
      stats.calls_per_second = 0.0;
    } else {
      stats.total_time_ms = std::accumulate(call_times.begin(), call_times.end(), 0.0);
      stats.avg_call_time_ms = stats.total_time_ms / call_times.size();
      stats.min_call_time_ms = *std::min_element(call_times.begin(), call_times.end());
      stats.max_call_time_ms = *std::max_element(call_times.begin(), call_times.end());
      stats.calls_per_second = (stats.total_time_ms > 0) ? 
                               (call_times.size() * 1000.0) / stats.total_time_ms : 0.0;
    }
    
    return stats;
  }

private:
  double performComplexCalculation(const Det& det_i, const Det& det_j) const {
    // Simulate realistic computational complexity
    if (det_i == det_j) {
      // Diagonal elements: more expensive
      double sum = 0.0;
      auto up_orbs = det_i.up.get_occupied_orbs();
      auto dn_orbs = det_i.dn.get_occupied_orbs();
      
      for (unsigned orb1 : up_orbs) {
        for (unsigned orb2 : dn_orbs) {
          sum += std::sin(orb1 * 0.1) * std::cos(orb2 * 0.1);
        }
      }
      return -1.0 - sum * 0.1;
    }
    
    // Off-diagonal elements
    unsigned up_diffs = det_i.up.n_diffs(det_j.up);
    unsigned dn_diffs = det_i.dn.n_diffs(det_j.dn);
    
    if (up_diffs == 2 && dn_diffs == 2) {
      // Opposite-spin: moderate complexity
      return 0.1 * std::exp(-0.1 * (up_diffs + dn_diffs));
    } else if ((up_diffs == 2 && dn_diffs == 0) || (up_diffs == 0 && dn_diffs == 2)) {
      // Same-spin: simpler
      return 0.05;
    }
    
    return 0.0;
  }
};

class HamiltonianPerformanceTest : public ::testing::Test {
protected:
  void SetUp() override {
    Config::set("n_up", 4u);
    Config::set("n_dn", 4u);
    Config::set("opposite_spin_debug_output", false);
  }
  
  void TearDown() override {
    Config::clear();
  }
  
  void createSystemOfSize(size_t target_size) {
    system.dets.clear();
    
    // Generate determinants for 6-orbital system with 4 electrons each spin
    // This gives us plenty of determinants to choose from
    for (unsigned up1 = 0; up1 < 6 && system.dets.size() < target_size; up1++) {
      for (unsigned up2 = up1 + 1; up2 < 6 && system.dets.size() < target_size; up2++) {
        for (unsigned up3 = up2 + 1; up3 < 6 && system.dets.size() < target_size; up3++) {
          for (unsigned up4 = up3 + 1; up4 < 6 && system.dets.size() < target_size; up4++) {
            for (unsigned dn1 = 0; dn1 < 6 && system.dets.size() < target_size; dn1++) {
              for (unsigned dn2 = dn1 + 1; dn2 < 6 && system.dets.size() < target_size; dn2++) {
                for (unsigned dn3 = dn2 + 1; dn3 < 6 && system.dets.size() < target_size; dn3++) {
                  for (unsigned dn4 = dn3 + 1; dn4 < 6 && system.dets.size() < target_size; dn4++) {
                    Det det;
                    det.up.set(up1); det.up.set(up2); det.up.set(up3); det.up.set(up4);
                    det.dn.set(dn1); det.dn.set(dn2); det.dn.set(dn3); det.dn.set(dn4);
                    system.dets.push_back(det);
                    
                    if (system.dets.size() >= target_size) break;
                  }
                  if (system.dets.size() >= target_size) break;
                }
                if (system.dets.size() >= target_size) break;
              }
              if (system.dets.size() >= target_size) break;
            }
            if (system.dets.size() >= target_size) break;
          }
          if (system.dets.size() >= target_size) break;
        }
        if (system.dets.size() >= target_size) break;
      }
      if (system.dets.size() >= target_size) break;
    }
  }
  
  struct BenchmarkResult {
    std::string algorithm_name;
    double execution_time_ms;
    size_t hamiltonian_calls;
    double calls_per_second;
    size_t matrix_elements;
    double setup_time_ms;
    double algorithm_time_ms;
  };
  
  BenchmarkResult benchmarkAlgorithm(const std::string& algorithm, const std::string& cost_model = "auto") {
    Config::set("opposite_spin_algorithm", algorithm);
    if (algorithm == "new") {
      Config::set("opposite_spin_cost_model", cost_model);
    }
    
    Hamiltonian<PerformanceTestSystem> hamiltonian;
    system.resetCounters();
    
    auto start = std::chrono::high_resolution_clock::now();
    
    hamiltonian.matrix.set_dim(system.dets.size());
    
    auto setup_start = std::chrono::high_resolution_clock::now();
    HamiltonianSetupData setup_data;
    if (algorithm == "new") {
      setup_data = hamiltonian.setup_variational_hamiltonian(system.dets);
    }
    auto setup_end = std::chrono::high_resolution_clock::now();
    
    auto alg_start = std::chrono::high_resolution_clock::now();
    if (algorithm == "new") {
      hamiltonian.find_opposite_spin_excitations_new(system, setup_data);
    } else {
      hamiltonian.find_opposite_spin_excitations_2018(system);
    }
    auto alg_end = std::chrono::high_resolution_clock::now();
    
    auto end = std::chrono::high_resolution_clock::now();
    
    BenchmarkResult result;
    result.algorithm_name = algorithm + (algorithm == "new" ? ("_" + cost_model) : "");
    result.execution_time_ms = std::chrono::duration<double>(end - start).count() * 1000.0;
    result.setup_time_ms = std::chrono::duration<double>(setup_end - setup_start).count() * 1000.0;
    result.algorithm_time_ms = std::chrono::duration<double>(alg_end - alg_start).count() * 1000.0;
    result.hamiltonian_calls = system.hamiltonian_calls;
    result.calls_per_second = (result.execution_time_ms > 0) ? 
                             (system.hamiltonian_calls * 1000.0) / result.execution_time_ms : 0.0;
    result.matrix_elements = hamiltonian.matrix.count_n_elems();
    
    return result;
  }
  
  PerformanceTestSystem system;
};

// Test 1: Basic performance benchmarks
TEST_F(HamiltonianPerformanceTest, BasicPerformanceBenchmarks) {
  createSystemOfSize(15);  // Medium-sized system
  
  std::vector<BenchmarkResult> results;
  
  // Benchmark all algorithms
  results.push_back(benchmarkAlgorithm("2018"));
  results.push_back(benchmarkAlgorithm("new", "auto"));
  results.push_back(benchmarkAlgorithm("new", "subalg1"));
  results.push_back(benchmarkAlgorithm("new", "subalg2"));
  results.push_back(benchmarkAlgorithm("new", "subalg3"));
  
  // Print results for analysis
  std::cout << "\nPerformance Benchmark Results (n_dets=" << system.dets.size() << "):\n";
  std::cout << "Algorithm        | Time(ms) | H_calls | Calls/s | Matrix_elems | Setup(ms) | Alg(ms)\n";
  std::cout << "-----------------|----------|---------|---------|--------------|-----------|--------\n";
  
  for (const auto& result : results) {
    std::cout << std::left << std::setw(16) << result.algorithm_name << " | "
              << std::right << std::setw(8) << std::fixed << std::setprecision(2) << result.execution_time_ms << " | "
              << std::setw(7) << result.hamiltonian_calls << " | "
              << std::setw(7) << std::fixed << std::setprecision(0) << result.calls_per_second << " | "
              << std::setw(12) << result.matrix_elements << " | "
              << std::setw(9) << std::setprecision(2) << result.setup_time_ms << " | "
              << std::setw(6) << result.algorithm_time_ms << "\n";
  }
  
  // Verify all algorithms complete successfully
  for (const auto& result : results) {
    EXPECT_GT(result.execution_time_ms, 0.0) << "Algorithm " << result.algorithm_name << " took no time";
    EXPECT_GT(result.hamiltonian_calls, 0) << "Algorithm " << result.algorithm_name << " made no calls";
    EXPECT_GT(result.matrix_elements, 0) << "Algorithm " << result.algorithm_name << " produced no matrix elements";
  }
  
  // Performance should be reasonable (< 5 seconds for this size)
  for (const auto& result : results) {
    EXPECT_LT(result.execution_time_ms, 5000.0) << "Algorithm " << result.algorithm_name << " too slow";
  }
}

// Test 2: Scaling behavior
TEST_F(HamiltonianPerformanceTest, ScalingBehavior) {
  std::vector<size_t> system_sizes = {5, 10, 15, 20};
  std::vector<std::string> algorithms = {"2018", "new"};
  
  std::cout << "\nScaling Behavior Analysis:\n";
  std::cout << "Size | Algorithm | Time(ms) | Calls | Scaling_factor\n";
  std::cout << "-----|-----------|----------|-------|---------------\n";
  
  for (const std::string& algorithm : algorithms) {
    std::vector<double> times;
    
    for (size_t size : system_sizes) {
      createSystemOfSize(size);
      BenchmarkResult result = benchmarkAlgorithm(algorithm);
      times.push_back(result.execution_time_ms);
      
      double scaling_factor = (times.size() > 1 && times[times.size()-2] > 0) ?
                             times.back() / times[times.size()-2] : 1.0;
      
      std::cout << std::setw(4) << size << " | "
                << std::left << std::setw(9) << algorithm << " | "
                << std::right << std::setw(8) << std::fixed << std::setprecision(2) << result.execution_time_ms << " | "
                << std::setw(5) << result.hamiltonian_calls << " | "
                << std::setw(13) << std::setprecision(2) << scaling_factor << "\n";
      
      // Check reasonable scaling (not exponential)
      if (times.size() > 1) {
        EXPECT_LT(scaling_factor, 5.0) << "Poor scaling for " << algorithm << " from size " 
                                       << system_sizes[times.size()-2] << " to " << size;
      }
    }
  }
}

// Test 3: Cost model effectiveness
TEST_F(HamiltonianPerformanceTest, CostModelEffectiveness) {
  createSystemOfSize(12);
  
  std::vector<std::string> cost_models = {"auto", "subalg1", "subalg2", "subalg3"};
  std::vector<BenchmarkResult> results;
  
  std::cout << "\nCost Model Comparison:\n";
  std::cout << "Cost_Model | Time(ms) | H_calls | Efficiency\n";
  std::cout << "-----------|----------|---------|----------\n";
  
  for (const std::string& model : cost_models) {
    BenchmarkResult result = benchmarkAlgorithm("new", model);
    results.push_back(result);
    
    double efficiency = (result.execution_time_ms > 0) ? 
                       result.hamiltonian_calls / result.execution_time_ms : 0.0;
    
    std::cout << std::left << std::setw(10) << model << " | "
              << std::right << std::setw(8) << std::fixed << std::setprecision(2) << result.execution_time_ms << " | "
              << std::setw(7) << result.hamiltonian_calls << " | "
              << std::setw(8) << std::setprecision(1) << efficiency << "\n";
  }
  
  // Auto model should not be significantly worse than the best specific model
  double auto_time = results[0].execution_time_ms;  // auto is first
  double best_time = auto_time;
  
  for (size_t i = 1; i < results.size(); i++) {
    best_time = std::min(best_time, results[i].execution_time_ms);
  }
  
  if (best_time > 0) {
    double auto_overhead = auto_time / best_time;
    EXPECT_LT(auto_overhead, 2.0) << "Auto model shows excessive overhead compared to best specific model";
  }
}

// Test 4: Memory allocation patterns
TEST_F(HamiltonianPerformanceTest, MemoryAllocationPatterns) {
  createSystemOfSize(10);
  
  // Test repeated allocations don't cause performance degradation
  std::vector<double> iteration_times;
  
  for (int i = 0; i < 5; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    
    BenchmarkResult result = benchmarkAlgorithm("new", "auto");
    
    auto end = std::chrono::high_resolution_clock::now();
    double iteration_time = std::chrono::duration<double>(end - start).count() * 1000.0;
    iteration_times.push_back(iteration_time);
    
    EXPECT_GT(result.hamiltonian_calls, 0) << "Iteration " << i << " failed";
  }
  
  // Performance should be consistent across iterations
  if (iteration_times.size() >= 2) {
    double avg_time = std::accumulate(iteration_times.begin(), iteration_times.end(), 0.0) / iteration_times.size();
    
    for (size_t i = 0; i < iteration_times.size(); i++) {
      double deviation = std::abs(iteration_times[i] - avg_time) / avg_time;
      EXPECT_LT(deviation, 0.5) << "High performance variation at iteration " << i 
                                << " (time=" << iteration_times[i] << "ms, avg=" << avg_time << "ms)";
    }
  }
}

// Test 5: Regression test against known benchmarks
TEST_F(HamiltonianPerformanceTest, RegressionTest) {
  createSystemOfSize(10);  // Fixed size for reproducible results
  
  // Known performance baselines (adjust these based on target hardware)
  struct PerformanceBaseline {
    std::string algorithm;
    double max_time_ms;        // Maximum acceptable time
    double min_calls_per_ms;   // Minimum calls per millisecond
  };
  
  std::vector<PerformanceBaseline> baselines = {
    {"2018", 1000.0, 0.1},           // Conservative baseline for 2018 algorithm
    {"new_auto", 1000.0, 0.1},       // New algorithm should be competitive
    {"new_subalg1", 1000.0, 0.1},
    {"new_subalg2", 1000.0, 0.1},
    {"new_subalg3", 1000.0, 0.1}
  };
  
  for (const auto& baseline : baselines) {
    std::string algorithm = baseline.algorithm.substr(0, baseline.algorithm.find('_'));
    std::string cost_model = (baseline.algorithm.find('_') != std::string::npos) ?
                            baseline.algorithm.substr(baseline.algorithm.find('_') + 1) : "auto";
    
    if (algorithm == "new" && cost_model == "new") cost_model = "auto";
    
    BenchmarkResult result = benchmarkAlgorithm(algorithm, cost_model);
    
    EXPECT_LT(result.execution_time_ms, baseline.max_time_ms) 
        << "Algorithm " << baseline.algorithm << " exceeded time baseline";
    
    if (result.execution_time_ms > 0) {
      double calls_per_ms = result.hamiltonian_calls / result.execution_time_ms;
      EXPECT_GT(calls_per_ms, baseline.min_calls_per_ms)
          << "Algorithm " << baseline.algorithm << " below efficiency baseline";
    }
  }
}

// Test 6: Timing statistics validation
TEST_F(HamiltonianPerformanceTest, TimingStatisticsValidation) {
  createSystemOfSize(8);
  
  Config::set("opposite_spin_algorithm", std::string("new"));
  Config::set("opposite_spin_cost_model", std::string("auto"));
  
  Hamiltonian<PerformanceTestSystem> hamiltonian;
  
  // Clear all timing statistics
  hamiltonian.total_opposite_spin_new_time = 0.0;
  hamiltonian.total_opposite_spin_subalg1_time = 0.0;
  hamiltonian.total_opposite_spin_subalg2_time = 0.0;
  hamiltonian.total_opposite_spin_subalg3_time = 0.0;
  hamiltonian.total_opposite_spin_new_calls = 0;
  hamiltonian.total_opposite_spin_subalg1_calls = 0;
  hamiltonian.total_opposite_spin_subalg2_calls = 0;
  hamiltonian.total_opposite_spin_subalg3_calls = 0;
  
  auto wall_start = std::chrono::high_resolution_clock::now();
  
  hamiltonian.matrix.set_dim(system.dets.size());
  HamiltonianSetupData setup_data = hamiltonian.setup_variational_hamiltonian(system.dets);
  hamiltonian.find_opposite_spin_excitations_new(system, setup_data);
  
  auto wall_end = std::chrono::high_resolution_clock::now();
  double wall_time = std::chrono::duration<double>(wall_end - wall_start).count();
  
  // Validate timing statistics
  EXPECT_GT(hamiltonian.total_opposite_spin_new_time, 0.0);
  EXPECT_GT(hamiltonian.total_opposite_spin_new_calls, 0);
  
  // Timing should be reasonable compared to wall time
  EXPECT_LT(hamiltonian.total_opposite_spin_new_time, wall_time * 2.0);  // Allow for overhead
  EXPECT_GT(hamiltonian.total_opposite_spin_new_time, wall_time * 0.01); // Should be significant
  
  // At least one sub-algorithm should have been used
  size_t total_subalg_calls = hamiltonian.total_opposite_spin_subalg1_calls +
                             hamiltonian.total_opposite_spin_subalg2_calls +
                             hamiltonian.total_opposite_spin_subalg3_calls;
  EXPECT_GT(total_subalg_calls, 0);
  
  double total_subalg_time = hamiltonian.total_opposite_spin_subalg1_time +
                            hamiltonian.total_opposite_spin_subalg2_time +
                            hamiltonian.total_opposite_spin_subalg3_time;
  EXPECT_GT(total_subalg_time, 0.0);
  
  // Sub-algorithm time should roughly match total time
  double time_ratio = total_subalg_time / hamiltonian.total_opposite_spin_new_time;
  EXPECT_GT(time_ratio, 0.1);  // Sub-algorithms should account for significant portion
  EXPECT_LT(time_ratio, 2.0);  // But timing overhead shouldn't be too high
}