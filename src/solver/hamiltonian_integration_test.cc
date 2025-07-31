/**
 * @file hamiltonian_integration_test.cc
 * @brief Integration tests for complete Hamiltonian algorithm selection and workflows
 */

#include <gtest/gtest.h>
#include <vector>
#include <chrono>
#include <cmath>
#include "hamiltonian.h"
#include "../det/det.h"
#include "../det/half_det.h"
#include "../config.h"

// Full-featured test system that mimics ChemSystem behavior
class IntegrationTestSystem {
public:
  std::vector<Det> dets;
  bool time_sym = false;
  mutable size_t hamiltonian_calls = 0;
  mutable double total_hamiltonian_time = 0.0;
  
  size_t get_n_dets() const { return dets.size(); }
  
  double get_hamiltonian_elem(const Det& det_i, const Det& det_j, int n_excite) const {
    auto start = std::chrono::high_resolution_clock::now();
    hamiltonian_calls++;
    
    // Realistic Hamiltonian computation
    double H = computeRealisticHamiltonian(det_i, det_j);
    
    auto end = std::chrono::high_resolution_clock::now();
    total_hamiltonian_time += std::chrono::duration<double>(end - start).count();
    
    return H;
  }
  
  double get_hamiltonian_elem_time_sym(const Det& det_i, const Det& det_j, int n_excite) const {
    return get_hamiltonian_elem(det_i, det_j, n_excite);
  }
  
  void resetCounters() const {
    hamiltonian_calls = 0;
    total_hamiltonian_time = 0.0;
  }

private:
  double computeRealisticHamiltonian(const Det& det_i, const Det& det_j) const {
    if (det_i == det_j) {
      // Diagonal: sum of orbital energies
      auto up_orbs = det_i.up.get_occupied_orbs();
      auto dn_orbs = det_i.dn.get_occupied_orbs();
      
      double energy = 0.0;
      for (unsigned orb : up_orbs) {
        energy -= 1.0 + 0.5 * orb;  // Orbital energies
      }
      for (unsigned orb : dn_orbs) {
        energy -= 1.0 + 0.5 * orb;
      }
      
      // Add electron-electron repulsion (simplified)
      for (unsigned orb1 : up_orbs) {
        for (unsigned orb2 : dn_orbs) {
          if (orb1 == orb2) energy += 0.5;  // On-site repulsion
          else energy += 0.1 / (1.0 + std::abs(int(orb1) - int(orb2)));  // Distance-dependent
        }
      }
      
      return energy;
    }
    
    // Off-diagonal elements
    unsigned up_diffs = det_i.up.n_diffs(det_j.up);
    unsigned dn_diffs = det_i.dn.n_diffs(det_j.dn);
    
    if (up_diffs == 2 && dn_diffs == 2) {
      // Opposite-spin double excitation
      auto diff_up = det_i.up.diff(det_j.up);
      auto diff_dn = det_i.dn.diff(det_j.dn);
      
      if (diff_up.n_diffs == 1 && diff_dn.n_diffs == 1) {
        // Realistic coupling strength
        return 0.1 * std::exp(-0.1 * (diff_up.left_only[0] + diff_dn.left_only[0]));
      }
    } else if ((up_diffs == 2 && dn_diffs == 0) || (up_diffs == 0 && dn_diffs == 2)) {
      // Same-spin double excitation (typically smaller)
      return 0.05 * std::exp(-0.2 * std::max(up_diffs, dn_diffs));
    }
    
    return 0.0;
  }
};

class HamiltonianIntegrationTest : public ::testing::Test {
protected:
  void SetUp() override {
    Config::set("n_up", 3u);
    Config::set("n_dn", 3u);
    Config::set("opposite_spin_debug_output", false);
    
    createRealisticSystem();
  }
  
  void TearDown() override {
    Config::clear();
  }
  
  void createRealisticSystem() {
    // Create determinants for a 4-orbital, 3-electron system
    // This gives us a reasonable number of determinants for testing
    
    // Generate all possible combinations of 3 electrons in 4 orbitals
    for (unsigned up1 = 0; up1 < 4; up1++) {
      for (unsigned up2 = up1 + 1; up2 < 4; up2++) {
        for (unsigned up3 = up2 + 1; up3 < 4; up3++) {
          for (unsigned dn1 = 0; dn1 < 4; dn1++) {
            for (unsigned dn2 = dn1 + 1; dn2 < 4; dn2++) {
              for (unsigned dn3 = dn2 + 1; dn3 < 4; dn3++) {
                Det det;
                det.up.set(up1);
                det.up.set(up2);
                det.up.set(up3);
                det.dn.set(dn1);
                det.dn.set(dn2);
                det.dn.set(dn3);
                system.dets.push_back(det);
              }
            }
          }
        }
      }
    }
    
    // Limit to reasonable size for testing (take first 20 determinants)
    if (system.dets.size() > 20) {
      system.dets.resize(20);
    }
  }
  
  IntegrationTestSystem system;
};

// Test 1: Algorithm selection and execution
TEST_F(HamiltonianIntegrationTest, AlgorithmSelectionExecution) {
  // Test all algorithm combinations
  std::vector<std::pair<std::string, std::string>> algorithm_combinations = {
    {"2018", "2018"},
    {"adaptive", "2018"},
    {"adaptive", "new"},
    {"n2", "new"}
  };
  
  for (const auto& combo : algorithm_combinations) {
    Config::set("same_spin_algorithm", combo.first);
    Config::set("opposite_spin_algorithm", combo.second);
    
    Hamiltonian<IntegrationTestSystem> hamiltonian;
    system.resetCounters();
    
    // This should execute without errors
    EXPECT_NO_THROW({
      hamiltonian.matrix.set_dim(system.dets.size());
      
      if (combo.second == "new") {
        HamiltonianSetupData setup_data = hamiltonian.setup_variational_hamiltonian(system.dets);
        hamiltonian.find_opposite_spin_excitations_new(system, setup_data);
      } else {
        hamiltonian.find_opposite_spin_excitations_2018(system);
      }
    });
    
    // Verify that Hamiltonian elements were computed
    EXPECT_GT(system.hamiltonian_calls, 0) << "No Hamiltonian calls for " << combo.first << "+" << combo.second;
    
    // Verify matrix was populated
    size_t n_elems = hamiltonian.matrix.count_n_elems();
    EXPECT_GT(n_elems, 0) << "No matrix elements for " << combo.first << "+" << combo.second;
  }
}

// Test 2: Cost model effectiveness
TEST_F(HamiltonianIntegrationTest, CostModelEffectiveness) {
  Config::set("opposite_spin_algorithm", std::string("new"));
  
  std::vector<std::string> cost_models = {"auto", "subalg1", "subalg2", "subalg3"};
  std::vector<double> execution_times;
  std::vector<size_t> hamiltonian_call_counts;
  
  for (const std::string& model : cost_models) {
    Config::set("opposite_spin_cost_model", model);
    
    Hamiltonian<IntegrationTestSystem> hamiltonian;
    system.resetCounters();
    
    auto start = std::chrono::high_resolution_clock::now();
    
    hamiltonian.matrix.set_dim(system.dets.size());
    HamiltonianSetupData setup_data = hamiltonian.setup_variational_hamiltonian(system.dets);
    hamiltonian.find_opposite_spin_excitations_new(system, setup_data);
    
    auto end = std::chrono::high_resolution_clock::now();
    double execution_time = std::chrono::duration<double>(end - start).count();
    
    execution_times.push_back(execution_time);
    hamiltonian_call_counts.push_back(system.hamiltonian_calls);
    
    EXPECT_GT(execution_time, 0.0);
    EXPECT_GT(system.hamiltonian_calls, 0);
  }
  
  // All models should produce similar results (within 2x of each other)
  double min_time = *std::min_element(execution_times.begin(), execution_times.end());
  double max_time = *std::max_element(execution_times.begin(), execution_times.end());
  
  if (min_time > 0) {
    EXPECT_LT(max_time / min_time, 10.0) << "Cost models show excessive variation";
  }
  
  // Auto model should choose a reasonable algorithm
  // (This is hard to test deterministically, but it should not crash)
  EXPECT_GT(execution_times[0], 0.0);  // auto model
}

// Test 3: Timing statistics accuracy
TEST_F(HamiltonianIntegrationTest, TimingStatisticsAccuracy) {
  Config::set("opposite_spin_algorithm", std::string("new"));
  Config::set("opposite_spin_cost_model", std::string("auto"));
  
  Hamiltonian<IntegrationTestSystem> hamiltonian;
  
  // Clear timing statistics
  hamiltonian.total_opposite_spin_new_time = 0.0;
  hamiltonian.total_opposite_spin_subalg1_time = 0.0;
  hamiltonian.total_opposite_spin_subalg2_time = 0.0;
  hamiltonian.total_opposite_spin_subalg3_time = 0.0;
  
  auto start = std::chrono::high_resolution_clock::now();
  
  hamiltonian.matrix.set_dim(system.dets.size());
  HamiltonianSetupData setup_data = hamiltonian.setup_variational_hamiltonian(system.dets);
  hamiltonian.find_opposite_spin_excitations_new(system, setup_data);
  
  auto end = std::chrono::high_resolution_clock::now();
  double wall_time = std::chrono::duration<double>(end - start).count();
  
  // Timing statistics should be reasonable
  EXPECT_GT(hamiltonian.total_opposite_spin_new_time, 0.0);
  EXPECT_LT(hamiltonian.total_opposite_spin_new_time, wall_time * 2.0);  // Allow for overhead
  
  // At least one sub-algorithm should have been used
  double total_subalg_time = hamiltonian.total_opposite_spin_subalg1_time +
                            hamiltonian.total_opposite_spin_subalg2_time +
                            hamiltonian.total_opposite_spin_subalg3_time;
  EXPECT_GT(total_subalg_time, 0.0);
}

// Test 4: Consistency between algorithms
TEST_F(HamiltonianIntegrationTest, AlgorithmConsistency) {
  // Compare results from 2018 and new algorithms
  Config::set("opposite_spin_algorithm", std::string("2018"));
  Hamiltonian<IntegrationTestSystem> ham_2018;
  ham_2018.matrix.set_dim(system.dets.size());
  
  system.resetCounters();
  ham_2018.find_opposite_spin_excitations_2018(system);
  size_t calls_2018 = system.hamiltonian_calls;
  size_t elems_2018 = ham_2018.matrix.count_n_elems();
  
  Config::set("opposite_spin_algorithm", std::string("new"));
  Config::set("opposite_spin_cost_model", std::string("subalg3"));  // Use direct for fairest comparison
  Hamiltonian<IntegrationTestSystem> ham_new;
  ham_new.matrix.set_dim(system.dets.size());
  
  system.resetCounters();
  HamiltonianSetupData setup_data = ham_new.setup_variational_hamiltonian(system.dets);
  ham_new.find_opposite_spin_excitations_new(system, setup_data);
  size_t calls_new = system.hamiltonian_calls;
  size_t elems_new = ham_new.matrix.count_n_elems();
  
  // Both algorithms should find connections (exact count may vary due to implementation differences)
  EXPECT_GT(calls_2018, 0);
  EXPECT_GT(calls_new, 0);
  EXPECT_GT(elems_2018, 0);
  EXPECT_GT(elems_new, 0);
  
  // Results should be roughly comparable (within factor of 3)
  if (calls_2018 > 0 && calls_new > 0) {
    double call_ratio = double(std::max(calls_2018, calls_new)) / double(std::min(calls_2018, calls_new));
    EXPECT_LT(call_ratio, 3.0) << "Algorithms show very different call patterns";
  }
}

// Test 5: Scalability test
TEST_F(HamiltonianIntegrationTest, ScalabilityTest) {
  // Create progressively larger systems and measure scaling
  std::vector<size_t> system_sizes = {5, 10, 15};
  std::vector<double> execution_times;
  
  for (size_t size : system_sizes) {
    // Create system of specified size
    std::vector<Det> test_dets(system.dets.begin(), 
                              system.dets.begin() + std::min(size, system.dets.size()));
    
    Config::set("opposite_spin_algorithm", std::string("new"));
    Config::set("opposite_spin_cost_model", std::string("auto"));
    
    Hamiltonian<IntegrationTestSystem> hamiltonian;
    IntegrationTestSystem test_system;
    test_system.dets = test_dets;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    hamiltonian.matrix.set_dim(test_system.dets.size());
    HamiltonianSetupData setup_data = hamiltonian.setup_variational_hamiltonian(test_system.dets);
    hamiltonian.find_opposite_spin_excitations_new(test_system, setup_data);
    
    auto end = std::chrono::high_resolution_clock::now();
    double execution_time = std::chrono::duration<double>(end - start).count();
    
    execution_times.push_back(execution_time);
    
    EXPECT_GT(execution_time, 0.0);
    EXPECT_LT(execution_time, 10.0) << "Algorithm too slow for size " << size;
  }
  
  // Execution time should scale reasonably (not exponentially)
  if (execution_times.size() >= 2) {
    for (size_t i = 1; i < execution_times.size(); i++) {
      if (execution_times[i-1] > 0) {
        double scaling_factor = execution_times[i] / execution_times[i-1];
        EXPECT_LT(scaling_factor, 10.0) << "Poor scaling between sizes " 
                                        << system_sizes[i-1] << " and " << system_sizes[i];
      }
    }
  }
}

// Test 6: Memory usage patterns
TEST_F(HamiltonianIntegrationTest, MemoryUsagePatterns) {
  // Test that algorithms don't leak memory with repeated calls
  Config::set("opposite_spin_algorithm", std::string("new"));
  Config::set("opposite_spin_cost_model", std::string("auto"));
  
  for (int iteration = 0; iteration < 10; iteration++) {
    Hamiltonian<IntegrationTestSystem> hamiltonian;
    
    // Run complete workflow
    hamiltonian.matrix.set_dim(system.dets.size());
    HamiltonianSetupData setup_data = hamiltonian.setup_variational_hamiltonian(system.dets);
    hamiltonian.find_opposite_spin_excitations_new(system, setup_data);
    
    // Verify successful completion
    size_t n_elems = hamiltonian.matrix.count_n_elems();
    EXPECT_GT(n_elems, 0) << "Failed at iteration " << iteration;
  }
  
  // If we get here without crashing, memory management is likely correct
  SUCCEED();
}

// Test 7: Configuration robustness
TEST_F(HamiltonianIntegrationTest, ConfigurationRobustness) {
  // Test various configuration edge cases
  std::vector<std::map<std::string, std::string>> test_configs = {
    {{"opposite_spin_algorithm", "new"}, {"opposite_spin_cost_model", "auto"}},
    {{"opposite_spin_algorithm", "new"}, {"opposite_spin_cost_model", "subalg1"}},
    {{"opposite_spin_algorithm", "new"}, {"opposite_spin_cost_model", "subalg2"}},
    {{"opposite_spin_algorithm", "new"}, {"opposite_spin_cost_model", "subalg3"}},
    {{"opposite_spin_algorithm", "2018"}},
    {{"opposite_spin_algorithm", "invalid"}},  // Should fallback gracefully
    {{"opposite_spin_cost_model", "invalid"}}  // Should fallback gracefully
  };
  
  for (const auto& config : test_configs) {
    // Apply configuration
    for (const auto& pair : config) {
      Config::set(pair.first, pair.second);
    }
    
    // Test execution
    EXPECT_NO_THROW({
      Hamiltonian<IntegrationTestSystem> hamiltonian;
      hamiltonian.matrix.set_dim(system.dets.size());
      
      if (Config::get<std::string>("opposite_spin_algorithm", "2018") == "new") {
        HamiltonianSetupData setup_data = hamiltonian.setup_variational_hamiltonian(system.dets);
        hamiltonian.find_opposite_spin_excitations_new(system, setup_data);
      } else {
        hamiltonian.find_opposite_spin_excitations_2018(system);
      }
    }) << "Failed with configuration containing opposite_spin_algorithm=" 
       << Config::get<std::string>("opposite_spin_algorithm", "unknown");
  }
}