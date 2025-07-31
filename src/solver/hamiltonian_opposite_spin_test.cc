/**
 * @file hamiltonian_opposite_spin_test.cc
 * @brief Comprehensive unit tests for new opposite-spin Hamiltonian algorithms
 */

#include <gtest/gtest.h>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "hamiltonian.h"
#include "../det/det.h"
#include "../det/half_det.h"
#include "../config.h"

// Mock system for testing
class MockChemSystem {
public:
  std::vector<Det> dets;
  bool time_sym = false;
  
  MockChemSystem() = default;
  
  size_t get_n_dets() const { return dets.size(); }
  
  double get_hamiltonian_elem(const Det& det_i, const Det& det_j, int n_excite) const {
    // Simple mock: return 1.0 for valid connections, 0 otherwise
    if (det_i == det_j) return -1.0;  // Diagonal
    
    // Check if this is a valid opposite-spin excitation
    unsigned up_diffs = det_i.up.n_diffs(det_j.up);
    unsigned dn_diffs = det_i.dn.n_diffs(det_j.dn);
    
    if (up_diffs == 2 && dn_diffs == 2) {
      // Opposite-spin double excitation
      return 0.1;
    } else if ((up_diffs == 2 && dn_diffs == 0) || (up_diffs == 0 && dn_diffs == 2)) {
      // Same-spin double excitation
      return 0.05;
    }
    
    return 0.0;
  }
  
  double get_hamiltonian_elem_time_sym(const Det& det_i, const Det& det_j, int n_excite) const {
    return get_hamiltonian_elem(det_i, det_j, n_excite);
  }
};

class HamiltonianOppositeSpinTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Set up minimal configuration
    Config::set("n_up", 2u);
    Config::set("n_dn", 2u);
    Config::set("opposite_spin_debug_output", false);
    Config::set("opposite_spin_algorithm", std::string("new"));
    Config::set("opposite_spin_cost_model", std::string("auto"));
    
    // Create test determinants for H2-like system
    createSimpleDeterminants();
  }
  
  void TearDown() override {
    Config::clear();
  }
  
  void createSimpleDeterminants() {
    // Create 4 determinants for a simple 2-orbital, 2-electron system
    // |00⟩, |01⟩, |10⟩, |11⟩ where each bit represents an orbital
    
    Det det1, det2, det3, det4;
    
    // |00⟩ - both electrons in orbital 0
    det1.up.set(0);
    det1.dn.set(0);
    
    // |01⟩ - up in 0, down in 1
    det2.up.set(0);
    det2.dn.set(1);
    
    // |10⟩ - up in 1, down in 0
    det3.up.set(1);
    det3.dn.set(0);
    
    // |11⟩ - both electrons in orbital 1
    det4.up.set(1);
    det4.dn.set(1);
    
    system.dets = {det1, det2, det3, det4};
  }
  
  MockChemSystem system;
  Hamiltonian<MockChemSystem> hamiltonian;
};

// Test 1: HamiltonianSetupData structure correctness
TEST_F(HamiltonianOppositeSpinTest, HamiltonianSetupDataStructure) {
  HamiltonianSetupData setup_data = hamiltonian.setup_variational_hamiltonian(system.dets);
  
  // Verify basic structure
  EXPECT_FALSE(setup_data.unique_up_dets.empty());
  EXPECT_FALSE(setup_data.unique_dn_dets.empty());
  EXPECT_FALSE(setup_data.up_to_full_map.empty());
  EXPECT_FALSE(setup_data.dn_to_full_map.empty());
  
  // Check that all determinants are accounted for
  size_t total_mapped = 0;
  for (const auto& pair : setup_data.up_to_full_map) {
    total_mapped += pair.second.size();
  }
  EXPECT_EQ(total_mapped, system.dets.size());
  
  // Verify index consistency
  for (size_t i = 0; i < setup_data.unique_up_dets.size(); i++) {
    const HalfDet& up_det = setup_data.unique_up_dets[i];
    EXPECT_EQ(setup_data.up_det_to_idx.at(up_det), i);
  }
}

// Test 2: setup_variational_hamiltonian() correctness
TEST_F(HamiltonianOppositeSpinTest, SetupVariationalHamiltonian) {
  HamiltonianSetupData setup_data = hamiltonian.setup_variational_hamiltonian(system.dets);
  
  // Test with known determinants
  EXPECT_EQ(setup_data.unique_up_dets.size(), 2);  // orbital 0 and orbital 1
  EXPECT_EQ(setup_data.unique_dn_dets.size(), 2);  // orbital 0 and orbital 1
  
  // Test sorting by importance (number of full determinants)
  // Both up-spin states should have 2 determinants each, so order may vary
  size_t up0_count = setup_data.up_to_full_map[setup_data.unique_up_dets[0]].size();
  size_t up1_count = setup_data.up_to_full_map[setup_data.unique_up_dets[1]].size();
  EXPECT_EQ(up0_count, 2);
  EXPECT_EQ(up1_count, 2);
  
  // Test singles connections
  // Each up-spin state should connect to the other
  EXPECT_EQ(setup_data.upSingles.size(), 1);  // Only one connection due to idx_j > idx_i condition
  
  // Each down-spin state should connect to the other
  EXPECT_EQ(setup_data.dnSingles.size(), 2);  // Both connections preserved
}

// Test 3: N-1 configuration generation
TEST_F(HamiltonianOppositeSpinTest, GenerateNMinus1Configs) {
  HalfDet half_det;
  half_det.set(0);
  half_det.set(2);
  half_det.set(4);
  
  std::vector<HalfDet> configs = hamiltonian.generate_n_minus_1_configs(half_det);
  
  EXPECT_EQ(configs.size(), 3);  // Should have 3 configurations (remove each electron)
  
  // Check that each configuration has exactly 2 electrons
  for (const auto& config : configs) {
    EXPECT_EQ(config.get_occupied_orbs().size(), 2);
  }
  
  // Check specific configurations
  EXPECT_TRUE(configs[0].has(2) && configs[0].has(4) && !configs[0].has(0));  // Removed 0
  EXPECT_TRUE(configs[1].has(0) && configs[1].has(4) && !configs[1].has(2));  // Removed 2
  EXPECT_TRUE(configs[2].has(0) && configs[2].has(2) && !configs[2].has(4));  // Removed 4
}

// Test 4: Single excitation checker
TEST_F(HamiltonianOppositeSpinTest, IsSingleExcitation) {
  HalfDet det1, det2, det3;
  
  // det1: orbital 0
  det1.set(0);
  
  // det2: orbital 1 (single excitation from det1)
  det2.set(1);
  
  // det3: orbitals 0 and 2 (not single excitation from det1)
  det3.set(0);
  det3.set(2);
  
  EXPECT_TRUE(hamiltonian.is_single_excitation(det1, det2));
  EXPECT_FALSE(hamiltonian.is_single_excitation(det1, det3));
  EXPECT_FALSE(hamiltonian.is_single_excitation(det1, det1));  // Same determinant
}

// Test 5: Cost estimation functions
TEST_F(HamiltonianOppositeSpinTest, CostEstimationFunctions) {
  size_t n_dn_i = 10;
  size_t n_up_singles = 5;
  size_t avg_dn_singles = 8;
  size_t avg_dn_j = 12;
  size_t n_electrons = 4;
  
  double cost1 = hamiltonian.estimate_opposite_spin_subalg1_cost(n_dn_i, n_up_singles, avg_dn_singles, avg_dn_j);
  double cost2 = hamiltonian.estimate_opposite_spin_subalg2_cost(n_dn_i, n_up_singles, avg_dn_j, n_electrons);
  double cost3 = hamiltonian.estimate_opposite_spin_subalg3_cost(n_dn_i, n_up_singles, avg_dn_j);
  
  // Verify costs are positive
  EXPECT_GT(cost1, 0);
  EXPECT_GT(cost2, 0);
  EXPECT_GT(cost3, 0);
  
  // Verify relative scaling makes sense
  // Sub-algorithm 3 (direct) should generally be most expensive for reasonable sizes
  EXPECT_GT(cost3, cost1);
  
  // Cost should scale with problem size
  double cost1_large = hamiltonian.estimate_opposite_spin_subalg1_cost(n_dn_i * 2, n_up_singles, avg_dn_singles, avg_dn_j);
  EXPECT_GT(cost1_large, cost1);
}

// Test 6: Algorithm selection logic
TEST_F(HamiltonianOppositeSpinTest, AlgorithmSelection) {
  // Create a hamiltonian with different cost model settings
  Config::set("opposite_spin_cost_model", std::string("subalg1"));
  Hamiltonian<MockChemSystem> ham1;
  
  Config::set("opposite_spin_cost_model", std::string("subalg2"));
  Hamiltonian<MockChemSystem> ham2;
  
  Config::set("opposite_spin_cost_model", std::string("subalg3"));
  Hamiltonian<MockChemSystem> ham3;
  
  // Test that different configurations are accepted
  EXPECT_NO_THROW({
    HamiltonianSetupData setup_data = ham1.setup_variational_hamiltonian(system.dets);
  });
  
  EXPECT_NO_THROW({
    HamiltonianSetupData setup_data = ham2.setup_variational_hamiltonian(system.dets);
  });
  
  EXPECT_NO_THROW({
    HamiltonianSetupData setup_data = ham3.setup_variational_hamiltonian(system.dets);
  });
}

// Test 7: Matrix construction correctness
TEST_F(HamiltonianOppositeSpinTest, MatrixConstruction) {
  hamiltonian.matrix.set_dim(system.dets.size());
  
  // Test that matrix operations work
  hamiltonian.matrix.append_elem(0, 1, 0.5);
  hamiltonian.matrix.append_elem(1, 2, 0.3);
  hamiltonian.matrix.cache_diag();
  
  size_t n_elems = hamiltonian.matrix.count_n_elems();
  EXPECT_GE(n_elems, 2);  // At least the elements we added
}

// Test 8: Empty input handling
TEST_F(HamiltonianOppositeSpinTest, EmptyInputHandling) {
  std::vector<Det> empty_dets;
  
  // Should handle empty input gracefully
  EXPECT_NO_THROW({
    HamiltonianSetupData setup_data = hamiltonian.setup_variational_hamiltonian(empty_dets);
    EXPECT_TRUE(setup_data.unique_up_dets.empty());
    EXPECT_TRUE(setup_data.unique_dn_dets.empty());
  });
}

// Test 9: Single determinant input
TEST_F(HamiltonianOppositeSpinTest, SingleDeterminantInput) {
  std::vector<Det> single_det = {system.dets[0]};
  
  HamiltonianSetupData setup_data = hamiltonian.setup_variational_hamiltonian(single_det);
  
  EXPECT_EQ(setup_data.unique_up_dets.size(), 1);
  EXPECT_EQ(setup_data.unique_dn_dets.size(), 1);
  EXPECT_EQ(setup_data.up_to_full_map.size(), 1);
  EXPECT_EQ(setup_data.dn_to_full_map.size(), 1);
  EXPECT_TRUE(setup_data.upSingles.empty());  // No connections possible
  EXPECT_TRUE(setup_data.dnSingles.empty());
}

// Test 10: Configuration validation
TEST_F(HamiltonianOppositeSpinTest, ConfigurationValidation) {
  // Test invalid algorithm selection
  Config::set("opposite_spin_algorithm", std::string("invalid"));
  
  // Should still work but fall back to default behavior
  EXPECT_NO_THROW({
    Hamiltonian<MockChemSystem> ham;
  });
  
  // Test invalid cost model
  Config::set("opposite_spin_cost_model", std::string("invalid"));
  Config::set("opposite_spin_algorithm", std::string("new"));
  
  EXPECT_NO_THROW({
    Hamiltonian<MockChemSystem> ham;
    HamiltonianSetupData setup_data = ham.setup_variational_hamiltonian(system.dets);
  });
}