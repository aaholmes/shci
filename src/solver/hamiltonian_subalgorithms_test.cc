/**
 * @file hamiltonian_subalgorithms_test.cc
 * @brief Detailed unit tests for the 3 opposite-spin sub-algorithms
 */

#include <gtest/gtest.h>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include "hamiltonian.h"
#include "../det/det.h"
#include "../det/half_det.h"
#include "../config.h"

// Test-specific mock system with deterministic Hamiltonian elements
class TestSystem {
public:
  std::vector<Det> dets;
  bool time_sym = false;
  mutable std::vector<std::tuple<size_t, size_t, double>> recorded_elements;
  
  size_t get_n_dets() const { return dets.size(); }
  
  double get_hamiltonian_elem(const Det& det_i, const Det& det_j, int n_excite) const {
    // Find indices
    size_t i = SIZE_MAX, j = SIZE_MAX;
    for (size_t idx = 0; idx < dets.size(); idx++) {
      if (dets[idx] == det_i) i = idx;
      if (dets[idx] == det_j) j = idx;
    }
    
    if (i == SIZE_MAX || j == SIZE_MAX) return 0.0;
    
    // Record this call for verification
    double H = computeTestHamiltonian(i, j);
    recorded_elements.push_back(std::make_tuple(i, j, H));
    return H;
  }
  
  double get_hamiltonian_elem_time_sym(const Det& det_i, const Det& det_j, int n_excite) const {
    return get_hamiltonian_elem(det_i, det_j, n_excite);
  }
  
  void clearRecordedElements() const {
    recorded_elements.clear();
  }
  
  bool wasElementComputed(size_t i, size_t j) const {
    for (const auto& elem : recorded_elements) {
      if ((std::get<0>(elem) == i && std::get<1>(elem) == j) ||
          (std::get<0>(elem) == j && std::get<1>(elem) == i)) {
        return true;
      }
    }
    return false;
  }

private:
  double computeTestHamiltonian(size_t i, size_t j) const {
    if (i == j) return -1.0 - 0.1 * i;  // Diagonal elements
    
    const Det& det_i = dets[i];
    const Det& det_j = dets[j];
    
    unsigned up_diffs = det_i.up.n_diffs(det_j.up);
    unsigned dn_diffs = det_i.dn.n_diffs(det_j.dn);
    
    if (up_diffs == 2 && dn_diffs == 2) {
      // Opposite-spin double excitation
      return 0.1 + 0.01 * (i + j);
    } else if ((up_diffs == 2 && dn_diffs == 0) || (up_diffs == 0 && dn_diffs == 2)) {
      // Same-spin double excitation
      return 0.05 + 0.005 * (i + j);
    }
    
    return 0.0;
  }
};

class SubAlgorithmsTest : public ::testing::Test {
protected:
  void SetUp() override {
    Config::set("n_up", 2u);
    Config::set("n_dn", 2u);
    Config::set("opposite_spin_debug_output", false);
    
    createTestDeterminants();
  }
  
  void TearDown() override {
    Config::clear();
  }
  
  void createTestDeterminants() {
    // Create a more complex set of determinants for thorough testing
    // 6 determinants covering different orbital patterns
    
    Det det0, det1, det2, det3, det4, det5;
    
    // det0: |00⟩ - both electrons in orbital 0
    det0.up.set(0);
    det0.dn.set(0);
    
    // det1: |01⟩ - up in 0, down in 1
    det1.up.set(0);
    det1.dn.set(1);
    
    // det2: |10⟩ - up in 1, down in 0
    det2.up.set(1);
    det2.dn.set(0);
    
    // det3: |11⟩ - both electrons in orbital 1
    det3.up.set(1);
    det3.dn.set(1);
    
    // det4: |02⟩ - up in 0, down in 2
    det4.up.set(0);
    det4.dn.set(2);
    
    // det5: |20⟩ - up in 2, down in 0
    det5.up.set(2);
    det5.dn.set(0);
    
    system.dets = {det0, det1, det2, det3, det4, det5};
  }
  
  TestSystem system;
  Hamiltonian<TestSystem> hamiltonian;
};

// Test 1: Sub-algorithm 1 (Hash existing connections)
TEST_F(SubAlgorithmsTest, SubAlgorithm1HashExistingConnections) {
  hamiltonian.matrix.set_dim(system.dets.size());
  HamiltonianSetupData setup_data = hamiltonian.setup_variational_hamiltonian(system.dets);
  
  system.clearRecordedElements();
  
  // Test sub-algorithm 1 with known data
  if (setup_data.upSingles.size() > 0) {
    auto it = setup_data.upSingles.begin();
    size_t up_idx = it->first;
    const std::vector<size_t>& up_singles = it->second;
    
    // Get down-spin indices for this up-spin
    const HalfDet& u_i = setup_data.unique_up_dets[up_idx];
    std::vector<size_t> dn_indices_i;
    
    auto up_it = setup_data.up_to_full_map.find(u_i);
    if (up_it != setup_data.up_to_full_map.end()) {
      for (size_t full_idx : up_it->second) {
        const Det& det = system.dets[full_idx];
        auto dn_idx_it = setup_data.dn_det_to_idx.find(det.dn);
        if (dn_idx_it != setup_data.dn_det_to_idx.end()) {
          dn_indices_i.push_back(dn_idx_it->second);
        }
      }
    }
    
    // Run sub-algorithm 1
    EXPECT_NO_THROW({
      hamiltonian.opposite_spin_subalg1(system, setup_data, up_idx, dn_indices_i, up_singles);
    });
    
    // Verify that Hamiltonian elements were computed
    EXPECT_GT(system.recorded_elements.size(), 0);
    
    // Check matrix has been populated
    size_t n_elems = hamiltonian.matrix.count_n_elems();
    EXPECT_GT(n_elems, 0);
  }
}

// Test 2: Sub-algorithm 2 (Hash N-1 configurations)
TEST_F(SubAlgorithmsTest, SubAlgorithm2HashNMinus1Configs) {
  hamiltonian.matrix.set_dim(system.dets.size());
  HamiltonianSetupData setup_data = hamiltonian.setup_variational_hamiltonian(system.dets);
  
  system.clearRecordedElements();
  
  if (setup_data.upSingles.size() > 0) {
    auto it = setup_data.upSingles.begin();
    size_t up_idx = it->first;
    const std::vector<size_t>& up_singles = it->second;
    
    // Get down-spin indices
    const HalfDet& u_i = setup_data.unique_up_dets[up_idx];
    std::vector<size_t> dn_indices_i;
    
    auto up_it = setup_data.up_to_full_map.find(u_i);
    if (up_it != setup_data.up_to_full_map.end()) {
      for (size_t full_idx : up_it->second) {
        const Det& det = system.dets[full_idx];
        auto dn_idx_it = setup_data.dn_det_to_idx.find(det.dn);
        if (dn_idx_it != setup_data.dn_det_to_idx.end()) {
          dn_indices_i.push_back(dn_idx_it->second);
        }
      }
    }
    
    // Run sub-algorithm 2
    EXPECT_NO_THROW({
      hamiltonian.opposite_spin_subalg2(system, setup_data, up_idx, dn_indices_i, up_singles);
    });
    
    // Verify operation
    EXPECT_GT(system.recorded_elements.size(), 0);
    size_t n_elems = hamiltonian.matrix.count_n_elems();
    EXPECT_GT(n_elems, 0);
  }
}

// Test 3: Sub-algorithm 3 (Direct comparison)
TEST_F(SubAlgorithmsTest, SubAlgorithm3DirectComparison) {
  hamiltonian.matrix.set_dim(system.dets.size());
  HamiltonianSetupData setup_data = hamiltonian.setup_variational_hamiltonian(system.dets);
  
  system.clearRecordedElements();
  
  if (setup_data.upSingles.size() > 0) {
    auto it = setup_data.upSingles.begin();
    size_t up_idx = it->first;
    const std::vector<size_t>& up_singles = it->second;
    
    // Get down-spin indices
    const HalfDet& u_i = setup_data.unique_up_dets[up_idx];
    std::vector<size_t> dn_indices_i;
    
    auto up_it = setup_data.up_to_full_map.find(u_i);
    if (up_it != setup_data.up_to_full_map.end()) {
      for (size_t full_idx : up_it->second) {
        const Det& det = system.dets[full_idx];
        auto dn_idx_it = setup_data.dn_det_to_idx.find(det.dn);
        if (dn_idx_it != setup_data.dn_det_to_idx.end()) {
          dn_indices_i.push_back(dn_idx_it->second);
        }
      }
    }
    
    // Run sub-algorithm 3
    EXPECT_NO_THROW({
      hamiltonian.opposite_spin_subalg3(system, setup_data, up_idx, dn_indices_i, up_singles);
    });
    
    // Verify operation
    EXPECT_GT(system.recorded_elements.size(), 0);
    size_t n_elems = hamiltonian.matrix.count_n_elems();
    EXPECT_GT(n_elems, 0);
  }
}

// Test 4: Compare sub-algorithm results
TEST_F(SubAlgorithmsTest, CompareSubAlgorithmResults) {
  // Run all three sub-algorithms on the same data and compare results
  HamiltonianSetupData setup_data = hamiltonian.setup_variational_hamiltonian(system.dets);
  
  if (setup_data.upSingles.empty()) {
    GTEST_SKIP() << "No up-spin singles available for comparison test";
  }
  
  auto it = setup_data.upSingles.begin();
  size_t up_idx = it->first;
  const std::vector<size_t>& up_singles = it->second;
  
  // Get down-spin indices
  const HalfDet& u_i = setup_data.unique_up_dets[up_idx];
  std::vector<size_t> dn_indices_i;
  
  auto up_it = setup_data.up_to_full_map.find(u_i);
  if (up_it != setup_data.up_to_full_map.end()) {
    for (size_t full_idx : up_it->second) {
      const Det& det = system.dets[full_idx];
      auto dn_idx_it = setup_data.dn_det_to_idx.find(det.dn);
      if (dn_idx_it != setup_data.dn_det_to_idx.end()) {
        dn_indices_i.push_back(dn_idx_it->second);
      }
    }
  }
  
  // Create three separate Hamiltonian instances
  Hamiltonian<TestSystem> ham1, ham2, ham3;
  ham1.matrix.set_dim(system.dets.size());
  ham2.matrix.set_dim(system.dets.size());
  ham3.matrix.set_dim(system.dets.size());
  
  // Run each sub-algorithm
  system.clearRecordedElements();
  ham1.opposite_spin_subalg1(system, setup_data, up_idx, dn_indices_i, up_singles);
  auto elements1 = system.recorded_elements;
  
  system.clearRecordedElements();
  ham2.opposite_spin_subalg2(system, setup_data, up_idx, dn_indices_i, up_singles);
  auto elements2 = system.recorded_elements;
  
  system.clearRecordedElements();
  ham3.opposite_spin_subalg3(system, setup_data, up_idx, dn_indices_i, up_singles);
  auto elements3 = system.recorded_elements;
  
  // All algorithms should find the same connections (though possibly in different order)
  // Extract unique pairs from each result
  std::set<std::pair<size_t, size_t>> pairs1, pairs2, pairs3;
  
  for (const auto& elem : elements1) {
    size_t i = std::get<0>(elem);
    size_t j = std::get<1>(elem);
    pairs1.insert({std::min(i, j), std::max(i, j)});
  }
  
  for (const auto& elem : elements2) {
    size_t i = std::get<0>(elem);
    size_t j = std::get<1>(elem);
    pairs2.insert({std::min(i, j), std::max(i, j)});
  }
  
  for (const auto& elem : elements3) {
    size_t i = std::get<0>(elem);
    size_t j = std::get<1>(elem);
    pairs3.insert({std::min(i, j), std::max(i, j)});
  }
  
  // All algorithms should find the same set of connections
  EXPECT_EQ(pairs1, pairs2) << "Sub-algorithms 1 and 2 found different connections";
  EXPECT_EQ(pairs2, pairs3) << "Sub-algorithms 2 and 3 found different connections";
  EXPECT_EQ(pairs1, pairs3) << "Sub-algorithms 1 and 3 found different connections";
}

// Test 5: Edge cases
TEST_F(SubAlgorithmsTest, EdgeCases) {
  hamiltonian.matrix.set_dim(system.dets.size());
  HamiltonianSetupData setup_data = hamiltonian.setup_variational_hamiltonian(system.dets);
  
  // Test with empty vectors
  std::vector<size_t> empty_dn_indices;
  std::vector<size_t> empty_up_singles;
  
  EXPECT_NO_THROW({
    hamiltonian.opposite_spin_subalg1(system, setup_data, 0, empty_dn_indices, empty_up_singles);
    hamiltonian.opposite_spin_subalg2(system, setup_data, 0, empty_dn_indices, empty_up_singles);
    hamiltonian.opposite_spin_subalg3(system, setup_data, 0, empty_dn_indices, empty_up_singles);
  });
  
  // Test with invalid indices (should handle gracefully)
  std::vector<size_t> invalid_indices = {SIZE_MAX, SIZE_MAX - 1};
  
  EXPECT_NO_THROW({
    hamiltonian.opposite_spin_subalg1(system, setup_data, SIZE_MAX, invalid_indices, invalid_indices);
    hamiltonian.opposite_spin_subalg2(system, setup_data, SIZE_MAX, invalid_indices, invalid_indices);
    hamiltonian.opposite_spin_subalg3(system, setup_data, SIZE_MAX, invalid_indices, invalid_indices);
  });
}

// Test 6: Performance characteristics
TEST_F(SubAlgorithmsTest, PerformanceCharacteristics) {
  // Create larger test case to observe performance differences
  std::vector<Det> large_dets;
  
  // Create determinants for 4 orbitals, 2 electrons each spin
  for (unsigned up1 = 0; up1 < 4; up1++) {
    for (unsigned up2 = up1 + 1; up2 < 4; up2++) {
      for (unsigned dn1 = 0; dn1 < 4; dn1++) {
        for (unsigned dn2 = dn1 + 1; dn2 < 4; dn2++) {
          Det det;
          det.up.set(up1);
          det.up.set(up2);
          det.dn.set(dn1);
          det.dn.set(dn2);
          large_dets.push_back(det);
        }
      }
    }
  }
  
  TestSystem large_system;
  large_system.dets = large_dets;
  
  Hamiltonian<TestSystem> large_ham;
  large_ham.matrix.set_dim(large_system.dets.size());
  
  HamiltonianSetupData setup_data = large_ham.setup_variational_hamiltonian(large_system.dets);
  
  if (!setup_data.upSingles.empty()) {
    auto it = setup_data.upSingles.begin();
    size_t up_idx = it->first;
    const std::vector<size_t>& up_singles = it->second;
    
    // Get down-spin indices
    const HalfDet& u_i = setup_data.unique_up_dets[up_idx];
    std::vector<size_t> dn_indices_i;
    
    auto up_it = setup_data.up_to_full_map.find(u_i);
    if (up_it != setup_data.up_to_full_map.end()) {
      for (size_t full_idx : up_it->second) {
        const Det& det = large_system.dets[full_idx];
        auto dn_idx_it = setup_data.dn_det_to_idx.find(det.dn);
        if (dn_idx_it != setup_data.dn_det_to_idx.end()) {
          dn_indices_i.push_back(dn_idx_it->second);
        }
      }
    }
    
    // Test that all algorithms complete in reasonable time
    auto start = std::chrono::high_resolution_clock::now();
    large_ham.opposite_spin_subalg1(large_system, setup_data, up_idx, dn_indices_i, up_singles);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    // Reset matrix for fair comparison
    large_ham.matrix.set_dim(large_system.dets.size());
    
    start = std::chrono::high_resolution_clock::now();
    large_ham.opposite_spin_subalg2(large_system, setup_data, up_idx, dn_indices_i, up_singles);
    end = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    large_ham.matrix.set_dim(large_system.dets.size());
    
    start = std::chrono::high_resolution_clock::now();
    large_ham.opposite_spin_subalg3(large_system, setup_data, up_idx, dn_indices_i, up_singles);
    end = std::chrono::high_resolution_clock::now();
    auto duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    // All should complete in reasonable time (< 1000ms for this test case)
    EXPECT_LT(duration1, 1000);
    EXPECT_LT(duration2, 1000);
    EXPECT_LT(duration3, 1000);
    
    // Record relative performance for debugging
    std::cout << "Algorithm performance (ms): Subalg1=" << duration1 
              << ", Subalg2=" << duration2 << ", Subalg3=" << duration3 << std::endl;
  }
}

// Test 7: Memory safety
TEST_F(SubAlgorithmsTest, MemorySafety) {
  // Test with various memory allocation patterns
  for (int run = 0; run < 5; run++) {
    Hamiltonian<TestSystem> test_ham;
    test_ham.matrix.set_dim(system.dets.size());
    
    HamiltonianSetupData setup_data = test_ham.setup_variational_hamiltonian(system.dets);
    
    if (!setup_data.upSingles.empty()) {
      auto it = setup_data.upSingles.begin();
      size_t up_idx = it->first;
      const std::vector<size_t>& up_singles = it->second;
      
      // Get down-spin indices
      const HalfDet& u_i = setup_data.unique_up_dets[up_idx];
      std::vector<size_t> dn_indices_i;
      
      auto up_it = setup_data.up_to_full_map.find(u_i);
      if (up_it != setup_data.up_to_full_map.end()) {
        for (size_t full_idx : up_it->second) {
          const Det& det = system.dets[full_idx];
          auto dn_idx_it = setup_data.dn_det_to_idx.find(det.dn);
          if (dn_idx_it != setup_data.dn_det_to_idx.end()) {
            dn_indices_i.push_back(dn_idx_it->second);
          }
        }
      }
      
      // Run all algorithms - should not crash or leak memory
      EXPECT_NO_THROW({
        test_ham.opposite_spin_subalg1(system, setup_data, up_idx, dn_indices_i, up_singles);
        test_ham.opposite_spin_subalg2(system, setup_data, up_idx, dn_indices_i, up_singles);
        test_ham.opposite_spin_subalg3(system, setup_data, up_idx, dn_indices_i, up_singles);
      });
    }
  }
}