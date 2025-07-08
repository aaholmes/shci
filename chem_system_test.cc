/**
 * @file chem_system_test.cc
 * @brief Unit tests for ChemSystem class
 */

#include <gtest/gtest.h>
#include <fstream>
#include <string>
#include "../chem/chem_system.h"
#include "../det/det.h"

/**
 * @class ChemSystemTest
 * @brief Test fixture for ChemSystem class
 */
class ChemSystemTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create temporary directory for test files
    system("mkdir -p chem_test_temp");
    
    // Create minimal H2 FCIDUMP for testing
    createH2FCIDUMP();
  }
  
  void TearDown() override {
    // Clean up test files
    system("rm -rf chem_test_temp");
  }
  
  /**
   * @brief Create minimal H2 FCIDUMP file for testing
   */
  void createH2FCIDUMP() {
    std::ofstream fcidump("chem_test_temp/FCIDUMP");
    fcidump << " &FCI NORB=2,NELEC=2,MS2=0,\n";
    fcidump << "  ORBSYM=1,1,\n";
    fcidump << "  ISYM=1,\n";
    fcidump << " &END\n";
    fcidump << "  1.2527200000E+00  1  1  1  1\n";
    fcidump << "  4.7181200000E-01  2  1  1  1\n";
    fcidump << "  1.2527200000E+00  2  2  1  1\n";
    fcidump << "  6.7577400000E-01  2  2  2  1\n";
    fcidump << "  1.2527200000E+00  2  2  2  2\n";
    fcidump << " -1.2527200000E+00  1  1  0  0\n";
    fcidump << " -1.2527200000E+00  2  2  0  0\n";
    fcidump << "  7.1132700000E-01  0  0  0  0\n";
    fcidump.close();
  }
  
  /**
   * @brief Create larger test FCIDUMP (BeH2 system)
   */
  void createBeH2FCIDUMP() {
    std::ofstream fcidump("chem_test_temp/FCIDUMP_BeH2");
    fcidump << " &FCI NORB=7,NELEC=6,MS2=0,\n";
    fcidump << "  ORBSYM=1,1,1,1,1,1,1,\n";
    fcidump << "  ISYM=1,\n";
    fcidump << " &END\n";
    // Add minimal integral data for BeH2
    fcidump << "  2.5000000000E+00  1  1  1  1\n";
    fcidump << "  1.8000000000E+00  2  2  2  2\n";
    fcidump << "  1.5000000000E+00  3  3  3  3\n";
    fcidump << "  1.2000000000E+00  4  4  4  4\n";
    fcidump << "  1.0000000000E+00  5  5  5  5\n";
    fcidump << "  0.8000000000E+00  6  6  6  6\n";
    fcidump << "  0.6000000000E+00  7  7  7  7\n";
    fcidump << " -2.5000000000E+00  1  1  0  0\n";
    fcidump << " -1.8000000000E+00  2  2  0  0\n";
    fcidump << " -1.5000000000E+00  3  3  0  0\n";
    fcidump << " -1.2000000000E+00  4  4  0  0\n";
    fcidump << " -1.0000000000E+00  5  5  0  0\n";
    fcidump << " -0.8000000000E+00  6  6  0  0\n";
    fcidump << " -0.6000000000E+00  7  7  0  0\n";
    fcidump << "  8.0000000000E+00  0  0  0  0\n";
    fcidump.close();
  }
};

/**
 * @brief Test basic ChemSystem initialization
 */
TEST_F(ChemSystemTest, BasicInitialization) {
  ChemSystem system;
  
  // Test default initialization
  EXPECT_EQ(system.get_n_orbs(), 0);
  EXPECT_EQ(system.get_n_up(), 0);
  EXPECT_EQ(system.get_n_dn(), 0);
}

/**
 * @brief Test FCIDUMP loading
 */
TEST_F(ChemSystemTest, FCIDUMPLoading) {
  ChemSystem system;
  
  // Change to test directory and load FCIDUMP
  ASSERT_EQ(chdir("chem_test_temp"), 0);
  
  EXPECT_NO_THROW(system.setup());
  
  // Check basic properties
  EXPECT_EQ(system.get_n_orbs(), 2);
  EXPECT_EQ(system.get_n_up(), 1);
  EXPECT_EQ(system.get_n_dn(), 1);
  EXPECT_EQ(system.get_ms2(), 0);
  
  // Test integral access
  double h_ii = system.get_one_body(0, 0);
  EXPECT_NE(h_ii, 0.0);  // Should have loaded one-body integrals
  
  double hij_kl = system.get_two_body(0, 0, 0, 0);
  EXPECT_NE(hij_kl, 0.0);  // Should have loaded two-body integrals
  
  chdir("..");
}

/**
 * @brief Test Hartree-Fock determinant generation
 */
TEST_F(ChemSystemTest, HartreeFockDeterminant) {
  ChemSystem system;
  
  ASSERT_EQ(chdir("chem_test_temp"), 0);
  system.setup();
  
  Det hf_det = system.get_hf_det();
  
  // For H2 system, HF should have orbitals 0 occupied for both spins
  EXPECT_TRUE(hf_det.up.has(0));
  EXPECT_TRUE(hf_det.dn.has(0));
  EXPECT_FALSE(hf_det.up.has(1));
  EXPECT_FALSE(hf_det.dn.has(1));
  
  chdir("..");
}

/**
 * @brief Test determinant connections
 */
TEST_F(ChemSystemTest, DeterminantConnections) {
  ChemSystem system;
  
  ASSERT_EQ(chdir("chem_test_temp"), 0);
  system.setup();
  
  Det hf_det = system.get_hf_det();
  std::vector<Det> connections;
  std::vector<double> matrix_elements;
  
  // Get connected determinants
  system.find_connections(hf_det, connections, matrix_elements);
  
  EXPECT_GT(connections.size(), 0);  // Should have some connections
  EXPECT_EQ(connections.size(), matrix_elements.size());
  
  // Check that matrix elements are reasonable
  for (double elem : matrix_elements) {
    EXPECT_TRUE(std::isfinite(elem));
    EXPECT_NE(elem, 0.0);
  }
  
  chdir("..");
}

/**
 * @brief Test Hamiltonian matrix elements
 */
TEST_F(ChemSystemTest, HamiltonianMatrixElements) {
  ChemSystem system;
  
  ASSERT_EQ(chdir("chem_test_temp"), 0);
  system.setup();
  
  Det hf_det = system.get_hf_det();
  
  // Test diagonal element (HF energy)
  double hf_energy = system.get_hamiltonian_elem(hf_det, hf_det);
  EXPECT_TRUE(std::isfinite(hf_energy));
  EXPECT_LT(hf_energy, 0.0);  // Should be negative for bound system
  
  // Test off-diagonal elements
  Det excited_det = hf_det;
  if (excited_det.up.has(0) && !excited_det.up.has(1)) {
    excited_det.up.unset(0);
    excited_det.up.set(1);
    
    double off_diag = system.get_hamiltonian_elem(hf_det, excited_det);
    EXPECT_TRUE(std::isfinite(off_diag));
    // Off-diagonal element can be positive or negative
  }
  
  chdir("..");
}

/**
 * @brief Test point group symmetry
 */
TEST_F(ChemSystemTest, PointGroupSymmetry) {
  ChemSystem system;
  
  ASSERT_EQ(chdir("chem_test_temp"), 0);
  system.setup();
  
  // Test orbital symmetries
  std::vector<int> orb_syms = system.get_orb_syms();
  EXPECT_EQ(orb_syms.size(), 2);
  
  // For H2, both orbitals should have same symmetry (irrep 1)
  EXPECT_EQ(orb_syms[0], 1);
  EXPECT_EQ(orb_syms[1], 1);
  
  // Test point group operations
  std::string point_group = system.get_point_group();
  EXPECT_FALSE(point_group.empty());
  
  chdir("..");
}

/**
 * @brief Test larger molecular system
 */
TEST_F(ChemSystemTest, LargerMolecularSystem) {
  createBeH2FCIDUMP();
  
  ChemSystem system;
  
  ASSERT_EQ(chdir("chem_test_temp"), 0);
  
  // Override FCIDUMP filename for this test
  system.set_fcidump_filename("FCIDUMP_BeH2");
  system.setup();
  
  // Check properties for BeH2 system
  EXPECT_EQ(system.get_n_orbs(), 7);
  EXPECT_EQ(system.get_n_up(), 3);
  EXPECT_EQ(system.get_n_dn(), 3);
  
  Det hf_det = system.get_hf_det();
  
  // Check that first 3 orbitals are occupied
  for (int i = 0; i < 3; ++i) {
    EXPECT_TRUE(hf_det.up.has(i));
    EXPECT_TRUE(hf_det.dn.has(i));
  }
  
  // Check that remaining orbitals are unoccupied
  for (int i = 3; i < 7; ++i) {
    EXPECT_FALSE(hf_det.up.has(i));
    EXPECT_FALSE(hf_det.dn.has(i));
  }
  
  chdir("..");
}

/**
 * @brief Test excitation generation
 */
TEST_F(ChemSystemTest, ExcitationGeneration) {
  ChemSystem system;
  
  ASSERT_EQ(chdir("chem_test_temp"), 0);
  system.setup();
  
  Det hf_det = system.get_hf_det();
  
  // Generate single excitations
  std::vector<Det> single_excitations;
  system.generate_single_excitations(hf_det, single_excitations);
  
  EXPECT_GT(single_excitations.size(), 0);
  
  // For H2, should have 2 single excitations (alpha 0->1, beta 0->1)
  EXPECT_EQ(single_excitations.size(), 2);
  
  // Generate double excitations
  std::vector<Det> double_excitations;
  system.generate_double_excitations(hf_det, double_excitations);
  
  EXPECT_GT(double_excitations.size(), 0);
  
  chdir("..");
}

/**
 * @brief Test integral caching and performance
 */
TEST_F(ChemSystemTest, IntegralCaching) {
  ChemSystem system;
  
  ASSERT_EQ(chdir("chem_test_temp"), 0);
  system.setup();
  
  // Time integral access
  auto start = std::chrono::high_resolution_clock::now();
  
  for (int i = 0; i < 100; ++i) {
    double integral = system.get_two_body(0, 0, 0, 0);
    EXPECT_TRUE(std::isfinite(integral));
  }
  
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  
  // Should be very fast due to caching
  EXPECT_LT(duration.count(), 10000);  // Less than 10ms for 100 accesses
  
  chdir("..");
}

/**
 * @brief Test memory management
 */
TEST_F(ChemSystemTest, MemoryManagement) {
  ChemSystem system;
  
  ASSERT_EQ(chdir("chem_test_temp"), 0);
  system.setup();
  
  // Check memory usage
  size_t memory_usage = system.get_memory_usage();
  EXPECT_GT(memory_usage, 0);
  
  // Test clearing caches
  system.clear_integral_cache();
  size_t memory_after_clear = system.get_memory_usage();
  EXPECT_LE(memory_after_clear, memory_usage);  // Should be same or less
  
  chdir("..");
}

/**
 * @brief Test configuration loading
 */
TEST_F(ChemSystemTest, ConfigurationLoading) {
  // Create test configuration
  std::ofstream config("chem_test_temp/config.json");
  config << "{\n";
  config << "  \"system\": \"chem\",\n";
  config << "  \"n_up\": 1,\n";
  config << "  \"n_dn\": 1,\n";
  config << "  \"chem\": {\n";
  config << "    \"point_group\": \"d2h\"\n";
  config << "  }\n";
  config << "}\n";
  config.close();
  
  ChemSystem system;
  
  ASSERT_EQ(chdir("chem_test_temp"), 0);
  
  // Load configuration
  Config& cfg = Config::get();
  cfg.load("config.json");
  
  system.setup();
  
  // Check that configuration was applied
  EXPECT_EQ(system.get_n_up(), 1);
  EXPECT_EQ(system.get_n_dn(), 1);
  
  chdir("..");
}

/**
 * @brief Test error handling
 */
TEST_F(ChemSystemTest, ErrorHandling) {
  ChemSystem system;
  
  // Test loading non-existent FCIDUMP
  ASSERT_EQ(chdir("chem_test_temp"), 0);
  system.set_fcidump_filename("nonexistent.fcidump");
  EXPECT_THROW(system.setup(), std::runtime_error);
  
  // Test invalid orbital indices
  system.set_fcidump_filename("FCIDUMP");
  system.setup();
  
  EXPECT_THROW(system.get_one_body(10, 0), std::out_of_range);
  EXPECT_THROW(system.get_two_body(0, 0, 10, 0), std::out_of_range);
  
  chdir("..");
}

/**
 * @brief Test serialization/deserialization
 */
TEST_F(ChemSystemTest, Serialization) {
  ChemSystem system1;
  
  ASSERT_EQ(chdir("chem_test_temp"), 0);
  system1.setup();
  
  // Serialize system
  std::vector<uint8_t> serialized = system1.serialize();
  EXPECT_GT(serialized.size(), 0);
  
  // Deserialize into new system
  ChemSystem system2;
  system2.deserialize(serialized);
  
  // Check that properties match
  EXPECT_EQ(system2.get_n_orbs(), system1.get_n_orbs());
  EXPECT_EQ(system2.get_n_up(), system1.get_n_up());
  EXPECT_EQ(system2.get_n_dn(), system1.get_n_dn());
  
  // Check that integrals match
  for (int i = 0; i < system1.get_n_orbs(); ++i) {
    for (int j = 0; j < system1.get_n_orbs(); ++j) {
      EXPECT_DOUBLE_EQ(system2.get_one_body(i, j), system1.get_one_body(i, j));
    }
  }
  
  chdir("..");
}