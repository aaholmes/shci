/**
 * @file config_test_fixed.cc
 * @brief Unit tests for configuration system (compatible with current API)
 */

#include <gtest/gtest.h>
#include <fstream>
#include <stdexcept>
#include "config.h"

/**
 * @class ConfigTest
 * @brief Test fixture for configuration system
 */
class ConfigTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create test directory
    system("mkdir -p config_test_temp");
    // Create a valid test config for the current working directory
    createValidConfig();
  }

  void TearDown() override {
    // Clean up
    system("rm -rf config_test_temp");
    system("rm -f config.json"); // Remove any test config in current dir
  }

  /**
   * @brief Create a valid test configuration
   */
  void createValidConfig() {
    std::ofstream config("config.json"); // Current API expects config.json in current dir
    config << "{\n";
    config << "  \"system\": \"chem\",\n";
    config << "  \"n_up\": 5,\n";
    config << "  \"n_dn\": 5,\n";
    config << "  \"eps_vars\": [1e-4, 5e-5, 2e-5],\n";
    config << "  \"eps_pt_dtm\": 1e-6,\n";
    config << "  \"target_error\": 1e-4,\n";
    config << "  \"time_sym\": true,\n";
    config << "  \"chem\": {\n";
    config << "    \"point_group\": \"c2v\"\n";
    config << "  }\n";
    config << "}\n";
    config.close();
  }

  /**
   * @brief Create minimal test configuration
   */
  void createMinimalConfig() {
    std::ofstream config("config.json");
    config << "{\n";
    config << "  \"system\": \"chem\",\n";
    config << "  \"n_up\": 2,\n";
    config << "  \"n_dn\": 2\n";
    config << "}\n";
    config.close();
  }
};

/**
 * @brief Test configuration parameter access
 */
TEST_F(ConfigTest, ParameterAccess) {
  // Test string parameter
  EXPECT_NO_THROW({
    std::string system = Config::get<std::string>("system");
    EXPECT_EQ(system, "chem");
  });
  
  // Test integer parameters
  EXPECT_NO_THROW({
    int n_up = Config::get<int>("n_up");
    int n_dn = Config::get<int>("n_dn");
    EXPECT_EQ(n_up, 5);
    EXPECT_EQ(n_dn, 5);
  });
  
  // Test double parameter
  EXPECT_NO_THROW({
    double eps_pt_dtm = Config::get<double>("eps_pt_dtm");
    double target_error = Config::get<double>("target_error");
    EXPECT_DOUBLE_EQ(eps_pt_dtm, 1e-6);
    EXPECT_DOUBLE_EQ(target_error, 1e-4);
  });
  
  // Test boolean parameter
  EXPECT_NO_THROW({
    bool time_sym = Config::get<bool>("time_sym");
    EXPECT_TRUE(time_sym);
  });
}

/**
 * @brief Test default values
 */
TEST_F(ConfigTest, DefaultValues) {
  // Test default value for non-existent parameter
  int default_val = Config::get<int>("nonexistent_param", 42);
  EXPECT_EQ(default_val, 42);
  
  // Test default value for existing parameter (should return actual value)
  int n_up = Config::get<int>("n_up", 99);
  EXPECT_EQ(n_up, 5); // Should return actual value, not default
}

/**
 * @brief Test nested parameter access
 */
TEST_F(ConfigTest, NestedParameters) {
  EXPECT_NO_THROW({
    std::string point_group = Config::get<std::string>("chem/point_group");
    EXPECT_EQ(point_group, "c2v");
  });
}

/**
 * @brief Test missing parameter handling
 */
TEST_F(ConfigTest, MissingParameters) {
  // Should throw for missing required parameter
  EXPECT_THROW({
    Config::get<std::string>("missing_required_param");
  }, std::runtime_error);
}

/**
 * @brief Test configuration printing
 */
TEST_F(ConfigTest, ConfigurationPrinting) {
  // Should not throw when printing configuration
  EXPECT_NO_THROW({
    Config::print();
  });
}

/**
 * @brief Test minimal configuration
 */
TEST_F(ConfigTest, MinimalConfiguration) {
  // Clean up current config and create minimal one
  system("rm -f config.json");
  createMinimalConfig();
  
  // Create new config instance by accessing it
  EXPECT_NO_THROW({
    std::string system = Config::get<std::string>("system");
    EXPECT_EQ(system, "chem");
  });
}

/**
 * @brief Test array parameter access
 */
TEST_F(ConfigTest, ArrayParameters) {
  EXPECT_NO_THROW({
    auto eps_vars = Config::get<std::vector<double>>("eps_vars");
    EXPECT_EQ(eps_vars.size(), 3);
    EXPECT_DOUBLE_EQ(eps_vars[0], 1e-4);
    EXPECT_DOUBLE_EQ(eps_vars[1], 5e-5);
    EXPECT_DOUBLE_EQ(eps_vars[2], 2e-5);
  });
}

/**
 * @brief Test set functionality
 */
TEST_F(ConfigTest, SetParameters) {
  // Test setting a new parameter
  EXPECT_NO_THROW({
    Config::set("test_param", 123);
    int value = Config::get<int>("test_param");
    EXPECT_EQ(value, 123);
  });
  
  // Test setting string parameter
  EXPECT_NO_THROW({
    Config::set("test_string", std::string("test_value"));
    std::string value = Config::get<std::string>("test_string");
    EXPECT_EQ(value, "test_value");
  });
}