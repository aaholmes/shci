/**
 * @file config_test.cc
 * @brief Unit tests for configuration system
 */

#include <gtest/gtest.h>
#include <fstream>
#include <stdexcept>
#include "../config.h"

/**
 * @class ConfigTest
 * @brief Test fixture for configuration system
 */
class ConfigTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create test directory
    system("mkdir -p config_test_temp");
  }

  void TearDown() override {
    // Clean up
    system("rm -rf config_test_temp");
  }

  /**
   * @brief Create a valid test configuration
   */
  void createValidConfig() {
    std::ofstream config("config_test_temp/valid_config.json");
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
   * @brief Create invalid JSON configuration
   */
  void createInvalidConfig() {
    std::ofstream config("config_test_temp/invalid_config.json");
    config << "{\n";
    config << "  \"system\": \"chem\",\n";
    config << "  \"n_up\": 5,\n";
    config << "  \"n_dn\": 5\n";  // Missing comma
    config << "  \"eps_vars\": [1e-4\n";  // Invalid JSON
    config.close();
  }

  /**
   * @brief Create configuration with missing required fields
   */
  void createIncompleteConfig() {
    std::ofstream config("config_test_temp/incomplete_config.json");
    config << "{\n";
    config << "  \"system\": \"chem\"\n";
    config << "}\n";  // Missing required fields
    config.close();
  }
};

/**
 * @brief Test valid configuration loading
 */
TEST_F(ConfigTest, ValidConfigurationLoading) {
  createValidConfig();
  
  Config& config = Config::get();
  EXPECT_NO_THROW(config.load("config_test_temp/valid_config.json"));
  
  // Test string parameter
  EXPECT_EQ(config.get<std::string>("system"), "chem");
  
  // Test integer parameters
  EXPECT_EQ(config.get<int>("n_up"), 5);
  EXPECT_EQ(config.get<int>("n_dn"), 5);
  
  // Test double parameter
  EXPECT_DOUBLE_EQ(config.get<double>("eps_pt_dtm"), 1e-6);
  EXPECT_DOUBLE_EQ(config.get<double>("target_error"), 1e-4);
  
  // Test boolean parameter
  EXPECT_TRUE(config.get<bool>("time_sym"));
  
  // Test array parameter
  auto eps_vars = config.get<std::vector<double>>("eps_vars");
  EXPECT_EQ(eps_vars.size(), 3);
  EXPECT_DOUBLE_EQ(eps_vars[0], 1e-4);
  EXPECT_DOUBLE_EQ(eps_vars[1], 5e-5);
  EXPECT_DOUBLE_EQ(eps_vars[2], 2e-5);
  
  // Test nested object
  EXPECT_EQ(config.get<std::string>("chem/point_group"), "c2v");
}

/**
 * @brief Test invalid JSON handling
 */
TEST_F(ConfigTest, InvalidJSONHandling) {
  createInvalidConfig();
  
  Config& config = Config::get();
  EXPECT_THROW(config.load("config_test_temp/invalid_config.json"), std::exception);
}

/**
 * @brief Test missing file handling
 */
TEST_F(ConfigTest, MissingFileHandling) {
  Config& config = Config::get();
  EXPECT_THROW(config.load("config_test_temp/nonexistent.json"), std::exception);
}

/**
 * @brief Test incomplete configuration
 */
TEST_F(ConfigTest, IncompleteConfiguration) {
  createIncompleteConfig();
  
  Config& config = Config::get();
  EXPECT_NO_THROW(config.load("config_test_temp/incomplete_config.json"));
  
  // Should be able to access existing parameter
  EXPECT_EQ(config.get<std::string>("system"), "chem");
  
  // Should throw or return default for missing parameters
  EXPECT_THROW(config.get<int>("n_up"), std::exception);
}

/**
 * @brief Test default value functionality
 */
TEST_F(ConfigTest, DefaultValues) {
  createIncompleteConfig();
  
  Config& config = Config::get();
  config.load("config_test_temp/incomplete_config.json");
  
  // Test default value provision
  EXPECT_EQ(config.get<int>("n_up", 10), 10);  // Should return default
  EXPECT_EQ(config.get<std::string>("system", "default"), "chem");  // Should return actual value
  
  // Test default for non-existent nested parameter
  EXPECT_EQ(config.get<std::string>("chem/point_group", "c1"), "c1");
}

/**
 * @brief Test parameter validation
 */
TEST_F(ConfigTest, ParameterValidation) {
  // Create config with invalid values
  std::ofstream config("config_test_temp/invalid_values.json");
  config << "{\n";
  config << "  \"system\": \"invalid_system\",\n";
  config << "  \"n_up\": -1,\n";
  config << "  \"n_dn\": 0,\n";
  config << "  \"eps_vars\": [],\n";
  config << "  \"eps_pt_dtm\": -1.0\n";
  config << "}\n";
  config.close();
  
  Config& cfg = Config::get();
  EXPECT_NO_THROW(cfg.load("config_test_temp/invalid_values.json"));
  
  // The config system should load but validation might happen elsewhere
  EXPECT_EQ(cfg.get<std::string>("system"), "invalid_system");
  EXPECT_EQ(cfg.get<int>("n_up"), -1);
  
  // In practice, validation would happen in the solver or system initialization
}

/**
 * @brief Test configuration modification
 */
TEST_F(ConfigTest, ConfigurationModification) {
  createValidConfig();
  
  Config& config = Config::get();
  config.load("config_test_temp/valid_config.json");
  
  // Test setting new values (if supported)
  EXPECT_NO_THROW(config.set("n_up", 7));
  EXPECT_EQ(config.get<int>("n_up"), 7);
  
  // Test setting nested values
  EXPECT_NO_THROW(config.set("chem/point_group", "d2h"));
  EXPECT_EQ(config.get<std::string>("chem/point_group"), "d2h");
}

/**
 * @brief Test type conversion errors
 */
TEST_F(ConfigTest, TypeConversionErrors) {
  createValidConfig();
  
  Config& config = Config::get();
  config.load("config_test_temp/valid_config.json");
  
  // Try to get string as int (should throw)
  EXPECT_THROW(config.get<int>("system"), std::exception);
  
  // Try to get int as string (might work with conversion)
  EXPECT_NO_THROW(config.get<std::string>("n_up"));
  
  // Try to get array as single value
  EXPECT_THROW(config.get<double>("eps_vars"), std::exception);
}

/**
 * @brief Test configuration serialization
 */
TEST_F(ConfigTest, ConfigurationSerialization) {
  createValidConfig();
  
  Config& config = Config::get();
  config.load("config_test_temp/valid_config.json");
  
  // Test saving configuration
  EXPECT_NO_THROW(config.save("config_test_temp/saved_config.json"));
  
  // Test loading saved configuration
  Config& config2 = Config::get();
  EXPECT_NO_THROW(config2.load("config_test_temp/saved_config.json"));
  
  // Should have same values
  EXPECT_EQ(config2.get<std::string>("system"), "chem");
  EXPECT_EQ(config2.get<int>("n_up"), 5);
}

/**
 * @brief Test singleton behavior
 */
TEST_F(ConfigTest, SingletonBehavior) {
  Config& config1 = Config::get();
  Config& config2 = Config::get();
  
  // Should be the same instance
  EXPECT_EQ(&config1, &config2);
  
  // Changes in one should affect the other
  createValidConfig();
  config1.load("config_test_temp/valid_config.json");
  
  EXPECT_EQ(config2.get<std::string>("system"), "chem");
}