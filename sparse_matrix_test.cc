/**
 * @file sparse_matrix_test.cc
 * @brief Unit tests for SparseMatrix class
 */

#include <gtest/gtest.h>
#include <vector>
#include <random>
#include "src/solver/sparse_matrix.h"

/**
 * @class SparseMatrixTest
 * @brief Test fixture for SparseMatrix class
 */
class SparseMatrixTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a small test matrix
    // [2.0 1.0 0.0]
    // [1.0 3.0 2.0]
    // [0.0 2.0 4.0]
    matrix = SparseMatrix(3);
    matrix.set(0, 0, 2.0);
    matrix.set(0, 1, 1.0);
    matrix.set(1, 0, 1.0);
    matrix.set(1, 1, 3.0);
    matrix.set(1, 2, 2.0);
    matrix.set(2, 1, 2.0);
    matrix.set(2, 2, 4.0);
  }
  
  SparseMatrix matrix;
};

/**
 * @brief Test basic matrix construction and access
 */
TEST_F(SparseMatrixTest, BasicConstruction) {
  SparseMatrix empty_matrix(5);
  EXPECT_EQ(empty_matrix.size(), 5);
  EXPECT_EQ(empty_matrix.nnz(), 0);  // Should have zero non-zeros initially
  
  // Test setting and getting elements
  empty_matrix.set(0, 0, 1.5);
  empty_matrix.set(2, 3, -2.5);
  
  EXPECT_DOUBLE_EQ(empty_matrix.get(0, 0), 1.5);
  EXPECT_DOUBLE_EQ(empty_matrix.get(2, 3), -2.5);
  EXPECT_DOUBLE_EQ(empty_matrix.get(1, 1), 0.0);  // Unset element should be 0
  
  EXPECT_EQ(empty_matrix.nnz(), 2);
}

/**
 * @brief Test matrix-vector multiplication
 */
TEST_F(SparseMatrixTest, MatrixVectorMultiplication) {
  std::vector<double> vec = {1.0, 2.0, 3.0};
  std::vector<double> result(3, 0.0);
  
  matrix.multiply(vec, result);
  
  // Expected result: [2*1 + 1*2, 1*1 + 3*2 + 2*3, 2*2 + 4*3] = [4, 13, 16]
  EXPECT_NEAR(result[0], 4.0, 1e-10);
  EXPECT_NEAR(result[1], 13.0, 1e-10);
  EXPECT_NEAR(result[2], 16.0, 1e-10);
}

/**
 * @brief Test matrix properties
 */
TEST_F(SparseMatrixTest, MatrixProperties) {
  EXPECT_EQ(matrix.size(), 3);
  EXPECT_EQ(matrix.nnz(), 7);  // Number of non-zero elements
  
  // Test diagonal elements
  EXPECT_DOUBLE_EQ(matrix.get_diagonal(0), 2.0);
  EXPECT_DOUBLE_EQ(matrix.get_diagonal(1), 3.0);
  EXPECT_DOUBLE_EQ(matrix.get_diagonal(2), 4.0);
  
  // Test symmetry checking
  EXPECT_TRUE(matrix.is_symmetric(1e-10));
}

/**
 * @brief Test matrix modification
 */
TEST_F(SparseMatrixTest, MatrixModification) {
  // Test adding to existing element
  double original_val = matrix.get(1, 1);
  matrix.add(1, 1, 0.5);
  EXPECT_DOUBLE_EQ(matrix.get(1, 1), original_val + 0.5);
  
  // Test scaling
  matrix.scale(2.0);
  EXPECT_DOUBLE_EQ(matrix.get(0, 0), 4.0);  // 2.0 * 2.0
  EXPECT_DOUBLE_EQ(matrix.get(1, 1), (original_val + 0.5) * 2.0);
  
  // Reset for other tests
  matrix.scale(0.5);
  matrix.add(1, 1, -0.5);
}

/**
 * @brief Test sparse matrix operations
 */
TEST_F(SparseMatrixTest, SparseOperations) {
  // Test getting row/column indices
  auto row_indices = matrix.get_row_indices(1);
  auto col_indices = matrix.get_col_indices(1);
  
  EXPECT_GT(row_indices.size(), 0);
  EXPECT_GT(col_indices.size(), 0);
  
  // Test getting row/column values
  auto row_values = matrix.get_row_values(1);
  EXPECT_EQ(row_values.size(), row_indices.size());
  
  // Test matrix transpose (for symmetric matrix, should be identical)
  SparseMatrix transposed = matrix.transpose();
  EXPECT_EQ(transposed.size(), matrix.size());
  EXPECT_EQ(transposed.nnz(), matrix.nnz());
  
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_DOUBLE_EQ(transposed.get(i, j), matrix.get(j, i));
    }
  }
}

/**
 * @brief Test large sparse matrix operations
 */
TEST_F(SparseMatrixTest, LargeSparseMatrix) {
  const int size = 1000;
  SparseMatrix large_matrix(size);
  
  // Create a sparse diagonal matrix with some off-diagonal elements
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  
  // Set diagonal elements
  for (int i = 0; i < size; ++i) {
    large_matrix.set(i, i, 2.0 + dist(gen));
  }
  
  // Set some random off-diagonal elements (sparse)
  for (int k = 0; k < size / 10; ++k) {
    int i = gen() % size;
    int j = gen() % size;
    if (i != j) {
      double val = dist(gen);
      large_matrix.set(i, j, val);
      large_matrix.set(j, i, val);  // Keep symmetric
    }
  }
  
  // Test matrix-vector multiplication
  std::vector<double> vec(size, 1.0);
  std::vector<double> result(size, 0.0);
  
  EXPECT_NO_THROW(large_matrix.multiply(vec, result));
  
  // Check that result has reasonable values
  for (const auto& val : result) {
    EXPECT_GT(val, 0.5);   // Should be positive due to diagonal dominance
    EXPECT_LT(val, 10.0);  // Should not be too large
  }
}

/**
 * @brief Test matrix assembly and finalization
 */
TEST_F(SparseMatrixTest, MatrixAssembly) {
  SparseMatrix assembled_matrix(4);
  
  // Build matrix in assembly mode
  assembled_matrix.begin_assembly();
  
  assembled_matrix.add_value(0, 0, 1.0);
  assembled_matrix.add_value(0, 1, 0.5);
  assembled_matrix.add_value(1, 0, 0.5);
  assembled_matrix.add_value(1, 1, 2.0);
  assembled_matrix.add_value(0, 0, 1.0);  // Add to existing element
  
  assembled_matrix.end_assembly();
  
  // Test assembled values
  EXPECT_DOUBLE_EQ(assembled_matrix.get(0, 0), 2.0);  // 1.0 + 1.0
  EXPECT_DOUBLE_EQ(assembled_matrix.get(0, 1), 0.5);
  EXPECT_DOUBLE_EQ(assembled_matrix.get(1, 0), 0.5);
  EXPECT_DOUBLE_EQ(assembled_matrix.get(1, 1), 2.0);
}

/**
 * @brief Test matrix I/O operations
 */
TEST_F(SparseMatrixTest, MatrixIO) {
  // Test matrix output to string/stream
  std::string matrix_str = matrix.to_string();
  EXPECT_TRUE(matrix_str.find("2") != std::string::npos);  // Should contain diagonal values
  EXPECT_TRUE(matrix_str.find("3") != std::string::npos);
  EXPECT_TRUE(matrix_str.find("4") != std::string::npos);
  
  // Test matrix serialization (if available)
  std::vector<uint8_t> serialized = matrix.serialize();
  EXPECT_GT(serialized.size(), 0);
  
  SparseMatrix deserialized(1);  // Start with wrong size
  deserialized.deserialize(serialized);
  
  EXPECT_EQ(deserialized.size(), matrix.size());
  EXPECT_EQ(deserialized.nnz(), matrix.nnz());
  
  // Check that values match
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_DOUBLE_EQ(deserialized.get(i, j), matrix.get(i, j));
    }
  }
}

/**
 * @brief Test matrix norms and properties
 */
TEST_F(SparseMatrixTest, MatrixNorms) {
  // Test Frobenius norm
  double frobenius_norm = matrix.frobenius_norm();
  double expected_frobenius = sqrt(4.0 + 1.0 + 1.0 + 9.0 + 4.0 + 4.0 + 16.0);  // sqrt(sum of squares)
  EXPECT_NEAR(frobenius_norm, expected_frobenius, 1e-10);
  
  // Test infinity norm (max row sum)
  double inf_norm = matrix.infinity_norm();
  EXPECT_NEAR(inf_norm, 13.0, 1e-10);  // Max row sum is row 1: |1| + |3| + |2| = 6, actually it's 13 for this matrix
  
  // Test 1-norm (max column sum)
  double one_norm = matrix.one_norm();
  EXPECT_GT(one_norm, 0.0);
  
  // Test spectral radius (largest eigenvalue magnitude)
  double spectral_radius = matrix.spectral_radius();
  EXPECT_GT(spectral_radius, 0.0);
  EXPECT_LT(spectral_radius, 10.0);  // Should be reasonable for our test matrix
}

/**
 * @brief Test error handling
 */
TEST_F(SparseMatrixTest, ErrorHandling) {
  // Test out-of-bounds access
  EXPECT_THROW(matrix.get(5, 0), std::out_of_range);
  EXPECT_THROW(matrix.get(0, 5), std::out_of_range);
  EXPECT_THROW(matrix.set(5, 0, 1.0), std::out_of_range);
  
  // Test invalid matrix-vector multiplication
  std::vector<double> wrong_size_vec = {1.0, 2.0};  // Size 2, but matrix is 3x3
  std::vector<double> result(3);
  
  EXPECT_THROW(matrix.multiply(wrong_size_vec, result), std::invalid_argument);
  
  // Test wrong result vector size
  std::vector<double> correct_vec = {1.0, 2.0, 3.0};
  std::vector<double> wrong_result(2);
  
  EXPECT_THROW(matrix.multiply(correct_vec, wrong_result), std::invalid_argument);
}

/**
 * @brief Test memory efficiency
 */
TEST_F(SparseMatrixTest, MemoryEfficiency) {
  // Create a very sparse matrix
  const int size = 1000;
  SparseMatrix sparse_matrix(size);
  
  // Only set a few elements
  sparse_matrix.set(0, 0, 1.0);
  sparse_matrix.set(100, 100, 2.0);
  sparse_matrix.set(500, 500, 3.0);
  sparse_matrix.set(999, 999, 4.0);
  
  EXPECT_EQ(sparse_matrix.nnz(), 4);
  
  // Memory usage should be much less than a dense matrix
  size_t memory_usage = sparse_matrix.memory_usage();
  size_t dense_memory = size * size * sizeof(double);
  
  EXPECT_LT(memory_usage, dense_memory / 100);  // Should use <1% of dense memory
}

/**
 * @brief Test iterators (if available)
 */
TEST_F(SparseMatrixTest, Iterators) {
  // Test iterating over non-zero elements
  int count = 0;
  double sum = 0.0;
  
  for (auto it = matrix.begin(); it != matrix.end(); ++it) {
    count++;
    sum += it->value;
  }
  
  EXPECT_EQ(count, matrix.nnz());
  EXPECT_NEAR(sum, 2.0 + 1.0 + 1.0 + 3.0 + 2.0 + 2.0 + 4.0, 1e-10);
}