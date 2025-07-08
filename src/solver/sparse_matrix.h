#pragma once

#include <climits>
#include <complex>
#include <vector>
#include <functional>
#include "../parallel.h"
#include "../timer.h"
#include "../util.h"
#include "sparse_vector.h"

// Sparse matrix implementation for SHCI Hamiltonian
// Uses compressed sparse row (CSR) format with parallel operations
class SparseMatrix {
 public:
  // Get diagonal element at position i
  double get_diag(const size_t i) const { return diag[i]; }
  
  // Get full diagonal vector
  std::vector<double> get_diag() const { return diag; }

  // Cache diagonal elements for fast access
  void cache_diag();

  // Count total number of non-zero elements
  size_t count_n_elems() const;
 
  // Get number of matrix rows
  size_t count_n_rows() const { return rows.size(); }

  // Matrix-vector multiplication (real)
  std::vector<double> mul(const std::vector<double>& vec) const;
 
  // Matrix-vector multiplication (complex)
  std::vector<std::complex<double>> mul(const std::vector<std::complex<double>>& vec) const;

  // In-place matrix-vector multiplication for complex vectors
  void mul(
      const std::vector<double>& input_real,
      const std::vector<double>& input_imag,
      std::vector<double>& output_real,
      std::vector<double>& output_imag) const;

  // Add element to matrix (used during construction)
  void append_elem(const size_t i, const size_t j, const double& elem);

  // Set matrix dimensions
  void set_dim(const size_t dim);

  // Clear all matrix data
  void clear();

  // Sort elements in row i by column index
  void sort_row(const size_t i);

  // Print row i for debugging
  void print_row(const size_t i) const { rows[i].print(); }

  // Get sparse vector for row i
  const SparseVector& get_row(const size_t i) const { return rows[i]; }

  // Zero out all elements in row i
  void zero_out_row(size_t i) { rows[i].zero_out_vector(); };

  // Get connectivity graph (which rows connect to which)
  std::vector<std::vector<size_t>> get_connections() const;

  // Update existing matrix elements using provided function
  void update_existing_elems(std::function<double(const size_t, const size_t, const int)>);

 private:
  std::vector<SparseVector> rows;    // Sparse matrix rows
  std::vector<double> diag_local;    // Local diagonal elements (for parallel)
  std::vector<double> diag;          // Full diagonal vector
  
  // Parallel reduction sum for matrix-vector operations
  std::vector<double> reduce_sum(const std::vector<double>& vec) const;
};
