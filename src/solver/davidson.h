#pragma once

#include <vector>
#include <eigen/Eigen/Dense>
#include "sparse_matrix.h"

// Off-diagonal element structure for dynamic preconditioner
struct OffDiagElement {
  size_t i, j;
  double h_ij;      // Actual Hamiltonian matrix element
  double magnitude; // |c_i * H_ij * c_j| for selection
  
  bool operator<(const OffDiagElement& other) const {
    return magnitude < other.magnitude;  // Max-heap ordering
  }
};

// Sparse column representation for U and V matrices
struct SparseColumn {
  int rowIndex;
  double value;
};

class Davidson {
 public:
  Davidson(const unsigned n_states) {
    lowest_eigenvalues.resize(n_states);
    lowest_eigenvectors.resize(n_states);
    preconditioner_cached = false;
    cached_eigenvalue = 0.0;
    block_preconditioner_cached = false;
    block_wf_preconditioner_cached = false;
  }

  void diagonalize(
      const SparseMatrix& matrix,
      const std::vector<std::vector<double>>& initial_vectors,
      const double target_error,
      const bool verbose = false);

  // Set off-diagonal elements for dynamic preconditioner
  void set_off_diagonal_elements(const std::vector<OffDiagElement>& elements) {
    off_diagonal_elements = elements;
    printf("Davidson: Received %zu off-diagonal elements\n", off_diagonal_elements.size());
  }

  std::vector<double> get_lowest_eigenvalues() const { return lowest_eigenvalues; }

  std::vector<std::vector<double>> get_lowest_eigenvectors() const { return lowest_eigenvectors; }

  bool converged = false;

 private:
  std::vector<double> lowest_eigenvalues;
  std::vector<std::vector<double>> lowest_eigenvectors;
  std::vector<OffDiagElement> off_diagonal_elements;
  
  // Sparse representation for dynamic preconditioner (cached between iterations)
  std::vector<SparseColumn> U_sparse;
  std::vector<SparseColumn> V_sparse;
  Eigen::MatrixXd M_inv;                   // Pre-computed (I + V^T D^(-1) U)^(-1)
  bool preconditioner_cached;
  double cached_eigenvalue;
  
  // Block inversion data members
  std::vector<size_t> core_indices;       // Sorted list of unique determinant indices in core subspace
  Eigen::MatrixXd H_block_inv;            // Inverted subblock of Hamiltonian
  bool block_preconditioner_cached;
  
  // Wavefunction-based block inversion data members
  std::vector<size_t> wf_core_indices;    // Sorted list of determinant indices with largest |c_i|
  Eigen::MatrixXd H_block_wf_inv;         // Inverted subblock based on wavefunction
  bool block_wf_preconditioner_cached;
  
  // Dynamic preconditioner implementation
  std::vector<double> apply_dynamic_preconditioner(
      const std::vector<double>& r, 
      const double eigenvalue,
      const SparseMatrix& matrix,
      const std::vector<double>& current_eigenvector = std::vector<double>());
      
  // Strategy 1: Woodbury formula (original implementation)
  void prepare_sparse_preconditioner(const double eigenvalue, const SparseMatrix& matrix);
  std::vector<double> apply_sparse_woodbury(const std::vector<double>& r, const SparseMatrix& matrix);
  
  // Strategy 2: Block inversion (off-diagonal elements)
  void prepare_block_preconditioner(const double eigenvalue, const SparseMatrix& matrix);
  std::vector<double> apply_block_inversion(const std::vector<double>& r, const double eigenvalue, const SparseMatrix& matrix);
  
  // Strategy 2b: Block inversion (wavefunction coefficients)
  void prepare_block_preconditioner_wavefunction(const std::vector<double>& wavefunction, const double eigenvalue, const SparseMatrix& matrix);
  std::vector<double> apply_block_inversion_wavefunction(const std::vector<double>& r, const double eigenvalue, const SparseMatrix& matrix, const std::vector<double>& current_eigenvector);
  
  // Strategy 3: Iterative refinement
  std::vector<double> apply_iterative_refinement(const std::vector<double>& r, const double eigenvalue, const SparseMatrix& matrix);
};
