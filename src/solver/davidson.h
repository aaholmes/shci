#pragma once

#include <vector>
#include "sparse_matrix.h"

// Off-diagonal element structure for dynamic preconditioner
struct OffDiagElement {
  size_t i, j;
  double magnitude;
  
  bool operator<(const OffDiagElement& other) const {
    return magnitude < other.magnitude;  // Max-heap ordering
  }
};

class Davidson {
 public:
  Davidson(const unsigned n_states) {
    lowest_eigenvalues.resize(n_states);
    lowest_eigenvectors.resize(n_states);
  }

  void diagonalize(
      const SparseMatrix& matrix,
      const std::vector<std::vector<double>>& initial_vectors,
      const double target_error,
      const bool verbose = false);

  // Set off-diagonal elements for dynamic preconditioner
  void set_off_diagonal_elements(const std::vector<OffDiagElement>& elements) {
    off_diagonal_elements = elements;
  }

  std::vector<double> get_lowest_eigenvalues() const { return lowest_eigenvalues; }

  std::vector<std::vector<double>> get_lowest_eigenvectors() const { return lowest_eigenvectors; }

  bool converged = false;

 private:
  std::vector<double> lowest_eigenvalues;
  std::vector<std::vector<double>> lowest_eigenvectors;
  std::vector<OffDiagElement> off_diagonal_elements;
  
  // Dynamic preconditioner implementation
  std::vector<double> apply_dynamic_preconditioner(
      const std::vector<double>& r, 
      const double eigenvalue,
      const SparseMatrix& matrix);
};
