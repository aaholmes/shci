#include "davidson.h"
#include <algorithm>
#include <cmath>
#include <eigen/Eigen/Dense>
#include "../config.h"
#include <random>
#include <set>
#include <unordered_map>

void Davidson::diagonalize(
    const SparseMatrix& matrix,
    const std::vector<std::vector<double>>& initial_vectors,
    const double target_error,
    const bool verbose) {
  const double TOLERANCE = target_error;
  const size_t N_ITERATIONS_STORE = 5; // storage per state.
  
  // Dynamic preconditioner configuration
  const bool use_dynamic_preconditioner = Config::get<bool>("davidson/use_dynamic_preconditioner", false);
  const std::string preconditioner_strategy = Config::get<std::string>("davidson/preconditioner_strategy", "woodbury");
  const int preconditioner_rank_k = Config::get<int>("davidson/preconditioner_rank_k", 200);

  const size_t dim = initial_vectors[0].size();
  const unsigned n_states = std::min(dim, initial_vectors.size());
  for (auto& eigenvec : lowest_eigenvectors) eigenvec.resize(dim);

  if (dim == 1) {
    lowest_eigenvalues[0] = matrix.get_diag(0);
    lowest_eigenvectors[0].resize(1);
    lowest_eigenvectors[0][0] = 1.0;
    converged = true;
    return;
  }

  const size_t n_iterations_store = std::min(dim, N_ITERATIONS_STORE);
  std::vector<double> lowest_eigenvalues_prev(n_states, 0.0);

  std::vector<std::vector<double>> v(n_states * n_iterations_store);
  std::vector<std::vector<double>> Hv(n_states * n_iterations_store);
  std::vector<std::vector<double>> w(n_states);
  std::vector<std::vector<double>> Hw(n_states);
  for (size_t i = 0; i < v.size(); i++) {
    v[i].resize(dim);
  }

  for (size_t i = 0; i < n_states; i++) {
    w[i].resize(dim);
    Hw[i].resize(dim);
  }

  for (unsigned i_state = 0; i_state < n_states; i_state++) {
    double norm = sqrt(Util::dot_omp(initial_vectors[i_state], initial_vectors[i_state]));
#pragma omp parallel for
    for (size_t j = 0; j < dim; j++) v[i_state][j] = initial_vectors[i_state][j] / norm;
    if (i_state > 0) {
      // Orthogonalize
      double inner_prod;
      for (unsigned k_state = 0; k_state < i_state; k_state++) {
        inner_prod = Util::dot_omp(v[i_state], v[k_state]);
        for (size_t j = 0; j < dim; j++) v[i_state][j] -= inner_prod * v[k_state][j];
      }
      // Normalize
      norm = sqrt(Util::dot_omp(v[i_state], v[i_state]));
      for (size_t j = 0; j < dim; j++) v[i_state][j] /= norm;
    }
  }

  Eigen::MatrixXd h_krylov =
      Eigen::MatrixXd::Zero(n_states * n_iterations_store, n_states * n_iterations_store);
  Eigen::MatrixXd eigenvector_krylov =
      Eigen::MatrixXd::Zero(n_states * n_iterations_store, n_states * n_iterations_store);
  converged = false;
  size_t n_converged = 0;

  for (unsigned i_state = 0; i_state < n_states; i_state++) {
    Hv[i_state] = matrix.mul(v[i_state]);
    lowest_eigenvalues[i_state] = Util::dot_omp(v[i_state], Hv[i_state]);
    h_krylov(i_state, i_state) = lowest_eigenvalues[i_state];
    w[i_state] = v[i_state];
    Hw[i_state] = Hv[i_state];
  }
  if (verbose) {
    printf("Davidson #0:");
    for (const auto& eigenval : lowest_eigenvalues) printf("  %.10f", eigenval);
    printf("\n");
  }
  lowest_eigenvalues_prev = lowest_eigenvalues;

  size_t it_real = 1;
  size_t i_state_precond = 0; // state preconditioning on
  for (size_t it = n_states; it < n_iterations_store * n_states * 2; it++) {
    size_t it_circ = it % (n_states * n_iterations_store);
    if (it >= n_states * n_iterations_store) {
      if (it_circ < n_states - 1) continue;
      if (it_circ == n_states - 1) {
        for (unsigned i_state = 0; i_state < n_states; i_state++) {
          v[i_state] = w[i_state];
          Hv[i_state] = Hw[i_state];
        }
        for (unsigned i_state = 0; i_state < n_states; i_state++) {
          lowest_eigenvalues[i_state] = Util::dot_omp(v[i_state], Hv[i_state]);
          h_krylov(i_state, i_state) = lowest_eigenvalues[i_state];
          for (unsigned k_state = i_state + 1; k_state < n_states; k_state++) {
            double element = Util::dot_omp(v[i_state], Hv[k_state]);
            h_krylov(i_state, k_state) = element;
            h_krylov(k_state, i_state) = element;
          }
        }
        continue;
      }
    }

    // Apply preconditioner (diagonal or dynamic)
    bool can_use_dynamic = use_dynamic_preconditioner && 
                          (!off_diagonal_elements.empty() || preconditioner_strategy == "block_inversion_wavefunction");
    if (can_use_dynamic) {
      if (it == n_states) {  // First iteration only
        printf("Davidson: Using dynamic preconditioner with %zu elements\n", off_diagonal_elements.size());
      }
      // Compute residual vector
      std::vector<double> residual(dim);
      for (size_t j = 0; j < dim; j++) {
        residual[j] = Hw[i_state_precond][j] - lowest_eigenvalues[i_state_precond] * w[i_state_precond][j];
      }
      
      // Apply dynamic preconditioner using Woodbury identity
      std::vector<double> preconditioned = apply_dynamic_preconditioner(residual, lowest_eigenvalues[i_state_precond], matrix, w[i_state_precond]);
      
      for (size_t j = 0; j < dim; j++) {
        v[it_circ][j] = preconditioned[j];
      }
    } else {
      // Original diagonal preconditioner
#pragma omp parallel for
      for (size_t j = 0; j < dim; j++) {
        const double diff_to_diag = lowest_eigenvalues[i_state_precond] - matrix.get_diag(j);  // diag_elems[j];
        if (std::abs(diff_to_diag) < 1.0e-8) {
          v[it_circ][j] = 0.;
        } else {
          v[it_circ][j] = (Hw[i_state_precond][j] - lowest_eigenvalues[i_state_precond] * w[i_state_precond][j]) / diff_to_diag;
        }
      }
    }

    // Orthogonalize and normalize.
    for (size_t i = 0; i < it_circ; i++) {
      double norm = Util::dot_omp(v[it_circ], v[i]);
#pragma omp parallel for
      for (size_t j = 0; j < dim; j++) {
        v[it_circ][j] -= norm * v[i][j];
      }
    }
    double norm = sqrt(Util::dot_omp(v[it_circ], v[it_circ]));
    if (norm<1e-12) { // corner case: norm gets small before eigenvalues converge
      converged = true;
      break;
    }

#pragma omp parallel for
    for (size_t j = 0; j < dim; j++) {
      v[it_circ][j] /= norm;
    }

    Hv[it_circ] = matrix.mul(v[it_circ]);

    // Construct subspace matrix.
    for (size_t i = 0; i <= it_circ; i++) {
      h_krylov(it_circ, i) = Util::dot_omp(v[i], Hv[it_circ]);
      //h_krylov(it_circ, i) = h_krylov(i, it_circ); // only lower trianguluar part is referenced
    }

    // Diagonalize subspace matrix.
    if (i_state_precond + 1 == n_states) {
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(
          h_krylov.leftCols(it_circ + 1).topRows(it_circ + 1));
      const auto& eigenvals = eigenSolver.eigenvalues();  // in ascending order
      const auto& eigenvecs = eigenSolver.eigenvectors();
      for (unsigned i_state = 0; i_state < n_states; i_state++) {
        lowest_eigenvalues[i_state] = eigenvals(i_state);
        double factor = 1.0;
        if (eigenvecs(0, i_state) < 0) factor = -1.0;
        for (size_t i = 0; i < it_circ + 1; i++)
          eigenvector_krylov(i, i_state) = eigenvecs(i, i_state) * factor;
#pragma omp parallel for
        for (size_t j = 0; j < dim; j++) {
          double w_j = 0.0;
          double Hw_j = 0.0;
          for (size_t i = 0; i < it_circ + 1; i++) {
            w_j += v[i][j] * eigenvector_krylov(i, i_state);
            Hw_j += Hv[i][j] * eigenvector_krylov(i, i_state);
          }
          w[i_state][j] = w_j;
          Hw[i_state][j] = Hw_j;
        }
      }

      if (verbose) {
        printf("Davidson #%zu:", it_real);
        for (const auto& eigenval : lowest_eigenvalues) printf("  %.10f", eigenval);
        printf("\n");
      }
      it_real++;
      for (unsigned i_state = n_converged; i_state < n_states; i_state++) {
        if (std::abs(lowest_eigenvalues[i_state] - lowest_eigenvalues_prev[i_state]) > TOLERANCE) {
          break;
        } else {
          n_converged++;
        }
      }
      if (n_converged == n_states) converged = true;

      if (!converged) lowest_eigenvalues_prev = lowest_eigenvalues;

      if (converged) break;
    }
    i_state_precond++;
    if (i_state_precond >= n_states) i_state_precond = n_converged;
  }
  lowest_eigenvectors = w;
  if (n_states < initial_vectors.size()) { // Corner case for excited states
    lowest_eigenvectors.resize(initial_vectors.size());
    for (unsigned i = n_states; i < initial_vectors.size(); i++) 
      lowest_eigenvectors[i] = initial_vectors[i];
  }
}

std::vector<double> Davidson::apply_dynamic_preconditioner(
    const std::vector<double>& r, 
    const double eigenvalue,
    const SparseMatrix& matrix,
    const std::vector<double>& current_eigenvector) {
  
  // Select strategy based on configuration
  const std::string strategy = Config::get<std::string>("davidson/preconditioner_strategy", "woodbury");
  printf("DEBUG: Using preconditioner strategy: '%s'\n", strategy.c_str());
  
  if (strategy == "block_inversion_wavefunction") {
    // Wavefunction-based block inversion doesn't require off-diagonal elements
    return apply_block_inversion_wavefunction(r, eigenvalue, matrix, current_eigenvector);
  } else if (strategy == "iterative_refinement") {
    // Iterative refinement doesn't require off-diagonal elements
    return apply_iterative_refinement(r, eigenvalue, matrix);
  }
  
  // For strategies that require off-diagonal elements, check if they are available
  const size_t k = off_diagonal_elements.size();
  
  if (k == 0) {
    // Fallback to diagonal preconditioner for strategies that need off-diagonal elements
    const size_t dim = r.size();
    std::vector<double> result(dim);
    for (size_t i = 0; i < dim; i++) {
      const double diff_to_diag = eigenvalue - matrix.get_diag(i);
      if (std::abs(diff_to_diag) < 1.0e-8) {
        result[i] = 0.0;
      } else {
        result[i] = r[i] / diff_to_diag;
      }
    }
    return result;
  }
  
  if (strategy == "block_inversion") {
    return apply_block_inversion(r, eigenvalue, matrix);
  } else {
    // Default: Woodbury formula
    if (!preconditioner_cached || std::abs(eigenvalue - cached_eigenvalue) > 1e-12) {
      prepare_sparse_preconditioner(eigenvalue, matrix);
      preconditioner_cached = true;
      cached_eigenvalue = eigenvalue;
    }
    return apply_sparse_woodbury(r, matrix);
  }
}

void Davidson::prepare_sparse_preconditioner(const double eigenvalue, const SparseMatrix& matrix) {
  const size_t k = off_diagonal_elements.size();
  
  // Build sparse U and V representations
  // Each off-diagonal element (i,j) contributes two columns: one for position (i,j) and one for (j,i)
  U_sparse.clear();
  V_sparse.clear();
  U_sparse.reserve(2 * k);
  V_sparse.reserve(2 * k);
  
  for (size_t idx = 0; idx < k; idx++) {
    const auto& elem = off_diagonal_elements[idx];
    
    // Column for (i,j): U has 1 at row i, V has H_ij at row j
    U_sparse.push_back({static_cast<int>(elem.i), 1.0});
    V_sparse.push_back({static_cast<int>(elem.j), elem.h_ij});
    
    // Column for (j,i): U has 1 at row j, V has H_ji at row i (symmetric)
    U_sparse.push_back({static_cast<int>(elem.j), 1.0});
    V_sparse.push_back({static_cast<int>(elem.i), elem.h_ij});  // H_ji = H_ij for symmetric Hamiltonian
  }
  
  const size_t two_k = 2 * k;
  
  // Pre-compute M_inner = V^T * D^(-1) * U (2k x 2k matrix)
  std::vector<std::vector<double>> M_inner(two_k, std::vector<double>(two_k, 0.0));
  
  for (size_t n = 0; n < two_k; n++) {
    for (size_t m = 0; m < two_k; m++) {
      // M_inner(n,m) = V_sparse[n]^T * D^(-1) * U_sparse[m]
      // Since both V and U columns have only one non-zero entry, this is simple
      if (V_sparse[n].rowIndex == U_sparse[m].rowIndex) {
        const double D_inv_val = 1.0 / (eigenvalue - matrix.get_diag(U_sparse[m].rowIndex));
        if (std::abs(eigenvalue - matrix.get_diag(U_sparse[m].rowIndex)) > 1e-8) {
          M_inner[n][m] = V_sparse[n].value * D_inv_val * U_sparse[m].value;
        }
      }
    }
  }
  
  // Form M = I + M_inner using Eigen
  Eigen::MatrixXd M = Eigen::MatrixXd::Identity(two_k, two_k);
  for (size_t i = 0; i < two_k; i++) {
    for (size_t j = 0; j < two_k; j++) {
      M(i, j) += M_inner[i][j];
    }
  }
  
  // Compute M^(-1) using Eigen's optimized inverse (O(k³))
  M_inv = M.inverse();
}

std::vector<double> Davidson::apply_sparse_woodbury(const std::vector<double>& r, const SparseMatrix& matrix) {
  const size_t dim = r.size();
  const size_t two_k = U_sparse.size();
  
  // Step 1: y1 = D^(-1) * r  (Cost: O(N))
  std::vector<double> y1(dim);
  for (size_t i = 0; i < dim; i++) {
    const double diff_to_diag = cached_eigenvalue - matrix.get_diag(i);
    if (std::abs(diff_to_diag) < 1.0e-8) {
      y1[i] = 0.0;
    } else {
      y1[i] = r[i] / diff_to_diag;
    }
  }
  
  // Step 2: y2 = V^T * y1  (Cost: O(k))
  std::vector<double> y2(two_k);
  for (size_t m = 0; m < two_k; m++) {
    y2[m] = V_sparse[m].value * y1[V_sparse[m].rowIndex];
  }
  
  // Step 3: y3 = M_inv * y2  (Cost: O(k²))
  Eigen::VectorXd y2_eigen = Eigen::Map<const Eigen::VectorXd>(y2.data(), two_k);
  Eigen::VectorXd y3_eigen = M_inv * y2_eigen;
  std::vector<double> y3(y3_eigen.data(), y3_eigen.data() + y3_eigen.size());
  
  // Step 4: y4 = U * y3  (Cost: O(k))
  std::vector<double> y4(dim, 0.0);
  for (size_t m = 0; m < two_k; m++) {
    y4[U_sparse[m].rowIndex] += U_sparse[m].value * y3[m];
  }
  
  // Step 5: result = y1 - D^(-1) * y4  (Cost: O(N))
  std::vector<double> result(dim);
  for (size_t i = 0; i < dim; i++) {
    const double diff_to_diag = cached_eigenvalue - matrix.get_diag(i);
    if (std::abs(diff_to_diag) < 1.0e-8) {
      result[i] = y1[i];
    } else {
      result[i] = y1[i] - y4[i] / diff_to_diag;
    }
  }
  
  // Debug: Check magnitude of correction
  double y4_norm = 0.0;
  double result_norm = 0.0;
  for (size_t i = 0; i < dim; i++) {
    y4_norm += y4[i] * y4[i];
    result_norm += result[i] * result[i];
  }
  y4_norm = std::sqrt(y4_norm);
  result_norm = std::sqrt(result_norm);
  
  static int debug_count = 0;
  if (debug_count++ < 5) {
    printf("DEBUG: Woodbury correction ||y4|| = %e, ||result|| = %e, k=%zu\n", 
           y4_norm, result_norm, off_diagonal_elements.size());
  }
  
  return result;
}

void Davidson::prepare_block_preconditioner(const double eigenvalue, const SparseMatrix& matrix) {
  const size_t k = off_diagonal_elements.size();
  
  // Step 1: Identify core subspace - collect all unique indices
  std::set<size_t> core_set;
  for (const auto& elem : off_diagonal_elements) {
    core_set.insert(elem.i);
    core_set.insert(elem.j);
  }
  
  // Convert to sorted vector
  core_indices.assign(core_set.begin(), core_set.end());
  const size_t M = core_indices.size();
  
  if (M == 0) return;
  
  // Step 2: Construct dense M x M block
  Eigen::MatrixXd H_block = Eigen::MatrixXd::Zero(M, M);
  
  // Fill diagonal elements
  for (size_t row = 0; row < M; row++) {
    H_block(row, row) = matrix.get_diag(core_indices[row]) - eigenvalue;
  }
  
  // Fill off-diagonal elements
  for (const auto& elem : off_diagonal_elements) {
    // Find positions in core_indices
    auto it_i = std::lower_bound(core_indices.begin(), core_indices.end(), elem.i);
    auto it_j = std::lower_bound(core_indices.begin(), core_indices.end(), elem.j);
    
    if (it_i != core_indices.end() && it_j != core_indices.end()) {
      size_t row = it_i - core_indices.begin();
      size_t col = it_j - core_indices.begin();
      
      H_block(row, col) = elem.h_ij;
      H_block(col, row) = elem.h_ij;  // Symmetric
    }
  }
  
  // Step 3: Invert the block
  try {
    H_block_inv = H_block.inverse();
    printf("DEBUG: Block inversion successful, M=%zu, k=%zu\n", M, k);
  } catch (const std::exception& e) {
    printf("ERROR: Block inversion failed: %s\n", e.what());
    H_block_inv = Eigen::MatrixXd::Identity(M, M);  // Fallback to identity
  }
}

std::vector<double> Davidson::apply_block_inversion(
    const std::vector<double>& r, 
    const double eigenvalue, 
    const SparseMatrix& matrix) {
  
  // Prepare block preconditioner if not cached
  if (!block_preconditioner_cached || std::abs(eigenvalue - cached_eigenvalue) > 1e-12) {
    prepare_block_preconditioner(eigenvalue, matrix);
    block_preconditioner_cached = true;
    cached_eigenvalue = eigenvalue;
  }
  
  const size_t dim = r.size();
  const size_t M = core_indices.size();
  std::vector<double> result(dim);
  
  if (M == 0) {
    // Fallback to diagonal preconditioner
    for (size_t i = 0; i < dim; i++) {
      const double diff_to_diag = eigenvalue - matrix.get_diag(i);
      if (std::abs(diff_to_diag) < 1.0e-8) {
        result[i] = 0.0;
      } else {
        result[i] = r[i] / diff_to_diag;
      }
    }
    return result;
  }
  
  // Apply block inversion preconditioner
  
  // Step 1: Extract core subspace residual
  Eigen::VectorXd r_core(M);
  for (size_t i = 0; i < M; i++) {
    r_core(i) = r[core_indices[i]];
  }
  
  // Step 2: Apply block inverse
  Eigen::VectorXd correction_core = H_block_inv * r_core;
  
  // Step 3: Apply to full vector
  for (size_t i = 0; i < dim; i++) {
    // Check if this index is in core subspace
    auto it = std::lower_bound(core_indices.begin(), core_indices.end(), i);
    
    if (it != core_indices.end() && *it == i) {
      // In core subspace - use block correction
      size_t core_pos = it - core_indices.begin();
      result[i] = correction_core(core_pos);
    } else {
      // Outside core subspace - use diagonal preconditioner
      const double diff_to_diag = eigenvalue - matrix.get_diag(i);
      if (std::abs(diff_to_diag) < 1.0e-8) {
        result[i] = 0.0;
      } else {
        result[i] = r[i] / diff_to_diag;
      }
    }
  }
  
  // Debug output
  static int debug_count = 0;
  if (debug_count++ < 3) {
    double core_norm = correction_core.norm();
    double result_norm = 0.0;
    for (size_t i = 0; i < dim; i++) {
      result_norm += result[i] * result[i];
    }
    result_norm = std::sqrt(result_norm);
    printf("DEBUG: Block inversion ||core_correction|| = %e, ||result|| = %e, M=%zu\n", 
           core_norm, result_norm, M);
  }
  
  return result;
}

std::vector<double> Davidson::apply_iterative_refinement(
    const std::vector<double>& r, 
    const double eigenvalue, 
    const SparseMatrix& matrix) {
  
  const size_t dim = r.size();
  std::vector<double> result(dim);
  
  // Step 1: Get initial diagonal correction
  std::vector<double> dx_diag(dim);
  for (size_t i = 0; i < dim; i++) {
    const double diff_to_diag = eigenvalue - matrix.get_diag(i);
    if (std::abs(diff_to_diag) < 1.0e-8) {
      dx_diag[i] = 0.0;
    } else {
      dx_diag[i] = r[i] / diff_to_diag;
    }
  }
  
  // Step 2: One iteration of Jacobi-like refinement
  result = dx_diag;  // Start with diagonal correction
  
  // Create a map for fast lookup of off-diagonal elements
  std::unordered_map<size_t, std::vector<std::pair<size_t, double>>> off_diag_map;
  for (const auto& elem : off_diagonal_elements) {
    off_diag_map[elem.i].emplace_back(elem.j, elem.h_ij);
    if (elem.i != elem.j) {  // Avoid double-counting diagonal
      off_diag_map[elem.j].emplace_back(elem.i, elem.h_ij);
    }
  }
  
  // Apply refinement
  for (size_t i = 0; i < dim; i++) {
    double correction = 0.0;
    
    // Sum off-diagonal contributions
    auto it = off_diag_map.find(i);
    if (it != off_diag_map.end()) {
      for (const auto& pair : it->second) {
        size_t j = pair.first;
        double h_ij = pair.second;
        if (j < dim) {
          correction += h_ij * dx_diag[j];
        }
      }
    }
    
    // Apply refined correction: (1/H_ii) * (r_i - sum_j H_ij * dx_j)
    const double h_ii = matrix.get_diag(i) - eigenvalue;
    if (std::abs(h_ii) > 1.0e-8) {
      result[i] = (r[i] - correction) / h_ii;
    }
  }
  
  // Debug output
  static int debug_count = 0;
  if (debug_count++ < 3) {
    double diag_norm = 0.0, result_norm = 0.0;
    for (size_t i = 0; i < dim; i++) {
      diag_norm += dx_diag[i] * dx_diag[i];
      result_norm += result[i] * result[i];
    }
    diag_norm = std::sqrt(diag_norm);
    result_norm = std::sqrt(result_norm);
    printf("DEBUG: Iterative refinement ||dx_diag|| = %e, ||result|| = %e, k=%zu\n", 
           diag_norm, result_norm, off_diagonal_elements.size());
  }
  
  return result;
}

void Davidson::prepare_block_preconditioner_wavefunction(const std::vector<double>& wavefunction, const double eigenvalue, const SparseMatrix& matrix) {
  const int preconditioner_rank_k = Config::get<int>("davidson/preconditioner_rank_k", 200);
  const size_t dim = wavefunction.size();
  
  printf("DEBUG: prepare_block_preconditioner_wavefunction called with dim=%zu, rank_k=%d\n", dim, preconditioner_rank_k);
  
  // Clear previous indices
  wf_core_indices.clear();
  
  // Special case: If the total number of determinants is less than or equal to the block size,
  // ignore the c_i coefficients and invert the entire matrix
  if (dim <= static_cast<size_t>(preconditioner_rank_k)) {
    printf("DEBUG: Small system detected (dim=%zu <= rank_k=%d), inverting entire matrix\n", dim, preconditioner_rank_k);
    
    // Populate core subspace with all determinant indices from 0 to dim-1
    wf_core_indices.reserve(dim);
    for (size_t i = 0; i < dim; i++) {
      wf_core_indices.push_back(i);
    }
    // Already sorted since we're adding indices in order
    
  } else {
    // Original logic: Find determinants with largest |c_i| coefficients
    printf("DEBUG: Large system detected (dim=%zu > rank_k=%d), using |c_i| selection\n", dim, preconditioner_rank_k);
    
    std::vector<std::pair<double, size_t>> coeff_index_pairs;
    coeff_index_pairs.reserve(dim);
    
    for (size_t i = 0; i < dim; i++) {
      coeff_index_pairs.emplace_back(std::abs(wavefunction[i]), i);
    }
    
    // Sort by coefficient magnitude (largest first)
    std::partial_sort(coeff_index_pairs.begin(), 
                      coeff_index_pairs.begin() + std::min(static_cast<size_t>(preconditioner_rank_k), dim),
                      coeff_index_pairs.end(),
                      [](const std::pair<double, size_t>& a, const std::pair<double, size_t>& b) { return a.first > b.first; });
    
    // Extract the indices and sort them for efficient lookup
    wf_core_indices.reserve(std::min(static_cast<size_t>(preconditioner_rank_k), dim));
    
    for (size_t i = 0; i < std::min(static_cast<size_t>(preconditioner_rank_k), dim); i++) {
      wf_core_indices.push_back(coeff_index_pairs[i].second);
    }
    std::sort(wf_core_indices.begin(), wf_core_indices.end());
  }
  
  const size_t M = wf_core_indices.size();
  
  if (M == 0) return;
  
  // Step 2: Construct dense M x M block from Hamiltonian
  Eigen::MatrixXd H_block = Eigen::MatrixXd::Zero(M, M);
  
  // Fill diagonal elements (H_ii - eigenvalue)
  for (size_t row = 0; row < M; row++) {
    H_block(row, row) = matrix.get_diag(wf_core_indices[row]) - eigenvalue;
  }
  
  // Fill off-diagonal elements by querying the sparse matrix
  // Note: This is expensive O(M^2 * matrix_access_cost) but necessary for wavefunction-based selection
  for (size_t row = 0; row < M; row++) {
    for (size_t col = row + 1; col < M; col++) {
      size_t i = wf_core_indices[row];
      size_t j = wf_core_indices[col];
      
      // Get H_ij from sparse matrix (this is the expensive part)
      double h_ij = 0.0;
      const auto& row_i = matrix.get_row(i);
      for (size_t k = 0; k < row_i.size(); k++) {
        if (row_i.get_index(k) == j) {
          h_ij = row_i.get_value(k);
          break;
        }
      }
      
      if (std::abs(h_ij) > 1e-12) {
        H_block(row, col) = h_ij;
        H_block(col, row) = h_ij;  // Symmetric
      }
    }
  }
  
  // Step 3: Invert the block
  try {
    H_block_wf_inv = H_block.inverse();
    if (dim <= static_cast<size_t>(preconditioner_rank_k)) {
      printf("DEBUG: Complete matrix inversion successful, M=%zu (all determinants)\n", M);
    } else {
      printf("DEBUG: Wavefunction block inversion successful, M=%zu (from |c_i| selection)\n", M);
      
      // Debug: Print coefficient range (only for coefficient-based selection)
      if (M > 0) {
        // Note: coeff_index_pairs is only available when using coefficient-based selection
        // For small systems, we don't create this array, so we can't print coefficient ranges
        printf("DEBUG: Used coefficient-based selection for large system\n");
      }
    }
  } catch (const std::exception& e) {
    printf("ERROR: Wavefunction block inversion failed: %s\n", e.what());
    H_block_wf_inv = Eigen::MatrixXd::Identity(M, M);  // Fallback to identity
  }
}

std::vector<double> Davidson::apply_block_inversion_wavefunction(
    const std::vector<double>& r, 
    const double eigenvalue, 
    const SparseMatrix& matrix,
    const std::vector<double>& current_eigenvector) {
  
  // Check if current eigenvector is available
  printf("DEBUG: apply_block_inversion_wavefunction called, eigenvector size=%zu\n", current_eigenvector.size());
  if (current_eigenvector.empty()) {
    printf("DEBUG: No current eigenvector available for block inversion, falling back to diagonal\n");
    // Fallback to diagonal preconditioner if no wavefunction available
    const size_t dim = r.size();
    std::vector<double> result(dim);
    for (size_t i = 0; i < dim; i++) {
      const double diff_to_diag = eigenvalue - matrix.get_diag(i);
      if (std::abs(diff_to_diag) < 1.0e-8) {
        result[i] = 0.0;
      } else {
        result[i] = r[i] / diff_to_diag;
      }
    }
    return result;
  }
  
  // Prepare block preconditioner if not cached
  if (!block_wf_preconditioner_cached || std::abs(eigenvalue - cached_eigenvalue) > 1e-12) {
    prepare_block_preconditioner_wavefunction(current_eigenvector, eigenvalue, matrix);
    block_wf_preconditioner_cached = true;
    cached_eigenvalue = eigenvalue;
  }
  
  const size_t dim = r.size();
  const size_t M = wf_core_indices.size();
  std::vector<double> result(dim);
  
  if (M == 0) {
    // Fallback to diagonal preconditioner
    for (size_t i = 0; i < dim; i++) {
      const double diff_to_diag = eigenvalue - matrix.get_diag(i);
      if (std::abs(diff_to_diag) < 1.0e-8) {
        result[i] = 0.0;
      } else {
        result[i] = r[i] / diff_to_diag;
      }
    }
    return result;
  }
  
  // Apply wavefunction-based block inversion preconditioner
  
  // Step 1: Extract core subspace residual
  Eigen::VectorXd r_core(M);
  for (size_t i = 0; i < M; i++) {
    r_core(i) = r[wf_core_indices[i]];
  }
  
  // Step 2: Apply block inverse
  Eigen::VectorXd correction_core = H_block_wf_inv * r_core;
  
  // Step 3: Apply to full vector
  for (size_t i = 0; i < dim; i++) {
    // Check if this index is in wavefunction core subspace
    auto it = std::lower_bound(wf_core_indices.begin(), wf_core_indices.end(), i);
    
    if (it != wf_core_indices.end() && *it == i) {
      // In core subspace - use block correction
      size_t core_pos = it - wf_core_indices.begin();
      result[i] = correction_core(core_pos);
    } else {
      // Outside core subspace - use diagonal preconditioner
      const double diff_to_diag = eigenvalue - matrix.get_diag(i);
      if (std::abs(diff_to_diag) < 1.0e-8) {
        result[i] = 0.0;
      } else {
        result[i] = r[i] / diff_to_diag;
      }
    }
  }
  
  // Debug output
  static int debug_count = 0;
  if (debug_count++ < 3) {
    double core_norm = correction_core.norm();
    double result_norm = 0.0;
    for (size_t i = 0; i < dim; i++) {
      result_norm += result[i] * result[i];
    }
    result_norm = std::sqrt(result_norm);
    printf("DEBUG: Wavefunction block inversion ||core_correction|| = %e, ||result|| = %e, M=%zu\n", 
           core_norm, result_norm, M);
  }
  
  return result;
}
