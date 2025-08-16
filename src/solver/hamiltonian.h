#pragma once

#include <fgpl/src/dist_range.h>
#include <string>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <chrono>
#include "../base_system.h"
#include "../parallel.h"
#include "../timer.h"
#include "../util.h"
#include "sparse_matrix.h"

// Data structure for new opposite-spin algorithm
struct HamiltonianSetupData {
  // Up-spin structures
  std::vector<HalfDet> unique_up_dets;
  std::unordered_map<HalfDet, std::vector<size_t>, HalfDetHasher> up_to_full_map;
  std::unordered_map<size_t, std::vector<size_t>> upSingles;  // idx -> connected indices
  
  // Down-spin structures  
  std::vector<HalfDet> unique_dn_dets;
  std::unordered_map<HalfDet, std::vector<size_t>, HalfDetHasher> dn_to_full_map;
  std::unordered_map<size_t, std::vector<size_t>> dnSingles;
  
  // Additional helper structures
  std::unordered_map<HalfDet, size_t, HalfDetHasher> up_det_to_idx;
  std::unordered_map<HalfDet, size_t, HalfDetHasher> dn_det_to_idx;
};

template <class S>
class Hamiltonian {
 public:
  Hamiltonian();

  SparseMatrix matrix;

  void update(const S& system);

  void update_existing_elems(const S& system);

  void clear();

 private:
  size_t n_dets = 0;

  size_t n_dets_prev = 0;

  unsigned n_up = 0;

  unsigned n_dn = 0;

  bool time_sym = false;

  bool sort_by_det_id = false;

  size_t samespin_hash_threshold = 999999;

  std::vector<HalfDet> unique_alphas;

  std::vector<HalfDet> unique_betas;

  std::unordered_map<HalfDet, size_t, HalfDetHasher> alpha_to_id;

  std::unordered_map<HalfDet, size_t, HalfDetHasher> beta_to_id;

  std::unordered_map<HalfDet, std::pair<std::vector<size_t>, std::vector<size_t>>, HalfDetHasher>
      abm1_to_ab_ids;

  std::vector<std::vector<size_t>> alpha_id_to_single_ids;

  std::vector<std::vector<size_t>> beta_id_to_single_ids;

  // Sorted by unique beta id.
  std::vector<std::vector<size_t>> alpha_id_to_beta_ids;

  // Sorted by unique beta id.
  std::vector<std::vector<size_t>> alpha_id_to_det_ids;

  // Sorted by unique alpha id.
  std::vector<std::vector<size_t>> beta_id_to_alpha_ids;

  // Sorted by unique alpha id.
  std::vector<std::vector<size_t>> beta_id_to_det_ids;

  // Augment unique alphas/betas and alpha/beta to det info.
  void update_abdet(const S& system);

  // Update unique alpha/beta minus one.
  void update_abm1(const S& system);

  // Update alpha/beta singles lists.
  void update_absingles(const S& system);

  void update_matrix(const S& system);

  void sort_by_first(std::vector<size_t>& vec1, std::vector<size_t>& vec2);

  // Removed: Non-batch same-spin algorithms (violated DESIGN_DOC.md architecture)
                                       
  // Batch versions for group-based processing
  void find_same_spin_excitations_loop_batch(const S& system, 
                                             const std::vector<size_t>& new_det_indices,
                                             const std::vector<size_t>& old_det_indices,
                                             bool is_alpha_excitation);
                                             
  void find_same_spin_excitations_hash_batch(const S& system,
                                             const std::vector<size_t>& new_det_indices, 
                                             const std::vector<size_t>& old_det_indices,
                                             bool is_alpha_excitation);

  void generate_n_minus_2_cores(const HalfDet& half_det, std::vector<HalfDet>& cores_out) const;
  
  // New opposite-spin algorithms
  HamiltonianSetupData setup_variational_hamiltonian(const std::vector<Det>& variational_space);
  
  void find_opposite_spin_excitations_candidates(const S& system);
  void find_opposite_spin_excitations_removal(const S& system);
  void find_opposite_spin_excitations_pairwise(const S& system);
  void find_opposite_spin_excitations_adaptive(const S& system);
  
  // Cornell-style 2018 algorithm refactored for alpha-grouping
  void find_excitations_cornell_style_2018(const S& system);
  
  // Removed: find_opposite_spin_excitations_new (functionality merged into adaptive)
  
  // Sub-algorithms for new opposite-spin method
  void opposite_spin_candidates(const S& system,
                               const HamiltonianSetupData& setup_data,
                               size_t up_idx,
                               const std::vector<size_t>& dn_indices_i,
                               const std::vector<size_t>& up_singles);
                            
  void opposite_spin_removal(const S& system,
                            const HamiltonianSetupData& setup_data,
                            size_t up_idx,
                            const std::vector<size_t>& dn_indices_i,
                            const std::vector<size_t>& up_singles);
                            
  void opposite_spin_pairwise(const S& system,
                             const HamiltonianSetupData& setup_data,
                             size_t up_idx,
                             const std::vector<size_t>& dn_indices_i,
                             const std::vector<size_t>& up_singles);
  
  // Cost estimation functions
  double estimate_opposite_spin_candidates_cost(size_t n_dn_i, size_t n_up_singles, 
                                               size_t avg_dn_singles, size_t avg_dn_j) const;
  double estimate_opposite_spin_removal_cost(size_t n_dn_i, size_t n_up_singles, 
                                            size_t avg_dn_j, size_t n_electrons) const;
  double estimate_opposite_spin_pairwise_cost(size_t n_dn_i, size_t n_up_singles, 
                                             size_t avg_dn_j) const;
  
  // Helper functions for new algorithms
  std::vector<HalfDet> generate_n_minus_1_configs(const HalfDet& half_det) const;
  bool is_single_excitation(const HalfDet& det1, const HalfDet& det2) const;
  
  // Timing statistics for benchmarking
  double total_hamiltonian_time = 0.0;
  
  // Same-spin timing (detailed breakdown)
  double total_same_spin_time = 0.0;
  double total_same_spin_2018_time = 0.0;
  double total_same_spin_n2_time = 0.0;
  double total_same_spin_adaptive_time = 0.0;
  size_t total_same_spin_2018_calls = 0;
  size_t total_same_spin_n2_calls = 0;
  size_t total_same_spin_adaptive_calls = 0;
  
  // Opposite-spin timing (detailed breakdown)
  double total_opposite_spin_time = 0.0;
  double total_opposite_spin_2018_time = 0.0;
  // Removed: timing variables for old "new" framework (merged into adaptive)
  size_t total_opposite_spin_2018_calls = 0;
  // Removed: call count variables for old "new" framework (merged into adaptive)
  
  // Legacy compatibility (maps to adaptive algorithm)
  double total_loop_time = 0.0;
  double total_hash_time = 0.0;
  double total_2018_batch_time = 0.0;
  size_t total_loop_calls = 0;
  size_t total_hash_calls = 0;
  size_t total_2018_batch_calls = 0;
  
  // Per-alpha-group timing data for Test Plan B
  std::vector<size_t> alpha_group_ids;
  std::vector<size_t> alpha_group_beta_counts;
  std::vector<std::string> alpha_group_algorithms;
  std::vector<double> alpha_group_times;
  
  // Auto-tuning framework for dynamic threshold determination
  size_t auto_tuning_samples = 20;
  size_t dynamic_threshold = 1000;  // Will be updated each iteration
  bool auto_tuning_enabled = false;
  bool use_reusable_hash_map = true;  // Memory optimization for N-2 hash algorithm
  
  // Algorithm selection parameters
  std::string same_spin_algorithm = "2018";  // "2018", "n2", "adaptive", "loop"
  std::string opposite_spin_algorithm = "2018";  // "2018", "new"
  std::string opposite_spin_cost_model = "auto"; // "auto", "subalg1", "subalg2", "subalg3"
  bool opposite_spin_debug_output = false;
  
  // Performance model coefficients
  double loop_coeff_a = 2.5e-6;   // Loop: time = a * (M_new * M) + b
  double loop_coeff_b = 0.001;
  double hash_coeff_a = 8.0e-6;   // Hash: time = a * M + b * M_new + c
  double hash_coeff_b = 5.0e-6;
  double hash_coeff_c = 0.015;
  
  // Reusable hash map for N-2 core hashing to eliminate allocation overhead
  std::unordered_map<HalfDet, std::vector<size_t>, HalfDetHasher> reusable_core_map_;
  
  struct CalibrationPoint {
    size_t M_total;      // Total beta determinants
    size_t M_new;        // New beta determinants this iteration
    double time_loop;    // Time for loop algorithm (ms)
    double time_hash;    // Time for hash algorithm (ms)
  };
  
  std::vector<CalibrationPoint> calibration_data;
  
  void perform_auto_tuning_calibration(const S& system);
  void select_calibration_targets(std::vector<std::pair<size_t, size_t>>& targets);
  void fit_performance_models_and_calculate_threshold();
  double time_algorithm_for_calibration(const S& system, size_t alpha_id, bool use_hash);
  
  void print_timing_summary() const;
  void print_per_alpha_summary() const;
  void print_auto_tuning_report() const;
};

template <class S>
Hamiltonian<S>::Hamiltonian() {
  n_up = Config::get<unsigned>("n_up");
  n_dn = Config::get<unsigned>("n_dn");
  samespin_hash_threshold = Config::get<size_t>("samespin_hash_threshold", 1000);
  auto_tuning_samples = Config::get<size_t>("auto_tuning_samples", 20);
  auto_tuning_enabled = Config::get<bool>("auto_tuning_enabled", false);
  use_reusable_hash_map = Config::get<bool>("use_reusable_hash_map", true);
  dynamic_threshold = samespin_hash_threshold;  // Initialize with static threshold
  
  // Algorithm selection configuration
  same_spin_algorithm = Config::get<std::string>("same_spin_algorithm", "2018");
  opposite_spin_algorithm = Config::get<std::string>("opposite_spin_algorithm", "2018");
  opposite_spin_cost_model = Config::get<std::string>("opposite_spin_cost_model", "auto");
  opposite_spin_debug_output = Config::get<bool>("opposite_spin_debug_output", false);
}

template <class S>
void Hamiltonian<S>::update(const S& system) {
  n_dets_prev = n_dets;
  n_dets = system.get_n_dets();
  if (n_dets_prev == n_dets) return;
  time_sym = system.time_sym;
  sort_by_det_id = (n_dets < n_dets_prev * 1.15);

  update_abdet(system);
  Timer::checkpoint("update unique ab");

  if (system.has_double_excitation) {
    update_abm1(system);
    Timer::checkpoint("create abm1");
    update_absingles(system);
    abm1_to_ab_ids.clear();
    Util::free(abm1_to_ab_ids);
    Timer::checkpoint("create absingles");
  }
  update_matrix(system);
  alpha_id_to_single_ids.clear();
  alpha_id_to_single_ids.shrink_to_fit();
  beta_id_to_single_ids.clear();
  beta_id_to_single_ids.shrink_to_fit();
  Timer::checkpoint("generate sparse hamiltonian");
}

template <class S>
void Hamiltonian<S>::update_existing_elems(const S& system) {
  using namespace std::placeholders;
  if (system.time_sym) {
    auto pf = static_cast<double (BaseSystem::*) (const size_t, const size_t, const int) const>
    	(&BaseSystem::get_hamiltonian_elem_time_sym);
    matrix.update_existing_elems(std::bind(pf, &system, _1, _2, _3));
  } else {
    auto pf = static_cast<double (BaseSystem::*) (const size_t, const size_t, const int) const>
    	(&BaseSystem::get_hamiltonian_elem);
    matrix.update_existing_elems(std::bind(pf, &system, _1, _2, _3));
  }
}

template <class S>
void Hamiltonian<S>::clear() {
  n_dets = 0;
  n_dets_prev = 0;
  unique_alphas.clear();
  unique_alphas.shrink_to_fit();
  unique_betas.clear();
  unique_betas.shrink_to_fit();
  alpha_to_id.clear();
  Util::free(alpha_to_id);
  beta_to_id.clear();
  Util::free(beta_to_id);
  abm1_to_ab_ids.clear();
  Util::free(abm1_to_ab_ids);
  alpha_id_to_single_ids.clear();
  alpha_id_to_single_ids.shrink_to_fit();
  beta_id_to_single_ids.clear();
  beta_id_to_single_ids.shrink_to_fit();
  alpha_id_to_beta_ids.clear();
  alpha_id_to_beta_ids.shrink_to_fit();
  alpha_id_to_det_ids.clear();
  alpha_id_to_det_ids.shrink_to_fit();
  beta_id_to_alpha_ids.clear();
  beta_id_to_alpha_ids.shrink_to_fit();
  beta_id_to_det_ids.clear();
  beta_id_to_det_ids.shrink_to_fit();
  matrix.clear();
}

template <class S>
void Hamiltonian<S>::update_abdet(const S& system) {
  std::unordered_set<size_t> updated_alphas;
  std::unordered_set<size_t> updated_betas;
  for (size_t i = n_dets_prev; i < n_dets; i++) {
    const auto& det = system.dets[i];

    // Obtain alpha id.
    const auto& alpha = det.up;
    size_t alpha_id;
    if (alpha_to_id.count(alpha) == 0) {
      alpha_id = alpha_to_id.size();
      alpha_to_id[alpha] = alpha_id;
      unique_alphas.push_back(alpha);
      if (alpha_id_to_beta_ids.capacity() < alpha_id + 1) {
        alpha_id_to_beta_ids.reserve(alpha_id_to_beta_ids.capacity() * 2);
        alpha_id_to_det_ids.reserve(alpha_id_to_det_ids.capacity() * 2);
      }
      alpha_id_to_beta_ids.resize(alpha_id + 1);
      alpha_id_to_det_ids.resize(alpha_id + 1);
    } else {
      alpha_id = alpha_to_id[alpha];
    }
    updated_alphas.insert(alpha_id);

    // Obtain beta id.
    const auto& beta = det.dn;
    size_t beta_id;
    if (time_sym) {
      if (alpha_to_id.count(beta) == 0) {
        beta_id = alpha_to_id.size();
        alpha_to_id[beta] = beta_id;
        unique_alphas.push_back(beta);
        if (beta_id_to_alpha_ids.capacity() < beta_id + 1) {
          beta_id_to_alpha_ids.reserve(beta_id_to_alpha_ids.capacity() * 2);
          beta_id_to_det_ids.reserve(beta_id_to_det_ids.capacity() * 2);
        }
        beta_id_to_alpha_ids.resize(beta_id + 1);
        beta_id_to_det_ids.resize(beta_id + 1);
      } else {
        beta_id = alpha_to_id[beta];
      }
    } else {
      if (beta_to_id.count(beta) == 0) {
        beta_id = beta_to_id.size();
        beta_to_id[beta] = beta_id;
        unique_betas.push_back(beta);
        if (beta_id_to_alpha_ids.capacity() < beta_id + 1) {
          beta_id_to_alpha_ids.reserve(beta_id_to_alpha_ids.capacity() * 2);
          beta_id_to_det_ids.reserve(beta_id_to_det_ids.capacity() * 2);
        }
        beta_id_to_alpha_ids.resize(beta_id + 1);
        beta_id_to_det_ids.resize(beta_id + 1);
      } else {
        beta_id = beta_to_id[beta];
      }
    }
    updated_betas.insert(beta_id);

    if (time_sym) {
      if (alpha_id_to_beta_ids.size() <= alpha_id) {
        alpha_id_to_beta_ids.resize(alpha_id + 1);
        alpha_id_to_det_ids.resize(alpha_id + 1);
      }
      if (beta_id_to_alpha_ids.size() <= beta_id) {
        beta_id_to_alpha_ids.resize(beta_id + 1);
        beta_id_to_det_ids.resize(beta_id + 1);
      }
    }

    // Update alpha/beta to det info.
    alpha_id_to_beta_ids[alpha_id].push_back(beta_id);
    alpha_id_to_det_ids[alpha_id].push_back(i);
    beta_id_to_alpha_ids[beta_id].push_back(alpha_id);
    beta_id_to_det_ids[beta_id].push_back(i);
  }

  // Sort updated alpha/beta to det info.
  if (sort_by_det_id) {
    for (const size_t alpha_id : updated_alphas) {
      Util::sort_by_first<size_t, size_t>(
          alpha_id_to_det_ids[alpha_id], alpha_id_to_beta_ids[alpha_id]);
    }
    for (const size_t beta_id : updated_betas) {
      Util::sort_by_first<size_t, size_t>(
          beta_id_to_det_ids[beta_id], beta_id_to_alpha_ids[beta_id]);
    }
  } else {
    for (const size_t alpha_id : updated_alphas) {
      Util::sort_by_first<size_t, size_t>(
          alpha_id_to_beta_ids[alpha_id], alpha_id_to_det_ids[alpha_id]);
    }
    for (const size_t beta_id : updated_betas) {
      Util::sort_by_first<size_t, size_t>(
          beta_id_to_alpha_ids[beta_id], beta_id_to_det_ids[beta_id]);
    }
  }
}

template <class S>
void Hamiltonian<S>::update_abm1(const S& system) {
  std::unordered_set<size_t> updated_alphas;
  std::unordered_set<size_t> updated_betas;
  for (size_t i = n_dets_prev; i < n_dets; i++) {
    const auto& det = system.dets[i];

    // Update alpha m1.
    const auto& alpha = det.up;
    const size_t alpha_id = alpha_to_id[alpha];
    if (updated_alphas.count(alpha_id) == 0) {
      const auto& up_elecs = det.up.get_occupied_orbs();
      HalfDet alpha_m1 = det.up;
      for (unsigned j = 0; j < n_up; j++) {
        alpha_m1.unset(up_elecs[j]);
        abm1_to_ab_ids[alpha_m1].first.push_back(alpha_id);
        alpha_m1.set(up_elecs[j]);
      }
      updated_alphas.insert(alpha_id);
    }

    // Update beta m1.
    const auto& beta = det.dn;
    if (time_sym) {
      const size_t beta_id = alpha_to_id[beta];
      if (updated_alphas.count(beta_id) == 0) {
        const auto& dn_elecs = det.dn.get_occupied_orbs();
        HalfDet beta_m1 = det.dn;
        for (unsigned j = 0; j < n_dn; j++) {
          beta_m1.unset(dn_elecs[j]);
          abm1_to_ab_ids[beta_m1].first.push_back(beta_id);
          beta_m1.set(dn_elecs[j]);
        }
        updated_alphas.insert(beta_id);
      }
    } else {
      const size_t beta_id = beta_to_id[beta];
      if (updated_betas.count(beta_id) == 0) {
        const auto& dn_elecs = det.dn.get_occupied_orbs();
        HalfDet beta_m1 = det.dn;
        for (unsigned j = 0; j < n_dn; j++) {
          beta_m1.unset(dn_elecs[j]);
          abm1_to_ab_ids[beta_m1].second.push_back(beta_id);
          beta_m1.set(dn_elecs[j]);
        }
        updated_betas.insert(beta_id);
      }
    }
  }
}

template <class S>
void Hamiltonian<S>::update_absingles(const S& system) {
  std::unordered_set<size_t> updated_alphas;
  std::unordered_set<size_t> updated_betas;
  alpha_id_to_single_ids.resize(alpha_to_id.size());
  beta_id_to_single_ids.resize(beta_to_id.size());

  for (size_t i = n_dets_prev; i < n_dets; i++) {
    const auto& det = system.dets[i];

    const auto& alpha = det.up;
    const size_t alpha_id = alpha_to_id[alpha];
    updated_alphas.insert(alpha_id);

    const auto& beta = det.dn;
    if (time_sym) {
      const size_t beta_id = alpha_to_id[beta];
      updated_alphas.insert(beta_id);
    } else {
      const size_t beta_id = beta_to_id[beta];
      updated_betas.insert(beta_id);
    }
  }

  const size_t n_unique_alphas = alpha_to_id.size();
  const size_t n_unique_betas = beta_to_id.size();

  std::vector<omp_lock_t> locks;
  const size_t n_locks = std::max(n_unique_alphas, n_unique_betas);
  locks.resize(n_locks);
  for (auto& lock : locks) omp_init_lock(&lock);

#pragma omp parallel for schedule(static, 1)
  for (size_t alpha_id = 0; alpha_id < n_unique_alphas; alpha_id++) {
    const auto& alpha = unique_alphas[alpha_id];
    HalfDet alpha_m1 = alpha;
    const auto& up_elecs = alpha.get_occupied_orbs();
    for (unsigned j = 0; j < n_up; j++) {
      alpha_m1.unset(up_elecs[j]);
      if (abm1_to_ab_ids.count(alpha_m1) == 1) {
        for (const size_t alpha_single : abm1_to_ab_ids[alpha_m1].first) {
          if (alpha_single == alpha_id) continue;
          if (alpha_id > alpha_single && updated_alphas.count(alpha_id) &&
              updated_alphas.count(alpha_single)) {
            continue;  // Delegate to the alpha_single outer iteration.
          }
          omp_set_lock(&locks[alpha_id]);
          alpha_id_to_single_ids[alpha_id].push_back(alpha_single);
          omp_unset_lock(&locks[alpha_id]);
          omp_set_lock(&locks[alpha_single]);
          alpha_id_to_single_ids[alpha_single].push_back(alpha_id);
          omp_unset_lock(&locks[alpha_single]);
        }
      }
      alpha_m1.set(up_elecs[j]);
    }
  }

#pragma omp parallel for schedule(static, 1)
  for (size_t beta_id = 0; beta_id < n_unique_betas; beta_id++) {
    const auto& beta = unique_betas[beta_id];
    HalfDet beta_m1 = beta;
    const auto& dn_elecs = beta.get_occupied_orbs();
    for (unsigned j = 0; j < n_dn; j++) {
      beta_m1.unset(dn_elecs[j]);
      if (abm1_to_ab_ids.count(beta_m1) == 1) {
        for (const size_t beta_single : abm1_to_ab_ids[beta_m1].second) {
          if (beta_single == beta_id) continue;
          if (beta_id > beta_single && updated_betas.count(beta_id) &&
              updated_betas.count(beta_single)) {
            continue;
          }
          omp_set_lock(&locks[beta_id]);
          beta_id_to_single_ids[beta_id].push_back(beta_single);
          omp_unset_lock(&locks[beta_id]);
          omp_set_lock(&locks[beta_single]);
          beta_id_to_single_ids[beta_single].push_back(beta_id);
          omp_unset_lock(&locks[beta_single]);
        }
      }
      beta_m1.set(dn_elecs[j]);
    }
  }

  for (auto& lock : locks) omp_destroy_lock(&lock);

  // Sort updated alpha/beta singles and keep uniques.
  unsigned long long singles_cnt = 0;
#pragma omp parallel for schedule(static, 1) reduction(+ : singles_cnt)
  for (size_t alpha_id = 0; alpha_id < n_unique_alphas; alpha_id++) {
    std::sort(alpha_id_to_single_ids[alpha_id].begin(), alpha_id_to_single_ids[alpha_id].end());
    singles_cnt += alpha_id_to_single_ids[alpha_id].size();
  }
#pragma omp parallel for schedule(static, 1) reduction(+ : singles_cnt)
  for (size_t beta_id = 0; beta_id < n_unique_betas; beta_id++) {
    std::sort(beta_id_to_single_ids[beta_id].begin(), beta_id_to_single_ids[beta_id].end());
    singles_cnt += beta_id_to_single_ids[beta_id].size();
  }

  if (Parallel::is_master()) {
    printf(
        "Outer size of a/b singles: %'zu / %'zu\n",
        alpha_id_to_single_ids.size(),
        beta_id_to_single_ids.size());
    printf("Full size of absingles: %'llu\n", singles_cnt);
  }
}

template <class S>
void Hamiltonian<S>::update_matrix(const S& system) {
  if (Parallel::is_master()) {
    // Removed debug output for performance
  }
  
  matrix.set_dim(system.get_n_dets());
  
  auto start_hamiltonian_time = std::chrono::high_resolution_clock::now();
  
  // Initialize reusable hash map for N-2 core hashing optimization
  if (use_reusable_hash_map) {
    reusable_core_map_.clear();
    // Reserve memory to avoid re-hashes during construction
    reusable_core_map_.reserve(100000);
  }
  
  // Perform auto-tuning calibration only for adaptive algorithm
  if (same_spin_algorithm == "adaptive" && auto_tuning_enabled && Parallel::is_master()) {
    perform_auto_tuning_calibration(system);
  }

  // Step 1: Group All Determinants by HalfDet
  // Always use half-det grouping approach
  if (Parallel::is_master()) {
    // Removed debug output for performance
  }

  // Map from a HalfDet spin-string to all determinant indices that contain it.
  std::unordered_map<HalfDet, std::vector<size_t>, HalfDetHasher> alpha_det_groups;
  std::unordered_map<HalfDet, std::vector<size_t>, HalfDetHasher> beta_det_groups;
  
  // Add diagonal elements for new determinants first
  for (size_t det_id = 0; det_id < n_dets; det_id++) {
    const auto& det = system.dets[det_id];
    const bool is_new_det = det_id >= n_dets_prev;
    if (is_new_det) {
      const double H = time_sym ? system.get_hamiltonian_elem_time_sym(det, det, 0)
                                : system.get_hamiltonian_elem(det, det, 0);
      matrix.append_elem(det_id, det_id, H);
    }
    
    // Group determinants by their alpha AND beta half-determinants
    const auto& alpha = det.up;
    const auto& beta = det.dn;
    alpha_det_groups[alpha].push_back(det_id);
    beta_det_groups[beta].push_back(det_id);
  }
  
  // Step 2a: Handle alpha excitations (dets with same beta)
  if (Parallel::is_master()) {
    printf("Processing alpha excitations using %s algorithm...\n", same_spin_algorithm.c_str());
  }
  bool processing_alpha_excitations = true;
  for (auto const& group_pair : beta_det_groups) {
    const auto& half_det = group_pair.first;
    const auto& det_indices = group_pair.second;
    // Step 3: Implement Per-Group Adaptive Logic
    
    // Partition det_indices into new and old determinants
    std::vector<size_t> new_det_indices;
    std::vector<size_t> old_det_indices;
    
    for (size_t det_id : det_indices) {
      if (det_id >= n_dets_prev) {
        new_det_indices.push_back(det_id);
      } else {
        old_det_indices.push_back(det_id);
      }
    }
    
    // If no new determinants in this group, skip
    if (new_det_indices.empty()) {
      continue;
    }
    
    // Calculate N and N_new for cost model
    size_t N = det_indices.size();
    // size_t N_new = new_det_indices.size();  // unused but kept for future cost model
    
    // Per-group timing and algorithm selection tracking
    auto start_group_time = std::chrono::high_resolution_clock::now();
    
    std::string algorithm_chosen;
    
    // Apply refined cost model to choose algorithm for this entire group
    // Cost models: 
    //   Loop: time = loop_coeff_a * (M_new * M) + loop_coeff_b
    //   Hash: time = hash_coeff_a * M + hash_coeff_b * M_new + hash_coeff_c
    size_t M_new = new_det_indices.size();
    
    // Check if we're forcing a specific algorithm
    if (same_spin_algorithm == "loop") {
      algorithm_chosen = "Loop";
      find_same_spin_excitations_loop_batch(system, new_det_indices, old_det_indices, processing_alpha_excitations);
      total_loop_calls++;
    } else if (same_spin_algorithm == "hash") {
      algorithm_chosen = "N-2 Hash";
      find_same_spin_excitations_hash_batch(system, new_det_indices, old_det_indices, processing_alpha_excitations);
      total_hash_calls++;
    } else if (same_spin_algorithm == "adaptive") {
      // Use adaptive selection
      if (auto_tuning_enabled && calibration_data.size() >= 3) {
        // Use fitted cost models for decision
        double time_loop = loop_coeff_a * (M_new * N) + loop_coeff_b;
        double time_hash = hash_coeff_a * N + hash_coeff_b * M_new + hash_coeff_c;
        
        if (time_loop <= time_hash) {
          algorithm_chosen = "Loop";
          find_same_spin_excitations_loop_batch(system, new_det_indices, old_det_indices, processing_alpha_excitations);
          total_loop_calls++;
        } else {
          algorithm_chosen = "N-2 Hash";
          find_same_spin_excitations_hash_batch(system, new_det_indices, old_det_indices, processing_alpha_excitations);
          total_hash_calls++;
        }
      } else {
        // Fallback to simple threshold-based decision
        size_t threshold_to_use = auto_tuning_enabled ? dynamic_threshold : samespin_hash_threshold;
        if (N < threshold_to_use) {
          algorithm_chosen = "Loop";
          find_same_spin_excitations_loop_batch(system, new_det_indices, old_det_indices, processing_alpha_excitations);
          total_loop_calls++;
        } else {
          algorithm_chosen = "N-2 Hash";
          find_same_spin_excitations_hash_batch(system, new_det_indices, old_det_indices, processing_alpha_excitations);
          total_hash_calls++;
        }
      }
    } else {
      // Default to 2018 algorithm
      algorithm_chosen = "2018";
      // Map 2018 to loop algorithm for backward compatibility
      find_same_spin_excitations_loop_batch(system, new_det_indices, old_det_indices, processing_alpha_excitations);
      total_2018_batch_calls++;
    }
    
    auto end_group_time = std::chrono::high_resolution_clock::now();
    double group_time = std::chrono::duration<double>(end_group_time - start_group_time).count();
    
    if (algorithm_chosen == "Loop") {
      total_loop_time += group_time;
    } else if (algorithm_chosen == "N-2 Hash") {
      total_hash_time += group_time;
    } else if (algorithm_chosen == "2018") {
      total_2018_batch_time += group_time;
    }
    
    // Store per-group data for reporting (using first determinant as representative)
    if (!det_indices.empty()) {
      size_t representative_det_id = det_indices[0];
      const auto& representative_det = system.dets[representative_det_id];
      const auto& alpha = representative_det.up;
      size_t alpha_id = alpha_to_id[alpha];
      
      alpha_group_ids.push_back(alpha_id);
      alpha_group_beta_counts.push_back(N);
      alpha_group_algorithms.push_back(algorithm_chosen);
      alpha_group_times.push_back(group_time * 1000.0); // Convert to ms
    }
  }
  
  // Step 2b: Handle beta excitations (dets with same alpha)
  if (Parallel::is_master()) {
    printf("Processing beta excitations using %s algorithm...\n", same_spin_algorithm.c_str());
  }
  processing_alpha_excitations = false;
  for (auto const& group_pair : alpha_det_groups) {
    const auto& half_det = group_pair.first;
    const auto& det_indices = group_pair.second;
    
    // Partition det_indices into new and old determinants
    std::vector<size_t> new_det_indices;
    std::vector<size_t> old_det_indices;
    
    for (size_t det_id : det_indices) {
      if (det_id >= n_dets_prev) {
        new_det_indices.push_back(det_id);
      } else {
        old_det_indices.push_back(det_id);
      }
    }
    
    // If no new determinants in this group, skip
    if (new_det_indices.empty()) {
      continue;
    }
    
    // Calculate N for cost model
    size_t N = det_indices.size();
    
    // Per-group timing and algorithm selection tracking
    auto start_group_time = std::chrono::high_resolution_clock::now();
    
    std::string algorithm_chosen;
    
    // Check if we're forcing a specific algorithm
    if (same_spin_algorithm == "loop") {
      algorithm_chosen = "Loop";
      find_same_spin_excitations_loop_batch(system, new_det_indices, old_det_indices, processing_alpha_excitations);
      total_loop_calls++;
    } else if (same_spin_algorithm == "hash") {
      algorithm_chosen = "N-2 Hash";
      find_same_spin_excitations_hash_batch(system, new_det_indices, old_det_indices, processing_alpha_excitations);
      total_hash_calls++;
    } else if (same_spin_algorithm == "adaptive") {
      // Use adaptive selection
      size_t M_new = new_det_indices.size();
      if (auto_tuning_enabled && calibration_data.size() >= 3) {
        // Use fitted cost models for decision
        double time_loop = loop_coeff_a * (M_new * N) + loop_coeff_b;
        double time_hash = hash_coeff_a * N + hash_coeff_b * M_new + hash_coeff_c;
        
        if (time_loop <= time_hash) {
          algorithm_chosen = "Loop";
          find_same_spin_excitations_loop_batch(system, new_det_indices, old_det_indices, processing_alpha_excitations);
          total_loop_calls++;
        } else {
          algorithm_chosen = "N-2 Hash";
          find_same_spin_excitations_hash_batch(system, new_det_indices, old_det_indices, processing_alpha_excitations);
          total_hash_calls++;
        }
      } else {
        // Fallback to simple threshold-based decision
        size_t threshold_to_use = auto_tuning_enabled ? dynamic_threshold : samespin_hash_threshold;
        if (N < threshold_to_use) {
          algorithm_chosen = "Loop";
          find_same_spin_excitations_loop_batch(system, new_det_indices, old_det_indices, processing_alpha_excitations);
          total_loop_calls++;
        } else {
          algorithm_chosen = "N-2 Hash";
          find_same_spin_excitations_hash_batch(system, new_det_indices, old_det_indices, processing_alpha_excitations);
          total_hash_calls++;
        }
      }
    } else {
      // Default to 2018 algorithm
      algorithm_chosen = "2018";
      // Map 2018 to loop algorithm for backward compatibility
      find_same_spin_excitations_loop_batch(system, new_det_indices, old_det_indices, processing_alpha_excitations);
      total_2018_batch_calls++;
    }
    
    auto end_group_time = std::chrono::high_resolution_clock::now();
    double group_time = std::chrono::duration<double>(end_group_time - start_group_time).count();
    
    if (algorithm_chosen == "Loop") {
      total_loop_time += group_time;
    } else if (algorithm_chosen == "N-2 Hash") {
      total_hash_time += group_time;
    } else if (algorithm_chosen == "2018") {
      total_2018_batch_time += group_time;
    }
  }

  // Handle opposite-spin excitations based on selected algorithm
  auto opposite_spin_start_time = std::chrono::high_resolution_clock::now();
  
  if (Parallel::is_master()) {
    // Removed debug output for performance
  }
  
  if (opposite_spin_algorithm == "2018") {
    // Map 2018 to candidates algorithm (closest equivalent)
    find_opposite_spin_excitations_candidates(system);
  } else if (opposite_spin_algorithm == "candidates") {
    find_opposite_spin_excitations_candidates(system);
  } else if (opposite_spin_algorithm == "removal") {
    find_opposite_spin_excitations_removal(system);
  } else if (opposite_spin_algorithm == "pairwise") {
    find_opposite_spin_excitations_pairwise(system);
  } else if (opposite_spin_algorithm == "adaptive") {
    find_opposite_spin_excitations_adaptive(system);
  } else if (opposite_spin_algorithm == "hash") {
    // Backward compatibility: map "hash" to "candidates"
    find_opposite_spin_excitations_candidates(system);
  } else if (opposite_spin_algorithm == "loop") {
    // Backward compatibility: map "loop" to "pairwise"
    find_opposite_spin_excitations_pairwise(system);
  } else {
    // ERROR: No fallback allowed!
    printf("ERROR: Invalid opposite_spin_algorithm: '%s'\n", opposite_spin_algorithm.c_str());
    printf("Valid options: 2018, candidates, removal, pairwise, adaptive, hash, loop\n");
    throw std::runtime_error("Invalid opposite_spin_algorithm: " + opposite_spin_algorithm);
  }
  
  auto opposite_spin_end_time = std::chrono::high_resolution_clock::now();
  double opposite_spin_time = std::chrono::duration<double>(opposite_spin_end_time - opposite_spin_start_time).count();
  
  if (Parallel::is_master() && opposite_spin_debug_output) {
    printf("Opposite-spin processing complete: %.3fs\n", opposite_spin_time);
  }

  const size_t n_elems = matrix.count_n_elems();
  if (Parallel::is_master()) {
    printf("Number of nonzero elems: %'zu\n", n_elems);
  }
  matrix.cache_diag();
  
  auto end_hamiltonian_time = std::chrono::high_resolution_clock::now();
  total_hamiltonian_time = std::chrono::duration<double>(end_hamiltonian_time - start_hamiltonian_time).count();
  
  // Print timing summary - disabled for performance
  // print_timing_summary();
  // print_per_alpha_summary();
  // print_auto_tuning_report();
}

// Removed: Non-batch same-spin algorithms that violated DESIGN_DOC.md architecture
// These functions used determinant-by-determinant processing instead of half-det grouping

// Generate all N-2 electron cores from a half-determinant
template <class S>
void Hamiltonian<S>::generate_n_minus_2_cores(const HalfDet& half_det, std::vector<HalfDet>& cores_out) const {
  cores_out.clear();
  auto occupied_orbs = half_det.get_occupied_orbs();
  
  // Generate all combinations of removing 2 electrons from N occupied orbitals
  for (size_t i = 0; i < occupied_orbs.size(); i++) {
    for (size_t j = i + 1; j < occupied_orbs.size(); j++) {
      HalfDet core = half_det;
      core.unset(occupied_orbs[i]);
      core.unset(occupied_orbs[j]);
      cores_out.push_back(core);
    }
  }
}

// Batch loop algorithm for group-based processing
template <class S>
void Hamiltonian<S>::find_same_spin_excitations_loop_batch(const S& system, 
                                                           const std::vector<size_t>& new_det_indices,
                                                           const std::vector<size_t>& old_det_indices,
                                                           bool processing_alpha_excitations) {
  // Step 4: Execute the Loop Algorithm on the Entire Batch
  // Simple O(N*N_new) nested loop over new determinants vs all determinants in group
  for (size_t new_det_id : new_det_indices) {
    const auto& new_det = system.dets[new_det_id];
    
    // Check connections to all other determinants in the group
    for (size_t other_det_id : old_det_indices) {
      if (other_det_id >= new_det_id) continue; // Avoid double counting
      
      const auto& other_det = system.dets[other_det_id];
      const double H = time_sym ? system.get_hamiltonian_elem_time_sym(new_det, other_det, -1)
                                : system.get_hamiltonian_elem(new_det, other_det, -1);
      if (std::abs(H) < Util::EPS) continue;
      matrix.append_elem(new_det_id, other_det_id, H);
    }
    
    // Check connections to other new determinants (upper triangle only)
    for (size_t other_new_det_id : new_det_indices) {
      if (other_new_det_id <= new_det_id) continue; // Avoid double counting and self
      
      const auto& other_det = system.dets[other_new_det_id];
      const double H = time_sym ? system.get_hamiltonian_elem_time_sym(new_det, other_det, -1)
                                : system.get_hamiltonian_elem(new_det, other_det, -1);
      if (std::abs(H) < Util::EPS) continue;
      matrix.append_elem(new_det_id, other_new_det_id, H);
    }
  }
}

// Batch hash algorithm for group-based processing  
template <class S>
void Hamiltonian<S>::find_same_spin_excitations_hash_batch(const S& system,
                                                           const std::vector<size_t>& new_det_indices, 
                                                           const std::vector<size_t>& old_det_indices,
                                                           bool is_alpha_excitation) {
  // Step 4: Execute the Hash Algorithm on the Entire Batch
  // Build hash table once using old determinants, probe with all new determinants
  
  std::vector<HalfDet> generated_cores;
  
  // Build Phase: Create hash table from old determinants
  std::unordered_map<HalfDet, std::vector<size_t>, HalfDetHasher> local_core_map;
  
  for (size_t old_det_id : old_det_indices) {
    const auto& old_det = system.dets[old_det_id];
    
    // Use correct spin channel based on excitation type
    const auto& spin_channel = is_alpha_excitation ? old_det.up : old_det.dn;
    generate_n_minus_2_cores(spin_channel, generated_cores);
    for (const auto& core : generated_cores) {
      local_core_map[core].push_back(old_det_id);
    }
  }
  
  // Query Phase: Find connections for all new determinants
  for (size_t new_det_id : new_det_indices) {
    const auto& new_det = system.dets[new_det_id];
    
    // Use correct spin channel based on excitation type
    const auto& spin_channel = is_alpha_excitation ? new_det.up : new_det.dn;
    generate_n_minus_2_cores(spin_channel, generated_cores);
    std::set<size_t> visited_pairs;  // Avoid duplicates from multiple cores
    
    for (const auto& core : generated_cores) {
      auto it = local_core_map.find(core);
      if (it != local_core_map.end()) {
        for (size_t connected_det_id : it->second) {
          if (visited_pairs.find(connected_det_id) != visited_pairs.end()) continue;
          visited_pairs.insert(connected_det_id);
          
          const auto& connected_det = system.dets[connected_det_id];
          const double H = time_sym ? system.get_hamiltonian_elem_time_sym(new_det, connected_det, -1)
                                    : system.get_hamiltonian_elem(new_det, connected_det, -1);
          if (std::abs(H) < Util::EPS) continue;
          matrix.append_elem(new_det_id, connected_det_id, H);
        }
      }
    }
  }
  
  // Handle new-to-new connections using hash approach
  // Build hash table from new determinants
  std::unordered_map<HalfDet, std::vector<size_t>, HalfDetHasher> new_core_map;
  
  for (size_t new_det_id : new_det_indices) {
    const auto& new_det = system.dets[new_det_id];
    
    // Use correct spin channel based on excitation type
    const auto& spin_channel = is_alpha_excitation ? new_det.up : new_det.dn;
    generate_n_minus_2_cores(spin_channel, generated_cores);
    for (const auto& core : generated_cores) {
      new_core_map[core].push_back(new_det_id);
    }
  }
  
  // Query for new-to-new connections (avoid double counting)
  for (size_t new_det_id : new_det_indices) {
    const auto& new_det = system.dets[new_det_id];
    
    // Use correct spin channel based on excitation type
    const auto& spin_channel = is_alpha_excitation ? new_det.up : new_det.dn;
    generate_n_minus_2_cores(spin_channel, generated_cores);
    std::set<size_t> visited_pairs;
    
    for (const auto& core : generated_cores) {
      auto it = new_core_map.find(core);
      if (it != new_core_map.end()) {
        for (size_t connected_det_id : it->second) {
          if (connected_det_id <= new_det_id) continue; // Avoid double counting and self
          if (visited_pairs.find(connected_det_id) != visited_pairs.end()) continue;
          visited_pairs.insert(connected_det_id);
          
          const auto& connected_det = system.dets[connected_det_id];
          const double H = time_sym ? system.get_hamiltonian_elem_time_sym(new_det, connected_det, -1)
                                    : system.get_hamiltonian_elem(new_det, connected_det, -1);
          if (std::abs(H) < Util::EPS) continue;
          matrix.append_elem(new_det_id, connected_det_id, H);
        }
      }
    }
  }
}


// Print timing summary for benchmarking
template <class S>
void Hamiltonian<S>::print_timing_summary() const {
  if (Parallel::is_master()) {
    printf("\n========== HAMILTONIAN TIMING SUMMARY ==========\n");
    printf("Total Hamiltonian construction time: %.6f seconds\n", total_hamiltonian_time);
    
    // Same-spin timing breakdown
    double computed_same_spin_time = total_same_spin_2018_time + total_same_spin_n2_time + total_loop_time + total_hash_time + total_2018_batch_time;
    
    printf("\nSame-spin excitation timing breakdown:\n");
    if (total_same_spin_2018_calls > 0) {
      printf("  2018 Algorithm:\n");
      printf("    Total calls: %zu\n", total_same_spin_2018_calls);
      printf("    Total time: %.6f seconds\n", total_same_spin_2018_time);
      printf("    Average time per call: %.9f seconds\n", total_same_spin_2018_time / total_same_spin_2018_calls);
    }
    if (total_same_spin_n2_calls > 0) {
      printf("  N-2 Hash Algorithm:\n");
      printf("    Total calls: %zu\n", total_same_spin_n2_calls);
      printf("    Total time: %.6f seconds\n", total_same_spin_n2_time);
      printf("    Average time per call: %.9f seconds\n", total_same_spin_n2_time / total_same_spin_n2_calls);
    }
    if (total_2018_batch_calls > 0) {
      printf("  2018 Batch Algorithm:\n");
      printf("    Total calls: %zu\n", total_2018_batch_calls);
      printf("    Total time: %.6f seconds\n", total_2018_batch_time);
      printf("    Average time per call: %.9f seconds\n", total_2018_batch_time / total_2018_batch_calls);
    }
    if (total_same_spin_adaptive_calls > 0) {
      printf("  Adaptive Algorithm (legacy compatibility):\n");
      printf("    Loop calls: %zu (time: %.6fs)\n", total_loop_calls, total_loop_time);
      printf("    Hash calls: %zu (time: %.6fs)\n", total_hash_calls, total_hash_time);
      printf("    Total adaptive calls: %zu\n", total_same_spin_adaptive_calls);
      printf("    Total adaptive time: %.6f seconds\n", total_loop_time + total_hash_time);
    }
    printf("  Total same-spin time: %.6f seconds\n", computed_same_spin_time);
    if (total_hamiltonian_time > 0) {
      printf("  Same-spin fraction: %.2f%%\n", 100.0 * computed_same_spin_time / total_hamiltonian_time);
    }
    
    // Opposite-spin timing breakdown
    double computed_opposite_spin_time = total_opposite_spin_2018_time;
    
    printf("\nOpposite-spin excitation timing breakdown:\n");
    if (total_opposite_spin_2018_calls > 0) {
      printf("  2018 Algorithm:\n");
      printf("    Total calls: %zu\n", total_opposite_spin_2018_calls);
      printf("    Total time: %.6f seconds\n", total_opposite_spin_2018_time);
      printf("    Average time per call: %.9f seconds\n", total_opposite_spin_2018_time / total_opposite_spin_2018_calls);
    }
    // Note: Adaptive algorithm timing now integrated into total_opposite_spin_time
    printf("  Total opposite-spin time: %.6f seconds\n", computed_opposite_spin_time);
    if (total_hamiltonian_time > 0) {
      printf("  Opposite-spin fraction: %.2f%%\n", 100.0 * computed_opposite_spin_time / total_hamiltonian_time);
    }
    
    printf("\nAlgorithm Performance Summary:\n");
    if (computed_same_spin_time > 0 && computed_opposite_spin_time > 0) {
      printf("  Same-spin / Opposite-spin ratio: %.2f\n", computed_same_spin_time / computed_opposite_spin_time);
    }
    if (total_hamiltonian_time > 0) {
      double other_time = total_hamiltonian_time - computed_same_spin_time - computed_opposite_spin_time;
      printf("  Other operations time: %.6f seconds (%.2f%%)\n", 
             other_time, 100.0 * other_time / total_hamiltonian_time);
    }
    
    printf("================================================\n\n");
  }
}

// Print per-alpha-group timing summary for Test Plan B
template <class S>
void Hamiltonian<S>::print_per_alpha_summary() const {
  if (!Parallel::is_master() || alpha_group_ids.empty()) return;
  
  printf("\n========= PER-ALPHA-GROUP TIMING SUMMARY =========\n");
  printf("%-15s %-20s %-15s %-10s\n", "alpha_string_id", "num_beta_determinants", "algorithm_chosen", "time_ms");
  printf("----------------------------------------------------------------\n");
  
  for (size_t i = 0; i < alpha_group_ids.size(); i++) {
    printf("%-15zu %-20zu %-15s %-10.3f\n", 
           alpha_group_ids[i], 
           alpha_group_beta_counts[i], 
           alpha_group_algorithms[i].c_str(), 
           alpha_group_times[i]);
  }
  printf("================================================================\n\n");
}

// Perform auto-tuning calibration to determine optimal threshold
template <class S>
void Hamiltonian<S>::perform_auto_tuning_calibration(const S& system) {
  if (!Parallel::is_master()) return;
  
  printf("\n========= AUTO-TUNING CALIBRATION PHASE =========\n");
  
  // Select calibration targets from 50th-80th percentile groups
  std::vector<std::pair<size_t, size_t>> targets;  // (alpha_id, beta_count)
  select_calibration_targets(targets);
  
  if (targets.empty()) {
    printf("No suitable calibration targets found. Using static threshold: %zu\n", samespin_hash_threshold);
    dynamic_threshold = samespin_hash_threshold;
    return;
  }
  
  printf("Selected %zu calibration targets from intermediate-sized groups\n", targets.size());
  
  // Clear previous calibration data
  calibration_data.clear();
  
  // Perform timing-only calibration runs
  for (const auto& target : targets) {
    size_t alpha_id = target.first;
    size_t M_total = target.second;
    size_t M_new = std::min(M_total, n_dets - n_dets_prev);  // Estimate new determinants
    
    CalibrationPoint point;
    point.M_total = M_total;
    point.M_new = M_new;
    
    // Time loop algorithm
    point.time_loop = time_algorithm_for_calibration(system, alpha_id, false);
    
    // Time hash algorithm  
    point.time_hash = time_algorithm_for_calibration(system, alpha_id, true);
    
    calibration_data.push_back(point);
    
    printf("Calibration: M=%zu, M_new=%zu, loop=%.3fms, hash=%.3fms\n",
           M_total, M_new, point.time_loop, point.time_hash);
  }
  
  // Print raw calibration data table
  printf("\nRaw Calibration Data:\n");
  printf("%-10s %-10s %-15s %-15s\n", "M", "M_new", "time_loop_ms", "time_hash_ms");
  printf("--------------------------------------------------------\n");
  for (const auto& point : calibration_data) {
    printf("%-10zu %-10zu %-15.3f %-15.3f\n", 
           point.M_total, point.M_new, point.time_loop, point.time_hash);
  }
  printf("--------------------------------------------------------\n\n");
  
  // Fit performance models and calculate optimal threshold
  fit_performance_models_and_calculate_threshold();
  
  printf("Dynamic threshold determined: %zu\n", dynamic_threshold);
  printf("================================================\n\n");
}

// Select calibration targets from 50th-80th percentile groups
template <class S>
void Hamiltonian<S>::select_calibration_targets(std::vector<std::pair<size_t, size_t>>& targets) {
  targets.clear();
  
  // Collect all alpha group sizes
  std::vector<std::pair<size_t, size_t>> all_groups;  // (alpha_id, beta_count)
  for (size_t alpha_id = 0; alpha_id < alpha_id_to_det_ids.size(); alpha_id++) {
    size_t beta_count = alpha_id_to_det_ids[alpha_id].size();
    if (beta_count > 0) {
      all_groups.push_back({alpha_id, beta_count});
    }
  }
  
  if (all_groups.size() < 10) {
    printf("Warning: Too few alpha groups (%zu) for meaningful calibration\n", all_groups.size());
    return;
  }
  
  // Select groups with absolute sizes between 50-3000 determinants
  // This broader range ensures we capture more groups for comprehensive calibration
  size_t min_size = 50;
  size_t max_size = 3000;
  
  // Removed debug output for performance
  
  // Filter groups within the target size range
  std::vector<std::pair<size_t, size_t>> filtered_groups;
  for (const auto& group : all_groups) {
    if (group.second >= min_size && group.second <= max_size) {
      filtered_groups.push_back(group);
    }
  }
  
  if (filtered_groups.empty()) {
    printf("Warning: No groups found in size range [%zu, %zu] for calibration\n", min_size, max_size);
    return;
  }
  
  // Sort by beta_count for systematic sampling
  std::sort(filtered_groups.begin(), filtered_groups.end(), 
            [](const std::pair<size_t, size_t>& a, const std::pair<size_t, size_t>& b) { 
              return a.second < b.second; 
            });
  
  // Sample uniformly across the size range
  size_t samples_to_take = std::min(auto_tuning_samples, filtered_groups.size());
  
  for (size_t i = 0; i < samples_to_take; i++) {
    size_t idx = (i * filtered_groups.size()) / samples_to_take;
    targets.push_back(filtered_groups[idx]);
  }
  
  printf("Calibration range: [%zu, %zu] determinants (absolute), sampling %zu groups from %zu candidates\n",
         min_size, max_size, targets.size(), filtered_groups.size());
         
  // Removed debug output for performance
}

// Time a single algorithm for calibration (timing-only, no matrix updates)
template <class S>
double Hamiltonian<S>::time_algorithm_for_calibration(const S& system, size_t alpha_id, bool use_hash) {
  if (alpha_id >= alpha_id_to_det_ids.size()) return 0.0;
  
  const auto& beta_dets = alpha_id_to_det_ids[alpha_id];
  if (beta_dets.empty()) return 0.0;
  
  // Use a representative determinant from this alpha group
  size_t det_id = beta_dets[0];
  const auto& det = system.dets[det_id];
  
  auto start_time = std::chrono::high_resolution_clock::now();
  
  if (use_hash) {
    // Time hash algorithm (without adding to matrix)
    // Single reusable vector for core generation
    std::vector<HalfDet> generated_cores;
    size_t n_electrons = det.dn.get_occupied_orbs().size();
    size_t n2_pairs = (n_electrons * (n_electrons - 1)) / 2;
    generated_cores.reserve(n2_pairs);
    
    // Generate N-2 cores for current determinant
    generate_n_minus_2_cores(det.dn, generated_cores);
    
    // Create temporary hash table for timing
    std::unordered_map<HalfDet, std::vector<size_t>, HalfDetHasher> core_to_indices;
    for (size_t i = 0; i < beta_dets.size(); i++) {
      const auto& beta_det = system.dets[beta_dets[i]];
      generate_n_minus_2_cores(beta_det.dn, generated_cores);
      for (const auto& core : generated_cores) {
        core_to_indices[core].push_back(i);
      }
    }
    
    // Perform lookups
    generate_n_minus_2_cores(det.dn, generated_cores);
    for (const auto& core : generated_cores) {
      auto it = core_to_indices.find(core);
      if (it != core_to_indices.end()) {
        // Process matches (timing only)
        for (size_t idx : it->second) {
          volatile double dummy = system.get_hamiltonian_elem(det, system.dets[beta_dets[idx]], -1);
          (void)dummy;  // Suppress unused variable warning
        }
      }
    }
  } else {
    // Time loop algorithm (without adding to matrix)
    for (auto it = beta_dets.begin(); it != beta_dets.end(); it++) {
      const size_t beta_det_id = *it;
      const auto& connected_det = system.dets[beta_det_id];
      volatile double dummy = system.get_hamiltonian_elem(det, connected_det, -1);
      (void)dummy;  // Suppress unused variable warning
    }
  }
  
  auto end_time = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double>(end_time - start_time).count() * 1000.0;  // Convert to ms
}

// Fit performance models and calculate optimal threshold
template <class S>
void Hamiltonian<S>::fit_performance_models_and_calculate_threshold() {
  if (calibration_data.size() < 3) {
    printf("Warning: Insufficient calibration data (%zu points) for model fitting\n", calibration_data.size());
    dynamic_threshold = samespin_hash_threshold;
    return;
  }
  
  // Perform multiple linear regression for both algorithms
  // Model: time_loop(M, M_new) = a1 * (M_new * M) + b1
  // Model: time_hash(M, M_new) = a2 * M + b2 * M_new + c2 (build + probe + setup)
  
  size_t n = calibration_data.size();
  
  // Calculate regression coefficients for loop algorithm: time = a * (M_new * M) + b
  double sum_x_loop = 0, sum_y_loop = 0, sum_xx_loop = 0, sum_xy_loop = 0;
  for (const auto& point : calibration_data) {
    double x = static_cast<double>(point.M_new * point.M_total);
    double y = point.time_loop;
    sum_x_loop += x;
    sum_y_loop += y;
    sum_xx_loop += x * x;
    sum_xy_loop += x * y;
  }
  
  loop_coeff_a = (n * sum_xy_loop - sum_x_loop * sum_y_loop) / (n * sum_xx_loop - sum_x_loop * sum_x_loop);
  loop_coeff_b = (sum_y_loop - loop_coeff_a * sum_x_loop) / n;
  
  // Calculate regression coefficients for hash algorithm: time = a2 * M + b2 * M_new + c2
  // Using multiple linear regression with three parameters
  double sum_m = 0, sum_m_new = 0, sum_y_hash = 0;
  double sum_mm = 0, sum_m_new_m_new = 0, sum_m_m_new = 0;
  double sum_m_y = 0, sum_m_new_y = 0;
  
  for (const auto& point : calibration_data) {
    double m = static_cast<double>(point.M_total);
    double m_new = static_cast<double>(point.M_new);
    double y = point.time_hash;
    
    sum_m += m;
    sum_m_new += m_new;
    sum_y_hash += y;
    sum_mm += m * m;
    sum_m_new_m_new += m_new * m_new;
    sum_m_m_new += m * m_new;
    sum_m_y += m * y;
    sum_m_new_y += m_new * y;
  }
  
  // Solve the normal equations for multiple linear regression
  // [sum_mm       sum_m_m_new  sum_m    ] [a2]   [sum_m_y    ]
  // [sum_m_m_new  sum_m_new^2  sum_m_new] [b2] = [sum_m_new_y]
  // [sum_m        sum_m_new    n        ] [c2]   [sum_y_hash ]
  
  double det = n * (sum_mm * sum_m_new_m_new - sum_m_m_new * sum_m_m_new) -
               sum_m * (sum_m * sum_m_new_m_new - sum_m_new * sum_m_m_new) +
               sum_m_new * (sum_m * sum_m_m_new - sum_mm * sum_m_new);
  
  if (std::abs(det) > 1e-12) {
    hash_coeff_a = (n * (sum_m_y * sum_m_new_m_new - sum_m_new_y * sum_m_m_new) -
                    sum_m_new * (sum_m * sum_m_new_y - sum_y_hash * sum_m_m_new) +
                    sum_y_hash * (sum_m * sum_m_new_m_new - sum_m_new * sum_m_m_new)) / det;
                    
    hash_coeff_b = (sum_mm * (n * sum_m_new_y - sum_m_new * sum_y_hash) -
                    sum_m * (sum_m * sum_m_new_y - sum_y_hash * sum_m_m_new) +
                    sum_y_hash * (sum_m * sum_m_new - sum_mm * sum_m_new)) / det;
                    
    hash_coeff_c = (sum_mm * (sum_m_new_m_new * sum_y_hash - sum_m_new * sum_m_new_y) -
                    sum_m_m_new * (sum_m_m_new * sum_y_hash - sum_m * sum_m_new_y) +
                    sum_m_y * (sum_m_m_new * sum_m_new - sum_mm * sum_m_new)) / det;
  } else {
    // Fallback to simple linear regression if matrix is singular
    printf("Warning: Singular matrix in hash regression, using fallback\n");
    double sum_x_hash = sum_m + sum_m_new;
    double sum_xx_hash = sum_mm + 2*sum_m_m_new + sum_m_new_m_new;
    double sum_xy_hash = sum_m_y + sum_m_new_y;
    hash_coeff_a = hash_coeff_b = (n * sum_xy_hash - sum_x_hash * sum_y_hash) / (n * sum_xx_hash - sum_x_hash * sum_x_hash);
    hash_coeff_c = (sum_y_hash - (hash_coeff_a + hash_coeff_b) * sum_x_hash) / n;
  }
  
  printf("Performance models fitted:\n");
  printf("  Loop: time = %.6f * (M_new * M) + %.6f\n", loop_coeff_a, loop_coeff_b);
  printf("  Hash: time = %.6f * M + %.6f * M_new + %.6f\n", hash_coeff_a, hash_coeff_b, hash_coeff_c);
  
  // Find crossover threshold where both algorithms have equal time
  // For each calibration point, calculate what M would give equal times
  // Then take the median or average of these crossover points
  std::vector<double> crossover_points;
  
  for (const auto& point : calibration_data) {
    // For this specific M_new, find M where:
    // loop_coeff_a * (M_new * M) + loop_coeff_b = hash_coeff_a * M + hash_coeff_b * M_new + hash_coeff_c
    // loop_coeff_a * M_new * M + loop_coeff_b = hash_coeff_a * M + hash_coeff_b * M_new + hash_coeff_c
    // loop_coeff_a * M_new * M - hash_coeff_a * M = hash_coeff_b * M_new + hash_coeff_c - loop_coeff_b
    // M * (loop_coeff_a * M_new - hash_coeff_a) = hash_coeff_b * M_new + hash_coeff_c - loop_coeff_b
    
    double coeff = loop_coeff_a * point.M_new - hash_coeff_a;
    double rhs = hash_coeff_b * point.M_new + hash_coeff_c - loop_coeff_b;
    
    if (std::abs(coeff) > 1e-12) {
      double crossover_M = rhs / coeff;
      if (crossover_M > 0 && crossover_M < 100000) {
        crossover_points.push_back(crossover_M);
      }
    }
  }
  
  if (!crossover_points.empty()) {
    // Use median of crossover points for robustness
    std::sort(crossover_points.begin(), crossover_points.end());
    size_t mid = crossover_points.size() / 2;
    double median_crossover = crossover_points[mid];
    dynamic_threshold = static_cast<size_t>(std::max(1.0, median_crossover));
    
    printf("Crossover points from calibration data: ");
    for (size_t i = 0; i < crossover_points.size(); i++) {
      printf("%.1f", crossover_points[i]);
      if (i < crossover_points.size() - 1) printf(", ");
    }
    printf("\n");
    printf("Median crossover point: %.1f\n", median_crossover);
  } else {
    // Fallback if no valid crossover points found
    dynamic_threshold = samespin_hash_threshold;
    printf("Warning: No valid crossover points found, using fallback threshold\n");
  }
  
  // Sanity check: keep threshold within reasonable bounds
  dynamic_threshold = std::max(static_cast<size_t>(1), std::min(dynamic_threshold, static_cast<size_t>(50000)));
  
  printf("Final calculated threshold: M = %zu\n", dynamic_threshold);
}

// Print auto-tuning calibration report
template <class S>
void Hamiltonian<S>::print_auto_tuning_report() const {
  if (!Parallel::is_master() || !auto_tuning_enabled) return;
  
  printf("\n========= AUTO-TUNING PERFORMANCE REPORT =========\n");
  printf("Auto-tuning enabled: %s\n", auto_tuning_enabled ? "Yes" : "No");
  printf("Calibration samples: %zu\n", auto_tuning_samples);
  printf("Dynamic threshold: %zu\n", dynamic_threshold);
  printf("Static threshold: %zu\n", samespin_hash_threshold);
  
  if (!calibration_data.empty()) {
    printf("\nCalibration data (%zu points):\n", calibration_data.size());
    printf("%-8s %-8s %-12s %-12s %-10s\n", "M_total", "M_new", "Loop(ms)", "Hash(ms)", "Speedup");
    printf("------------------------------------------------------------\n");
    
    for (const auto& point : calibration_data) {
      double speedup = (point.time_hash > 0) ? point.time_loop / point.time_hash : 1.0;
      printf("%-8zu %-8zu %-12.3f %-12.3f %-10.2fx\n",
             point.M_total, point.M_new, point.time_loop, point.time_hash, speedup);
    }
  }
  
  printf("==================================================\n\n");
}

// Implementation of new opposite-spin algorithms

template <class S>
HamiltonianSetupData Hamiltonian<S>::setup_variational_hamiltonian(const std::vector<Det>& variational_space) {
  HamiltonianSetupData setup_data;
  
  if (Parallel::is_master() && opposite_spin_debug_output) {
    printf("Setting up variational Hamiltonian data structures...\n");
  }
  
  // Step 1: Extract unique half-determinants and build maps
  for (size_t i = 0; i < variational_space.size(); i++) {
    const Det& det = variational_space[i];
    
    // Process up-spin
    setup_data.up_to_full_map[det.up].push_back(i);
    if (setup_data.up_det_to_idx.find(det.up) == setup_data.up_det_to_idx.end()) {
      setup_data.up_det_to_idx[det.up] = setup_data.unique_up_dets.size();
      setup_data.unique_up_dets.push_back(det.up);
    }
    
    // Process down-spin
    setup_data.dn_to_full_map[det.dn].push_back(i);
    if (setup_data.dn_det_to_idx.find(det.dn) == setup_data.dn_det_to_idx.end()) {
      setup_data.dn_det_to_idx[det.dn] = setup_data.unique_dn_dets.size();
      setup_data.unique_dn_dets.push_back(det.dn);
    }
  }
  
  // Step 2: Sort unique determinants by number of corresponding full determinants (descending)
  std::vector<std::pair<size_t, size_t>> up_counts;
  std::vector<std::pair<size_t, size_t>> dn_counts;
  
  for (size_t i = 0; i < setup_data.unique_up_dets.size(); i++) {
    up_counts.push_back({i, setup_data.up_to_full_map[setup_data.unique_up_dets[i]].size()});
  }
  for (size_t i = 0; i < setup_data.unique_dn_dets.size(); i++) {
    dn_counts.push_back({i, setup_data.dn_to_full_map[setup_data.unique_dn_dets[i]].size()});
  }
  
  std::sort(up_counts.begin(), up_counts.end(), 
            [](const auto& a, const auto& b) { return a.second > b.second; });
  std::sort(dn_counts.begin(), dn_counts.end(), 
            [](const auto& a, const auto& b) { return a.second > b.second; });
  
  // Reorder unique determinants based on sorted counts
  std::vector<HalfDet> sorted_up_dets;
  std::vector<HalfDet> sorted_dn_dets;
  std::unordered_map<size_t, size_t> up_idx_map;  // old_idx -> new_idx
  std::unordered_map<size_t, size_t> dn_idx_map;
  
  for (size_t i = 0; i < up_counts.size(); i++) {
    size_t old_idx = up_counts[i].first;
    up_idx_map[old_idx] = i;
    sorted_up_dets.push_back(setup_data.unique_up_dets[old_idx]);
  }
  for (size_t i = 0; i < dn_counts.size(); i++) {
    size_t old_idx = dn_counts[i].first;
    dn_idx_map[old_idx] = i;
    sorted_dn_dets.push_back(setup_data.unique_dn_dets[old_idx]);
  }
  
  setup_data.unique_up_dets = sorted_up_dets;
  setup_data.unique_dn_dets = sorted_dn_dets;
  
  // Update index maps with new ordering
  setup_data.up_det_to_idx.clear();
  setup_data.dn_det_to_idx.clear();
  for (size_t i = 0; i < setup_data.unique_up_dets.size(); i++) {
    setup_data.up_det_to_idx[setup_data.unique_up_dets[i]] = i;
  }
  for (size_t i = 0; i < setup_data.unique_dn_dets.size(); i++) {
    setup_data.dn_det_to_idx[setup_data.unique_dn_dets[i]] = i;
  }
  
  // Step 3: Build single excitation constructor maps
  std::unordered_map<HalfDet, std::vector<size_t>, HalfDetHasher> upSingleExciteConstructor;
  std::unordered_map<HalfDet, std::vector<size_t>, HalfDetHasher> dnSingleExciteConstructor;
  
  // For up-spin
  for (size_t idx_i = 0; idx_i < setup_data.unique_up_dets.size(); idx_i++) {
    const HalfDet& u_i = setup_data.unique_up_dets[idx_i];
    std::vector<HalfDet> n_minus_1_configs = generate_n_minus_1_configs(u_i);
    
    for (const HalfDet& config : n_minus_1_configs) {
      upSingleExciteConstructor[config].push_back(idx_i);
    }
  }
  
  // For down-spin
  for (size_t idx_i = 0; idx_i < setup_data.unique_dn_dets.size(); idx_i++) {
    const HalfDet& d_i = setup_data.unique_dn_dets[idx_i];
    std::vector<HalfDet> n_minus_1_configs = generate_n_minus_1_configs(d_i);
    
    for (const HalfDet& config : n_minus_1_configs) {
      dnSingleExciteConstructor[config].push_back(idx_i);
    }
  }
  
  // Step 4: Populate singles maps
  setup_data.upSingles.clear();
  setup_data.dnSingles.clear();
  
  // For up-spin (with idx_j > idx_i condition)
  for (size_t idx_i = 0; idx_i < setup_data.unique_up_dets.size(); idx_i++) {
    const HalfDet& u_i = setup_data.unique_up_dets[idx_i];
    std::vector<HalfDet> n_minus_1_configs = generate_n_minus_1_configs(u_i);
    
    std::set<size_t> connected_indices;  // Use set to avoid duplicates
    for (const HalfDet& config : n_minus_1_configs) {
      auto it = upSingleExciteConstructor.find(config);
      if (it != upSingleExciteConstructor.end()) {
        for (size_t idx_j : it->second) {
          if (idx_j > idx_i) {  // Only add connections to later indices
            connected_indices.insert(idx_j);
          }
        }
      }
    }
    
    if (!connected_indices.empty()) {
      setup_data.upSingles[idx_i] = std::vector<size_t>(connected_indices.begin(), connected_indices.end());
    }
  }
  
  // For down-spin (all connections)
  for (size_t idx_i = 0; idx_i < setup_data.unique_dn_dets.size(); idx_i++) {
    const HalfDet& d_i = setup_data.unique_dn_dets[idx_i];
    std::vector<HalfDet> n_minus_1_configs = generate_n_minus_1_configs(d_i);
    
    std::set<size_t> connected_indices;
    for (const HalfDet& config : n_minus_1_configs) {
      auto it = dnSingleExciteConstructor.find(config);
      if (it != dnSingleExciteConstructor.end()) {
        for (size_t idx_j : it->second) {
          if (idx_j != idx_i) {  // All connections except self
            connected_indices.insert(idx_j);
          }
        }
      }
    }
    
    if (!connected_indices.empty()) {
      setup_data.dnSingles[idx_i] = std::vector<size_t>(connected_indices.begin(), connected_indices.end());
    }
  }
  
  if (Parallel::is_master() && opposite_spin_debug_output) {
    printf("Variational Hamiltonian setup complete:\n");
    printf("  Unique up-spin dets: %zu\n", setup_data.unique_up_dets.size());
    printf("  Unique dn-spin dets: %zu\n", setup_data.unique_dn_dets.size());
    printf("  Up-spin singles connections: %zu\n", setup_data.upSingles.size());
    printf("  Dn-spin singles connections: %zu\n", setup_data.dnSingles.size());
  }
  
  return setup_data;
}

// Helper function to generate N-1 electron configurations
template <class S>
std::vector<HalfDet> Hamiltonian<S>::generate_n_minus_1_configs(const HalfDet& half_det) const {
  std::vector<HalfDet> configs;
  auto occupied_orbs = half_det.get_occupied_orbs();
  
  configs.reserve(occupied_orbs.size());
  for (unsigned orb : occupied_orbs) {
    HalfDet config = half_det;
    config.unset(orb);
    configs.push_back(config);
  }
  
  return configs;
}

// Helper function to check if two half-determinants differ by a single excitation
template <class S>
bool Hamiltonian<S>::is_single_excitation(const HalfDet& det1, const HalfDet& det2) const {
  return det1.n_diffs(det2) == 2;  // 2 differences = 1 electron moved
}

// Cost estimation functions for opposite-spin sub-algorithms
template <class S>
double Hamiltonian<S>::estimate_opposite_spin_candidates_cost(size_t n_dn_i, size_t n_up_singles, 
                                                          size_t avg_dn_singles, size_t avg_dn_j) const {
  // Sub-algorithm 1: Hash existing connections
  // Cost = Build hash table from dn singles + Query for each up single's dn dets
  double build_cost = n_dn_i * avg_dn_singles;  // Building dnCandidates
  double query_cost = n_up_singles * avg_dn_j;  // Querying for each up single
  return build_cost + query_cost;
}

template <class S>
double Hamiltonian<S>::estimate_opposite_spin_removal_cost(size_t n_dn_i, size_t n_up_singles, 
                                                          size_t avg_dn_j, size_t n_electrons) const {
  // Sub-algorithm 2: Hash N-1 configurations
  // Cost = Build N-1 hash table + Query for each up single's dn dets
  double build_cost = n_dn_i * n_electrons;  // Building dnSingleExciteConstructor
  double query_cost = n_up_singles * avg_dn_j * n_electrons;  // Querying with N-1 configs
  return build_cost + query_cost;
}

template <class S>
double Hamiltonian<S>::estimate_opposite_spin_pairwise_cost(size_t n_dn_i, size_t n_up_singles, 
                                                          size_t avg_dn_j) const {
  // Sub-algorithm 3: Direct comparison
  // Cost = Triple nested loop
  return n_up_singles * n_dn_i * avg_dn_j;
}

// Removed: find_opposite_spin_excitations_new (functionality merged into adaptive)

// Sub-algorithm 1: Hash existing connections
template <class S>
void Hamiltonian<S>::opposite_spin_candidates(const S& system,
                                          const HamiltonianSetupData& setup_data,
                                          size_t up_idx,
                                          const std::vector<size_t>& dn_indices_i,
                                          const std::vector<size_t>& up_singles) {
  const HalfDet& u_i = setup_data.unique_up_dets[up_idx];
  
  // Build hash table of down-spin candidates
  std::unordered_map<size_t, std::vector<size_t>> dnCandidates;  // dn_idx -> full_det_indices
  
  for (size_t dn_idx : dn_indices_i) {
    auto dn_singles_it = setup_data.dnSingles.find(dn_idx);
    if (dn_singles_it == setup_data.dnSingles.end()) continue;
    
    const HalfDet& d_j = setup_data.unique_dn_dets[dn_idx];
    
    // Find full determinant index for (u_i, d_j)
    size_t full_idx_ij = SIZE_MAX;
    auto up_it = setup_data.up_to_full_map.find(u_i);
    if (up_it != setup_data.up_to_full_map.end()) {
      for (size_t idx : up_it->second) {
        if (system.dets[idx].dn == d_j) {
          full_idx_ij = idx;
          break;
        }
      }
    }
    
    if (full_idx_ij == SIZE_MAX) continue;
    
    // Add all connected down-spins to candidates
    for (size_t connected_dn_idx : dn_singles_it->second) {
      dnCandidates[connected_dn_idx].push_back(full_idx_ij);
    }
  }
  
  // Query for each up-spin single
  for (size_t up_j_idx : up_singles) {
    const HalfDet& u_j = setup_data.unique_up_dets[up_j_idx];
    
    auto up_j_it = setup_data.up_to_full_map.find(u_j);
    if (up_j_it == setup_data.up_to_full_map.end()) continue;
    
    // Check each down-spin determinant for u_j
    for (size_t full_idx_jk : up_j_it->second) {
      const Det& det_jk = system.dets[full_idx_jk];
      
      auto dn_idx_it = setup_data.dn_det_to_idx.find(det_jk.dn);
      if (dn_idx_it == setup_data.dn_det_to_idx.end()) continue;
      
      size_t dn_k_idx = dn_idx_it->second;
      
      // Check if this down-spin is in our candidates
      auto candidates_it = dnCandidates.find(dn_k_idx);
      if (candidates_it != dnCandidates.end()) {
        // Found connections: (u_j, d_k) is connected to all dets in candidates
        for (size_t full_idx_ij : candidates_it->second) {
          // Only process connections involving at least one NEW determinant
          // This is crucial for incremental updates
          if (full_idx_ij < n_dets_prev && full_idx_jk < n_dets_prev) {
            continue;  // Both determinants are old, skip
          }
          
          // Ensure upper triangular matrix storage
          size_t row = std::min(full_idx_ij, full_idx_jk);
          size_t col = std::max(full_idx_ij, full_idx_jk);
          
          if (row != col) {  // Skip diagonal elements
            const double H = time_sym ? system.get_hamiltonian_elem_time_sym(system.dets[row], system.dets[col], -1)
                                      : system.get_hamiltonian_elem(system.dets[row], system.dets[col], -1);
            if (std::abs(H) >= Util::EPS) {
              matrix.append_elem(row, col, H);
              // Removed debug output for performance
            }
          }
        }
      }
    }
  }
}

// Sub-algorithm 2: Hash N-1 configurations
template <class S>
void Hamiltonian<S>::opposite_spin_removal(const S& system,
                                          const HamiltonianSetupData& setup_data,
                                          size_t up_idx,
                                          const std::vector<size_t>& dn_indices_i,
                                          const std::vector<size_t>& up_singles) {
  const HalfDet& u_i = setup_data.unique_up_dets[up_idx];
  
  size_t connections_found = 0;
  
  // Build hash table of N-1 configurations
  std::unordered_map<HalfDet, std::vector<size_t>, HalfDetHasher> dnSingleExciteConstructor;
  
  for (size_t dn_idx : dn_indices_i) {
    const HalfDet& d_j = setup_data.unique_dn_dets[dn_idx];
    
    // Find full determinant index for (u_i, d_j)
    size_t full_idx_ij = SIZE_MAX;
    auto up_it = setup_data.up_to_full_map.find(u_i);
    if (up_it != setup_data.up_to_full_map.end()) {
      for (size_t idx : up_it->second) {
        if (system.dets[idx].dn == d_j) {
          full_idx_ij = idx;
          break;
        }
      }
    }
    
    if (full_idx_ij == SIZE_MAX) continue;
    
    // Generate all N-1 configurations for this down-spin
    std::vector<HalfDet> n_minus_1_configs = generate_n_minus_1_configs(d_j);
    for (const HalfDet& config : n_minus_1_configs) {
      dnSingleExciteConstructor[config].push_back(full_idx_ij);
    }
  }
  
  // Query for each up-spin single
  for (size_t up_j_idx : up_singles) {
    const HalfDet& u_j = setup_data.unique_up_dets[up_j_idx];
    
    auto up_j_it = setup_data.up_to_full_map.find(u_j);
    if (up_j_it == setup_data.up_to_full_map.end()) continue;
    
    // Check each down-spin determinant for u_j
    for (size_t full_idx_jk : up_j_it->second) {
      const Det& det_jk = system.dets[full_idx_jk];
      
      // Generate N-1 configurations for this down-spin
      std::vector<HalfDet> n_minus_1_configs = generate_n_minus_1_configs(det_jk.dn);
      
      std::set<size_t> visited_connections;  // Avoid duplicates
      
      for (const HalfDet& config : n_minus_1_configs) {
        auto it = dnSingleExciteConstructor.find(config);
        if (it != dnSingleExciteConstructor.end()) {
          // Found connections via this N-1 configuration
          for (size_t full_idx_ij : it->second) {
            if (visited_connections.find(full_idx_ij) != visited_connections.end()) continue;
            visited_connections.insert(full_idx_ij);
            
            // CRITICAL FIX: Only process connections involving at least one NEW determinant
            if (full_idx_ij < n_dets_prev && full_idx_jk < n_dets_prev) {
              continue;  // Both determinants are old, skip
            }
            
            // CRITICAL FIX: Ensure upper triangular matrix storage
            size_t row = std::min(full_idx_ij, full_idx_jk);
            size_t col = std::max(full_idx_ij, full_idx_jk);
            
            if (row != col) {  // Skip diagonal elements
              // CRITICAL CHECK: Verify this is actually a double excitation (both spins change)
              const Det& det_row = system.dets[row];
              const Det& det_col = system.dets[col];
              unsigned up_diffs = det_row.up.n_diffs(det_col.up);
              unsigned dn_diffs = det_row.dn.n_diffs(det_col.dn);
              
              // For opposite-spin double excitations: up_diffs = 2 AND dn_diffs = 2
              if (up_diffs == 2 && dn_diffs == 2) {
                const double H = time_sym ? system.get_hamiltonian_elem_time_sym(det_row, det_col, -1)
                                          : system.get_hamiltonian_elem(det_row, det_col, -1);
                if (std::abs(H) >= Util::EPS) {
                  matrix.append_elem(row, col, H);
                  connections_found++;
                }
              }
            }
          }
        }
      }
    }
  }
  
}

// Sub-algorithm 3: Direct comparison
template <class S>
void Hamiltonian<S>::opposite_spin_pairwise(const S& system,
                                          const HamiltonianSetupData& setup_data,
                                          size_t up_idx,
                                          const std::vector<size_t>& dn_indices_i,
                                          const std::vector<size_t>& up_singles) {
  const HalfDet& u_i = setup_data.unique_up_dets[up_idx];
  
  size_t connections_found = 0;
  size_t old_old_skipped = 0;
  size_t total_comparisons = 0;
  
  // Get all full determinants with up-spin u_i
  auto up_i_it = setup_data.up_to_full_map.find(u_i);
  if (up_i_it == setup_data.up_to_full_map.end()) return;
  
  
  // Loop over up-spin singles from u_i
  for (size_t up_j_idx : up_singles) {
    const HalfDet& u_j = setup_data.unique_up_dets[up_j_idx];
    
    // Get all full determinants with up-spin u_j
    auto up_j_it = setup_data.up_to_full_map.find(u_j);
    if (up_j_it == setup_data.up_to_full_map.end()) continue;
    
    // DEBUG: Print which up-spin pair is being processed and their down-spin lists
    if (false /* was debug_counter */ <= 2) {
      printf("  Processing up_pair: %zu -> %zu\n", up_idx, up_j_idx);
      printf("    u_i group has %zu dets: [", up_i_it->second.size());
      for (size_t i = 0; i < up_i_it->second.size(); i++) {
        size_t det_idx = up_i_it->second[i];
        printf("%zu%s", det_idx, (i == up_i_it->second.size()-1) ? "" : ",");
      }
      printf("]\n");
      printf("    u_j group has %zu dets: [", up_j_it->second.size());
      for (size_t i = 0; i < up_j_it->second.size(); i++) {
        size_t det_idx = up_j_it->second[i];
        printf("%zu%s", det_idx, (i == up_j_it->second.size()-1) ? "" : ",");
      }
      printf("]\n");
    }
    
    // Now compare ALL determinants from u_i group with ALL determinants from u_j group
    for (size_t full_idx_i : up_i_it->second) {
      for (size_t full_idx_j : up_j_it->second) {
        // CRITICAL FIX: Only process connections involving at least one NEW determinant
        if (full_idx_i < n_dets_prev && full_idx_j < n_dets_prev) {
          old_old_skipped++;
          continue;  // Both determinants are old, skip
        }
        
        total_comparisons++;
        
        const Det& det_i = system.dets[full_idx_i];
        const Det& det_j = system.dets[full_idx_j];
        
        // Debug: Print first few comparisons
        if (false /* was debug_counter */ <= 1 && total_comparisons <= 8) {
          int n_diffs = det_i.dn.n_diffs(det_j.dn);
          printf("    Comparing [%zu] vs [%zu]: up_diffs=%d, dn_diffs=%d\n", 
                 full_idx_i, full_idx_j, det_i.up.n_diffs(det_j.up), n_diffs);
          if (n_diffs == 2) {
            printf("      -> FOUND SINGLE EXCITATION!\n");
          }
        }
        
        // Check if this is a valid opposite-spin double excitation:
        // u_i -> u_j is single excitation (already ensured)
        // d_i -> d_j must also be single excitation
        bool is_dn_single = is_single_excitation(det_i.dn, det_j.dn);
        if (false /* was debug_counter */ <= 1 && total_comparisons <= 8) {
          int n_diffs_dn = det_i.dn.n_diffs(det_j.dn);
          printf("      n_diffs(dn)=%d, is_single_excitation(dn)=%s\n", 
                 n_diffs_dn, is_dn_single ? "true" : "false");
        }
        
        if (is_dn_single) {
          // CRITICAL FIX: Ensure upper triangular matrix storage
          size_t row = std::min(full_idx_i, full_idx_j);
          size_t col = std::max(full_idx_i, full_idx_j);
          
          if (row != col) {  // Skip diagonal elements
            const double H = time_sym ? system.get_hamiltonian_elem_time_sym(system.dets[row], system.dets[col], -1)
                                      : system.get_hamiltonian_elem(system.dets[row], system.dets[col], -1);
            if (false /* was debug_counter */ <= 1 && total_comparisons <= 8) {
              printf("      H_elem[%zu,%zu] = %.8f, EPS=%.2e\n", row, col, H, Util::EPS);
            }
            if (std::abs(H) >= Util::EPS) {
              matrix.append_elem(row, col, H);
              connections_found++;
            } else if (false /* was debug_counter */ <= 1 && total_comparisons <= 8) {
              printf("      SKIPPED: |H|=%.8f < EPS=%.2e\n", std::abs(H), Util::EPS);
            }
          } else if (false /* was debug_counter */ <= 1 && total_comparisons <= 8) {
            printf("      SKIPPED: diagonal element\n");
          }
        }
      }
    }
  }
  
}


// Wrapper: Candidates algorithm (dnCandidates method)
template <class S>
void Hamiltonian<S>::find_opposite_spin_excitations_candidates(const S& system) {
  auto start_time = std::chrono::high_resolution_clock::now();
  
  if (Parallel::is_master()) {
    // Removed debug output for performance
  }
  
  // Setup data structures
  HamiltonianSetupData setup_data = setup_variational_hamiltonian(system.dets);
  
  // Process each unique up-spin determinant
  for (size_t up_idx = 0; up_idx < setup_data.unique_up_dets.size(); up_idx++) {
    const HalfDet& u_i = setup_data.unique_up_dets[up_idx];
    
    // Get down-spin determinants for this up-spin
    auto up_it = setup_data.up_to_full_map.find(u_i);
    if (up_it == setup_data.up_to_full_map.end()) continue;
    
    std::vector<size_t> dn_indices_i;
    for (size_t full_idx : up_it->second) {
      const Det& det = system.dets[full_idx];
      auto dn_idx_it = setup_data.dn_det_to_idx.find(det.dn);
      if (dn_idx_it != setup_data.dn_det_to_idx.end()) {
        dn_indices_i.push_back(dn_idx_it->second);
      }
    }
    
    // Get up-spin singles for this determinant
    auto singles_it = setup_data.upSingles.find(up_idx);
    if (singles_it == setup_data.upSingles.end()) continue;
    
    const std::vector<size_t>& up_singles = singles_it->second;
    if (up_singles.empty()) continue;
    
    // Call the candidates sub-algorithm
    opposite_spin_candidates(system, setup_data, up_idx, dn_indices_i, up_singles);
  }
  
  auto end_time = std::chrono::high_resolution_clock::now();
  double total_time = std::chrono::duration<double>(end_time - start_time).count();
  
  if (Parallel::is_master()) {
    // Removed debug output for performance
  }
}

// Wrapper: Removal algorithm (N-1 electron removal method)
template <class S>
void Hamiltonian<S>::find_opposite_spin_excitations_removal(const S& system) {
  auto start_time = std::chrono::high_resolution_clock::now();
  
  if (Parallel::is_master()) {
    // Removed debug output for performance
  }
  
  // Setup data structures
  HamiltonianSetupData setup_data = setup_variational_hamiltonian(system.dets);
  
  // Process each unique up-spin determinant (same pattern as candidates)
  for (size_t up_idx = 0; up_idx < setup_data.unique_up_dets.size(); up_idx++) {
    const HalfDet& u_i = setup_data.unique_up_dets[up_idx];
    
    auto up_it = setup_data.up_to_full_map.find(u_i);
    if (up_it == setup_data.up_to_full_map.end()) continue;
    
    std::vector<size_t> dn_indices_i;
    for (size_t full_idx : up_it->second) {
      const Det& det = system.dets[full_idx];
      auto dn_idx_it = setup_data.dn_det_to_idx.find(det.dn);
      if (dn_idx_it != setup_data.dn_det_to_idx.end()) {
        dn_indices_i.push_back(dn_idx_it->second);
      }
    }
    
    auto singles_it = setup_data.upSingles.find(up_idx);
    if (singles_it == setup_data.upSingles.end()) continue;
    
    const std::vector<size_t>& up_singles = singles_it->second;
    if (up_singles.empty()) continue;
    
    // Call the removal sub-algorithm
    opposite_spin_removal(system, setup_data, up_idx, dn_indices_i, up_singles);
  }
  
  auto end_time = std::chrono::high_resolution_clock::now();
  double total_time = std::chrono::duration<double>(end_time - start_time).count();
  
  if (Parallel::is_master()) {
    // Removed debug output for performance
  }
}

// Wrapper: Pairwise algorithm (direct comparison method)
template <class S>
void Hamiltonian<S>::find_opposite_spin_excitations_pairwise(const S& system) {
  auto start_time = std::chrono::high_resolution_clock::now();
  
  if (Parallel::is_master()) {
    // Removed debug output for performance
  }
  
  // Setup data structures
  HamiltonianSetupData setup_data = setup_variational_hamiltonian(system.dets);
  
  // Process each unique up-spin determinant (same pattern as others)
  for (size_t up_idx = 0; up_idx < setup_data.unique_up_dets.size(); up_idx++) {
    const HalfDet& u_i = setup_data.unique_up_dets[up_idx];
    
    auto up_it = setup_data.up_to_full_map.find(u_i);
    if (up_it == setup_data.up_to_full_map.end()) continue;
    
    std::vector<size_t> dn_indices_i;
    for (size_t full_idx : up_it->second) {
      const Det& det = system.dets[full_idx];
      auto dn_idx_it = setup_data.dn_det_to_idx.find(det.dn);
      if (dn_idx_it != setup_data.dn_det_to_idx.end()) {
        dn_indices_i.push_back(dn_idx_it->second);
      }
    }
    
    auto singles_it = setup_data.upSingles.find(up_idx);
    if (singles_it == setup_data.upSingles.end()) continue;
    
    const std::vector<size_t>& up_singles = singles_it->second;
    if (up_singles.empty()) continue;
    
    // Call the pairwise sub-algorithm
    opposite_spin_pairwise(system, setup_data, up_idx, dn_indices_i, up_singles);
  }
  
  auto end_time = std::chrono::high_resolution_clock::now();
  double total_time = std::chrono::duration<double>(end_time - start_time).count();
  
  if (Parallel::is_master()) {
    // Removed debug output for performance
    printf("DEBUG: Pairwise algorithm complete (time: %.3fs)\n", total_time);
  }
}

// Wrapper: Adaptive algorithm (cost-based selection)
template <class S>
void Hamiltonian<S>::find_opposite_spin_excitations_adaptive(const S& system) {
  auto start_time = std::chrono::high_resolution_clock::now();
  
  if (Parallel::is_master()) {
    printf("DEBUG: *** INSIDE find_opposite_spin_excitations_adaptive ***\n");
    printf("DEBUG: Using adaptive cost-based selection for each group\n");
  }
  
  // Setup data structures
  HamiltonianSetupData setup_data = setup_variational_hamiltonian(system.dets);
  
  // Statistics for algorithm selection
  size_t total_candidates_calls = 0;
  size_t total_removal_calls = 0;
  size_t total_pairwise_calls = 0;
  
  // Process each unique up-spin determinant
  for (size_t up_idx = 0; up_idx < setup_data.unique_up_dets.size(); up_idx++) {
    const HalfDet& u_i = setup_data.unique_up_dets[up_idx];
    
    auto up_it = setup_data.up_to_full_map.find(u_i);
    if (up_it == setup_data.up_to_full_map.end()) continue;
    
    std::vector<size_t> dn_indices_i;
    for (size_t full_idx : up_it->second) {
      const Det& det = system.dets[full_idx];
      auto dn_idx_it = setup_data.dn_det_to_idx.find(det.dn);
      if (dn_idx_it != setup_data.dn_det_to_idx.end()) {
        dn_indices_i.push_back(dn_idx_it->second);
      }
    }
    
    auto singles_it = setup_data.upSingles.find(up_idx);
    if (singles_it == setup_data.upSingles.end()) continue;
    
    const std::vector<size_t>& up_singles = singles_it->second;
    if (up_singles.empty()) continue;
    
    // Calculate cost estimates for algorithm selection with proper estimation
    size_t n_dn_i = dn_indices_i.size();
    size_t n_up_singles = up_singles.size();
    
    // Estimate average down-spin properties (merged from "new" framework)
    size_t avg_dn_singles = 0;
    size_t avg_dn_j = 0;
    size_t sample_size = std::min(size_t(5), up_singles.size());
    
    for (size_t i = 0; i < sample_size; i++) {
      const HalfDet& u_j = setup_data.unique_up_dets[up_singles[i]];
      auto up_j_it = setup_data.up_to_full_map.find(u_j);
      if (up_j_it != setup_data.up_to_full_map.end()) {
        avg_dn_j += up_j_it->second.size();
      }
    }
    avg_dn_j = (sample_size > 0) ? avg_dn_j / sample_size : 1;
    
    // Estimate average dn singles connections
    sample_size = std::min(size_t(5), dn_indices_i.size());
    for (size_t i = 0; i < sample_size; i++) {
      auto dn_singles_it = setup_data.dnSingles.find(dn_indices_i[i]);
      if (dn_singles_it != setup_data.dnSingles.end()) {
        avg_dn_singles += dn_singles_it->second.size();
      }
    }
    avg_dn_singles = (sample_size > 0) ? avg_dn_singles / sample_size : 10;
    
    // Choose algorithm based on configuration or cost model (merged from "new" framework)
    if (opposite_spin_cost_model == "subalg1" || opposite_spin_cost_model == "candidates") {
      opposite_spin_candidates(system, setup_data, up_idx, dn_indices_i, up_singles);
      total_candidates_calls++;
    } else if (opposite_spin_cost_model == "subalg2" || opposite_spin_cost_model == "removal") {
      opposite_spin_removal(system, setup_data, up_idx, dn_indices_i, up_singles);
      total_removal_calls++;
    } else if (opposite_spin_cost_model == "subalg3" || opposite_spin_cost_model == "pairwise") {
      opposite_spin_pairwise(system, setup_data, up_idx, dn_indices_i, up_singles);
      total_pairwise_calls++;
    } else {  // "auto" - choose based on estimated costs
      double cost1 = estimate_opposite_spin_candidates_cost(n_dn_i, n_up_singles, avg_dn_singles, avg_dn_j);
      double cost2 = estimate_opposite_spin_removal_cost(n_dn_i, n_up_singles, avg_dn_j, n_dn);
      double cost3 = estimate_opposite_spin_pairwise_cost(n_dn_i, n_up_singles, avg_dn_j);
      
      if (cost1 <= cost2 && cost1 <= cost3) {
        opposite_spin_candidates(system, setup_data, up_idx, dn_indices_i, up_singles);
        total_candidates_calls++;
      } else if (cost2 <= cost3) {
        opposite_spin_removal(system, setup_data, up_idx, dn_indices_i, up_singles);
        total_removal_calls++;
      } else {
        opposite_spin_pairwise(system, setup_data, up_idx, dn_indices_i, up_singles);
        total_pairwise_calls++;
      }
    }
  }
  
  auto end_time = std::chrono::high_resolution_clock::now();
  double total_time = std::chrono::duration<double>(end_time - start_time).count();
  
  if (Parallel::is_master()) {
    printf("DEBUG: *** EXITING find_opposite_spin_excitations_adaptive ***\n");
    printf("DEBUG: Adaptive algorithm complete (time: %.3fs)\n", total_time);
    printf("DEBUG: Algorithm selection: candidates=%zu, removal=%zu, pairwise=%zu\n", 
           total_candidates_calls, total_removal_calls, total_pairwise_calls);
  }
}

// Cornell-style 2018 algorithm refactored for alpha-grouping
// This is mathematically equivalent to Cornell's algorithm but organized by alpha groups
template <class S>
void Hamiltonian<S>::find_excitations_cornell_style_2018(const S& system) {
  auto start_time = std::chrono::high_resolution_clock::now();
  
  if (Parallel::is_master()) {
    printf("Using Cornell-style 2018 algorithm (alpha-grouped)...\n");
  }
  
  // Process determinants in order (same as Cornell) 
  // Note: This could be optimized with alpha-grouping later, but for correctness we maintain exact ordering
  for (size_t det_id = 0; det_id < n_dets; det_id++) {
      const auto& det = system.dets[det_id];
      const bool is_new_det = det_id >= n_dets_prev;
      
      // Add diagonal element for new determinants
      if (is_new_det) {
        const double H = time_sym ? system.get_hamiltonian_elem_time_sym(det, det, 0)
                                  : system.get_hamiltonian_elem(det, det, 0);
        matrix.append_elem(det_id, det_id, H);
      }
      
      const size_t start_id = is_new_det ? det_id + 1 : n_dets_prev;

      // === SAME-SPIN EXCITATIONS ===
      
      // 1. Alpha excitations (single or double alpha excitations)
      const auto& beta = det.dn;
      const size_t beta_id = time_sym ? alpha_to_id[beta] : beta_to_id[beta];
      const auto& alpha_dets = beta_id_to_det_ids[beta_id];
      for (auto it = alpha_dets.begin(); it != alpha_dets.end(); it++) {
        const size_t alpha_det_id = *it;
        if (alpha_det_id < start_id) continue;
        const auto& connected_det = system.dets[alpha_det_id];
        const double H = time_sym ? system.get_hamiltonian_elem_time_sym(det, connected_det, -1)
                                  : system.get_hamiltonian_elem(det, connected_det, -1);
        if (std::abs(H) < Util::EPS) continue;
        matrix.append_elem(det_id, alpha_det_id, H);
      }
      if (time_sym && alpha_id_to_det_ids.size() > beta_id && det.up != det.dn) {
        const auto& alpha_dets = alpha_id_to_det_ids[beta_id];
        for (auto it = alpha_dets.begin(); it != alpha_dets.end(); it++) {
          const size_t alpha_det_id = *it;
          if (alpha_det_id < start_id) continue;
          const auto& connected_det = system.dets[alpha_det_id];
          if (connected_det.up == connected_det.dn) continue;
          const double H = system.get_hamiltonian_elem_time_sym(det, connected_det, -1);
          if (std::abs(H) < Util::EPS) continue;
          matrix.append_elem(det_id, alpha_det_id, H);
        }
      }

      // 2. Beta excitations (single or double beta excitations)
      const auto& alpha = det.up;
      const size_t alpha_id = alpha_to_id[alpha];
      const auto& beta_dets = alpha_id_to_det_ids[alpha_id];
      for (auto it = beta_dets.begin(); it != beta_dets.end(); it++) {
        const size_t beta_det_id = *it;
        if (beta_det_id < start_id) continue;
        const auto& connected_det = system.dets[beta_det_id];
        const double H = time_sym ? system.get_hamiltonian_elem_time_sym(det, connected_det, -1)
                                  : system.get_hamiltonian_elem(det, connected_det, -1);
        if (std::abs(H) < Util::EPS) continue;
        matrix.append_elem(det_id, beta_det_id, H);
      }
      if (time_sym && beta_id_to_det_ids.size() > alpha_id && det.up != det.dn) {
        const auto& beta_dets = beta_id_to_det_ids[alpha_id];
        for (auto it = beta_dets.begin(); it != beta_dets.end(); it++) {
          const size_t beta_det_id = *it;
          if (beta_det_id < start_id) continue;
          const auto& connected_det = system.dets[beta_det_id];
          if (connected_det.up == connected_det.dn) continue;
          const double H = system.get_hamiltonian_elem_time_sym(det, connected_det, -1);
          if (std::abs(H) < Util::EPS) continue;
          matrix.append_elem(det_id, beta_det_id, H);
        }
      }

      // === OPPOSITE-SPIN EXCITATIONS (MIXED DOUBLE) ===
      
      // 3. Mixed double excitations
      if (!system.has_double_excitation && !system.time_sym) continue;
      const auto& alpha_singles = alpha_id_to_single_ids[alpha_id];
      const auto& beta_singles =
          time_sym ? alpha_id_to_single_ids[beta_id] : beta_id_to_single_ids[beta_id];
      for (const auto alpha_single : alpha_singles) {
        if (alpha_id_to_beta_ids.size() <= alpha_single) continue;
        if (time_sym && alpha_single == beta_id) continue;
        const auto& related_beta_ids = alpha_id_to_beta_ids[alpha_single];
        const auto& related_det_ids = alpha_id_to_det_ids[alpha_single];
        const size_t n_related_dets = related_beta_ids.size();
        if (sort_by_det_id) {
          const auto& start_ptr =
              std::lower_bound(related_det_ids.begin(), related_det_ids.end(), start_id);
          const size_t start_related_id = start_ptr - related_det_ids.begin();
          for (size_t related_id = start_related_id; related_id < n_related_dets; related_id++) {
            const size_t related_beta = related_beta_ids[related_id];
            if (time_sym && related_beta == alpha_id) continue;
            if (std::binary_search(beta_singles.begin(), beta_singles.end(), related_beta)) {
              const size_t related_det_id = related_det_ids[related_id];
              const auto& connected_det = system.dets[related_det_id];
              const double H = time_sym
                                   ? system.get_hamiltonian_elem_time_sym(det, connected_det, 2)
                                   : system.get_hamiltonian_elem(det, connected_det, 2);
              if (std::abs(H) < Util::EPS) continue;
              matrix.append_elem(det_id, related_det_id, H);
            }
          }
        } else {
          size_t ptr = 0;
          for (auto it = beta_singles.begin(); it != beta_singles.end(); it++) {
            const size_t beta_single = *it;
            if (time_sym && beta_single == alpha_id) continue;
            while (ptr < n_related_dets && related_beta_ids[ptr] < beta_single) {
              ptr++;
            }
            if (ptr == n_related_dets) break;
            if (related_beta_ids[ptr] == beta_single) {
              const size_t related_det_id = related_det_ids[ptr];
              ptr++;
              if (related_det_id < start_id) continue;
              const auto& connected_det = system.dets[related_det_id];
              const double H = time_sym
                                   ? system.get_hamiltonian_elem_time_sym(det, connected_det, 2)
                                   : system.get_hamiltonian_elem(det, connected_det, 2);
              if (std::abs(H) < Util::EPS) continue;
              matrix.append_elem(det_id, related_det_id, H);
            }
          }
        }
      }
    }
  
  const size_t n_elems = matrix.count_n_elems();
  if (Parallel::is_master()) {
    printf("Number of nonzero elems: %'zu\n", n_elems);
  }
  matrix.cache_diag();
  
  auto end_time = std::chrono::high_resolution_clock::now();
  double total_time = std::chrono::duration<double>(end_time - start_time).count();
  total_same_spin_2018_time += total_time;
  total_same_spin_2018_calls++;
  
  if (Parallel::is_master()) {
    printf("Cornell-style 2018 algorithm complete (time: %.3fs)\n", total_time);
  }
}
