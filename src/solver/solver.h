#pragma once

#include <fgpl/src/broadcast.h>
#include <fgpl/src/dist_hash_map.h>
#include <fgpl/src/dist_hash_set.h>
#include <fgpl/src/dist_range.h>
#include <fgpl/src/hash_set.h>
#include <fgpl/src/reducer.h>
#include <hps/src/hps.h>
#include <omp_hash_map/src/omp_hash_map.h>
#include <omp_hash_map/src/omp_hash_set.h>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <queue>
#include <random>
#include <mutex>
#include <unordered_map>

#include "../config.h"
#include "../det/det.h"
#include "../math_vector.h"
#include "../parallel.h"
#include "../result.h"
#include "../timer.h"
#include "../util.h"
#include "davidson.h"
#include "green.h"
#include "hamiltonian.h"
#include "hc_server.h"
#include "uncert_result.h"

// Main SHCI solver class
// Performs variational optimization and perturbative corrections
template <class S>
class Solver {
 public:
  // Run complete SHCI calculation (variation + perturbation)
  void run();

  // Run orbital optimization (natural orbitals + optimized orbitals)
  void optimization_run();

 private:
  S system;                                      // Quantum system (ChemSystem, HegSystem, etc.)
  Hamiltonian<S> hamiltonian;                    // Sparse Hamiltonian matrix
  std::vector<double> eps_tried_prev;            // Previous screening thresholds
  fgpl::HashSet<Det, DetHasher> var_dets;        // Variational determinant set
  
  // Memory and convergence parameters
  size_t pt_mem_avail;                          // Available memory for perturbation
  size_t var_iteration_global;                  // Global variation iteration counter
  double eps_var_min;                           // Minimum variational threshold
  double eps_pt;                                // Perturbation threshold (stochastic)
  double eps_pt_dtm;                            // Perturbation threshold (deterministic)
  double eps_pt_psto;                           // Perturbation threshold (pseudo-stochastic)
  double target_error;                          // Target energy error
  double eps_pt_max;                            // Maximum perturbation threshold
  size_t bytes_per_det;                         // Memory per determinant

  // Dynamic preconditioner data members
  std::priority_queue<OffDiagElement> off_diag_heap;  // Max-heap for top-k elements
  std::vector<OffDiagElement> collected_off_diagonal_elements;  // Final collected elements
  bool off_diagonal_collection_enabled;               // Whether to collect elements
  std::mutex off_diag_mutex;                          // Thread safety for collection
  
  // Shadow wavefunction data members
  struct Connection {
    Det new_det;   // New determinant
    size_t j;      // Old determinant index
    double H_ij;   // Hamiltonian matrix element
  };
  std::vector<Connection> important_connections;      // Connections from variational selection
  std::mutex connections_mutex;                       // Thread safety for connections
  bool use_shadow_wavefunction;                       // Whether to use shadow wavefunction technique

  void run_all_variations();

  void run_variation(const double eps_var, const bool until_converged = true);

  void run_all_perturbations();

  void run_perturbation(const double eps_var);

  double get_energy_pt_dtm(const double eps_var, const unsigned i_state);

  UncertResult get_energy_pt_psto(
      const double eps_var, const unsigned i_state, const double energy_pt_dtm);

  UncertResult get_energy_pt_sto(
      const double eps_var, const unsigned i_state, const UncertResult& get_energy_pt_sto);

  bool load_variation_result(const std::string& filename);

  void save_variation_result(const std::string& filename);

  void save_pair_contrib(const double eps_var);

  void print_dets_info() const;

  std::string get_state_suffix(const unsigned i_state) const;

  std::string get_wf_filename(const double eps_var) const;

  // Generate automated epsilon schedule based on Hamiltonian matrix elements
  std::vector<double> generate_automated_epsilon_schedule(const double eps_final);


  // Collect off-diagonal elements during heat-bath step
  void collect_off_diagonal_elements(
      const size_t i, const size_t j, 
      const double H_ij, const double c_i, const double c_j);

  // Finalize collected off-diagonal elements for Davidson
  void finalize_off_diagonal_collection();

  // Brute force verification of off-diagonal collection (debug only)
  void verify_off_diagonal_collection_brute_force();
  
  // Prepare shadow wavefunction and select top-k off-diagonal elements
  void prepare_shadow_wavefunction_preconditioner();

  template <class C>
  std::array<double, 2> mapreduce_sum(
      const fgpl::DistHashMap<Det, C, DetHasher>& map,
      const std::function<double(const Det& det, const C& hc_sum)>& mapper) const;
};

template <class S>
void Solver<S>::run() {
  Timer::start("setup");
  std::setlocale(LC_ALL, "en_US.UTF-8");
  system.setup();
  target_error = Config::get<double>("target_error", 5.0e-5);
  
  // Initialize dynamic preconditioner collection
  off_diagonal_collection_enabled = Config::get<bool>("davidson/use_dynamic_preconditioner", false);
  use_shadow_wavefunction = Config::get<bool>("davidson/use_shadow_wavefunction", true);
  
  Result::put("energy_hf", system.energy_hf);
  Timer::end();

  std::vector<std::vector<size_t>> connections;

  if (!Config::get<bool>("skip_var", false)) {
    Timer::start("variation");
    run_all_variations();

    if (Config::get<bool>("2rdm", false) || Config::get<bool>("get_2rdm_csv", false) ||
        Config::get<bool>("optorb", false)) {
      hamiltonian.update(system);
      connections = hamiltonian.matrix.get_connections();
    }

    hamiltonian.clear();
    Timer::end();
  }

  Timer::start("post variation");

  if (Config::get<bool>("hc_server_mode", false)) {
    if (system.time_sym) throw std::invalid_argument("time sym hc server not implemented");
    const auto& wf_filename = get_wf_filename(eps_var_min);
    if (!load_variation_result(wf_filename)) throw std::runtime_error("failed to load wf");
    hamiltonian.update(system);
    HcServer<S> server(system, hamiltonian);
    server.run();
    return;
  }

  if (Config::get<bool>("get_green", false)) {
    if (system.time_sym) throw std::invalid_argument("time sym green not implemented");
    Timer::start("green");
    Green<S> green(system, hamiltonian);
    green.run();
    Timer::end();
  }

  system.post_variation(connections);
  connections.clear();
  connections.shrink_to_fit();
  hamiltonian.clear();
  eps_tried_prev.clear();
  var_dets.clear_and_shrink();

  Timer::end();

  if (Config::get<bool>("var_only", false)) return;

  Timer::start("perturbation");
  run_all_perturbations();
  system.post_perturbation();
  Timer::end();
}

template <class S>
void Solver<S>::optimization_run() {
  std::setlocale(LC_ALL, "en_US.UTF-8");

  Config::set<bool>("force_var", true);

  unsigned natorb_iter = Config::get<unsigned>("optimization/natorb_iter", 1);
  unsigned optorb_iter = Config::get<unsigned>("optimization/optorb_iter", 20);

  if (Parallel::is_master())
    std::cout << "\n==== Optimization started: " << natorb_iter << " natorb iterations and "
              << optorb_iter << " optorb iterations ====" << std::endl;

  std::vector<std::vector<size_t>> connections;
  target_error = Config::get<double>("target_error", 5.0e-5);

  unsigned i_iter = 0;
  double prev_energy_var = 0., energy_var, diff_energy_var;

  while (i_iter < natorb_iter) {
    if (Parallel::is_master())
      std::cout << "\n== Iteration " << i_iter << ": natural orbitals ==\n";

    Timer::start("Iteration");
    Timer::start("setup");
    if (i_iter == 0) {
      system.setup();
    } else {
      system.setup(false);  // load_integrals_from_file = false
    }
    Result::put("energy_hf", system.energy_hf);
    Timer::end();

    run_all_variations();
    hamiltonian.clear();

    energy_var = std::accumulate(system.energy_var.begin(), system.energy_var.end(), 0.) / system.n_states;
    diff_energy_var = energy_var - prev_energy_var;

    if (Parallel::is_master()) {
      if (i_iter == 0) {
        std::printf(
            "\nIter 0: HF orbitals E: %.8f ndet: %'zu\n", energy_var, system.dets.size());
      } else {
        std::printf(
            "\nIter %d: natural orbitals E: %.8f ndet: %'zu dE: %.8f\n",
            i_iter,
            energy_var,
            system.dets.size(),
            diff_energy_var);
      }
    }

    prev_energy_var = energy_var;

    SparseMatrix place_holder;
    system.post_variation_optimization(place_holder, "natorb");

    eps_tried_prev.clear();
    var_dets.clear_and_shrink();
    i_iter++;
    Timer::end();
  }

  if (natorb_iter > 0) {
    system.dump_integrals("FCIDUMP_natorb");
  }

  double min_energy_var = prev_energy_var;
  unsigned iters_bt_dumps = 1;

  const std::string method = Config::get<std::string>("optimization/method", "app_newton");
  const unsigned n_micro = Config::get<unsigned>("optimization/micro_iter", 0);

  while (i_iter < natorb_iter + optorb_iter) {
    if (Parallel::is_master())
      std::cout << "\n== Iteration " << i_iter << ": optimized orbitals (" << method
                << ") ==" << std::endl;

    Timer::start("setup");
    if (i_iter == 0) {
      system.setup();
    } else {
      system.setup(false);  // load_integrals_from_file = false
    }
    Result::put("energy_hf", system.energy_hf);
    Timer::end();

    run_all_variations();
    
    energy_var = std::accumulate(system.energy_var.begin(), system.energy_var.end(), 0.) / system.n_states;
    diff_energy_var = energy_var - prev_energy_var;
    if (Parallel::is_master()) {
      if (i_iter == 0) {
        std::printf(
            "\nIter 0: HF orbitals E: %.8f ndet: %'zu\n", energy_var, system.dets.size());
      } else if (natorb_iter != 0 && i_iter == natorb_iter) {
        std::printf(
            "\nIter %d: natural orbitals E: %.8f ndet: %'zu dE: %.8f\n",
            i_iter,
            energy_var,
            system.dets.size(),
            diff_energy_var);
      } else {
        std::printf(
            "\nIter %d: optimized orbitals (%s) E: %.8f ndet: %'zu dE: %.8f\n",
            i_iter,
            method.c_str(),
            energy_var,
            system.dets.size(),
            diff_energy_var);
      }
    }

    if (diff_energy_var < 0 &&
        (diff_energy_var > -1e-7 || i_iter == natorb_iter + optorb_iter - 1)) {
      // dump integrals if converged or max iter reached
      system.dump_integrals("FCIDUMP_optorb");
      break;
    }

    if (energy_var < min_energy_var &&
        iters_bt_dumps > 3) {  // dump integrals at most every 4 iterations
      system.dump_integrals("FCIDUMP_optorb");
      min_energy_var = energy_var;
      iters_bt_dumps = 1;
    } else {
      iters_bt_dumps++;
    }

    prev_energy_var = energy_var;

    for (unsigned i_micro = 0; i_micro < n_micro; i_micro++) {
      system.optimization_microiteration(hamiltonian.matrix, method);

      hamiltonian.update_existing_elems(system);

      Timer::checkpoint("update Hamiltonian elements");

      Davidson davidson(system.n_states);
      while (!davidson.converged) {
        davidson.diagonalize(hamiltonian.matrix, system.coefs, 1e-8, Parallel::is_master());
        system.coefs = davidson.get_lowest_eigenvectors();
      }
      system.energy_var = davidson.get_lowest_eigenvalues();
      system.coefs = davidson.get_lowest_eigenvectors();
    
      energy_var = std::accumulate(system.energy_var.begin(), system.energy_var.end(), 0.) / system.n_states;
      diff_energy_var = energy_var - prev_energy_var;

      if (Parallel::is_master()) {
        std::printf(
            "\nMicro-iteration %d: E: %.8f ndet: %'zu dE: %.8f\n",
            i_micro,
            energy_var,
            system.dets.size(),
            diff_energy_var);
      }
      prev_energy_var = energy_var;
    }


    system.post_variation_optimization(hamiltonian.matrix, method);

    hamiltonian.clear();
    connections.clear();
    connections.shrink_to_fit();

    eps_tried_prev.clear();
    var_dets.clear_and_shrink();
    i_iter++;
  }

  if (Parallel::is_master()) std::cout << "\n==== Optimization finished ====\n";
}

template <class S>
void Solver<S>::run_all_variations() {
  if (Parallel::is_master()) {
    printf("Final iteration 0 HF ndets= 1 energy= %.8f\n", system.energy_hf);
  }
  
  // Get configuration parameters for automated epsilon schedule
  const bool use_auto_epsilon_schedule = Config::get<bool>("use_auto_epsilon_schedule", true);
  auto eps_vars = Config::get<std::vector<double>>("eps_vars");
  auto eps_vars_schedule = Config::get<std::vector<double>>("eps_vars_schedule", std::vector<double>());
  
  // Implement automated epsilon schedule if conditions are met
  if (use_auto_epsilon_schedule && eps_vars_schedule.empty()) {
    eps_vars_schedule = generate_automated_epsilon_schedule(eps_vars[0]);
    // Replace eps_vars with the automated schedule + original eps_vars
    auto combined_schedule = eps_vars_schedule;
    combined_schedule.insert(combined_schedule.end(), eps_vars.begin(), eps_vars.end());
    eps_vars = combined_schedule;
    eps_vars_schedule.clear(); // Clear to avoid double processing
    if (Parallel::is_master()) {
      printf("Using automated epsilon schedule: ");
      for (const auto& eps : eps_vars) {
        printf("%#.2e ", eps);
      }
      printf("\n");
    }
  }
  double eps_var_prev = Util::INF;
  for (const auto& det : system.dets) var_dets.set(det);
  auto it_schedule = eps_vars_schedule.begin();
  var_iteration_global = 0;
  eps_var_min = eps_vars.back();
  const bool get_pair_contrib = Config::get<bool>("get_pair_contrib", false);
  for (const double eps_var : eps_vars) {
    Timer::start(Util::str_printf("eps_var=%#.2e", eps_var));
    const auto& filename = get_wf_filename(eps_var);
    if (Config::get<bool>("force_var", false) || !load_variation_result(filename)) {
      // Perform extra scheduled eps.
      while (it_schedule != eps_vars_schedule.end() && *it_schedule >= eps_var_prev) it_schedule++;
      while (it_schedule != eps_vars_schedule.end() && *it_schedule > eps_var) {
        const double eps_var_extra = *it_schedule;
        Timer::start(Util::str_printf("extra=%#.2e", eps_var_extra));
        run_variation(eps_var_extra, false);
        Timer::end();
        it_schedule++;
      }

      Timer::start("main");
      run_variation(eps_var);
      for (unsigned i_state = 0; i_state < system.n_states; i_state++) {
        Result::put<double>(
            Util::str_printf("energy_var%s/%#.2e", get_state_suffix(i_state).c_str(), eps_var),
            system.energy_var[i_state]);
      }
      Timer::end();
      save_variation_result(filename);
    } else {
      eps_tried_prev.clear();
      var_dets.clear();
      for (const auto& det : system.dets) var_dets.set(det);
      //      hamiltonian.clear();
      for (unsigned i_state = 0; i_state < system.n_states; i_state++) {
        Result::put<double>(
            Util::str_printf("energy_var%s/%#.2e", get_state_suffix(i_state).c_str(), eps_var),
            system.energy_var[i_state]);
      }
    }

    if (Parallel::is_master() && get_pair_contrib) {
      save_pair_contrib(eps_var);
    }
    eps_var_prev = eps_var;
    Timer::end();
  }

  //  hamiltonian.clear();
  eps_tried_prev.clear();
  eps_tried_prev.shrink_to_fit();
  var_dets.clear_and_shrink();
}

template <class S>
void Solver<S>::run_all_perturbations() {
  const auto& eps_vars = Config::get<std::vector<double>>("eps_vars");
  bytes_per_det = N_CHUNKS * 16;  // 2 * 8 * N_CHKS
#ifdef INF_ORBS
  bytes_per_det += 128;  // 4 * 32. Assume 4 avg excitations beyond bit-rep CHUNKS per Half Det.
#endif
  for (const double eps_var : eps_vars) {
    Timer::start(Util::str_printf("eps_var=%#.2e", eps_var));
    run_perturbation(eps_var);
    Timer::end();
  }
}

template <class S>
void Solver<S>::run_variation(const double eps_var, const bool until_converged) {
  Davidson davidson(system.n_states);
  fgpl::DistHashSet<Det, DetHasher> dist_new_dets;
  size_t n_dets = system.get_n_dets();
  size_t n_dets_new = n_dets;
  std::vector<double> energy_var_prev(system.n_states, 0.0);
  bool converged = false;
  size_t iteration = 0;
  bool dets_converged = false;
  const bool get_pair_contrib = Config::get<bool>("get_pair_contrib", false);
  bool var_sd = Config::get<bool>("var_sd", get_pair_contrib);

  const bool second_rejection = Config::get<bool>("second_rejection", false);
  system.energy_hf_1b = second_rejection ? system.get_e_hf_1b() : 0.;
  system.second_rejection_factor = Config::get<double>("second_rejection_factor", 0.2);

  while (!converged) {
    eps_tried_prev.resize(n_dets, Util::INF);
    if (until_converged) Timer::start(Util::str_printf("#%zu", iteration + 1));
    
    // Clear important connections for shadow wavefunction
    if (use_shadow_wavefunction) {
      important_connections.clear();
    }

    // Random execution and broadcast.
    if (!dets_converged) {
      n_dets_new = n_dets;
      for (size_t j = 0; j < 5; j++) {
        fgpl::DistRange<size_t>(j, n_dets, 5).for_each([&](const size_t i) {
          const auto& det = system.dets[i];
          double max_coef = system.coefs[0][i];
          for (unsigned i_state = 1; i_state < system.n_states; i_state++) {
            const double coef = system.coefs[i_state][i];
            if (std::abs(coef) > std::abs(max_coef)) max_coef = coef;
          }
          double eps_min = eps_var / std::abs(max_coef);
          if (i == 0 && var_sd) eps_min = 0.0;
          if (system.time_sym && det.up != det.dn) eps_min *= Util::SQRT2;
          if (eps_min >= eps_tried_prev[i]) return;
          Det connected_det_reg;
          const auto& connected_det_handler = [&](const Det& connected_det, const int n_excite) {
            connected_det_reg = connected_det;
            if (system.time_sym && connected_det.up > connected_det.dn) {
              connected_det_reg.reverse_spin();
            }
            
            // Compute Hamiltonian matrix element first
            double h_ij = 0.0;
            if (n_excite == 1) {
              h_ij = system.get_hamiltonian_elem(det, connected_det, 1);
              if (std::abs(h_ij) < eps_min) return;  // Filter out small single excitation.
            } else if (n_excite == 2) {
              h_ij = system.get_hamiltonian_elem(det, connected_det, 2);
            }
            
            // Collect off-diagonal element if this determinant is already in variational space
            if (var_dets.has(connected_det_reg)) {
              if (off_diagonal_collection_enabled && h_ij != 0.0) {
                // Find index of connected determinant in current variational space
                for (size_t j_idx = 0; j_idx < system.dets.size(); j_idx++) {
                  if (system.dets[j_idx] == connected_det_reg) {
                    double c_j = system.coefs[0][j_idx];
                    // Find max coefficient for connected det across all states
                    for (unsigned i_state = 1; i_state < system.n_states; i_state++) {
                      const double coef = system.coefs[i_state][j_idx];
                      if (std::abs(coef) > std::abs(c_j)) c_j = coef;
                    }
                    
                    // Collect off-diagonal element: |c_j * H_ij * c_i|
                    collect_off_diagonal_elements(i, j_idx, h_ij, max_coef, c_j);
                    break;
                  }
                }
              }
              return;
            }
            
            // Store connection for shadow wavefunction technique
            if (use_shadow_wavefunction && h_ij != 0.0) {
              std::lock_guard<std::mutex> lock(connections_mutex);
              important_connections.push_back({connected_det_reg, i, h_ij});
            }
            
            
            dist_new_dets.async_set(connected_det_reg);
          };
          if (second_rejection) {
            double prev_max_rejection = system.find_connected_dets(
                det, eps_tried_prev[i], eps_min, connected_det_handler, true);
            eps_tried_prev[i] = 1.000000001 * prev_max_rejection;
          } else {
	    static_cast<void>(system.find_connected_dets(det, eps_tried_prev[i], eps_min, connected_det_handler));
            eps_tried_prev[i] = 1.000000001 * eps_min;
          }
        });
        dist_new_dets.sync();
        n_dets_new += dist_new_dets.get_n_keys();
        system.dets.reserve(n_dets_new);
        for (auto& state_coefs : system.coefs) state_coefs.reserve(n_dets_new);
        dist_new_dets.for_each_serial([&](const Det& connected_det, const size_t) {
          var_dets.set(connected_det);
          system.dets.push_back(connected_det);
          
          // Populate orbital partitioning screener for new determinant
          system.populate_screener(connected_det, system.dets.size() - 1);
          
	  // initialize as unit vectors 
          for (unsigned i_state = 0; i_state < system.n_states; i_state++) {
            if (i_state == 0 || n_dets > 50) {
              system.coefs[i_state].push_back(1e-16);
            } else {
              const double elem = (system.coefs[i_state].size() == i_state) ? 1.0: 1e-16;
              system.coefs[i_state].push_back(elem);
	    }
          }
        });
        dist_new_dets.clear();
        if (Parallel::is_master()) printf("%zu%% ", (j + 1) * 20);
      }

      if (Parallel::is_master()) {
        printf("\nNumber of dets / new dets: %'zu / %'zu\n", n_dets_new, n_dets_new - n_dets);
      }
      Timer::checkpoint("get next det list");

      hamiltonian.update(system);
    }

    // Finalize off-diagonal element collection for dynamic preconditioner
    if (use_shadow_wavefunction && iteration == 0) {
      // Use shadow wavefunction for first iteration
      prepare_shadow_wavefunction_preconditioner();
    } else {
      // Use regular collection for subsequent iterations
      finalize_off_diagonal_collection();
    }

    // Verify collection with brute force (debug only)
    verify_off_diagonal_collection_brute_force();

    // Pass collected off-diagonal elements to Davidson
    if (off_diagonal_collection_enabled) {
      davidson.set_off_diagonal_elements(collected_off_diagonal_elements);
    }

    const double davidson_target_error =
        until_converged ? target_error / 500000 : target_error / 50;
    davidson.diagonalize(
        hamiltonian.matrix, system.coefs, davidson_target_error, Parallel::is_master());
    const std::vector<double> energy_var_new = davidson.get_lowest_eigenvalues();
    system.coefs = davidson.get_lowest_eigenvectors();
    Timer::checkpoint("diagonalize sparse hamiltonian");
    var_iteration_global++;
    if (Parallel::is_master()) {
      printf("Iteration %zu ", var_iteration_global);
      printf("eps1= %#.2e ndets= %'zu energy=", eps_var, n_dets_new);
      for (const auto& energy : energy_var_new) printf(" %.8f", energy);
      printf("\n");
    }
    for (unsigned i_state = 0; i_state < system.n_states; i_state++) {
      if (std::abs(energy_var_new[i_state] - energy_var_prev[i_state]) > target_error * 0.001)
        break;
      if (i_state == system.n_states) converged = true;
    }
    if (n_dets_new < n_dets * 1.001) {
      dets_converged = true;
    }
    if (dets_converged && davidson.converged) {
      converged = true;
    }
    n_dets = n_dets_new;
    energy_var_prev = energy_var_new;
    if (!until_converged) break;
    Timer::end();
    iteration++;
  }
  system.energy_var = energy_var_prev;
  if (Parallel::is_master() && until_converged) {
    printf("Final iteration %zu ", var_iteration_global);
    printf("eps1= %#.2e ndets= %'zu energy=", eps_var, n_dets);
    for (const auto& energy : system.energy_var) printf("  %.8f", energy);
    printf("\n");
    print_dets_info();
  }
}

template <class S>
void Solver<S>::run_perturbation(const double eps_var) {
  double eps_pt_dtm_min = 2.0e-6;
  double eps_pt_psto_min = 1.0e-7;
  double eps_pt_dtm_ratio = 1.0e-1;
  double eps_pt_psto_ratio = 1.0e-2;
  double eps_pt_ratio = 1.0e-3;

  eps_pt_dtm_ratio = Config::get<double>("eps_pt_dtm_ratio", eps_pt_dtm_ratio);
  eps_pt_psto_ratio = Config::get<double>("eps_pt_psto_ratio", eps_pt_psto_ratio);
  eps_pt_ratio = Config::get<double>("eps_pt_ratio", eps_pt_ratio);

  eps_pt_dtm = std::max(eps_pt_dtm_min, eps_pt_dtm_ratio*eps_var);
  eps_pt_psto = std::max(eps_pt_psto_min, eps_pt_psto_ratio*eps_var);
  eps_pt = eps_pt_ratio*eps_var; // no max here because we want eps_pt propto eps_var

  eps_pt_dtm = Config::get<double>("eps_pt_dtm", eps_pt_dtm);
  eps_pt_psto = Config::get<double>("eps_pt_psto", eps_pt_psto);
  eps_pt = Config::get<double>("eps_pt", eps_pt);

  if (eps_pt_psto < eps_pt) eps_pt_psto = eps_pt;
  if (eps_pt_dtm < eps_pt_psto) eps_pt_dtm = eps_pt_psto;

  // If results for all states of current eps_var already exist, return.
  bool missing_pt = false;
  for (unsigned i_state = 0; i_state < system.n_states; i_state++) {
    const auto& value_entry = Util::str_printf(
        "energy_total%s/%#.2e/%#.2e/value", get_state_suffix(i_state).c_str(), eps_var, eps_pt);
    const auto& uncert_entry = Util::str_printf(
        "energy_total%s/%#.2e/%#.2e/uncert", get_state_suffix(i_state).c_str(), eps_var, eps_pt);
    UncertResult res(Result::get<double>(value_entry, 0.0));
    if (res.value != 0.0) {
      if (Parallel::is_master()) {
        res.uncert = Result::get<double>(uncert_entry, 0.0);
        printf(
            "Total energy: %s (state %d, loaded from result file)\n",
            res.to_string().c_str(),
            i_state);
      }
    } else {
      missing_pt = true;
    }
  }

  if (!missing_pt && !Config::get<bool>("force_pt", false)) return;

  // Load var wf.
  const auto& var_filename = get_wf_filename(eps_var);
  if (!load_variation_result(var_filename)) {
    throw new std::runtime_error("cannot load variation results");
  }
  system.update_diag_helper();
  if (system.time_sym) system.unpack_time_sym();

  // Perform multi stage PT.
  system.dets.shrink_to_fit();
  for (auto& coefs : system.coefs) coefs.shrink_to_fit();
  var_dets.clear_and_shrink();
  var_dets.reserve(system.get_n_dets());
  for (const auto& det : system.dets) var_dets.set(det);
  size_t mem_total = Config::get<double>("mem_total", Util::get_mem_total());
#ifdef INF_ORBS
  mem_total *= 0.8;
#endif
  const size_t mem_var = system.get_n_dets() * (bytes_per_det * 3 + 8);
  const double mem_left = mem_total * 0.7 - mem_var - system.helper_size;
  assert(mem_left > 0);
  pt_mem_avail = mem_left;
  const size_t n_procs = Parallel::get_n_procs();
  if (n_procs >= 2) {
    pt_mem_avail = static_cast<size_t>(pt_mem_avail * 0.7 * n_procs);
  }
  if (Parallel::is_master()) {
    printf("Memory total: %.1fGB\n", mem_total * 1.0e-9);
    printf("Helper size: %.1fGB\n", system.helper_size * 1.0e-9);
    printf("Bytes per det: %zu\n", bytes_per_det);
    printf("Memory var: %.1fGB\n", mem_var * 1.0e-9);
    printf("Memory PT limit: %.1fGB\n", pt_mem_avail * 1.0e-9);
  }
  for (unsigned i_state = 0; i_state < system.n_states; i_state++) {
    const auto& value_entry = Util::str_printf(
        "energy_total%s/%#.2e/%#.2e/value", get_state_suffix(i_state).c_str(), eps_var, eps_pt);
    const auto& uncert_entry = Util::str_printf(
        "energy_total%s/%#.2e/%#.2e/uncert", get_state_suffix(i_state).c_str(), eps_var, eps_pt);
    const double energy_pt_dtm = get_energy_pt_dtm(eps_var, i_state);
    const UncertResult energy_pt_psto = get_energy_pt_psto(eps_var, i_state, energy_pt_dtm);
    const UncertResult energy_pt = get_energy_pt_sto(eps_var, i_state, energy_pt_psto);
    if (Parallel::is_master()) {
      printf("Total energy: %s Ha (state %d)\n", energy_pt.to_string().c_str(), i_state);
    }
    Result::put(value_entry, energy_pt.value);
    Result::put(uncert_entry, energy_pt.uncert);
  }
}

template <class S>
double Solver<S>::get_energy_pt_dtm(const double eps_var, const unsigned i_state) {
//We cannot return if eps_pt_dtm >= eps_var because: a) some c_i may go up in mag. during the last HCI iteration,
//b) New dets may be added during last HCI iteration, c) if second_rejection=true then not all dets
//that pass the usual HCI criterion will be included in the variational wavefn.
//So, to be safe we just set eps_pt_max=infinity
  eps_pt_max=Util::INF;
//if (eps_pt_dtm >= eps_pt_max) return system.energy_var;

  Timer::start(Util::str_printf("dtm %#.2e (state %d)", eps_pt_dtm, i_state));
  const size_t n_var_dets = system.get_n_dets();
  size_t n_batches = Config::get<size_t>("n_batches_pt_dtm", 0);
  fgpl::DistHashMap<Det, MathVector<double, 1>, DetHasher> hc_sums;
  size_t bytes_per_entry = bytes_per_det + 8;
  const DetHasher det_hasher;

  // Estimate best n batches.
  if (n_batches == 0) {
    fgpl::DistRange<size_t>(50, n_var_dets, 100).for_each([&](const size_t i) {
      const Det& det = system.dets[i];
      const double coef = system.coefs[i_state][i];
      const auto& pt_det_handler = [&](const Det& det_a, const int n_excite) {
        if (var_dets.has(det_a)) return;
        const size_t det_a_hash = det_hasher(det_a);
        const size_t batch_hash = Util::rehash(det_a_hash);
        if ((batch_hash & 127) != 0) return;  // For n a power of 2, "% n" = "& (n-1)"
        if (n_excite == 1) {
          const double h_ai = system.get_hamiltonian_elem(det, det_a, n_excite);
          const double hc = h_ai * coef;
          if (std::abs(hc) < eps_pt_dtm) return;  // Filter out small single excitation.
        }
        MathVector<double, 1> contrib;
        hc_sums.async_set(det_a, contrib);
      };
      static_cast<void>(system.find_connected_dets(
          det, eps_pt_max / std::abs(coef), eps_pt_dtm / std::abs(coef), pt_det_handler));
    });
    hc_sums.sync();
    const size_t n_pt_dets = hc_sums.get_n_keys();
    // 128 batches during estimation. 1 out of 100 var dets used. Hash table filling rate < 1 / 2.5
    n_batches =
        static_cast<size_t>(ceil(2.5 * 128 * 100 * n_pt_dets * bytes_per_entry / pt_mem_avail));
    if (n_batches == 0) n_batches = 1;
    size_t n_batches_node = n_batches;
    fgpl::broadcast(n_batches);
    if (n_batches_node > n_batches) {
      printf("Warning: there may insufficient memory on node id %d.\n", Parallel::get_proc_id());
    }
    if (Parallel::is_master()) {
      printf("Number of dtm batches: %zu\n", n_batches);
    }
    Timer::checkpoint("determine number of dtm batches");
    hc_sums.clear();
  }

  double energy_sum = 0.0;
  double energy_sq_sum = 0.0;
  size_t n_pt_dets_sum = 0;
  UncertResult energy_pt_dtm;

  for (size_t batch_id = 0; batch_id < n_batches; batch_id++) {
    Timer::start(Util::str_printf("#%zu/%zu", batch_id + 1, n_batches));

    for (size_t j = 0; j < 5; j++) {
      fgpl::DistRange<size_t>(j, n_var_dets, 5).for_each([&](const size_t i) {
        const Det& det = system.dets[i];
        const double coef = system.coefs[i_state][i];
        const auto& pt_det_handler = [&](const Det& det_a, const int n_excite) {
          const size_t det_a_hash = det_hasher(det_a);
          const size_t batch_hash = Util::rehash(det_a_hash);
          if (batch_hash % n_batches != batch_id) return;
          if (var_dets.has(det_a)) return;
          const double h_ai = system.get_hamiltonian_elem(det, det_a, n_excite);
          const double hc = h_ai * coef;
          if (std::abs(hc) < eps_pt_dtm) return;  // Filter out small single excitation.
          const MathVector<double, 1> contrib(hc);
          hc_sums.async_set(det_a, contrib, fgpl::Reducer<MathVector<double, 1>>::sum);
        };
        static_cast<void>(system.find_connected_dets(
            det, eps_pt_max / std::abs(coef), eps_pt_dtm / std::abs(coef), pt_det_handler));
      });
      hc_sums.sync(fgpl::Reducer<MathVector<double, 1>>::sum);
      if (Parallel::is_master()) printf("%zu%% ", (j + 1) * 20);
    }
    const size_t n_pt_dets = hc_sums.get_n_keys();
    if (Parallel::is_master()) {
      printf("\nNumber of dtm pt dets: %'zu\n", n_pt_dets);
    }
    n_pt_dets_sum += n_pt_dets;
    Timer::checkpoint("create hc sums");

    const auto& energy_pt_dtm_batch = mapreduce_sum<MathVector<double, 1>>(
        hc_sums, [&](const Det& det_a, const MathVector<double, 1>& hc_sum) {
          const double H_aa = system.get_hamiltonian_elem(det_a, det_a, 0);
          const double contrib = hc_sum[0] * hc_sum[0] / (system.energy_var[i_state] - H_aa);
          return contrib;
        });
    energy_sum += energy_pt_dtm_batch[0];
    energy_sq_sum += energy_pt_dtm_batch[1];
    energy_pt_dtm.value = energy_sum / (batch_id + 1) * n_batches;
    if (batch_id == n_batches - 1) {
      energy_pt_dtm.uncert = 0.0;
    } else {
      const double energy_avg = energy_sum / n_pt_dets_sum;
      const double sample_stdev = sqrt(energy_sq_sum / n_pt_dets_sum - energy_avg * energy_avg);
      energy_pt_dtm.uncert =
          sample_stdev * sqrt(n_pt_dets_sum) / (batch_id + 1) * (n_batches - batch_id - 1);
    }

    if (Parallel::is_master()) {
      printf("PT dtm batch correction: " ENERGY_FORMAT "\n", energy_pt_dtm_batch[0]);
      printf("PT dtm correction (eps1= %.2e, eps_pt_dtm= %.2e):", eps_var, eps_pt_dtm);
      printf(" %s Ha\n", energy_pt_dtm.to_string().c_str());
      printf("PT dtm total energy (eps1= %.2e, eps_pt_dtm= %.2e):", eps_var, eps_pt_dtm);
      printf(" %s Ha\n", (energy_pt_dtm + system.energy_var[i_state]).to_string().c_str());
      printf("Correlation energy (eps1= %.2e, eps_pt_dtm= %.2e):", eps_var, eps_pt_dtm);
      printf(
          " %s Ha\n",
          (energy_pt_dtm + system.energy_var[i_state] - system.energy_hf).to_string().c_str());
    }

    hc_sums.clear();
    Timer::end();  // batch
  }

  hc_sums.clear_and_shrink();
  Timer::end();  // dtm
  return energy_pt_dtm.value + system.energy_var[i_state];
}

template <class S>
UncertResult Solver<S>::get_energy_pt_psto(
    const double eps_var, const unsigned i_state, const double energy_pt_dtm) {
  if (eps_pt_psto >= eps_pt_dtm) return UncertResult(energy_pt_dtm, 0.0);

  Timer::start(Util::str_printf("psto %#.2e (state %d)", eps_pt_psto, i_state));
  const size_t n_var_dets = system.get_n_dets();
  size_t n_batches = Config::get<size_t>("n_batches_pt_psto", 0);
  fgpl::DistHashMap<Det, MathVector<double, 2>, DetHasher> hc_sums;
  const size_t bytes_per_entry = bytes_per_det + 16;
  const DetHasher det_hasher;

  // Estimate best n batches.
  if (n_batches == 0) {
    fgpl::DistRange<size_t>(50, n_var_dets, 100).for_each([&](const size_t i) {
      const Det& det = system.dets[i];
      const double coef = system.coefs[i_state][i];
      const auto& pt_det_handler = [&](const Det& det_a, const int n_excite) {
        if (var_dets.has(det_a)) return;
        const size_t det_a_hash = det_hasher(det_a);
        const size_t batch_hash = Util::rehash(det_a_hash);
        if ((batch_hash & 127) != 0) return;  // For n a power of 2, "% n" = "& (n-1)"
        if (n_excite == 1) {
          const double h_ai = system.get_hamiltonian_elem(det, det_a, n_excite);
          const double hc = h_ai * coef;
          if (std::abs(hc) < eps_pt_psto) return;  // Filter out small single excitation.
        }
        MathVector<double, 2> contrib;
        hc_sums.async_set(det_a, contrib);
      };
      static_cast<void>(system.find_connected_dets(
          det, eps_pt_max / std::abs(coef), eps_pt_psto / std::abs(coef), pt_det_handler));
    });
    hc_sums.sync();
    const size_t n_pt_dets = hc_sums.get_n_keys();
    const double mem_usage = Config::get<double>("pt_psto_mem_usage", 1.0);
    // 128 batches during estimation. 1 out of 100 var dets used. Hash table filling rate < 1 / 2.5
    n_batches = static_cast<size_t>(
        ceil(2.5 * 128 * 100 * n_pt_dets * bytes_per_entry / (pt_mem_avail * mem_usage)));
    if (n_batches < 16) n_batches = 16;
    size_t n_batches_node = n_batches;
    fgpl::broadcast(n_batches);
    if (n_batches_node > n_batches) {
      printf("Warning: there may be insufficient memory on node id %d.\n", Parallel::get_proc_id());
    }
    if (Parallel::is_master()) {
      printf("Number of psto batches: %zu\n", n_batches);
    }
    Timer::checkpoint("determine number of psto batches");
    hc_sums.clear();
  }

  double energy_sum = 0.0;
  double energy_sq_sum = 0.0;
  size_t n_pt_dets_sum = 0;
  UncertResult energy_pt_psto;

  for (size_t batch_id = 0; batch_id < n_batches; batch_id++) {
    Timer::start(Util::str_printf("#%zu/%zu", batch_id + 1, n_batches));

    for (size_t j = 0; j < 5; j++) {
      fgpl::DistRange<size_t>(j, n_var_dets, 5).for_each([&](const size_t i) {
        const Det& det = system.dets[i];
        const double coef = system.coefs[i_state][i];
        const auto& pt_det_handler = [&](const Det& det_a, const int n_excite) {
          const size_t det_a_hash = det_hasher(det_a);
          const size_t batch_hash = Util::rehash(det_a_hash);
          if (batch_hash % n_batches != batch_id) return;
          if (var_dets.has(det_a)) return;
          const double h_ai = system.get_hamiltonian_elem(det, det_a, n_excite);
          const double hc = h_ai * coef;
          if (std::abs(hc) < eps_pt_psto) return;  // Filter out small single excitation.
          MathVector<double, 2> contrib;
          contrib[0] = hc;
          if (std::abs(hc) >= eps_pt_dtm) contrib[1] = hc;
          hc_sums.async_set(det_a, contrib, fgpl::Reducer<MathVector<double, 2>>::sum);
        };
        static_cast<void>(system.find_connected_dets(
            det, eps_pt_max / std::abs(coef), eps_pt_psto / std::abs(coef), pt_det_handler));
      });
      hc_sums.sync(fgpl::Reducer<MathVector<double, 2>>::sum);
      if (Parallel::is_master()) printf("%zu%% ", (j + 1) * 20);
    }
    const size_t n_pt_dets = hc_sums.get_n_keys();
    if (Parallel::is_master()) {
      printf("\nNumber of psto pt dets: %'zu\n", n_pt_dets);
    }
    n_pt_dets_sum += n_pt_dets;
    Timer::checkpoint("create hc sums");

    const auto& energy_pt_psto_batch = mapreduce_sum<MathVector<double, 2>>(
        hc_sums, [&](const Det& det_a, const MathVector<double, 2>& hc_sum) {
          const double hc_sum_sq_diff = hc_sum[0] * hc_sum[0] - hc_sum[1] * hc_sum[1];
          const double H_aa = system.get_hamiltonian_elem(det_a, det_a, 0);
          const double contrib = hc_sum_sq_diff / (system.energy_var[i_state] - H_aa);
          return contrib;
        });
    energy_sum += energy_pt_psto_batch[0];
    energy_sq_sum += energy_pt_psto_batch[1];
    energy_pt_psto.value = energy_sum / (batch_id + 1) * n_batches;
    if (batch_id == n_batches - 1) {
      energy_pt_psto.uncert = 0.0;
    } else {
      const double energy_avg = energy_sum / n_pt_dets_sum;
      const double sample_stdev = sqrt(energy_sq_sum / n_pt_dets_sum - energy_avg * energy_avg);
      const double mean_stdev = sample_stdev / sqrt(n_pt_dets_sum);
      energy_pt_psto.uncert =
          mean_stdev * n_pt_dets_sum / (batch_id + 1) * (n_batches - batch_id - 1);
      // energy_pt_psto.uncert = sample_stdev * sqrt(n_pt_dets_sum) / (batch_id + 1) * n_batches;
    }

    if (Parallel::is_master()) {
      printf("PT psto batch correction: " ENERGY_FORMAT "\n", energy_pt_psto_batch[0]);
      printf("PT psto correction (eps1= %.2e, eps_pt_psto= %.2e):", eps_var, eps_pt_psto);
      printf(" %s Ha\n", energy_pt_psto.to_string().c_str());
      printf("PT psto total energy (eps1= %.2e, eps_pt_psto= %.2e):", eps_var, eps_pt_psto);
      printf(" %s Ha\n", (energy_pt_psto + energy_pt_dtm).to_string().c_str());
      printf("Correlation energy (eps1= %.2e, eps_pt_psto= %.2e):", eps_var, eps_pt_psto);
      printf(" %s Ha\n", (energy_pt_psto + energy_pt_dtm - system.energy_hf).to_string().c_str());
    }

    hc_sums.clear();
    Timer::end();  // batch

    if (energy_pt_psto.uncert <= target_error * 0.5) break;
    if (eps_pt_psto <= eps_pt && energy_pt_psto.uncert <= target_error) break;
  }

  Timer::end();  // psto
  return energy_pt_psto + energy_pt_dtm;
}

template <class S>
UncertResult Solver<S>::get_energy_pt_sto(
    const double eps_var, const unsigned i_state, const UncertResult& energy_pt_psto) {
  if (eps_pt >= eps_pt_psto) return energy_pt_psto;

  const size_t max_pt_iterations = Config::get<size_t>("max_pt_iterations", 100);
  fgpl::DistHashMap<Det, MathVector<double, 5>, DetHasher> hc_sums;
  const size_t bytes_per_entry = bytes_per_det + 40;
  const size_t n_var_dets = system.get_n_dets();
  size_t n_batches = Config::get<size_t>("n_batches_pt_sto", 0);
  if (n_batches == 0) n_batches = 64;
  size_t n_dets_in_sample = Config::get<size_t>("n_dets_in_sample_pt_sto", 0);
  size_t n_unique_target = 0;
  std::vector<double> probs(n_var_dets);
  std::unordered_map<size_t, unsigned> sample_dets_sto;  // only contains stochastic dets
  std::vector<size_t> sample_dets_list;  // dtm dets followed by stochastic dets
  size_t n_dtm_dets = 0;
  size_t iteration = 0;
  const DetHasher det_hasher;

  UncertResult energy_pt_sto;
  std::vector<double> energy_pt_sto_loops;

  // Contruct probs.
  double sum_weights = 0.0;
  for (size_t i = 0; i < n_var_dets; i++) {
    sum_weights += std::abs(system.coefs[i_state][i]);
  }
  std::vector<double> cum_probs(n_var_dets);  // For sampling.
  for (size_t i = 0; i < n_var_dets; i++) {
    probs[i] = std::abs(system.coefs[i_state][i]) / sum_weights;
    if (i > 0)
      cum_probs[i] = probs[i] + cum_probs[i - 1];
    else
      cum_probs[i] = probs[i];
  }

  Timer::start(Util::str_printf("sto %#.2e (state %d)", eps_pt, i_state));

  //const unsigned random_seed = Config::get<unsigned>("random_seed", time(nullptr));
  const unsigned random_seed = Config::get<unsigned>("random_seed", 347634253);
  if (Parallel::is_master()) printf("\nrandom_seed= %d\n", random_seed);
  std::default_random_engine generator(random_seed);
  std::uniform_real_distribution<double> unif_real_distr(0., 1.);

  // Estimate best n_dets_in_sample.
  if (n_dets_in_sample == 0) {
    for (size_t i = 0; i < 1000; i++) {
      const double rand_01 = unif_real_distr(generator);
      const size_t sample_det_id =
          std::lower_bound(cum_probs.begin(), cum_probs.end(), rand_01) - cum_probs.begin();
      if (sample_dets_sto.count(sample_det_id) == 0) sample_dets_list.push_back(sample_det_id);
      sample_dets_sto[sample_det_id]++;
    }
    fgpl::broadcast(sample_dets_sto);
    fgpl::broadcast(sample_dets_list);
    size_t n_unique_dets_in_sample = sample_dets_list.size();
    fgpl::DistRange<size_t>(0, n_unique_dets_in_sample).for_each([&](const size_t sample_id) {
      const size_t i = sample_dets_list[sample_id];
      const Det& det = system.dets[i];
      const double coef = system.coefs[0][i];
      const auto& pt_det_handler = [&](const Det& det_a, const int n_excite) {
        if (var_dets.has(det_a)) return;
        const size_t det_a_hash = det_hasher(det_a);
        const size_t batch_hash = Util::rehash(det_a_hash);
        if ((batch_hash & 127) != 0) return;  // For n a power of 2, "% n" = "& (n-1)"
        if (n_excite == 1) {
          const double h_ai = system.get_hamiltonian_elem(det, det_a, n_excite);
          const double hc = h_ai * coef;
          if (std::abs(hc) < eps_pt) return;  // Filter out small single excitation.
        }
        MathVector<double, 5> contrib;
        hc_sums.async_set(det_a, contrib);
      };
      static_cast<void>(system.find_connected_dets(
          det, eps_pt_max / std::abs(coef), eps_pt / std::abs(coef), pt_det_handler));
    });
    hc_sums.sync();
    const size_t n_pt_dets = hc_sums.get_n_keys();
    hc_sums.clear();
    const size_t n_pt_dets_batch = n_pt_dets * 128 / n_batches;
    double default_mem_usage = 0.4;
    if (system.type == SystemType::HEG) default_mem_usage = 1.0;
    const double mem_usage = Config::get<double>("pt_sto_mem_usage", default_mem_usage);
    n_unique_target = pt_mem_avail * mem_usage * n_unique_dets_in_sample / bytes_per_entry / 5.0 /
                      n_pt_dets_batch;
    const size_t max_unique_targets = n_var_dets / 8 + 1;
    if (n_unique_target >= max_unique_targets) n_unique_target = max_unique_targets;
    if (Parallel::is_master()) printf("Number of unique var dets in sample target: %'zu\n", n_unique_target);
    sample_dets_sto.clear();
    sample_dets_list.clear();
    n_dets_in_sample = 0;
    n_unique_dets_in_sample = 0;
    while (n_unique_dets_in_sample < n_unique_target) {
      const double rand_01 = unif_real_distr(generator);
      const int sample_det_id =
          std::lower_bound(cum_probs.begin(), cum_probs.end(), rand_01) - cum_probs.begin();
      if (sample_dets_sto.count(sample_det_id) == 0) {
        n_unique_dets_in_sample++;
      }
      n_dets_in_sample++;
      sample_dets_sto[sample_det_id]++;
    }
    sample_dets_sto.clear();
    fgpl::broadcast(n_dets_in_sample);
    if (Parallel::is_master()) {
      printf("Number of dets chosen: %'zu\n", n_dets_in_sample);
    }
    Timer::checkpoint("determine n samples");
  }

  if (Config::get<bool>("semisto_sto", true)) {
    // Top 25% of dets are made deterministic.
    n_dtm_dets = n_unique_target / 4;
    // Use a min heap to find the 25% quantile as threshold.
    std::priority_queue<double, std::vector<double>, std::greater<double>> probs_heap;
    for (size_t i = 0; i < n_dtm_dets; i++) probs_heap.push(probs[i]);
    for (size_t i = n_dtm_dets; i < n_var_dets; i++) {
      if (probs_heap.top() < probs[i]) {
        probs_heap.pop();
        probs_heap.push(probs[i]);
      }
    }
    double C = probs_heap.top();  // C is threshold for deterministic dets
    double sum_weights = 0.;
    for (size_t i = 0; i < n_var_dets; i++) {
      if (probs[i] > C) {
        probs[i] = 0.;
        sample_dets_list.push_back(i);
      } else {
        sum_weights += probs[i];
      }
    }
    n_dtm_dets = sample_dets_list.size(); // Might have changed due to decimal comparison.
    if (Parallel::is_master()) printf("Number of deterministic var dets: %'zu\n", n_dtm_dets);
    for (auto& prob : probs) prob /= sum_weights;

    fgpl::broadcast(n_dtm_dets);
  }

  std::vector<double> alias_probs;
  std::vector<size_t> aliases;
  Util::setup_alias_arrays(probs, alias_probs, aliases);

  std::uniform_int_distribution<size_t> unif_int_distr_n_var_dets(0, n_var_dets - 1);
  std::uniform_int_distribution<size_t> unif_int_distr_n_batches(0, n_batches - 1);

  while (iteration < max_pt_iterations) {
    Timer::start(Util::str_printf("#%zu", iteration + 1));

    // Generate random sample
    for (size_t i = 0; i < n_dets_in_sample - n_dtm_dets; i++) {
      const size_t rand_01 = unif_int_distr_n_var_dets(generator);  // rand int in [0, n_var_dets - 1]
      const double rand_02 = unif_real_distr(generator);  // rand real in [0., 1.)
      size_t sample_det_id;
      if (rand_02 < alias_probs[rand_01])
        sample_det_id = rand_01;
      else
        sample_det_id = aliases[rand_01];

      if (sample_dets_sto.count(sample_det_id) == 0) sample_dets_list.push_back(sample_det_id);
      sample_dets_sto[sample_det_id]++;
    }
    fgpl::broadcast(sample_dets_sto);
    fgpl::broadcast(sample_dets_list);
    if (Parallel::is_master()) {
      printf(
          "Number of unique variational determinants in sample: %'zu\n", sample_dets_list.size());
    }

    // Select random batch.
    size_t batch_id = unif_int_distr_n_batches(generator);
    fgpl::broadcast(batch_id);
    const size_t n_unique_dets_in_sample = sample_dets_list.size();
    if (Parallel::is_master()) printf("Batch id: %zu / %zu\n", batch_id, n_batches);

    const double factor = 1. / (n_dets_in_sample - n_dtm_dets - 1.);
    for (size_t j = 0; j < 5; j++) {
      fgpl::DistRange<size_t>(j, n_unique_dets_in_sample, 5).for_each([&](const size_t sample_id) {
        const size_t i = sample_dets_list[sample_id];
        const Det& det = system.dets[i];
        const double coef = system.coefs[i_state][i];
        const bool is_dtm_det = sample_id < n_dtm_dets;
        const double count = is_dtm_det ? 1. : static_cast<double>(sample_dets_sto[i]);  // w_i
        const double weight =
            is_dtm_det ? 1. : (n_dets_in_sample - n_dtm_dets) * probs[i];  // <w_i>
        const auto& pt_det_handler = [&](const Det& det_a, const int n_excite) {
          const size_t det_a_hash = det_hasher(det_a);
          const size_t batch_hash = Util::rehash(det_a_hash);
          if (batch_hash % n_batches != batch_id) return;
          if (var_dets.has(det_a)) return;
          const double h_ai = system.get_hamiltonian_elem(det, det_a, n_excite);
          const double hc = h_ai * coef;
          if (std::abs(hc) < eps_pt) return;  // Filter out small single excitation.

          MathVector<double, 5> contrib;
          contrib[0] = count * hc / weight;
          if (std::abs(hc) > eps_pt_psto) contrib[1] = contrib[0];
          if (!is_dtm_det) {
            contrib[2] = count * hc / weight * sqrt(factor);
            if (std::abs(hc) > eps_pt_psto)
              contrib[3] = contrib[2];
            else
              contrib[4] = pow(count * hc / weight, 2) * (1. - weight / count + factor);
          }
          hc_sums.async_set(det_a, contrib, fgpl::Reducer<MathVector<double, 5>>::sum);
        };
        static_cast<void>(system.find_connected_dets(
            det, eps_pt_max / std::abs(coef), eps_pt / std::abs(coef), pt_det_handler));
      });
      hc_sums.sync(fgpl::Reducer<MathVector<double, 5>>::sum);
      if (Parallel::is_master()) printf("%zu%% ", (j + 1) * 20);
    }
    const size_t n_pt_dets = hc_sums.get_n_keys();
    if (Parallel::is_master()) printf("\nNumber of sto pt dets: %'zu\n", n_pt_dets);
    sample_dets_sto.clear();
    sample_dets_list.resize(n_dtm_dets);
    Timer::checkpoint("create hc sums");

    const double energy_pt_sto_loop = mapreduce_sum<MathVector<double, 5>>(
        hc_sums, [&](const Det& det_a, const MathVector<double, 5>& hc_sum) {
          const double h_aa = system.get_hamiltonian_elem(det_a, det_a, 0);
          const double factor = static_cast<double>(n_batches) / (system.energy_var[i_state] - h_aa);
          return (pow(hc_sum[0], 2) - pow(hc_sum[1], 2) + pow(hc_sum[2], 2) - pow(hc_sum[3], 2) -
                  hc_sum[4]) *
                 factor;
        })[0];

    energy_pt_sto_loops.push_back(energy_pt_sto_loop);
    energy_pt_sto.value = Util::avg(energy_pt_sto_loops);
    energy_pt_sto.uncert = Util::stdev(energy_pt_sto_loops) / sqrt(iteration + 1.0);
    if (Parallel::is_master()) {
      printf("PT sto loop correction: " ENERGY_FORMAT "\n", energy_pt_sto_loop);
      printf("PT sto correction (eps1= %.2e, eps_pt= %.2e):", eps_var, eps_pt);
      printf(" %s Ha\n", energy_pt_sto.to_string().c_str());
      printf("PT sto total energy (eps1= %.2e, eps_pt= %.2e):", eps_var, eps_pt);
      printf(" %s Ha\n", (energy_pt_sto + energy_pt_psto).to_string().c_str());
      printf("Correlation energy (eps1= %.2e, eps_pt= %.2e):", eps_var, eps_pt);
      printf(" %s Ha\n", (energy_pt_sto + energy_pt_psto - system.energy_hf).to_string().c_str());
    }

    hc_sums.clear();
    Timer::end();
    iteration++;
    if (iteration >= 6 && energy_pt_sto.uncert <= target_error * 0.7) {
      break;
    }
    if (iteration >= 10 && (energy_pt_sto + energy_pt_psto).uncert <= target_error) {
      break;
    }
  }

  hc_sums.clear_and_shrink();
  Timer::end();
  return energy_pt_sto + energy_pt_psto;
}

template <class S>
template <class C>
std::array<double, 2> Solver<S>::mapreduce_sum(
    const fgpl::DistHashMap<Det, C, DetHasher>& map,
    const std::function<double(const Det& det, const C& hc_sum)>& mapper) const {
  const int n_threads = omp_get_max_threads();
  std::vector<double> res_sq_thread(n_threads, 0.0);
  std::vector<double> res_thread(n_threads, 0.0);
  map.for_each([&](const Det& key, const size_t, const C& value) {
    const int thread_id = omp_get_thread_num();
    const double mapped = mapper(key, value);
    res_thread[thread_id] += mapped;
    res_sq_thread[thread_id] += mapped * mapped;
  });
  std::array<double, 2> res_local = {0.0, 0.0};
  std::array<double, 2> res = {0.0, 0.0};
  for (int i = 0; i < n_threads; i++) {
    res_local[0] += res_thread[i];
    res_local[1] += res_sq_thread[i];
  }
  MPI_Allreduce(&res_local, &res, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return res;
}

template <class S>
bool Solver<S>::load_variation_result(const std::string& filename) {
  if (Parallel::is_master()) {
    printf("Try Loading Wavefunction %s\n", filename.c_str());
    fflush(stdout);
  }
  std::string serialized;
  const int TRUNK_SIZE = 1 << 20;
  char buffer[TRUNK_SIZE];
  MPI_File file;
  int error;
  error = MPI_File_open(
      MPI_COMM_WORLD, filename.c_str(), MPI_MODE_RDONLY | MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
  if (error) return false;
  MPI_Offset size;
  MPI_File_get_size(file, &size);
  MPI_Status status;
  while (size > TRUNK_SIZE) {
    // Parallel File loading and tree distribution.
    MPI_File_read_all(file, buffer, TRUNK_SIZE, MPI_CHAR, &status);
    serialized.append(buffer, TRUNK_SIZE);
    size -= TRUNK_SIZE;
  }
  MPI_File_read_all(file, buffer, size, MPI_CHAR, &status);
  serialized.append(buffer, size);
  MPI_File_close(&file);
  hps::from_string(serialized, system);
  if (Parallel::is_master()) {
    printf("Loaded %'zu dets from: %s\n", system.get_n_dets(), filename.c_str());
    print_dets_info();
    printf("HF energy: " ENERGY_FORMAT "\n", system.energy_hf);
    printf("Variational energy: ");
    for (const double energy_var : system.energy_var) printf(ENERGY_FORMAT "\t", energy_var);
    printf("\n");
  }
  return true;
}

template <class S>
void Solver<S>::save_variation_result(const std::string& filename) {
  if (Parallel::is_master()) {
    std::ofstream file(filename, std::ofstream::binary);
    hps::to_stream(system, file);
    printf("Variational results saved to: %s\n", filename.c_str());
  }
}

template <class S>
void Solver<S>::save_pair_contrib(const double eps_var) {
  const auto& det_hf = system.dets[0];
  // const size_t n_elecs = system.n_elecs;
  const size_t n_up = system.n_up;
  // const size_t n_dn = system.n_dn;
  if (det_hf.up != det_hf.dn) {
    throw std::invalid_argument("non sym det_hf not implemented");
  }
  std::vector<std::vector<double>> contribs(n_up);
  std::string contrib_filename = Util::str_printf("pair_contrib_%#.2e.csv", eps_var);
  const auto& contrib_entry = Util::str_printf("pair_contrib/%#.2e", eps_var);
  Result::put<std::string>(contrib_entry, contrib_filename);
  std::ofstream contrib_file(contrib_filename);
  contrib_file << "i,j,pair_contrib" << std::endl;
  for (size_t i = 0; i < n_up; i++) {
    contribs[i].assign(n_up, 0.0);
  }
  const double c0 = system.coefs[0][0];
  for (size_t det_id = 1; det_id < system.dets.size(); det_id++) {
    const auto& det = system.dets[det_id];
    const double coef = system.coefs[0][det_id];
    const auto& diff_up = det_hf.up.diff(det.up);
    const auto& diff_dn = det_hf.dn.diff(det.dn);
    const unsigned n_excite = diff_up.n_diffs + diff_dn.n_diffs;
    if (n_excite > 2) continue;
    size_t i = 0;
    size_t j = 0;
    const auto& H = system.get_hamiltonian_elem(det_hf, det, -1);
    if (diff_up.n_diffs == 2) {
      i = diff_up.left_only[0];
      j = diff_up.left_only[1];
    } else if (diff_up.n_diffs == 1) {
      i = diff_up.left_only[0];
      if (diff_dn.n_diffs == 1) {
        j = diff_dn.left_only[0];
        if (j < i) {
          std::swap(i, j);
        }
      } else {
        j = i;
      }
    } else {
      i = diff_dn.left_only[0];
      if (diff_dn.n_diffs == 2) {
        j = diff_dn.left_only[1];
        if (j < i) {
          std::swap(i, j);
        }
      } else {
        j = i;
      }
    }
    if (det.up == det.dn) {
      contribs[i][j] += H * coef / c0;
    } else if (system.time_sym) {
      contribs[i][j] += H * coef / c0 * Util::SQRT2;
    } else {
      contribs[i][j] += H * coef / c0;
    }
  }
  contrib_file.precision(15);
  for (size_t i = 0; i < n_up; i++) {
    for (size_t j = i; j < n_up; j++) {
      if (i != j) {
        contribs[i][j] /= 2;
      }
      contrib_file << i << "," << j << "," << contribs[i][j] << std::endl;
    }
  }
  contrib_file.close();
}

template <class S>
void Solver<S>::print_dets_info() const {
  if (system.time_sym) {
    // Print effective dets for unpacked time sym.
    size_t n_eff_dets = 0;
    for (const auto& det : system.dets) {
      if (det.up == det.dn) {
        n_eff_dets += 1;
      } else if (det.up < det.dn) {
        n_eff_dets += 2;
      } else {
        throw std::runtime_error("wf has unvalid det for time sym");
      }
    }
    printf("Effect dets (without time sym): %'zu\n", n_eff_dets);
  }

  for (unsigned i_state = 0; i_state < system.n_states; i_state++) {
    printf("State %d:\n", i_state);
    // Print excitations.
    std::unordered_map<unsigned, size_t> excitations;
    std::unordered_map<unsigned, double> weights;
    unsigned highest_excitation = 0;
    const auto& det_hf = system.dets[0];
    for (size_t i = 0; i < system.dets.size(); i++) {
      const auto& det = system.dets[i];
      const double coef = system.coefs[i_state][i];
      const unsigned n_excite = det_hf.up.n_diffs(det.up) + det_hf.dn.n_diffs(det.dn);
      if (det.up != det.dn && system.time_sym) {
        excitations[n_excite] += 2;
      } else {
        excitations[n_excite] += 1;
      }
      weights[n_excite] += coef * coef;
      if (highest_excitation < n_excite) highest_excitation = n_excite;
    }
    printf("----------------------------------------\n");
    printf("%-10s%12s%16s\n", "Excite Lv", "# dets", "Sum c^2");
    for (unsigned i = 0; i <= highest_excitation; i++) {
      if (excitations.count(i) == 0) {
        excitations[i] = 0;
        weights[i] = 0.0;
      }
      printf("%-10u%12zu%16.8f\n", i, excitations[i], weights[i]);
    }

    // Print orb occupations.
    std::vector<double> orb_occupations(system.n_orbs, 0.0);
#pragma omp parallel for schedule(static, 1)
    for (unsigned j = 0; j < system.n_orbs; j++) {
      for (size_t i = 0; i < system.dets.size(); i++) {
        const auto& det = system.dets[i];
        const double coef = system.coefs[i_state][i];
        if (det.up.has(j)) {
          orb_occupations[j] += coef * coef;
        }
        if (det.dn.has(j)) {
          orb_occupations[j] += coef * coef;
        }
      }
    }
    printf("----------------------------------------\n");
    printf("%-10s%12s%16s\n", "Orbital", "", "Sum c^2");
    for (unsigned j = 0; j < system.n_orbs && j < 50; j++) {
      printf("%-10u%12s%16.8f\n", j, "", orb_occupations[j]);
    }
    double sum_orb_occupation = std::accumulate(orb_occupations.begin(), orb_occupations.end(), 0.0);
    printf("Sum orbitals c^2: %.8f\n", sum_orb_occupation);
  
    // Analyze determinants sharing same alpha/beta half-determinants
    if (i_state == 0) {  // Only print for first state to avoid redundancy
      std::unordered_map<HalfDet, size_t, HalfDetHasher> alpha_counts;
      std::unordered_map<HalfDet, size_t, HalfDetHasher> beta_counts;
      
      for (const auto& det : system.dets) {
        alpha_counts[det.up]++;
        beta_counts[det.dn]++;
      }
      
      // Find maximum sharing
      size_t max_alpha_share = 0;
      size_t max_beta_share = 0;
      HalfDet most_shared_alpha;
      HalfDet most_shared_beta;
      
      for (const auto& pair : alpha_counts) {
        if (pair.second > max_alpha_share) {
          max_alpha_share = pair.second;
          most_shared_alpha = pair.first;
        }
      }
      
      for (const auto& pair : beta_counts) {
        if (pair.second > max_beta_share) {
          max_beta_share = pair.second;
          most_shared_beta = pair.first;
        }
      }
      
      printf("----------------------------------------\n");
      printf("Half-determinant sharing analysis:\n");
      printf("Unique alpha half-dets: %zu\n", alpha_counts.size());
      printf("Unique beta half-dets: %zu\n", beta_counts.size());
      printf("Max dets sharing same alpha: %zu\n", max_alpha_share);
      printf("Max dets sharing same beta: %zu\n", max_beta_share);
      
      // Print distribution of sharing
      std::map<size_t, size_t> alpha_share_dist;
      std::map<size_t, size_t> beta_share_dist;
      
      for (const auto& pair : alpha_counts) {
        alpha_share_dist[pair.second]++;
      }
      for (const auto& pair : beta_counts) {
        beta_share_dist[pair.second]++;
      }
      
      printf("\nAlpha sharing distribution (# shared -> count):\n");
      for (const auto& pair : alpha_share_dist) {
        if (pair.first >= 10 || pair.second >= 10) {  // Only print significant entries
          printf("  %zu dets shared -> %zu alpha half-dets\n", pair.first, pair.second);
        }
      }
      
      printf("\nBeta sharing distribution (# shared -> count):\n");
      for (const auto& pair : beta_share_dist) {
        if (pair.first >= 10 || pair.second >= 10) {  // Only print significant entries
          printf("  %zu dets shared -> %zu beta half-dets\n", pair.first, pair.second);
        }
      }
    }
  
    // Print most important dets.
    printf("----------------------------------------\n");
    printf("Most important dets:\n");
    std::vector<size_t> det_order(system.dets.size());
    for (size_t i = 0; i < system.dets.size(); i++) {
      det_order[i] = i;
    }
    const auto& comp = [&](const size_t a, const size_t b) {
      if (std::abs(system.coefs[i_state][a]) != std::abs(system.coefs[i_state][b])) {
        return std::abs(system.coefs[i_state][a]) < std::abs(system.coefs[i_state][b]);
      }
      return a > b;
    };
    std::priority_queue<size_t, std::vector<size_t>, decltype(comp)> det_ordered(comp, det_order);
    printf("%-10s%12s      %-12s\n", "Excite Lv", "Coef", "Det (Reordered orb)");
    for (size_t i = 0; i < std::min((size_t)20, system.dets.size()); i++) {
      size_t ordered_i = det_ordered.top();
      det_ordered.pop();
      const double coef = system.coefs[i_state][ordered_i];
      const auto& det = system.dets[ordered_i];
      const auto& occs_up = det.up.get_occupied_orbs();
      const auto& occs_dn = det.dn.get_occupied_orbs();
      const unsigned n_excite = det_hf.up.n_diffs(det.up) + det_hf.dn.n_diffs(det.dn);
      printf("%-10u%12.8f", n_excite, coef);
      printf("      | ");
      for (unsigned j = 0; j < system.n_up; j++) {
        printf("%2u ", occs_up[j]);
      }
      printf("| ");
      for (unsigned j = 0; j < system.n_dn; j++) {
        printf("%2u ", occs_dn[j]);
      }
      printf("|\n");
    }
    printf("----------------------------------------\n");
  }
}

template <class S>
std::string Solver<S>::get_state_suffix(const unsigned i_state) const {
  if (i_state == 0)
    return "";
  else
    return "_" + std::to_string(i_state);
}

template <class S>
std::string Solver<S>::get_wf_filename(const double eps_var) const {
  return Util::str_printf("wf_eps1_%#.2e.dat", eps_var);
}

template <class S>
std::vector<double> Solver<S>::generate_automated_epsilon_schedule(const double eps_final) {
  if (Parallel::is_master()) {
    printf("Generating automated epsilon schedule with eps_final = %#.2e\n", eps_final);
  }
  
  // Initialize with HF determinant to find connected determinants
  const Det& det_hf = system.dets[0];
  
  double max_H_same = 0.0;   // Maximum |H_ij| for same-spin double excitations
  double max_H_opp = 0.0;    // Maximum |H_ij| for opposite-spin double excitations
  
  // Handler to find maximum Hamiltonian elements for different excitation types
  const auto& connection_handler = [&](const Det& connected_det, const int n_excite) {
    if (n_excite == 2) {
      const double h_ij = std::abs(system.get_hamiltonian_elem(det_hf, connected_det, 2));
      
      // Determine if this is a same-spin or opposite-spin excitation
      const unsigned n_up_diffs = det_hf.up.n_diffs(connected_det.up);
      const unsigned n_dn_diffs = det_hf.dn.n_diffs(connected_det.dn);
      
      if (n_up_diffs == 2 && n_dn_diffs == 0) {
        // Same-spin up excitation
        max_H_same = std::max(max_H_same, h_ij);
      } else if (n_up_diffs == 0 && n_dn_diffs == 2) {
        // Same-spin down excitation  
        max_H_same = std::max(max_H_same, h_ij);
      } else if (n_up_diffs == 1 && n_dn_diffs == 1) {
        // Opposite-spin excitation
        max_H_opp = std::max(max_H_opp, h_ij);
      }
    }
  };
  
  // Find all double excitations from HF determinant
  static_cast<void>(system.find_connected_dets(det_hf, Util::INF, 0.0, connection_handler, false));
  
  // Synchronize maximum values across MPI processes
  double max_H_same_global, max_H_opp_global;
  MPI_Allreduce(&max_H_same, &max_H_same_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&max_H_opp, &max_H_opp_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  max_H_same = max_H_same_global;
  max_H_opp = max_H_opp_global;
  
  // Calculate eps0 as minimum of the two maximum values
  const double eps0 = std::min(max_H_same, max_H_opp);
  
  // Get the epsilon schedule algorithm from config
  const std::string schedule_algorithm = Config::get<std::string>("epsilon_schedule_algorithm", "geometric_mean");
  
  std::vector<double> schedule;
  
  if (schedule_algorithm == "geometric_mean") {
    // Original algorithm: use geometric mean for intermediate value
    const double eps_intermediate = std::sqrt(eps0 * eps_final);
    
    if (Parallel::is_master()) {
      printf("Automated schedule (geometric_mean): eps0 = %#.2e (max_H_same = %#.2e, max_H_opp = %#.2e)\n", 
             eps0, max_H_same, max_H_opp);
      printf("                                     eps_intermediate = %#.2e\n", eps_intermediate);
      printf("                                     eps_final = %#.2e\n", eps_final);
    }
    
    // Only add intermediate stage if it's meaningfully different from endpoints
    if (eps_intermediate > eps_final * 2.0 && eps_intermediate < eps0 * 0.5) {
      schedule = {eps0, eps_intermediate};
    } else {
      schedule = {eps0};
    }
  } else if (schedule_algorithm == "divide_by_2") {
    // New algorithm: divide by 2 each iteration until reaching target
    double current_eps = eps0;
    schedule.push_back(current_eps);
    
    while (current_eps > eps_final * 2.0) {
      current_eps /= 2.0;
      if (current_eps >= eps_final) {
        schedule.push_back(current_eps);
      }
    }
    
    if (Parallel::is_master()) {
      printf("Automated schedule (divide_by_2): eps0 = %#.2e (max_H_same = %#.2e, max_H_opp = %#.2e)\n", 
             eps0, max_H_same, max_H_opp);
      printf("                                  intermediate values: ");
      for (size_t i = 1; i < schedule.size(); ++i) {
        printf("%#.2e ", schedule[i]);
      }
      printf("\n                                  eps_final = %#.2e\n", eps_final);
    }
  } else {
    throw std::invalid_argument("Invalid epsilon_schedule_algorithm: " + schedule_algorithm + 
                                ". Valid options are: geometric_mean, divide_by_2");
  }
  
  return schedule;
}

template <class S>
void Solver<S>::collect_off_diagonal_elements(
    const size_t i, const size_t j, 
    const double H_ij, const double c_i, const double c_j) {
  
  if (!off_diagonal_collection_enabled) return;
  
  // Bounds checking to prevent crashes
  if (i >= system.dets.size() || j >= system.dets.size()) return;
  if (i == j) return;  // Skip diagonal elements
  
  const double magnitude = std::abs(c_i * H_ij * c_j);
  const int preconditioner_rank_k = Config::get<int>("davidson/preconditioner_rank_k", 200);
  
  // Filter out very small elements for numerical stability
  if (magnitude < 1e-12 || !std::isfinite(magnitude)) return;
  
  OffDiagElement element{i, j, H_ij, magnitude};
  
  // Thread-safe collection
  std::lock_guard<std::mutex> lock(off_diag_mutex);
  
  if (off_diag_heap.size() < static_cast<size_t>(preconditioner_rank_k)) {
    // Heap not full, add directly
    off_diag_heap.push(element);
  } else if (magnitude > off_diag_heap.top().magnitude) {
    // New element is larger than smallest in heap, replace
    off_diag_heap.pop();
    off_diag_heap.push(element);
  }
}

template <class S>
void Solver<S>::finalize_off_diagonal_collection() {
  if (!off_diagonal_collection_enabled) return;
  
  // Convert heap to sorted vector
  collected_off_diagonal_elements.clear();
  collected_off_diagonal_elements.reserve(off_diag_heap.size());
  
  while (!off_diag_heap.empty()) {
    collected_off_diagonal_elements.push_back(off_diag_heap.top());
    off_diag_heap.pop();
  }
  
  // Sort by magnitude (largest first) for optimal preconditioner construction
  std::sort(collected_off_diagonal_elements.begin(), collected_off_diagonal_elements.end(),
            [](const OffDiagElement& a, const OffDiagElement& b) {
              return a.magnitude > b.magnitude;
            });
  
  if (Parallel::is_master() && !collected_off_diagonal_elements.empty()) {
    printf("Collected %zu off-diagonal elements for dynamic preconditioner\n", 
           collected_off_diagonal_elements.size());
    printf("Largest magnitude: %#.6e, smallest: %#.6e\n",
           collected_off_diagonal_elements.front().magnitude,
           collected_off_diagonal_elements.back().magnitude);
    
    // Debug: Show H_ij values
    double min_h = std::abs(collected_off_diagonal_elements[0].h_ij);
    double max_h = std::abs(collected_off_diagonal_elements[0].h_ij);
    for (const auto& elem : collected_off_diagonal_elements) {
      min_h = std::min(min_h, std::abs(elem.h_ij));
      max_h = std::max(max_h, std::abs(elem.h_ij));
    }
    printf("DEBUG: |H_ij| range: [%#.6e, %#.6e]\n", min_h, max_h);
    
    // Show first few elements with H_ij values
    printf("DEBUG: First 3 elements: ");
    for (size_t i = 0; i < std::min(size_t(3), collected_off_diagonal_elements.size()); i++) {
      const auto& elem = collected_off_diagonal_elements[i];
      printf("(i=%zu,j=%zu,H=%.3e,mag=%.3e) ", elem.i, elem.j, elem.h_ij, elem.magnitude);
    }
    printf("\n");
    
    // Debug: Print top 10 and bottom 10 collected elements
    const int preconditioner_rank_k = Config::get<int>("davidson/preconditioner_rank_k", 200);
    if (preconditioner_rank_k <= 50) {  // Only for small debug runs
      printf("DEBUG: Top 10 collected elements:\n");
      for (size_t i = 0; i < std::min(10UL, collected_off_diagonal_elements.size()); i++) {
        const auto& elem = collected_off_diagonal_elements[i];
        printf("  [%zu] i=%zu, j=%zu, magnitude=%#.6e\n", i, elem.i, elem.j, elem.magnitude);
      }
      if (collected_off_diagonal_elements.size() > 10) {
        printf("DEBUG: Bottom 10 collected elements:\n");
        size_t start = std::max(10UL, collected_off_diagonal_elements.size() - 10);
        for (size_t i = start; i < collected_off_diagonal_elements.size(); i++) {
          const auto& elem = collected_off_diagonal_elements[i];
          printf("  [%zu] i=%zu, j=%zu, magnitude=%#.6e\n", i, elem.i, elem.j, elem.magnitude);
        }
      }
    }
  }
}

template <class S>
void Solver<S>::verify_off_diagonal_collection_brute_force() {
  if (!off_diagonal_collection_enabled) return;
  
  const int preconditioner_rank_k = Config::get<int>("davidson/preconditioner_rank_k", 200);
  if (preconditioner_rank_k > 50 || system.dets.size() > 1000) return;  // Only for small debug runs
  
  const size_t n_dets = system.dets.size();
  if (n_dets < 2) return;
  
  // Get current eigenvector coefficients
  const auto& coefs = system.coefs[0];
  
  // Brute force: collect ALL off-diagonal elements
  std::vector<OffDiagElement> all_elements;
  
  for (size_t i = 0; i < n_dets; i++) {
    const auto& row = hamiltonian.matrix.get_row(i);
    
    for (size_t k = 0; k < row.size(); k++) {
      const size_t j = row.get_index(k);
      const double H_ij = row.get_value(k);
      
      if (i == j) continue;  // Skip diagonal elements
      if (j <= i) continue;  // Only upper triangular
      
      if (std::abs(H_ij) < 1e-12) continue;
      
      // Get max coefficients for both determinants
      double c_i_max = coefs[i];
      double c_j_max = coefs[j];
      
      for (unsigned i_state = 1; i_state < system.n_states; i_state++) {
        if (std::abs(system.coefs[i_state][i]) > std::abs(c_i_max)) c_i_max = system.coefs[i_state][i];
        if (std::abs(system.coefs[i_state][j]) > std::abs(c_j_max)) c_j_max = system.coefs[i_state][j];
      }
      
      const double magnitude = std::abs(c_i_max * H_ij * c_j_max);
      
      if (magnitude > 1e-12) {
        all_elements.push_back({i, j, H_ij, magnitude});
      }
    }
  }
  
  // Sort all elements by magnitude (largest first)
  std::sort(all_elements.begin(), all_elements.end(),
            [](const OffDiagElement& a, const OffDiagElement& b) {
              return a.magnitude > b.magnitude;
            });
  
  if (Parallel::is_master()) {
    printf("BRUTE FORCE VERIFICATION:\n");
    printf("Total off-diagonal elements found: %zu\n", all_elements.size());
    printf("Expected to collect top %d elements\n", preconditioner_rank_k);
    
    if (!all_elements.empty()) {
      printf("Actual largest magnitude: %#.6e\n", all_elements.front().magnitude);
      size_t top_k = std::min(static_cast<size_t>(preconditioner_rank_k), all_elements.size());
      if (top_k > 0) {
        printf("Actual %zu-th largest magnitude: %#.6e\n", top_k, all_elements[top_k-1].magnitude);
      }
      
      printf("Top 10 actual largest elements:\n");
      for (size_t i = 0; i < std::min(10UL, all_elements.size()); i++) {
        const auto& elem = all_elements[i];
        printf("  [%zu] i=%zu, j=%zu, magnitude=%#.6e\n", i, elem.i, elem.j, elem.magnitude);
      }
    }
  }
}

template <class S>
void Solver<S>::prepare_shadow_wavefunction_preconditioner() {
  if (!use_shadow_wavefunction || important_connections.empty()) {
    return;
  }
  
  const int preconditioner_rank_k = Config::get<int>("davidson/preconditioner_rank_k", 200);
  const size_t n_dets = system.get_n_dets();
  
  // Step 1: Build a map from determinant to index for fast lookup
  std::unordered_map<Det, size_t, DetHasher> det_to_index;
  for (size_t i = 0; i < n_dets; i++) {
    det_to_index[system.dets[i]] = i;
  }
  
  // Step 2: Construct shadow CI vector c_shadow
  std::vector<double> c_shadow(n_dets);
  
  // Copy old coefficients (already have correct values)
  for (size_t i = 0; i < n_dets; i++) {
    c_shadow[i] = system.coefs[0][i];
    for (unsigned i_state = 1; i_state < system.n_states; i_state++) {
      if (std::abs(system.coefs[i_state][i]) > std::abs(c_shadow[i])) {
        c_shadow[i] = system.coefs[i_state][i];
      }
    }
  }
  
  // For new determinants, compute perturbative estimates
  // Group connections by new determinant for efficiency
  std::unordered_map<Det, std::vector<const Connection*>, DetHasher> connections_by_det;
  for (const auto& conn : important_connections) {
    connections_by_det[conn.new_det].push_back(&conn);
  }
  
  // Get previous variational energy (use HF if first iteration)
  double E_var_old = system.energy_var.empty() ? system.energy_hf : system.energy_var[0];
  
  for (const auto& pair : connections_by_det) {
    const Det& new_det = pair.first;
    const std::vector<const Connection*>& conn_ptrs = pair.second;
    
    auto it = det_to_index.find(new_det);
    if (it != det_to_index.end()) {
      size_t new_idx = it->second;
      
      // Compute perturbative estimate for new determinant
      double numerator = 0.0;
      for (const Connection* conn : conn_ptrs) {
        // c_j_old is the coefficient of the old determinant
        double c_j_old = c_shadow[conn->j];
        numerator += conn->H_ij * c_j_old;
      }
      
      // Get diagonal element H_aa
      double H_aa = hamiltonian.matrix.get_diag(new_idx);
      double denominator = E_var_old - H_aa;
      
      if (std::abs(denominator) > 1e-10) {
        c_shadow[new_idx] = numerator / denominator;
      }
    }
  }
  
  // Step 3: Efficiently select top k off-diagonal elements using shadow wavefunction
  std::priority_queue<OffDiagElement> shadow_heap;
  
  // Process all connections
  for (const auto& conn : important_connections) {
    auto it = det_to_index.find(conn.new_det);
    if (it != det_to_index.end()) {
      size_t i = it->second;
      size_t j = conn.j;
      
      // Calculate importance metric: |c_shadow[i] * H_ij * c_shadow[j]|
      double metric = std::abs(c_shadow[i] * conn.H_ij * c_shadow[j]);
      
      if (metric > 1e-12) {
        OffDiagElement elem = {i, j, conn.H_ij, metric};
        
        if (shadow_heap.size() < static_cast<size_t>(preconditioner_rank_k)) {
          shadow_heap.push(elem);
        } else if (metric > shadow_heap.top().magnitude) {
          shadow_heap.pop();
          shadow_heap.push(elem);
        }
      }
    }
  }
  
  // Step 4: Replace collected elements with shadow-based selection
  collected_off_diagonal_elements.clear();
  collected_off_diagonal_elements.reserve(shadow_heap.size());
  
  while (!shadow_heap.empty()) {
    collected_off_diagonal_elements.push_back(shadow_heap.top());
    shadow_heap.pop();
  }
  
  // Sort by magnitude (largest first) for optimal preconditioner construction
  std::sort(collected_off_diagonal_elements.begin(), collected_off_diagonal_elements.end(),
            [](const OffDiagElement& a, const OffDiagElement& b) {
              return a.magnitude > b.magnitude;
            });
  
  if (Parallel::is_master() && !collected_off_diagonal_elements.empty()) {
    printf("Shadow wavefunction: Collected %zu off-diagonal elements for dynamic preconditioner\n", 
           collected_off_diagonal_elements.size());
    printf("Largest magnitude: %#.6e, smallest: %#.6e\n",
           collected_off_diagonal_elements.front().magnitude,
           collected_off_diagonal_elements.back().magnitude);
  }
}
