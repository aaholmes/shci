#pragma once

#include <fgpl/src/hash_map.h>
#include <hps/src/hps.h>
#include <functional>
#include <string>
#include <vector>
#include "det/det.h"
#include "solver/sparse_matrix.h"
#include "system_type.h"
#include "util.h"

// Base class for quantum systems (chemical, HEG, etc.)
// Defines common interface for SHCI calculations
class BaseSystem {
 public:
  SystemType type;  // Type of quantum system (chem, heg, etc.)

  unsigned n_up = 0;    // Number of up-spin electrons
  unsigned n_dn = 0;    // Number of down-spin electrons
  unsigned n_orbs = 0;  // Total number of orbitals
  unsigned n_elecs = 0; // Total number of electrons
  unsigned n_states = 1; // Number of states to compute

  bool time_sym = false;              // Use time-reversal symmetry
  bool has_single_excitation = true;  // Allow single excitations
  bool has_double_excitation = true;  // Allow double excitations

  double energy_hf = 0.0;  // Hartree-Fock energy
  std::vector<double> energy_var = std::vector<double>(n_states, 0.0);  // Variational energies

  size_t helper_size = 0;           // Size of diagonal helper table
  double energy_hf_1b = 0.0;        // One-body HF energy (for second rejection)
  double second_rejection_factor = 0.2;  // Rejection screening parameter
  
  std::vector<Det> dets;  // List of determinants in variational space
  std::vector<std::vector<double>> coefs;  // Coefficients for each state
  fgpl::HashMap<HalfDet, double, HalfDetHasher> diag_helper;  // Cache for diagonal elements

  // Get number of determinants in variational space
  size_t get_n_dets() const { return dets.size(); }

  // System-specific setup (load integrals, etc.)
  virtual void setup(const bool){};

  // Get one-body HF energy for second rejection
  virtual double get_e_hf_1b() const { return 0.; }

  // Find all determinants connected to given det within eps range
  virtual double find_connected_dets(
      const Det& det,
      const double eps_max,
      const double eps_min,
      const std::function<void(const Det&, const int n_excite)>& handler,
      const bool second_rejection) const = 0;

  // Calculate Hamiltonian matrix element between two determinants
  virtual double get_hamiltonian_elem(
      const Det& det_i, const Det& det_j, const int n_excite) const = 0;

  // Get Hamiltonian element using indices into dets array
  virtual double get_hamiltonian_elem(
      const size_t i, const size_t j, const int n_excite) const {
    return get_hamiltonian_elem(dets[i], dets[j], n_excite);
  };

  // Update diagonal helper cache
  virtual void update_diag_helper() = 0;

  virtual void post_variation(std::vector<std::vector<size_t>>&){};

  virtual void post_variation_optimization(
      SparseMatrix&, const std::string&) {};

  virtual void optimization_microiteration(
      SparseMatrix&, const std::string&) {};

  virtual void dump_integrals(const char*){};

  virtual void post_perturbation(){};

  // Calculate Hamiltonian element with time-reversal symmetry
  double get_hamiltonian_elem_time_sym(
      const Det& det_i, const Det& det_j, const int n_excite) const {
    double h = get_hamiltonian_elem(det_i, det_j, n_excite);
    if (det_i.up == det_i.dn) {
      if (det_j.up != det_j.dn) h *= Util::SQRT2;
    } else {
      if (det_j.up == det_j.dn) {
        h *= Util::SQRT2;
      } else {
        Det det_i_rev = det_i;
        det_i_rev.reverse_spin();
        h += get_hamiltonian_elem(det_i_rev, det_j, -1);
      }
    }
    return h;
  }

  // Time-symmetric Hamiltonian element using indices
  double get_hamiltonian_elem_time_sym(
      const size_t i, const size_t j, const int n_excite) const {
    return get_hamiltonian_elem_time_sym(dets[i], dets[j], n_excite);
  }

  // Expand time-symmetric wavefunction to full space
  void unpack_time_sym() {
    const size_t n_dets_old = get_n_dets();
    for (size_t i = 0; i < n_dets_old; i++) {
      const auto& det = dets[i];
      if (det.up < det.dn) {
        Det det_rev = det;
        det_rev.reverse_spin();
	for (auto& state_coefs: coefs) {
          const double coef_new = state_coefs[i] * Util::SQRT2_INV;
          state_coefs[i] = coef_new;
          state_coefs.push_back(coef_new);
	}
        dets.push_back(det_rev);
      }
    }
  }

  template <class B>
  void serialize(B& buf) const {
    buf << n_up << n_dn << dets << coefs << energy_hf << energy_var << time_sym;
  }

  template <class B>
  void parse(B& buf) {
    buf >> n_up >> n_dn >> dets >> coefs >> energy_hf >> energy_var >> time_sym;
  }

  virtual void variation_cleanup(){};
};
