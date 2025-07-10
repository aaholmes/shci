#pragma once

#include <unordered_map>
#include "../base_system.h"
#include "k_points.h"

class HegSystem : public BaseSystem {
 public:
  void setup(const bool load_integrals_from_file = true) override;

  double find_connected_dets(
      const Det&,
      const double,
      const double,
      const std::function<void(const Det&, const int)>&,
      const bool second_rejection = false) const override;

  double get_hamiltonian_elem(const Det&, const Det&, const int) const override;

  void update_diag_helper() override {}

  // Orbital partitioning stubs (not implemented for HEG)
  void populate_screener(const Det& det, int det_id) {}
  uint64_t compute_occupation_key(const Det& det, int group_id) const { return 0; }
  bool use_orbital_partitioning = false;
  int partitioning_threshold = 500;
  std::vector<std::unordered_map<uint64_t, std::vector<int>>> same_spin_screener;

 private:
  double r_cut;

  double r_s;

  double k_unit;

  double H_unit;

  KPoints k_points;

  double max_abs_H;

  std::unordered_map<KPoint, std::vector<std::pair<KPoint, double>>, KPointHasher> hci_queue;

  void setup_hci_queue();

  void setup_hf();

  double get_hamiltonian_elem_no_time_sym(const Det& det_i, const Det& det_j, int n_excite) const;

  double get_one_body_diag(const Det& det) const;

  double get_two_body_diag(const Det& det) const;

  double get_two_body_double(const DiffResult& diff_up, const DiffResult& diff_dn) const;
};
