#include "chem_system.h"

#include <fgpl/src/concurrent_hash_map.h>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include "../parallel.h"
#include "../result.h"
#include "../timer.h"
#include "../util.h"
#include "dooh_util.h"
#include "optimization.h"
#include "rdm.h"
#include <eigen/Eigen/Dense>
#include <fstream>

void ChemSystem::setup(const bool load_integrals_from_file) {
  if (load_integrals_from_file) { // during optimization, no need to reload
    type = SystemType::Chemistry;
    n_up = Config::get<unsigned>("n_up");
    n_dn = Config::get<unsigned>("n_dn");
    n_elecs = n_up + n_dn;
    Result::put("n_elecs", n_elecs);
    n_states = Config::get<unsigned>("n_states", 1);
  
    point_group = get_point_group(Config::get<std::string>("chem/point_group"));
    product_table.set_point_group(point_group);
    time_sym = Config::get<bool>("time_sym", false);
    has_double_excitation = Config::get<bool>("has_double_excitation", true);
    enforce_active_space = Config::get<bool>("chem/active_space", false);

    Timer::start("load integrals");
    integrals.load();
    integrals.set_point_group(point_group);
    n_orbs = integrals.n_orbs;
    orb_sym = integrals.orb_sym;
    check_group_elements();
    Timer::end();
    setup_sym_orbs();
  }

  Timer::start("setup hci queue");
  setup_hci_queue();
  Timer::end();

  Timer::start("setup singles_queue");
  setup_singles_queue();
  Timer::end();

  // Initialize orbital partitioning configuration
  use_orbital_partitioning = Config::get<bool>("use_orbital_partitioning", false);
  partitioning_threshold = Config::get<int>("partitioning_threshold", 500);
  
  if (use_orbital_partitioning) {
    Timer::start("setup orbital partitioning");
    setup_orbital_partitioning();
    Timer::end();
  }

  dets.push_back(integrals.det_hf);
  
  // Populate orbital partitioning screener for HF determinant
  if (use_orbital_partitioning) {
    populate_screener(integrals.det_hf, 0);
  }

  coefs.resize(n_states);
  coefs[0].push_back(1.0);
  for (unsigned i_state = 1; i_state < n_states; i_state++)  {
    coefs[i_state].push_back(1e-16);
  }
  energy_hf = get_hamiltonian_elem(integrals.det_hf, integrals.det_hf, 0);
  if (Parallel::is_master()) {
    printf("HF energy: " ENERGY_FORMAT "\n", energy_hf);
  }

  if (rotation_matrix.size() != n_orbs * n_orbs) rotation_matrix = Eigen::MatrixXd::Identity(n_orbs, n_orbs);
}

void ChemSystem::setup_sym_orbs() {
  sym_orbs.resize(product_table.get_n_syms() + 1);  // Symmetry starts from 1.
  for (unsigned orb = 0; orb < n_orbs; orb++) {
    unsigned sym = orb_sym[orb];
    if (sym >= sym_orbs.size()) sym_orbs.resize(sym + 1);  // For Dooh.
    sym_orbs[sym].push_back(orb);
  }
}

void ChemSystem::setup_singles_queue() {
  size_t n_entries = 0;
  max_singles_queue_elem = 0.0;

  const int n_threads = Parallel::get_n_threads();
  std::vector<size_t> n_entries_local(n_threads, 0);
  std::vector<double> max_singles_queue_elem_local(n_threads, 0.0);

  singles_queue.resize(n_orbs);
#pragma omp parallel for schedule(dynamic, 5)
  for (unsigned p = 0; p < n_orbs; p++) {
    const int thread_id = omp_get_thread_num();
    const unsigned sym_p = orb_sym[p];
    for (const unsigned r : sym_orbs[sym_p]) {
      const double S = get_singles_queue_elem(p, r);
      if (S == 0.0) continue;
      singles_queue.at(p).push_back(Sr(S, r));
    }
    if (singles_queue.at(p).size() > 0) {
      std::sort(
          singles_queue.at(p).begin(), singles_queue.at(p).end(), [](const Sr& a, const Sr& b) {
            return a.S > b.S;
          });
      n_entries_local[thread_id] += singles_queue.at(p).size();
      max_singles_queue_elem_local[thread_id] =
          std::max(max_singles_queue_elem_local[thread_id], singles_queue.at(p).front().S);
    }
  }
  for (int i = 0; i < n_threads; i++) {
    n_entries += n_entries_local[i];
    max_singles_queue_elem = std::max(max_singles_queue_elem, max_singles_queue_elem_local[i]);
  }

  const int proc_id = Parallel::get_proc_id();
  if (proc_id == 0) {
    printf("Max singles_queue elem: " ENERGY_FORMAT "\n", max_singles_queue_elem);
    printf("Number of entries in singles_queue: %'zu\n", n_entries);
  }
  helper_size += n_entries * 16 * 2;  // vector size <= 2 * number of elements
}

void ChemSystem::setup_hci_queue() {
  size_t n_entries = 0;
  max_hci_queue_elem = 0.0;

  const int n_threads = Parallel::get_n_threads();
  std::vector<size_t> n_entries_local(n_threads, 0);
  std::vector<double> max_hci_queue_elem_local(n_threads, 0.0);

  // Same spin.
  hci_queue.resize(Integrals::combine2(n_orbs, 2 * n_orbs));
#pragma omp parallel for schedule(dynamic, 5)
  for (unsigned p = 0; p < n_orbs; p++) {
    const int thread_id = omp_get_thread_num();
    const unsigned sym_p = orb_sym[p];
    for (unsigned q = p + 1; q < n_orbs; q++) {
      const size_t pq = Integrals::combine2(p, q);
      const unsigned sym_q = product_table.get_product(sym_p, orb_sym[q]);
      for (unsigned r = 0; r < n_orbs; r++) {
        unsigned sym_r = orb_sym[r];
        if (point_group == PointGroup::Dooh) sym_r = DoohUtil::get_inverse(sym_r);
        sym_r = product_table.get_product(sym_q, sym_r);
        if (sym_r >= sym_orbs.size()) continue;
        for (const unsigned s : sym_orbs[sym_r]) {
          if (s < r) continue;
          const double H = get_hci_queue_elem(p, q, r, s);
          if (H == 0.0) continue;
          hci_queue.at(pq).push_back(Hrs(H, r, s));
        }
      }
      if (hci_queue.at(pq).size() > 0) {
        std::sort(hci_queue.at(pq).begin(), hci_queue.at(pq).end(), [](const Hrs& a, const Hrs& b) {
          return a.H > b.H;
        });
        n_entries_local[thread_id] += hci_queue.at(pq).size();
        max_hci_queue_elem_local[thread_id] =
            std::max(max_hci_queue_elem_local[thread_id], hci_queue.at(pq).front().H);
      }
    }
  }

// Opposite spin.
#pragma omp parallel for schedule(dynamic, 5)
  for (unsigned p = 0; p < n_orbs; p++) {
    const int thread_id = omp_get_thread_num();
    const unsigned sym_p = orb_sym[p];
    for (unsigned q = n_orbs + p; q < n_orbs * 2; q++) {
      const size_t pq = Integrals::combine2(p, q);
      const unsigned sym_q = product_table.get_product(sym_p, orb_sym[q - n_orbs]);
      for (unsigned r = 0; r < n_orbs; r++) {
        unsigned sym_r = orb_sym[r];
        if (point_group == PointGroup::Dooh) sym_r = DoohUtil::get_inverse(sym_r);
        sym_r = product_table.get_product(sym_q, sym_r);
        if (sym_r >= sym_orbs.size()) continue;
        for (const unsigned s : sym_orbs[sym_r]) {
          const double H = get_hci_queue_elem(p, q, r, s + n_orbs);
          if (H == 0.0) continue;
          hci_queue.at(pq).push_back(Hrs(H, r, s + n_orbs));
        }
      }
      if (hci_queue.at(pq).size() > 0) {
        std::sort(hci_queue.at(pq).begin(), hci_queue.at(pq).end(), [](const Hrs& a, const Hrs& b) {
          return a.H > b.H;
        });
        n_entries_local[thread_id] += hci_queue.at(pq).size();
        max_hci_queue_elem_local[thread_id] =
            std::max(max_hci_queue_elem_local[thread_id], hci_queue.at(pq).front().H);
      }
    }
  }

  for (int i = 0; i < n_threads; i++) {
    n_entries += n_entries_local[i];
    max_hci_queue_elem = std::max(max_hci_queue_elem, max_hci_queue_elem_local[i]);
  }

  const int proc_id = Parallel::get_proc_id();
  if (proc_id == 0) {
    printf("Max hci queue elem: " ENERGY_FORMAT "\n", max_hci_queue_elem);
    printf("Number of entries in hci queue: %'zu\n", n_entries);
  }
  helper_size += n_entries * 16 * 2;  // vector size <= 2 * number of elements
}

PointGroup ChemSystem::get_point_group(const std::string& str) const {
  if (Util::str_equals_ci("C1", str)) {
    return PointGroup::C1;
  } else if (Util::str_equals_ci("C2", str)) {
    return PointGroup::C2;
  } else if (Util::str_equals_ci("Cs", str)) {
    return PointGroup::Cs;
  } else if (Util::str_equals_ci("Ci", str)) {
    return PointGroup::Ci;
  } else if (Util::str_equals_ci("C2v", str)) {
    return PointGroup::C2v;
  } else if (Util::str_equals_ci("C2h", str)) {
    return PointGroup::C2h;
  } else if (Util::str_equals_ci("Coov", str) || Util::str_equals_ci("Civ", str)) {
    return PointGroup::Dooh;
  } else if (Util::str_equals_ci("D2", str)) {
    return PointGroup::D2;
  } else if (Util::str_equals_ci("D2h", str)) {
    return PointGroup::D2h;
  } else if (Util::str_equals_ci("Dooh", str) || Util::str_equals_ci("Dih", str)) {
    return PointGroup::Dooh;
  }
  throw new std::runtime_error("No point group provided");
}

void ChemSystem::check_group_elements() const {
  unsigned num_group_elems =
      std::set<unsigned>(integrals.orb_sym.begin(), integrals.orb_sym.end()).size();

  if (point_group == PointGroup::C1) assert(num_group_elems == 1);
  if (point_group == PointGroup::C2) assert(num_group_elems <= 2);
  if (point_group == PointGroup::Cs) assert(num_group_elems <= 2);
  if (point_group == PointGroup::Ci) assert(num_group_elems <= 2);
  if (point_group == PointGroup::C2v) assert(num_group_elems <= 4);
  if (point_group == PointGroup::C2h) assert(num_group_elems <= 4);
  if (point_group == PointGroup::D2) assert(num_group_elems <= 4);
  if (point_group == PointGroup::D2h) assert(num_group_elems <= 8);
}

double ChemSystem::get_singles_queue_elem(const unsigned orb_i, const unsigned orb_j) const {
  if (orb_i == orb_j) return 0.0;
  std::vector<double> singles_queue_elems;
  singles_queue_elems.reserve(2*n_orbs-2);
  for (unsigned orb = 0; orb < n_orbs; orb++) {
    const double exchange = integrals.get_2b(orb_i, orb, orb, orb_j);
    const double direct = integrals.get_2b(orb_i, orb_j, orb, orb);
    if (orb == orb_i or orb == orb_j) {
      singles_queue_elems.push_back(direct); // opposite spin only
    } else {
      singles_queue_elems.push_back(direct); // opposite spin
      singles_queue_elems.push_back(direct - exchange); //same spin
    }
  }
  std::sort(singles_queue_elems.begin(), singles_queue_elems.end());
  double S_min = integrals.get_1b(orb_i, orb_j);
  double S_max = S_min;
  for (unsigned i = 0; i < std::min(n_elecs - 1, 2*n_orbs-2); i++) {
    S_min += singles_queue_elems[i];
    S_max += singles_queue_elems[2*n_orbs-3-i];
  } 
  return std::max(std::abs(S_min), std::abs(S_max));
}

double ChemSystem::get_hci_queue_elem(
    const unsigned p, const unsigned q, const unsigned r, const unsigned s) {
  if (p == q || r == s || p == r || q == s || p == s || q == r) return 0.0;
  DiffResult diff_up;
  DiffResult diff_dn;
  if (p < n_orbs && q < n_orbs) {
    assert(r < n_orbs);
    assert(s < n_orbs);
    diff_up.left_only[0] = p;
    diff_up.left_only[1] = q;
    diff_up.right_only[0] = r;
    diff_up.right_only[1] = s;
    diff_up.n_diffs = 2;
  } else if (p < n_orbs && q >= n_orbs) {
    assert(r < n_orbs);
    assert(s >= n_orbs);
    diff_up.left_only[0] = p;
    diff_dn.left_only[0] = q - n_orbs;
    diff_up.right_only[0] = r;
    diff_dn.right_only[0] = s - n_orbs;
    diff_up.n_diffs = 1;
    diff_dn.n_diffs = 1;
  } else {
    throw std::runtime_error("impossible pqrs for getting hci queue elem");
  }
  return std::abs(get_two_body_double(diff_up, diff_dn));
}

double ChemSystem::find_connected_dets(
    const Det& det,
    const double eps_max,
    const double eps_min,
    const std::function<void(const Det&, const int n_excite)>& handler,
    const bool second_rejection) const {
  if (eps_max < eps_min) return eps_min;

  auto occ_orbs_up = det.up.get_occupied_orbs();
  auto occ_orbs_dn = det.dn.get_occupied_orbs();

  double diff_from_hf = - energy_hf_1b;
  if (second_rejection) {
    // Find approximate energy difference of spawning det from HF det.
    // Later the energy difference of the spawned det from the HF det requires at most 4 1-body energies.
    // Note: Using 1-body energies as a proxy for det energies
    for (const auto p: occ_orbs_up) diff_from_hf += integrals.get_1b(p, p);
    for (const auto p: occ_orbs_dn) diff_from_hf += integrals.get_1b(p, p);
  }

  double max_rejection = 0.;

  // Filter such that S < epsilon not allowed
  if (eps_min <= max_singles_queue_elem) {
    for (unsigned p_id = 0; p_id < n_elecs; p_id++) {
      const unsigned p = p_id < n_up ? occ_orbs_up[p_id] : occ_orbs_dn[p_id - n_up];
      for (const auto& connected_sr : singles_queue.at(p)) {
        auto S = connected_sr.S;
        if (S < eps_min) break;
//      if (S >= eps_max) continue; // This line is incorrect because for single excitations we compute H_ij and have some additional rejections.
        unsigned r = connected_sr.r;
        if (second_rejection) {
          double denominator = diff_from_hf - integrals.get_1b(p, p) + integrals.get_1b(r, r);
          if (denominator > 0. && S * S / denominator < second_rejection_factor * eps_min * eps_min) {
            max_rejection = std::max(max_rejection, S);
            continue;
          }
        }
	if (enforce_active_space) {
          if (r > integrals.highest_occ_orb_in_irrep[orb_sym[r] - 1]) continue;
	}
        Det connected_det(det);
        if (p_id < n_up) {
          if (det.up.has(r)) continue;
          connected_det.up.unset(p).set(r);
          handler(connected_det, 1);
        } else {
          if (det.dn.has(r)) continue;
          connected_det.dn.unset(p).set(r);
          handler(connected_det, 1);
        }
      }
    }
  }

  // Add double excitations.
  if (!has_double_excitation) return eps_min;
  if (eps_min > max_hci_queue_elem) return eps_min;
  for (unsigned p_id = 0; p_id < n_elecs; p_id++) {
    for (unsigned q_id = p_id + 1; q_id < n_elecs; q_id++) {
      const unsigned p = p_id < n_up ? occ_orbs_up[p_id] : occ_orbs_dn[p_id - n_up] + n_orbs;
      const unsigned q = q_id < n_up ? occ_orbs_up[q_id] : occ_orbs_dn[q_id - n_up] + n_orbs;
      double p2 = p;
      double q2 = q;
      if (p >= n_orbs && q >= n_orbs) {
        p2 -= n_orbs;
        q2 -= n_orbs;
      } else if (p < n_orbs && q >= n_orbs && p > q - n_orbs) {
        p2 = q - n_orbs;
        q2 = p + n_orbs;
      }
      const unsigned pq = Integrals::combine2(p2, q2);
      for (const auto& hrs : hci_queue.at(pq)) {
        const double H = hrs.H;
        if (H < eps_min) break;
        if (H >= eps_max) continue;
        unsigned r = hrs.r;
        unsigned s = hrs.s;
        if (p >= n_orbs && q >= n_orbs) {
          r += n_orbs;
          s += n_orbs;
        } else if (p < n_orbs && q >= n_orbs && p > q - n_orbs) {
          const unsigned tmp_r = s - n_orbs;
          s = r + n_orbs;
          r = tmp_r;
        }
        if (second_rejection) {
          double denominator = diff_from_hf - integrals.get_1b(p%n_orbs, p%n_orbs) - integrals.get_1b(q%n_orbs, q%n_orbs)
                                            + integrals.get_1b(r%n_orbs, r%n_orbs) + integrals.get_1b(s%n_orbs, s%n_orbs);
          if (denominator > 0. && H * H / denominator < second_rejection_factor * eps_min * eps_min) {
            max_rejection = std::max(max_rejection, H);
            continue;
          }
        }
	if (enforce_active_space) {
          const unsigned rr = r%n_orbs, ss = s%n_orbs;
          if (rr > integrals.highest_occ_orb_in_irrep[orb_sym[rr] - 1]) continue;
          if (ss > integrals.highest_occ_orb_in_irrep[orb_sym[ss] - 1]) continue;
	}
        const bool occ_r = r < n_orbs ? det.up.has(r) : det.dn.has(r - n_orbs);
        if (occ_r) continue;
        const bool occ_s = s < n_orbs ? det.up.has(s) : det.dn.has(s - n_orbs);
        if (occ_s) continue;
        Det connected_det(det);
        p < n_orbs ? connected_det.up.unset(p) : connected_det.dn.unset(p - n_orbs);
        q < n_orbs ? connected_det.up.unset(q) : connected_det.dn.unset(q - n_orbs);
        r < n_orbs ? connected_det.up.set(r) : connected_det.dn.set(r - n_orbs);
        s < n_orbs ? connected_det.up.set(s) : connected_det.dn.set(s - n_orbs);
        handler(connected_det, 2);
      }
    }
  }
  return std::max(max_rejection, eps_min);
}

double ChemSystem::get_hamiltonian_elem(
    const Det& det_i, const Det& det_j, const int n_excite) const {
  return get_hamiltonian_elem_no_time_sym(det_i, det_j, n_excite);
}

double ChemSystem::get_hamiltonian_elem_no_time_sym(
    const Det& det_i, const Det& det_j, int n_excite) const {
  DiffResult diff_up;
  DiffResult diff_dn;
  if (n_excite < 0) {
    diff_up = det_i.up.diff(det_j.up);
    if (diff_up.n_diffs > 2) return 0.0;
    diff_dn = det_i.dn.diff(det_j.dn);
    n_excite = diff_up.n_diffs + diff_dn.n_diffs;
    if (n_excite > 2) return 0.0;
  } else if (n_excite > 0) {
    diff_up = det_i.up.diff(det_j.up);
    if (diff_up.n_diffs < static_cast<unsigned>(n_excite)) {
      diff_dn = det_i.dn.diff(det_j.dn);
    }
  }

  if (n_excite == 0) {
    const double one_body_energy = get_one_body_diag(det_i);
    const double two_body_energy = get_two_body_diag(det_i);
//  if (Parallel::is_master()) { printf("one_body_energy: " ENERGY_FORMAT "\n", one_body_energy); }
//  if (Parallel::is_master()) { printf("two_body_energy: " ENERGY_FORMAT "\n", two_body_energy); }
//  if (Parallel::is_master()) { printf("integrals.energy_core: " ENERGY_FORMAT "\n", integrals.energy_core); }
    return one_body_energy + two_body_energy + integrals.energy_core;
  } else if (n_excite == 1) {
    const double one_body_energy = get_one_body_single(diff_up, diff_dn);
    const double two_body_energy = get_two_body_single(det_i, diff_up, diff_dn);
    return one_body_energy + two_body_energy;
  } else if (n_excite == 2) {
    return get_two_body_double(diff_up, diff_dn);
  }

  throw new std::runtime_error("Calling hamiltonian with >2 exicitation");
}

double ChemSystem::get_one_body_diag(const Det& det) const {
  double energy = 0.0;
  for (const auto& orb : det.up.get_occupied_orbs()) {
    energy += integrals.get_1b(orb, orb);
  }
  if (det.up == det.dn) {
    energy *= 2;
  } else {
    for (const auto& orb : det.dn.get_occupied_orbs()) {
      energy += integrals.get_1b(orb, orb);
    }
  }
  return energy;
}

void ChemSystem::update_diag_helper() {
  const size_t n_dets = get_n_dets();
  const auto& get_two_body = [&](const HalfDet& half_det) {
    const auto& occ_orbs_up = half_det.get_occupied_orbs();
    double direct_energy = 0.0;
    double exchange_energy = 0.0;
    for (unsigned i = 0; i < occ_orbs_up.size(); i++) {
      const unsigned orb_i = occ_orbs_up[i];
      for (unsigned j = i + 1; j < occ_orbs_up.size(); j++) {
        const unsigned orb_j = occ_orbs_up[j];
        direct_energy += integrals.get_2b(orb_i, orb_i, orb_j, orb_j);
        exchange_energy -= integrals.get_2b(orb_i, orb_j, orb_j, orb_i);
      }
    }
    return direct_energy + exchange_energy;
  };

  fgpl::ConcurrentHashMap<HalfDet, double, HalfDetHasher> parallel_helper;
#pragma omp parallel for schedule(dynamic, 5)
  for (size_t i = 0; i < n_dets; i++) {
    const Det& det = dets[i];
    if (!parallel_helper.has(det.up)) {
      const double two_body = get_two_body(det.up);
      parallel_helper.set(det.up, two_body);
    }
    if (!parallel_helper.has(det.dn)) {
      const double two_body = get_two_body(det.dn);
      parallel_helper.set(det.dn, two_body);
    }
  }

  parallel_helper.for_each_serial([&](const HalfDet& half_det, const size_t, const double value) {
    diag_helper.set(half_det, value);
  });
}

double ChemSystem::get_two_body_diag(const Det& det) const {
  const auto& occ_orbs_up = det.up.get_occupied_orbs();
  const auto& occ_orbs_dn = det.dn.get_occupied_orbs();
  double direct_energy = 0.0;
  double exchange_energy = 0.0;
  // up to up.
  if (diag_helper.has(det.up)) {
    direct_energy += diag_helper.get(det.up);
  } else {
    for (unsigned i = 0; i < occ_orbs_up.size(); i++) {
      const unsigned orb_i = occ_orbs_up[i];
      for (unsigned j = i + 1; j < occ_orbs_up.size(); j++) {
        const unsigned orb_j = occ_orbs_up[j];
        direct_energy += integrals.get_2b(orb_i, orb_i, orb_j, orb_j);
        exchange_energy -= integrals.get_2b(orb_i, orb_j, orb_j, orb_i);
      }
    }
  }
  if (det.up == det.dn) {
    direct_energy *= 2;
    exchange_energy *= 2;
  } else {
    // dn to dn.
    if (diag_helper.has(det.dn)) {
      direct_energy += diag_helper.get(det.dn);
    } else {
      for (unsigned i = 0; i < occ_orbs_dn.size(); i++) {
        const unsigned orb_i = occ_orbs_dn[i];
        for (unsigned j = i + 1; j < occ_orbs_dn.size(); j++) {
          const unsigned orb_j = occ_orbs_dn[j];
          direct_energy += integrals.get_2b(orb_i, orb_i, orb_j, orb_j);
          exchange_energy -= integrals.get_2b(orb_i, orb_j, orb_j, orb_i);
        }
      }
    }
  }
  // up to dn.
  for (unsigned i = 0; i < occ_orbs_up.size(); i++) {
    const unsigned orb_i = occ_orbs_up[i];
    for (unsigned j = 0; j < occ_orbs_dn.size(); j++) {
      const unsigned orb_j = occ_orbs_dn[j];
      direct_energy += integrals.get_2b(orb_i, orb_i, orb_j, orb_j);
    }
  }
//if (Parallel::is_master()) { printf("direct_energy: " ENERGY_FORMAT "\n", direct_energy); }
//if (Parallel::is_master()) { printf("exchange_energy: " ENERGY_FORMAT "\n", exchange_energy); }
  return direct_energy + exchange_energy;
}

double ChemSystem::get_one_body_single(const DiffResult& diff_up, const DiffResult& diff_dn) const {
  const bool is_up_single = diff_up.n_diffs == 1;
  const auto& diff = is_up_single ? diff_up : diff_dn;
  const unsigned orb_i = diff.left_only[0];
  const unsigned orb_j = diff.right_only[0];
  return diff.permutation_factor * integrals.get_1b(orb_i, orb_j);
}

double ChemSystem::get_two_body_single(
    const Det& det_i, const DiffResult& diff_up, const DiffResult& diff_dn) const {
  const bool is_up_single = diff_up.n_diffs == 1;
  const auto& diff = is_up_single ? diff_up : diff_dn;
  const unsigned orb_i = diff.left_only[0];
  const unsigned orb_j = diff.right_only[0];
  const auto& same_spin_half_det = is_up_single ? det_i.up : det_i.dn;
  auto oppo_spin_half_det = is_up_single ? det_i.dn : det_i.up;
  double energy = 0.0;
  for (const unsigned orb : same_spin_half_det.get_occupied_orbs()) {
    if (orb == orb_i || orb == orb_j) continue;
    energy -= integrals.get_2b(orb_i, orb, orb, orb_j);  // Exchange.
    const double direct = integrals.get_2b(orb_i, orb_j, orb, orb);  // Direct.
    if (oppo_spin_half_det.has(orb)) {
      oppo_spin_half_det.unset(orb);
      energy += 2 * direct;
    } else {
      energy += direct;
    }
  }
  for (const unsigned orb : oppo_spin_half_det.get_occupied_orbs()) {
    energy += integrals.get_2b(orb_i, orb_j, orb, orb);  // Direct.
  }
  energy *= diff.permutation_factor;
  return energy;
}

double ChemSystem::get_two_body_double(const DiffResult& diff_up, const DiffResult& diff_dn) const {
  double energy = 0.0;
  if (diff_up.n_diffs == 0) {
    if (diff_dn.n_diffs != 2) return 0.0;
    const unsigned orb_i1 = diff_dn.left_only[0];
    const unsigned orb_i2 = diff_dn.left_only[1];
    const unsigned orb_j1 = diff_dn.right_only[0];
    const unsigned orb_j2 = diff_dn.right_only[1];
    energy = integrals.get_2b(orb_i1, orb_j1, orb_i2, orb_j2) -
             integrals.get_2b(orb_i1, orb_j2, orb_i2, orb_j1);
    energy *= diff_dn.permutation_factor;
  } else if (diff_dn.n_diffs == 0) {
    if (diff_up.n_diffs != 2) return 0.0;
    const unsigned orb_i1 = diff_up.left_only[0];
    const unsigned orb_i2 = diff_up.left_only[1];
    const unsigned orb_j1 = diff_up.right_only[0];
    const unsigned orb_j2 = diff_up.right_only[1];
    energy = integrals.get_2b(orb_i1, orb_j1, orb_i2, orb_j2) -
             integrals.get_2b(orb_i1, orb_j2, orb_i2, orb_j1);
    energy *= diff_up.permutation_factor;
  } else {
    if (diff_up.n_diffs != 1 || diff_dn.n_diffs != 1) return 0.0;
    const unsigned orb_i1 = diff_up.left_only[0];
    const unsigned orb_i2 = diff_dn.left_only[0];
    const unsigned orb_j1 = diff_up.right_only[0];
    const unsigned orb_j2 = diff_dn.right_only[0];
    energy = integrals.get_2b(orb_i1, orb_j1, orb_i2, orb_j2);
    energy *= diff_up.permutation_factor * diff_dn.permutation_factor;
  }
  return energy;
}

void ChemSystem::post_variation(std::vector<std::vector<size_t>>& connections) {
  if (Config::get<bool>("get_1rdm_csv", false)) {
    RDM rdm(integrals, dets, coefs);
    rdm.get_1rdm();
    rdm.dump_1rdm();
  }

  if (Config::get<bool>("2rdm", false) || Config::get<bool>("get_2rdm_csv", false)) {
    RDM rdm(integrals, dets, coefs);
    rdm.get_2rdm(connections);
    connections.clear();
    rdm.dump_2rdm(Config::get<bool>("get_2rdm_csv", false));
  }

  bool unpacked = false;

  if (Config::get<bool>("s2", false)) {
    if (time_sym && !unpacked) {
      unpack_time_sym();
      unpacked = true;
    }
    const double s2 = get_s2(coefs[0]);
    Result::put("s2", s2);
  }
  
  // Recalculate orbital partitioning with full variational wavefunction
  if (use_orbital_partitioning) {
    recalculate_orbital_partitioning();
  }
  
  // Analyze orbital partitioning screening effectiveness
  analyze_screening_effectiveness();
}

void ChemSystem::post_variation_optimization(
    SparseMatrix& hamiltonian_matrix,
    const std::string& method) {

  if (method == "natorb") {  // natorb optimization
    hamiltonian_matrix.clear();
    Optimization natorb_optimizer(integrals, hamiltonian_matrix, dets, coefs);

    Timer::start("Natorb optimization");
    natorb_optimizer.get_natorb_rotation_matrix();
    Timer::end();

    variation_cleanup();

    rotation_matrix *= natorb_optimizer.rotation_matrix();
    natorb_optimizer.rotate_and_rewrite_integrals();

  } else {  // optorb optimization
    Optimization optorb_optimizer(integrals, hamiltonian_matrix, dets, coefs);

    if (Util::str_equals_ci("newton", method)) {
      Timer::start("Newton optimization");
      optorb_optimizer.get_optorb_rotation_matrix_from_newton();
    } else if (Util::str_equals_ci("grad_descent", method)) {
      Timer::start("Gradient descent optimization");
      optorb_optimizer.get_optorb_rotation_matrix_from_grad_descent();
    } else if (Util::str_equals_ci("amsgrad", method)) {
      Timer::start("AMSGrad optimization");
      optorb_optimizer.get_optorb_rotation_matrix_from_amsgrad();
    } else if (Util::str_equals_ci("bfgs", method)) {
      Timer::start("BFGS optimization");
      optorb_optimizer.generate_optorb_integrals_from_bfgs();
    } else {
      Timer::start("Approximate Newton optimization");
      optorb_optimizer.get_optorb_rotation_matrix_from_approximate_newton();
    }
    Timer::end();
   
    hamiltonian_matrix.clear(); 
    variation_cleanup();

    rotation_matrix *= optorb_optimizer.rotation_matrix();
    optorb_optimizer.rotate_and_rewrite_integrals();
  }
}

void ChemSystem::optimization_microiteration(
    SparseMatrix& hamiltonian_matrix,
    const std::string& method) {
  Optimization optorb_optimizer(integrals, hamiltonian_matrix, dets, coefs);

  if (Util::str_equals_ci("newton", method)) {
    Timer::start("Newton microiteration");
    optorb_optimizer.get_optorb_rotation_matrix_from_newton();
  } else if (Util::str_equals_ci("grad_descent", method)) {
    Timer::start("Gradient descent microiteration");
    optorb_optimizer.get_optorb_rotation_matrix_from_grad_descent();
  } else if (Util::str_equals_ci("amsgrad", method)) {
    Timer::start("AMSGrad microiteration");
    optorb_optimizer.get_optorb_rotation_matrix_from_amsgrad();
  } else if (Util::str_equals_ci("bfgs", method)) {
    Timer::start("BFGS microiteration");
    optorb_optimizer.generate_optorb_integrals_from_bfgs();
  } else {
    Timer::start("Approximate Newton microiteration");
    optorb_optimizer.get_optorb_rotation_matrix_from_approximate_newton();
  }
  Timer::end();

  rotation_matrix *= optorb_optimizer.rotation_matrix();
  optorb_optimizer.rotate_and_rewrite_integrals();
}

void ChemSystem::variation_cleanup() {
  energy_hf = 0.;
  energy_var = std::vector<double>(n_states, 0.);
  helper_size = 0;
  dets.clear();
  dets.shrink_to_fit();
  for (auto& state: coefs) {
    state.clear();
    state.shrink_to_fit();
  }
  max_hci_queue_elem = 0.;
  max_singles_queue_elem = 0.;
  hci_queue.clear();
  singles_queue.clear();
}

void ChemSystem::dump_integrals(const char* filename) {
  integrals.dump_integrals(filename);
  if (Config::get<bool>("optimization/rotation_matrix", false)) {
    FILE *fp;
    fp = fopen("rotation_matrix", "w");
    const size_t n_rotation_params = rotation_matrix.rows();
    for (int i = 0; i < n_rotation_params; i++) {
      for (int j = 0; j < n_rotation_params; j++) {
        const int I = integrals.orb_order[i];
        const int J = integrals.orb_order[j];
        fprintf(fp, "%.10E ", rotation_matrix(I, J));
      }
      fprintf(fp, "\n");
    }
    fclose(fp);
  }
}

//======================================================
double ChemSystem::get_s2(std::vector<double> state_coefs) const {
  // Calculates <S^2> of the variation wf.
  // s^2 = n_up -n_doub - 1/2*(n_up-n_dn) + 1/4*(n_up - n_dn)^2
  //  - sum_{p != q} c_{q,dn}^{+} c_{p,dn} c_{p,up}^{+} c_{q,up}
  //
  // Created: Y. Yao, May 2018
  //======================================================
  double s2 = 0.;

  // Create hash table; used for looking up the coef of a det
  std::unordered_map<Det, double, DetHasher> det2coef;
  for (size_t i = 0; i < dets.size(); i++) {
    det2coef[dets[i]] = state_coefs[i];
  }

#pragma omp parallel for reduction(+ : s2)
  for (size_t i_det = 0; i_det < dets.size(); i_det++) {
    Det this_det = dets[i_det];

    const auto& occ_orbs = this_det.up.get_occupied_orbs();
    unsigned num_db_occ = 0;  // number of doubly occupied orbs
    for (unsigned i = 0; i < occ_orbs.size(); i++) {
      if (this_det.dn.has(occ_orbs[i])) num_db_occ++;
    }

    // diagonal terms
    double diag = 0.5 * n_up - num_db_occ + 0.5 * n_dn;
    diag += 0.25 * (pow(n_up, 2) + pow(n_dn, 2)) - 0.5 * n_up * n_dn;
    diag *= pow(state_coefs[i_det], 2);
    s2 += diag;

    // double excitations
    for (unsigned i_orb = 0; i_orb < n_orbs; i_orb++) {
      if (!this_det.dn.has(i_orb)) continue;
      if (this_det.up.has(i_orb)) continue;

      for (unsigned j_orb = i_orb + 1; j_orb < n_orbs; j_orb++) {
        if (!this_det.up.has(j_orb)) continue;
        if (this_det.dn.has(j_orb)) continue;

        Det new_det = this_det;
        new_det.up.unset(j_orb);
        new_det.up.set(i_orb);
        new_det.dn.unset(i_orb);
        new_det.dn.set(j_orb);

        double coef;
        if (det2coef.count(new_det) == 1) {
          coef = det2coef[new_det];
        } else {
          continue;
        }

        const double perm_up = this_det.up.diff(new_det.up).permutation_factor;
        const double perm_dn = this_det.dn.diff(new_det.dn).permutation_factor;
        double off_diag = -2 * coef * state_coefs[i_det] * perm_up * perm_dn;
        s2 += off_diag;
      }  // j_orb
    }  // i_orb
  }  // i_det

  if (Parallel::is_master()) {
    printf("s_squared: %15.10f\n", s2);
  }
  return s2;
}

double ChemSystem::get_e_hf_1b() const {
  double e_hf_1b = 0.;
  auto occ_orbs_up = dets[0].up.get_occupied_orbs();
  for (const auto p : occ_orbs_up) e_hf_1b += integrals.get_1b(p, p);
  if (Config::get<bool>("time_sym", false)) {
    e_hf_1b *= 2.;
  } else {
    auto occ_orbs_dn = dets[0].dn.get_occupied_orbs();
    for (const auto p : occ_orbs_dn) e_hf_1b += integrals.get_1b(p, p);
  }
  return e_hf_1b;
}

void ChemSystem::setup_orbital_partitioning() {
  if (Parallel::is_master()) {
    printf("Setting up orbital partitioning for same-spin screening...\n");
  }
  
  // Initialize with simple ordering initially - will recalculate after variation
  sorted_beta_orbitals.clear();
  for (unsigned orb = 0; orb < n_orbs; orb++) {
    sorted_beta_orbitals.push_back(orb);
  }
  
  // Initialize 5 hash maps for the screening
  same_spin_screener.clear();
  same_spin_screener.resize(5);
  
  if (Parallel::is_master()) {
    printf("Orbital partitioning setup complete. %d orbitals in 5 groups.\n", n_orbs);
    printf("Note: Orbital ordering will be optimized after variational calculation.\n\n");
  }
}

void ChemSystem::recalculate_orbital_partitioning() {
  if (!use_orbital_partitioning || dets.empty()) return;
  
  if (Parallel::is_master()) {
    printf("Recalculating orbital partitioning with full variational wavefunction...\n");
  }
  
  // Calculate true occupation probabilities from all determinants
  std::vector<double> orbital_occupations(n_orbs, 0.0);
  double total_coef_squared = 0.0;
  
  // Sum over all determinants weighted by coefficient squared
  for (size_t det_idx = 0; det_idx < dets.size(); det_idx++) {
    double coef_sq = coefs[0][det_idx] * coefs[0][det_idx];  // Use first state
    total_coef_squared += coef_sq;
    
    // Count beta orbital occupations
    for (unsigned orb = 0; orb < n_orbs; orb++) {
      if (dets[det_idx].dn.has(orb)) {
        orbital_occupations[orb] += coef_sq;
      }
    }
  }
  
  // Normalize to get probabilities
  for (unsigned orb = 0; orb < n_orbs; orb++) {
    orbital_occupations[orb] /= total_coef_squared;
  }
  
  // Create pairs of |p_i - 0.5| and orbital index for sorting
  std::vector<std::pair<double, unsigned>> orbital_probs;
  for (unsigned orb = 0; orb < n_orbs; orb++) {
    double p_i = orbital_occupations[orb];
    orbital_probs.push_back({std::abs(p_i - 0.5), orb});
  }
  
  // Sort by increasing |p_i - 0.5| to prioritize most delocalized orbitals
  std::sort(orbital_probs.begin(), orbital_probs.end());
  
  // Update sorted orbital indices
  sorted_beta_orbitals.clear();
  for (const auto& pair : orbital_probs) {
    sorted_beta_orbitals.push_back(pair.second);
  }
  
  if (Parallel::is_master()) {
    printf("\n=== ORBITAL PARTITIONING DISTRIBUTION (from %zu determinants) ===\n", dets.size());
    printf("Orbitals sorted by increasing |p_i - 0.5| (most delocalized first):\n");
    
    std::vector<std::vector<unsigned>> groups(5);
    for (int orb_idx = 0; orb_idx < static_cast<int>(n_orbs); orb_idx++) {
      unsigned orb = sorted_beta_orbitals[orb_idx];
      double p_i = orbital_occupations[orb];
      
      // Determine group assignment using serpentine pattern
      int assigned_group;
      int cycle_pos = orb_idx % 10;
      if (cycle_pos < 5) {
        assigned_group = cycle_pos;
      } else {
        assigned_group = 9 - cycle_pos;
      }
      
      groups[assigned_group].push_back(orb);
      printf("Orbital %2d: p_i = %.4f, |p_i - 0.5| = %.4f, assigned to group %d\n", 
             orb, p_i, std::abs(p_i - 0.5), assigned_group);
    }
    
    printf("\nGroup distributions:\n");
    for (int g = 0; g < 5; g++) {
      printf("Group %d (%zu orbitals): ", g, groups[g].size());
      for (unsigned orb : groups[g]) {
        printf("%d(%.3f) ", orb, orbital_occupations[orb]);
      }
      printf("\n");
    }
    printf("===============================================\n\n");
  }
  
  // Clear and rebuild hash maps with new orbital ordering
  for (int group = 0; group < 5; group++) {
    same_spin_screener[group].clear();
  }
  
  // Repopulate screener with all determinants using new orbital ordering
  for (size_t det_idx = 0; det_idx < dets.size(); det_idx++) {
    populate_screener(dets[det_idx], static_cast<int>(det_idx));
  }
}

void ChemSystem::populate_screener(const Det& det, int det_id) {
  if (!use_orbital_partitioning) return;
  
  // Compute occupation keys for all 5 groups and add det_id to appropriate buckets
  for (int group = 0; group < 5; group++) {
    uint64_t key = compute_occupation_key(det, group);
    same_spin_screener[group][key].push_back(det_id);
  }
}

uint64_t ChemSystem::compute_occupation_key(const Det& det, int group_id) const {
  uint64_t key = 0;
  
  // Serpentine round-robin dealing: 1,2,3,4,5,5,4,3,2,1,...
  int orb_idx = 0;
  for (unsigned orb : sorted_beta_orbitals) {
    int assigned_group;
    int cycle_pos = orb_idx % 10;  // 10-element cycle: 0,1,2,3,4,4,3,2,1,0
    if (cycle_pos < 5) {
      assigned_group = cycle_pos;
    } else {
      assigned_group = 9 - cycle_pos;  // 4,3,2,1,0
    }
    
    if (assigned_group == group_id) {
      // Add this orbital's occupation to the key
      if (det.dn.has(orb)) {
        key |= (1ULL << (orb % 64));  // Set bit for occupied orbital
      }
    }
    orb_idx++;
  }
  
  return key;
}

void ChemSystem::analyze_screening_effectiveness() const {
  if (!use_orbital_partitioning) {
    if (Parallel::is_master()) {
      printf("Orbital partitioning is disabled - no analysis available.\n");
    }
    return;
  }
  
  if (Parallel::is_master()) {
    printf("\n=== ORBITAL PARTITIONING SCREENING ANALYSIS ===\n");
    printf("Total determinants: %zu\n", dets.size());
    printf("Partitioning threshold: %d\n", partitioning_threshold);
    
    // Group determinants by alpha string
    std::unordered_map<HalfDet, std::vector<int>, HalfDetHasher> alpha_to_det_ids;
    for (size_t i = 0; i < dets.size(); i++) {
      alpha_to_det_ids[dets[i].up].push_back(i);
    }
    
    printf("Number of unique alpha strings: %zu\n", alpha_to_det_ids.size());
    
    int large_alpha_groups = 0;
    int max_alpha_group_size = 0;
    int total_screening_benefit = 0;
    int max_screening_compression = 0;
    
    for (const auto& entry : alpha_to_det_ids) {
      const auto& beta_det_ids = entry.second;
      int alpha_group_size = beta_det_ids.size();
      max_alpha_group_size = std::max(max_alpha_group_size, alpha_group_size);
      
      if (alpha_group_size >= partitioning_threshold) {
        large_alpha_groups++;
        
        // Analyze screening compression for this alpha group
        std::vector<std::unordered_map<uint64_t, int>> group_key_counts(5);
        
        // Count determinants per screening key for each group
        for (int det_id : beta_det_ids) {
          const Det& det = dets[det_id];
          for (int group = 0; group < 5; group++) {
            uint64_t key = compute_occupation_key(det, group);
            group_key_counts[group][key]++;
          }
        }
        
        // Calculate screening effectiveness: find unique pairs across all groups
        std::set<std::pair<int, int>> candidate_pairs;
        for (int group = 0; group < 5; group++) {
          for (const auto& bucket : group_key_counts[group]) {
            const auto& det_ids_in_bucket = [&]() {
              std::vector<int> ids_in_bucket;
              for (int det_id : beta_det_ids) {
                if (compute_occupation_key(dets[det_id], group) == bucket.first) {
                  ids_in_bucket.push_back(det_id);
                }
              }
              return ids_in_bucket;
            }();
            
            // Add all unique pairs within this bucket
            for (size_t i = 0; i < det_ids_in_bucket.size(); i++) {
              for (size_t j = i + 1; j < det_ids_in_bucket.size(); j++) {
                int det1 = det_ids_in_bucket[i];
                int det2 = det_ids_in_bucket[j];
                candidate_pairs.insert({std::min(det1, det2), std::max(det1, det2)});
              }
            }
          }
        }
        
        int screened_pair_count = candidate_pairs.size();
        
        // Original: M*(M-1)/2 pairs, Screened: actual unique pairs found
        int original_pairs = alpha_group_size * (alpha_group_size - 1) / 2;
        int benefit = original_pairs - screened_pair_count;
        
        total_screening_benefit += benefit;
        max_screening_compression = std::max(max_screening_compression, benefit);
        
        printf("Alpha group size: %d, Screened pairs: %d, Benefit: %d pairs avoided (%.1f%% reduction)\n", 
               alpha_group_size, screened_pair_count, benefit, 
               100.0 * benefit / original_pairs);
      }
    }
    
    printf("\nSCREENING STATISTICS:\n");
    printf("Alpha groups >= threshold: %d\n", large_alpha_groups);
    printf("Max alpha group size: %d\n", max_alpha_group_size);
    printf("Total screening benefit: %d pair checks avoided\n", total_screening_benefit);
    printf("Max single group compression: %d pairs\n", max_screening_compression);
    
    // Analysis of screening hash map utilization
    printf("\nSCREENING HASH MAP STATISTICS:\n");
    for (int group = 0; group < 5; group++) {
      printf("Group %d: %zu unique keys, %zu total entries\n", 
             group, same_spin_screener[group].size(),
             [&]() {
               size_t total = 0;
               for (const auto& bucket : same_spin_screener[group]) {
                 total += bucket.second.size();
               }
               return total;
             }());
    }
    
    printf("===============================================\n\n");
  }
}
