#pragma once

#include <chrono>
#include <string>
#include <vector>

// Hierarchical timing system for SHCI performance profiling
// Provides nested timing with memory usage tracking
class Timer {
 public:
  // Get singleton timer instance
  static Timer& get_instance() {
    static Timer instance;
    return instance;
  }

  // Start timing a named event (can be nested)
  static void start(const std::string& event);

  // Record checkpoint within current timing event
  static void checkpoint(const std::string& event);

  // End current timing event and print results
  static void end();

 private:
  Timer();

  void print_status() const;

  void print_mem() const;

  void print_time() const;

  double get_duration(
      const std::chrono::high_resolution_clock::time_point start,
      const std::chrono::high_resolution_clock::time_point end) const;

  std::chrono::high_resolution_clock::time_point init_time;

  std::chrono::high_resolution_clock::time_point prev_time;

  std::vector<std::pair<std::string, std::chrono::high_resolution_clock::time_point>> start_times;

  size_t init_mem;

  bool is_master;
};
