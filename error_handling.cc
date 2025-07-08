/**
 * @file error_handling.cc
 * @brief Implementation of enhanced error handling system
 */

#include "error_handling.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <sstream>
#include <algorithm>
#include <ctime>
#include <sys/utsname.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// MPI stubs for compilation without MPI
#ifndef MPI_SUCCESS
#define MPI_SUCCESS 0
#define MPI_MAX_PROCESSOR_NAME 256
inline int MPI_Get_processor_name(char* name, int* len) { 
  strcpy(name, "localhost"); 
  *len = 9; 
  return MPI_SUCCESS; 
}
inline int MPI_Comm_rank(int comm, int* rank) { *rank = 0; return MPI_SUCCESS; }
inline int MPI_Comm_size(int comm, int* size) { *size = 1; return MPI_SUCCESS; }
inline int MPI_Initialized(int* flag) { *flag = 0; return MPI_SUCCESS; }
#define MPI_COMM_WORLD 0
#endif

namespace error {

void SHCIException::format_message() {
  std::ostringstream oss;
  
  // Add severity and category header
  oss << "[" << ErrorHandler::instance().severity_to_string(severity_) 
      << "/" << ErrorHandler::instance().category_to_string(category_) << "] ";
  
  // Add main message
  oss << message_;
  
  // Add context information if available
  if (!context_.function_name.empty()) {
    oss << "\n  Location: " << context_.function_name;
    if (!context_.file_name.empty()) {
      // Extract just the filename from full path
      size_t pos = context_.file_name.find_last_of("/\\");
      std::string filename = (pos != std::string::npos) ? 
                            context_.file_name.substr(pos + 1) : context_.file_name;
      oss << " (" << filename;
      if (context_.line_number > 0) {
        oss << ":" << context_.line_number;
      }
      oss << ")";
    }
  }
  
  // Add variable information
  if (!context_.variables.empty()) {
    oss << "\n  Context variables:";
    for (const auto& var : context_.variables) {
      oss << "\n    " << var.first << " = " << var.second;
    }
  }
  
  // Add call stack if available
  if (!context_.call_stack.empty()) {
    oss << "\n  Call stack:";
    for (size_t i = 0; i < context_.call_stack.size(); ++i) {
      oss << "\n    " << i << ": " << context_.call_stack[i];
    }
  }
  
  // Add diagnostic suggestions
  auto suggestions = ErrorHandler::instance().get_suggestions(category_, message_);
  if (!suggestions.empty()) {
    oss << "\n  Suggestions:";
    for (const auto& suggestion : suggestions) {
      oss << "\n    • " << suggestion;
    }
  }
  
  formatted_message_ = oss.str();
}

void ErrorHandler::log(ErrorSeverity severity, ErrorCategory category,
                      const std::string& message, const ErrorContext& context) {
  // Check if we should log this severity level
  if (severity < min_log_level_) {
    return;
  }
  
  // Create timestamp
  auto now = std::chrono::system_clock::now();
  auto time_t = std::chrono::system_clock::to_time_t(now);
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
              now.time_since_epoch()) % 1000;
  
  std::ostringstream log_message;
  log_message << "[" << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
  log_message << "." << std::setfill('0') << std::setw(3) << ms.count() << "] ";
  
  // Add severity and category
  log_message << "[" << severity_to_string(severity) 
              << "/" << category_to_string(category) << "] ";
  
  // Add message
  log_message << message;
  
  // Add context if available
  if (!context.function_name.empty()) {
    log_message << " [" << context.function_name << "]";
  }
  
  std::string final_message = log_message.str();
  
  // Output to console if enabled
  if (console_output_) {
    if (severity >= ErrorSeverity::ERROR) {
      std::cerr << final_message << std::endl;
    } else {
      std::cout << final_message << std::endl;
    }
  }
  
  // Output to log file if configured
  if (!log_filename_.empty()) {
    std::ofstream log_file(log_filename_, std::ios::app);
    if (log_file.is_open()) {
      log_file << final_message << std::endl;
    }
  }
}

void ErrorHandler::set_log_file(const std::string& filename) {
  log_filename_ = filename;
  
  // Create initial log entry
  std::ofstream log_file(filename, std::ios::app);
  if (log_file.is_open()) {
    log_file << "\n=== SHCI Error Log Started ===\n";
    log_file << "System Info: " << get_system_info() << "\n";
    log_file << "==============================\n\n";
  }
}

std::vector<std::string> ErrorHandler::get_suggestions(ErrorCategory category, 
                                                       const std::string& message) {
  std::vector<std::string> suggestions;
  
  // Get category-specific suggestions
  auto it = error_suggestions_.find(category);
  if (it != error_suggestions_.end()) {
    suggestions = it->second;
  }
  
  // Add message-specific suggestions based on keywords
  std::string lower_message = message;
  std::transform(lower_message.begin(), lower_message.end(), 
                lower_message.begin(), ::tolower);
  
  if (lower_message.find("memory") != std::string::npos) {
    suggestions.push_back("Try increasing the number of MPI processes to distribute memory");
    suggestions.push_back("Reduce eps_vars values to use fewer determinants");
    suggestions.push_back("Enable memory compression in configuration");
  }
  
  if (lower_message.find("convergence") != std::string::npos) {
    suggestions.push_back("Try increasing the target_error tolerance");
    suggestions.push_back("Check for linear dependencies in the basis set");
    suggestions.push_back("Increase max_iterations in Davidson configuration");
  }
  
  if (lower_message.find("file") != std::string::npos || 
      lower_message.find("fcidump") != std::string::npos) {
    suggestions.push_back("Check that the file exists and has correct permissions");
    suggestions.push_back("Verify the file format matches FCIDUMP specification");
    suggestions.push_back("Ensure the file is not corrupted or truncated");
  }
  
  if (lower_message.find("mpi") != std::string::npos) {
    suggestions.push_back("Check MPI installation and network connectivity");
    suggestions.push_back("Try running with fewer MPI processes");
    suggestions.push_back("Verify that all nodes have access to input files");
  }
  
  // Remove duplicates
  std::sort(suggestions.begin(), suggestions.end());
  suggestions.erase(std::unique(suggestions.begin(), suggestions.end()), suggestions.end());
  
  return suggestions;
}

std::string ErrorHandler::get_system_info() {
  std::ostringstream info;
  
  // Operating system information
  struct utsname sys_info;
  if (uname(&sys_info) == 0) {
    info << "OS: " << sys_info.sysname << " " << sys_info.release 
         << " (" << sys_info.machine << ")";
  }
  
  // Compiler information
  info << ", Compiler: ";
#ifdef __GNUC__
  info << "GCC " << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__;
#elif defined(__clang__)
  info << "Clang " << __clang_major__ << "." << __clang_minor__ << "." << __clang_patchlevel__;
#else
  info << "Unknown";
#endif

  // OpenMP information
#ifdef _OPENMP
  info << ", OpenMP: " << _OPENMP;
  info << " (threads: " << omp_get_max_threads() << ")";
#else
  info << ", OpenMP: Not available";
#endif

  // MPI information (basic check)
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  int rank, size;
  
  if (MPI_Get_processor_name(processor_name, &name_len) == MPI_SUCCESS) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    info << ", MPI: " << size << " processes";
  }
  
  return info.str();
}

void ErrorHandler::validate_system_requirements() {
  std::vector<std::string> warnings;
  
  // Check available memory
  long pages = sysconf(_SC_PHYS_PAGES);
  long page_size = sysconf(_SC_PAGE_SIZE);
  if (pages > 0 && page_size > 0) {
    double total_memory_gb = (double)(pages * page_size) / (1024.0 * 1024.0 * 1024.0);
    if (total_memory_gb < 2.0) {
      warnings.push_back("Low system memory detected (" + 
                        std::to_string(total_memory_gb) + " GB). Consider using more nodes.");
    }
  }
  
  // Check OpenMP availability
#ifndef _OPENMP
  warnings.push_back("OpenMP not available. Performance may be reduced.");
#endif

  // Check MPI setup
  int mpi_initialized;
  MPI_Initialized(&mpi_initialized);
  if (!mpi_initialized) {
    warnings.push_back("MPI not initialized. Parallel execution may not work correctly.");
  }
  
  // Log warnings
  for (const auto& warning : warnings) {
    log(ErrorSeverity::WARNING, ErrorCategory::SYSTEM, warning);
  }
}

void ErrorHandler::initialize_suggestions() {
  // Configuration suggestions
  error_suggestions_[ErrorCategory::CONFIGURATION] = {
    "Check JSON syntax and formatting",
    "Verify all required parameters are present",
    "Ensure parameter values are within valid ranges",
    "Check parameter types (string, number, boolean, array)",
    "Review configuration documentation for valid options"
  };
  
  // File I/O suggestions
  error_suggestions_[ErrorCategory::FILE_IO] = {
    "Check file permissions and accessibility",
    "Verify file path is correct and complete",
    "Ensure sufficient disk space is available",
    "Check if file is being used by another process",
    "Verify file format and structure"
  };
  
  // Memory suggestions
  error_suggestions_[ErrorCategory::MEMORY] = {
    "Increase the number of MPI processes to distribute memory load",
    "Reduce variational space size (increase eps_vars values)",
    "Enable memory compression options",
    "Use more nodes with distributed memory",
    "Monitor memory usage during calculation"
  };
  
  // Convergence suggestions
  error_suggestions_[ErrorCategory::CONVERGENCE] = {
    "Increase convergence tolerance (target_error)",
    "Check for numerical instabilities in the system",
    "Try different initial guess vectors",
    "Increase maximum number of iterations",
    "Check for near-linear dependencies in basis"
  };
  
  // MPI suggestions
  error_suggestions_[ErrorCategory::MPI] = {
    "Check MPI installation and configuration",
    "Verify network connectivity between nodes",
    "Ensure all processes have access to input files",
    "Check MPI process binding and affinity",
    "Monitor for deadlocks or communication failures"
  };
  
  // Mathematical suggestions
  error_suggestions_[ErrorCategory::MATHEMATICAL] = {
    "Check input data for numerical errors",
    "Verify calculation parameters are reasonable",
    "Look for overflow or underflow conditions",
    "Check matrix conditioning and stability",
    "Ensure algorithm assumptions are met"
  };
  
  // User input suggestions
  error_suggestions_[ErrorCategory::USER_INPUT] = {
    "Review input parameter documentation",
    "Check parameter ranges and constraints",
    "Verify file formats and structure",
    "Ensure electron count matches system",
    "Check orbital and symmetry specifications"
  };
}

std::string ErrorHandler::severity_to_string(ErrorSeverity severity) {
  switch (severity) {
    case ErrorSeverity::DEBUG:   return "DEBUG";
    case ErrorSeverity::INFO:    return "INFO";
    case ErrorSeverity::WARNING: return "WARN";
    case ErrorSeverity::ERROR:   return "ERROR";
    case ErrorSeverity::FATAL:   return "FATAL";
    default: return "UNKNOWN";
  }
}

std::string ErrorHandler::category_to_string(ErrorCategory category) {
  switch (category) {
    case ErrorCategory::CONFIGURATION: return "CONFIG";
    case ErrorCategory::FILE_IO:       return "FILE_IO";
    case ErrorCategory::MEMORY:        return "MEMORY";
    case ErrorCategory::CONVERGENCE:   return "CONVERGENCE";
    case ErrorCategory::MPI:           return "MPI";
    case ErrorCategory::MATHEMATICAL:  return "MATH";
    case ErrorCategory::SYSTEM:        return "SYSTEM";
    case ErrorCategory::USER_INPUT:    return "INPUT";
    case ErrorCategory::INTERNAL:      return "INTERNAL";
    default: return "UNKNOWN";
  }
}

} // namespace error