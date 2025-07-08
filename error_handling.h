/**
 * @file error_handling.h
 * @brief Enhanced error handling system with diagnostic messages
 * 
 * This file provides a comprehensive error handling framework for SHCI,
 * including custom exception types, diagnostic message formatting, and
 * context-aware error reporting.
 */

#pragma once

#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <memory>
#include <fstream>

/**
 * @namespace error
 * @brief Error handling utilities and exception classes
 */
namespace error {

/**
 * @enum ErrorSeverity
 * @brief Severity levels for errors and warnings
 */
enum class ErrorSeverity {
  DEBUG = 0,    ///< Debug information
  INFO = 1,     ///< Informational messages
  WARNING = 2,  ///< Warnings that don't stop execution
  ERROR = 3,    ///< Errors that stop current operation
  FATAL = 4     ///< Fatal errors that terminate program
};

/**
 * @enum ErrorCategory
 * @brief Categories of errors for better organization
 */
enum class ErrorCategory {
  CONFIGURATION,    ///< Configuration file and parameter errors
  FILE_IO,         ///< File reading/writing errors
  MEMORY,          ///< Memory allocation and management errors
  CONVERGENCE,     ///< Convergence and numerical errors
  MPI,             ///< MPI communication errors
  MATHEMATICAL,    ///< Mathematical operation errors
  SYSTEM,          ///< System-level errors
  USER_INPUT,      ///< User input validation errors
  INTERNAL         ///< Internal logic errors
};

/**
 * @struct ErrorContext
 * @brief Context information for better error diagnostics
 */
struct ErrorContext {
  std::string function_name;      ///< Function where error occurred
  std::string file_name;          ///< Source file name
  int line_number = -1;           ///< Line number in source file
  std::string module_name;        ///< Module/component name
  std::map<std::string, std::string> variables;  ///< Relevant variable values
  std::vector<std::string> call_stack;  ///< Function call stack
  
  ErrorContext() = default;
  
  ErrorContext(const std::string& func, const std::string& file, int line) 
    : function_name(func), file_name(file), line_number(line) {}
    
  /**
   * @brief Add variable value to context
   */
  template<typename T>
  void add_variable(const std::string& name, const T& value) {
    std::ostringstream oss;
    oss << value;
    variables[name] = oss.str();
  }
  
  /**
   * @brief Add function to call stack
   */
  void push_function(const std::string& func_name) {
    call_stack.push_back(func_name);
  }
};

/**
 * @class SHCIException
 * @brief Base exception class for all SHCI errors
 */
class SHCIException : public std::exception {
 public:
  /**
   * @brief Constructor with basic error information
   */
  SHCIException(const std::string& message, 
                ErrorSeverity severity = ErrorSeverity::ERROR,
                ErrorCategory category = ErrorCategory::INTERNAL)
    : message_(message), severity_(severity), category_(category) {
    format_message();
  }
  
  /**
   * @brief Constructor with context information
   */
  SHCIException(const std::string& message,
                const ErrorContext& context,
                ErrorSeverity severity = ErrorSeverity::ERROR,
                ErrorCategory category = ErrorCategory::INTERNAL)
    : message_(message), context_(context), severity_(severity), category_(category) {
    format_message();
  }
  
  const char* what() const noexcept override {
    return formatted_message_.c_str();
  }
  
  /**
   * @brief Get error severity
   */
  ErrorSeverity get_severity() const { return severity_; }
  
  /**
   * @brief Get error category
   */
  ErrorCategory get_category() const { return category_; }
  
  /**
   * @brief Get error context
   */
  const ErrorContext& get_context() const { return context_; }
  
  /**
   * @brief Get original message
   */
  const std::string& get_message() const { return message_; }
  
  /**
   * @brief Get formatted diagnostic message
   */
  const std::string& get_formatted_message() const { return formatted_message_; }
  
 private:
  std::string message_;                ///< Original error message
  std::string formatted_message_;     ///< Formatted diagnostic message
  ErrorContext context_;              ///< Error context information
  ErrorSeverity severity_;            ///< Error severity level
  ErrorCategory category_;            ///< Error category
  
  /**
   * @brief Format comprehensive error message with context
   */
  void format_message();
};

/**
 * @class ConfigurationException
 * @brief Exception for configuration-related errors
 */
class ConfigurationException : public SHCIException {
 public:
  ConfigurationException(const std::string& message, const ErrorContext& context = ErrorContext())
    : SHCIException(message, context, ErrorSeverity::ERROR, ErrorCategory::CONFIGURATION) {}
};

/**
 * @class FileIOException
 * @brief Exception for file I/O errors
 */
class FileIOException : public SHCIException {
 public:
  FileIOException(const std::string& message, const ErrorContext& context = ErrorContext())
    : SHCIException(message, context, ErrorSeverity::ERROR, ErrorCategory::FILE_IO) {}
};

/**
 * @class MemoryException
 * @brief Exception for memory-related errors
 */
class MemoryException : public SHCIException {
 public:
  MemoryException(const std::string& message, const ErrorContext& context = ErrorContext())
    : SHCIException(message, context, ErrorSeverity::ERROR, ErrorCategory::MEMORY) {}
};

/**
 * @class ConvergenceException
 * @brief Exception for convergence failures
 */
class ConvergenceException : public SHCIException {
 public:
  ConvergenceException(const std::string& message, const ErrorContext& context = ErrorContext())
    : SHCIException(message, context, ErrorSeverity::ERROR, ErrorCategory::CONVERGENCE) {}
};

/**
 * @class MPIException
 * @brief Exception for MPI communication errors
 */
class MPIException : public SHCIException {
 public:
  MPIException(const std::string& message, const ErrorContext& context = ErrorContext())
    : SHCIException(message, context, ErrorSeverity::ERROR, ErrorCategory::MPI) {}
};

/**
 * @class MathematicalException
 * @brief Exception for mathematical operation errors
 */
class MathematicalException : public SHCIException {
 public:
  MathematicalException(const std::string& message, const ErrorContext& context = ErrorContext())
    : SHCIException(message, context, ErrorSeverity::ERROR, ErrorCategory::MATHEMATICAL) {}
};

/**
 * @class UserInputException
 * @brief Exception for user input validation errors
 */
class UserInputException : public SHCIException {
 public:
  UserInputException(const std::string& message, const ErrorContext& context = ErrorContext())
    : SHCIException(message, context, ErrorSeverity::ERROR, ErrorCategory::USER_INPUT) {}
};

/**
 * @class ErrorHandler
 * @brief Central error handling and logging system
 */
class ErrorHandler {
 public:
  /**
   * @brief Get singleton instance
   */
  static ErrorHandler& instance() {
    static ErrorHandler handler;
    return handler;
  }
  
  /**
   * @brief Log error or warning message
   */
  void log(ErrorSeverity severity, ErrorCategory category, 
           const std::string& message, const ErrorContext& context = ErrorContext());
  
  /**
   * @brief Set minimum severity level for logging
   */
  void set_log_level(ErrorSeverity level) { min_log_level_ = level; }
  
  /**
   * @brief Enable/disable error logging to file
   */
  void set_log_file(const std::string& filename);
  
  /**
   * @brief Enable/disable console output
   */
  void set_console_output(bool enabled) { console_output_ = enabled; }
  
  /**
   * @brief Get diagnostic suggestions for common errors
   */
  std::vector<std::string> get_suggestions(ErrorCategory category, const std::string& message);
  
  /**
   * @brief Generate system information for debugging
   */
  std::string get_system_info();
  
  /**
   * @brief Check system requirements and warn about potential issues
   */
  void validate_system_requirements();
  
  /**
   * @brief Format severity level as string
   */
  std::string severity_to_string(ErrorSeverity severity);
  
  /**
   * @brief Format category as string
   */
  std::string category_to_string(ErrorCategory category);
  
 private:
  ErrorSeverity min_log_level_ = ErrorSeverity::WARNING;
  bool console_output_ = true;
  std::string log_filename_;
  std::map<ErrorCategory, std::vector<std::string>> error_suggestions_;
  
  ErrorHandler() { initialize_suggestions(); }
  
  /**
   * @brief Initialize diagnostic suggestions
   */
  void initialize_suggestions();
};

/**
 * @brief Assertion macro with context information
 */
#define SHCI_ASSERT(condition, message) \
  do { \
    if (!(condition)) { \
      error::ErrorContext ctx(__FUNCTION__, __FILE__, __LINE__); \
      throw error::SHCIException(message, ctx, error::ErrorSeverity::FATAL); \
    } \
  } while(0)

/**
 * @brief Assertion macro with custom exception type
 */
#define SHCI_ASSERT_TYPE(condition, message, exception_type) \
  do { \
    if (!(condition)) { \
      error::ErrorContext ctx(__FUNCTION__, __FILE__, __LINE__); \
      throw exception_type(message, ctx); \
    } \
  } while(0)

/**
 * @brief Warning macro
 */
#define SHCI_WARN(message) \
  do { \
    error::ErrorContext ctx(__FUNCTION__, __FILE__, __LINE__); \
    error::ErrorHandler::instance().log(error::ErrorSeverity::WARNING, \
                                       error::ErrorCategory::INTERNAL, \
                                       message, ctx); \
  } while(0)

/**
 * @brief Info macro
 */
#define SHCI_INFO(message) \
  do { \
    error::ErrorContext ctx(__FUNCTION__, __FILE__, __LINE__); \
    error::ErrorHandler::instance().log(error::ErrorSeverity::INFO, \
                                       error::ErrorCategory::INTERNAL, \
                                       message, ctx); \
  } while(0)

/**
 * @brief Debug macro (only active in debug builds)
 */
#ifdef DEBUG
#define SHCI_DEBUG(message) \
  do { \
    error::ErrorContext ctx(__FUNCTION__, __FILE__, __LINE__); \
    error::ErrorHandler::instance().log(error::ErrorSeverity::DEBUG, \
                                       error::ErrorCategory::INTERNAL, \
                                       message, ctx); \
  } while(0)
#else
#define SHCI_DEBUG(message) do {} while(0)
#endif

/**
 * @brief Convenience function to create error context
 */
inline error::ErrorContext make_context(const std::string& func, 
                                       const std::string& file, 
                                       int line) {
  return error::ErrorContext(func, file, line);
}

/**
 * @brief Enhanced error checking for file operations
 */
inline void check_file_exists(const std::string& filename) {
  if (!std::ifstream(filename.c_str()).good()) {
    error::ErrorContext ctx(__FUNCTION__, __FILE__, __LINE__);
    ctx.add_variable("filename", filename);
    throw error::FileIOException("File not found or cannot be opened", ctx);
  }
}

/**
 * @brief Enhanced error checking for memory allocations
 */
template<typename T>
inline void check_allocation(const T* ptr, size_t size) {
  if (ptr == nullptr && size > 0) {
    error::ErrorContext ctx(__FUNCTION__, __FILE__, __LINE__);
    ctx.add_variable("requested_size", size);
    ctx.add_variable("size_mb", size / (1024.0 * 1024.0));
    throw error::MemoryException("Memory allocation failed", ctx);
  }
}

/**
 * @brief Enhanced convergence checking
 */
inline void check_convergence(bool converged, int iteration, int max_iterations, 
                             double error, double tolerance) {
  if (!converged && iteration >= max_iterations) {
    error::ErrorContext ctx(__FUNCTION__, __FILE__, __LINE__);
    ctx.add_variable("final_error", error);
    ctx.add_variable("tolerance", tolerance);
    ctx.add_variable("iterations", iteration);
    ctx.add_variable("max_iterations", max_iterations);
    throw error::ConvergenceException("Maximum iterations reached without convergence", ctx);
  }
}

/**
 * @brief Enhanced parameter validation
 */
template<typename T>
inline void validate_range(const T& value, const T& min_val, const T& max_val, 
                          const std::string& param_name) {
  if (value < min_val || value > max_val) {
    error::ErrorContext ctx(__FUNCTION__, __FILE__, __LINE__);
    ctx.add_variable("parameter", param_name);
    ctx.add_variable("value", value);
    ctx.add_variable("min_allowed", min_val);
    ctx.add_variable("max_allowed", max_val);
    throw error::UserInputException("Parameter value out of valid range", ctx);
  }
}

} // namespace error