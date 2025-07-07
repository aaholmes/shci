# SHCI API Documentation

## Overview

This document provides comprehensive API documentation for the SHCI (Semistochastic Heat Bath Configuration Interaction) codebase. The code is organized into several key modules:

## Core Modules

### 1. Determinant Representation (`src/det/`)

#### HalfDet Class
- **Purpose**: Efficient bit-based representation of α or β electron configurations
- **Key Features**: 
  - O(1) orbital operations using bit manipulation
  - Support for up to 128 orbitals (default)
  - Fast comparison and hashing for hash table usage
- **File**: `src/det/half_det.h`

#### Det Class  
- **Purpose**: Complete determinant (α + β electrons)
- **Key Features**:
  - Combines two HalfDet objects
  - Sign calculation for electron permutations
  - Excitation operators for generating connected determinants
- **File**: `src/det/det.h`

### 2. Solver Framework (`src/solver/`)

#### Solver<S> Template Class
- **Purpose**: Main SHCI algorithm implementation
- **Template Parameter**: S = System type (ChemSystem, HegSystem, etc.)
- **Key Methods**:
  - `run()`: Complete SHCI calculation
  - `run_variation()`: Variational determinant selection
  - `run_perturbation()`: Multi-stage perturbation theory
- **File**: `src/solver/solver.h`

#### Davidson Class
- **Purpose**: Iterative diagonalization of large sparse matrices
- **Key Features**:
  - Memory-efficient subspace method
  - Fast convergence for lowest eigenvalues
  - Parallel matrix-vector operations
- **File**: `src/solver/davidson.h`

#### SparseMatrix Class
- **Purpose**: Efficient storage and operations for Hamiltonian matrices
- **Key Features**:
  - Compressed sparse row format
  - Parallel matrix-vector multiplication
  - Memory-optimized for quantum chemistry applications
- **File**: `src/solver/sparse_matrix.h`

### 3. System Abstraction (`src/chem/`, `src/heg/`)

#### BaseSystem Class
- **Purpose**: Abstract interface for quantum systems
- **Key Methods**:
  - `get_connections()`: Generate connected determinants
  - `get_hamiltonian_elem()`: Compute matrix elements
  - Virtual interface for different system types

#### ChemSystem Class
- **Purpose**: Molecular quantum chemistry systems
- **Key Features**:
  - FCIDUMP integral file support
  - Point group symmetry exploitation
  - Molecular orbital basis
- **File**: `src/chem/chem_system.h`

#### HegSystem Class  
- **Purpose**: Homogeneous Electron Gas systems
- **Key Features**:
  - Plane wave basis
  - Momentum conservation
  - Thermodynamic limit extrapolation
- **File**: `src/heg/heg_system.h`

## Algorithm Overview

### SHCI Algorithm Flow

```
1. Configuration Loading
   ├── JSON parameter parsing
   ├── System setup (ChemSystem/HegSystem)
   └── Integral loading (FCIDUMP/momentum space)

2. Variational Phase
   ├── Initial determinant selection (HF + singles/doubles)
   ├── For each ε_var value:
   │   ├── Heat-bath determinant screening
   │   ├── Hamiltonian matrix construction
   │   ├── Davidson diagonalization
   │   └── Convergence checking
   └── Wavefunction storage

3. Perturbation Theory Phase
   ├── DTM (Deterministic): Most important determinants
   ├── PSTO (Partitioned Stochastic): Hybrid treatment
   ├── STO (Stochastic): Remaining determinant space
   └── Energy correction accumulation

4. Results and Analysis
   ├── Energy extrapolation
   ├── Error bar estimation
   ├── JSON result output
   └── Wavefunction serialization
```

## Key Data Structures

### Determinant Storage
```cpp
// Efficient determinant representation
class HalfDet {
    std::array<uint64_t, N_CHUNKS> chunks;  // Bit storage
};

class Det {
    HalfDet up, dn;  // α and β electrons
    bool has_time_sym;  // Time-reversal symmetry flag
};
```

### Distributed Collections
```cpp
// Parallel hash tables for large-scale storage
fgpl::DistHashMap<Det, double, DetHasher> wf_map;
fgpl::DistHashSet<Det, DetHasher> var_dets;
omp_hash_map<Det, HcSum, DetHasher> local_map;
```

### Configuration Parameters
```cpp
// JSON-based configuration system
struct Config {
    std::string system;           // "chem" or "heg"
    std::vector<double> eps_vars; // Variational thresholds
    double eps_pt_dtm;           // DTM PT threshold
    bool time_sym;               // Time-reversal symmetry
    std::string load_wf_file;    // Wavefunction loading
};
```

## Parallel Programming Model

### MPI + OpenMP Hybrid
- **MPI**: Distributed memory parallelism across nodes
- **OpenMP**: Shared memory parallelism within nodes  
- **Load Balancing**: Dynamic work distribution
- **Communication**: Optimized collective operations

### Memory Management
- **Distributed Storage**: Hash tables spread across MPI ranks
- **Memory Monitoring**: Automatic memory usage tracking
- **Batch Processing**: Large PT calculations in memory-limited chunks
- **Cache Optimization**: Data structure layout for performance

## Configuration System

### JSON Parameter Format
```json
{
  "system": "chem",
  "n_up": 5,
  "n_dn": 5,
  "eps_vars": [1e-4, 5e-5, 2e-5],
  "eps_pt_dtm": 1e-6,
  "target_error": 1e-4,
  "chem": {
    "point_group": "c2v"
  }
}
```

### Key Parameters
- **eps_vars**: Variational selection thresholds (decreasing sequence)
- **eps_pt_dtm**: Deterministic perturbation theory threshold
- **target_error**: Convergence criterion for Davidson diagonalization
- **n_up/n_dn**: Number of α/β electrons
- **time_sym**: Enable time-reversal symmetry optimization

## Error Handling

### Exception Hierarchy
```cpp
// Custom exception types for different error categories
class SHCIException : public std::exception;
class ConfigException : public SHCIException;
class ConvergenceException : public SHCIException;
class MemoryException : public SHCIException;
```

### Validation Systems
- **Input Validation**: Configuration parameter checking
- **Convergence Monitoring**: Automatic stagnation detection
- **Memory Limits**: Graceful degradation when memory constrained
- **MPI Error Handling**: Robust parallel error propagation

## Performance Considerations

### Scalability
- **Strong Scaling**: Efficient up to 1000+ MPI ranks
- **Weak Scaling**: Problem size grows with number of processors
- **Memory Scaling**: Distributed storage enables large systems
- **Load Balancing**: Dynamic work redistribution

### Optimization Techniques
- **Vectorization**: SIMD operations for critical loops
- **Cache Optimization**: Memory layout for high cache hit rates
- **Communication Reduction**: Minimized MPI message passing
- **Algorithmic Optimizations**: Second rejection, time symmetry

## Usage Examples

### Basic Usage
```cpp
#include "solver/solver.h"
#include "chem/chem_system.h"

// Create and run SHCI calculation
Solver<ChemSystem> solver;
solver.run();  // Reads config.json automatically
```

### Advanced Configuration
```cpp
// Custom configuration
Config config;
config.system = "chem";
config.eps_vars = {1e-4, 5e-5, 2e-5};
config.eps_pt_dtm = 1e-6;

// System-specific setup
ChemSystem system;
system.load_integrals("FCIDUMP");

Solver<ChemSystem> solver;
solver.run();
```

## File I/O Formats

### Input Files
- **FCIDUMP**: Molecular orbital integrals (chemistry)
- **config.json**: Calculation parameters
- **wavefunction files**: Previously computed wavefunctions

### Output Files
- **result.json**: Final energies and statistics
- **wf_eps1_*.dat**: Wavefunction files for each ε value
- **integrals_cache.dat**: Cached integral transformations
- **energy**, **energy_var**, **energy_tot**: Energy progression files

This API documentation provides the foundation for understanding and extending the SHCI codebase. For detailed implementation examples, see the test suite and example calculations.