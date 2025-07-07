# SHCI User Manual

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start Guide](#quick-start-guide)
4. [Configuration System](#configuration-system)
5. [Input Files](#input-files)
6. [Running Calculations](#running-calculations)
7. [Understanding Output](#understanding-output)
8. [Advanced Usage](#advanced-usage)
9. [Performance Optimization](#performance-optimization)
10. [Troubleshooting](#troubleshooting)
11. [Examples](#examples)

## Introduction

SHCI (Semistochastic Heat Bath Configuration Interaction) is a highly efficient quantum chemistry method for computing accurate ground and excited state energies of molecules and materials. This manual will guide you through setting up and running SHCI calculations.

### What is SHCI?

SHCI combines the best aspects of:
- **Variational methods**: Systematic determinant selection using heat-bath algorithm
- **Perturbation theory**: Multi-stage corrections (DTM, PSTO, STO) for remaining determinant space
- **Parallel computing**: Efficient MPI+OpenMP implementation for large-scale calculations

### Key Features

- **High accuracy**: Achieves near-exact energies for molecules with 10-50 electrons
- **Scalability**: Runs efficiently on 1-1000+ CPU cores
- **Memory efficiency**: Distributed storage enables calculations on systems with limited memory
- **Flexibility**: Supports both molecular (ChemSystem) and solid-state (HegSystem) calculations

## Installation

### Prerequisites

- **C++ Compiler**: GCC 7+ or Clang 10+ with C++11 support
- **MPI**: OpenMPI 3.0+ or Intel MPI
- **OpenMP**: For shared-memory parallelization
- **Git**: For downloading dependencies

### Build Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/QMC-Cornell/shci.git
   cd shci
   ```

2. **Initialize submodules:**
   ```bash
   git submodule update --init --recursive
   ```

3. **Build the code:**
   ```bash
   make -j$(nproc)
   ```

4. **Test the installation:**
   ```bash
   make test
   ```

### Platform-Specific Notes

#### macOS (Apple Silicon)
```bash
# Install native ARM64 tools
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
/opt/homebrew/bin/brew install open-mpi libomp

# Create local.mk for macOS configuration
cat > local.mk << EOF
CXX := /opt/homebrew/bin/mpic++
CXXFLAGS := -std=c++11 -O3 -Xpreprocessor -fopenmp -Wall -Wextra -Wno-unused-result
LDLIBS := -pthread -lpthread -L/opt/homebrew/lib -lomp
CXXFLAGS := \$(CXXFLAGS) -I/opt/homebrew/include -I \$(LIB_DIR)
EOF
```

#### Linux (Intel/AMD)
```bash
# Ubuntu/Debian
sudo apt-get install build-essential libopenmpi-dev libomp-dev

# CentOS/RHEL
sudo yum install gcc-c++ openmpi-devel libgomp
```

## Quick Start Guide

### Step 1: Prepare Input Files

Create a working directory with two essential files:

1. **FCIDUMP**: Molecular orbital integrals (from quantum chemistry packages)
2. **config.json**: Calculation parameters

### Step 2: Basic Configuration

Create `config.json`:
```json
{
  "system": "chem",
  "n_up": 5,
  "n_dn": 5,
  "eps_vars": [1e-4, 5e-5, 2e-5],
  "eps_pt_dtm": 1e-6,
  "target_error": 1e-4,
  "time_sym": true
}
```

### Step 3: Run Calculation

```bash
# Single-core run
./shci

# Parallel run (8 MPI processes, 4 OpenMP threads each)
mpirun -np 8 ./shci
export OMP_NUM_THREADS=4
```

### Step 4: Check Results

Look for output files:
- `result.json`: Final energies and statistics
- `energy_var`: Variational energy progression
- `energy_tot`: Total energy including PT corrections

## Configuration System

SHCI uses JSON configuration files for maximum flexibility and reproducibility.

### Basic Parameters

```json
{
  "system": "chem",           // System type: "chem" or "heg"
  "n_up": 5,                  // Number of α electrons
  "n_dn": 5,                  // Number of β electrons
  "eps_vars": [1e-4, 5e-5],   // Variational selection thresholds
  "eps_pt_dtm": 1e-6,         // DTM perturbation threshold
  "target_error": 1e-4,       // Davidson convergence criterion
  "time_sym": true,           // Use time-reversal symmetry
  "max_pt_iterations": 10,    // Maximum PT iterations
  "n_batches_pt": 1000        // PT batch size for memory management
}
```

### Chemistry-Specific Parameters

```json
{
  "chem": {
    "point_group": "d2h",     // Point group symmetry
    "irrep": 1,               // Target irreducible representation
    "fcidump_filename": "FCIDUMP",  // Integral file name
    "load_integrals_cache": true    // Cache integrals to disk
  }
}
```

### Advanced Parameters

```json
{
  "davidson": {
    "n_states": 1,            // Number of states to compute
    "max_iterations": 100,    // Maximum Davidson iterations
    "max_subspace_size": 50,  // Subspace size limit
    "preconditioner": "diagonal"  // Preconditioning method
  },
  "parallel": {
    "load_balance_frequency": 1000,  // Load balancing interval
    "communication_scheme": "alltoall"  // MPI communication pattern
  },
  "memory": {
    "max_memory_gb": 16,      // Memory limit per MPI process
    "integral_storage": "distributed"  // Integral storage strategy
  }
}
```

## Input Files

### FCIDUMP Format

SHCI reads molecular orbital integrals in FCIDUMP format:

```
 &FCI NORB=4,NELEC=4,MS2=0,
  ORBSYM=1,1,1,1,
  ISYM=1,
 &END
  1.0000000000E+00  1  1  1  1
  5.0000000000E-01  2  1  1  1
  ...
  0.0000000000E+00  0  0  0  0
```

**Key fields:**
- `NORB`: Number of molecular orbitals
- `NELEC`: Total number of electrons  
- `MS2`: 2 × (S_z), where S_z is the spin projection
- `ORBSYM`: Orbital symmetries (irreducible representations)

### Generating FCIDUMP Files

#### From PySCF
```python
from pyscf import gto, scf, mcscf

# Define molecule
mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='cc-pvdz')

# Run SCF calculation
mf = scf.RHF(mol).run()

# Generate FCIDUMP
from pyscf.tools import fcidump
fcidump.from_scf(mf, 'FCIDUMP')
```

#### From Molpro
```
{hf; wf,10,1,0}                    ! SCF calculation
{multi; closed,2; occ,5; wf,10,1,0}  ! CASSCF if needed
put,molden,molden.out              ! Save orbitals
{fcidump,FCIDUMP}                  ! Generate integrals
```

#### From Gaussian
1. Run calculation with `gfinput` and `iop(6/7=3)` keywords
2. Use external tools like `fchk2fcidump` to convert

## Running Calculations

### Single-Point Energy

Basic ground state energy calculation:

```bash
# Configure calculation
cat > config.json << EOF
{
  "system": "chem",
  "n_up": 5,
  "n_dn": 5,
  "eps_vars": [1e-4, 5e-5, 2e-5],
  "eps_pt_dtm": 1e-6,
  "target_error": 1e-4
}
EOF

# Run calculation
mpirun -np 4 ./shci > output.log 2>&1
```

### Excited States

Calculate multiple electronic states:

```json
{
  "davidson": {
    "n_states": 3,
    "state_specific": false
  },
  "eps_vars": [1e-3, 5e-4, 2e-4],
  "target_error": 1e-5
}
```

### Wavefunction Analysis

Save and analyze wavefunctions:

```json
{
  "save_wavefunction": true,
  "wavefunction_filename": "wf_final.dat",
  "analyze_wavefunction": {
    "natural_orbitals": true,
    "ci_coefficients": true,
    "determinant_weights": true
  }
}
```

### Parameter Sweeps

Systematic convergence studies:

```bash
#!/bin/bash
# Convergence study script

eps_values=(1e-3 5e-4 2e-4 1e-4 5e-5)

for eps in "${eps_values[@]}"; do
    mkdir -p "eps_${eps}"
    cd "eps_${eps}"
    
    # Create configuration
    cat > config.json << EOF
{
  "system": "chem",
  "n_up": 5,
  "n_dn": 5,
  "eps_vars": [${eps}],
  "eps_pt_dtm": 1e-6
}
EOF
    
    # Copy integrals and run
    cp ../FCIDUMP .
    mpirun -np 8 ../shci > output.log
    
    cd ..
done
```

## Understanding Output

### Console Output

SHCI provides detailed progress information:

```
=== SHCI Calculation ===
System: Chem (10 electrons, 20 orbitals)
Configuration: 5 α, 5 β electrons

=== Variational Phase ===
Epsilon: 1.0e-04
  Initial determinants: 1
  Heat-bath selection: 15234 determinants
  Davidson iteration 1: E = -108.923456 Ha
  Davidson iteration 2: E = -108.954321 Ha
  Converged: E = -108.954356 Ha (10 iterations)

=== Perturbation Theory Phase ===
DTM correction: -0.012345 Ha (9876 determinants)
PSTO correction: -0.003456 Ha 
STO correction: -0.001234 ± 0.000123 Ha

=== Final Results ===
Variational energy: -108.954356 Ha
PT correction: -0.017035 Ha
Total energy: -108.971391 ± 0.000123 Ha
```

### Output Files

#### result.json
Contains final results in structured format:
```json
{
  "energies": {
    "variational": -108.954356,
    "pt_correction": -0.017035,
    "total": -108.971391,
    "error_bar": 0.000123
  },
  "determinants": {
    "variational": 15234,
    "dtm": 9876,
    "total_visited": 2456789
  },
  "timing": {
    "variational": 123.45,
    "perturbation": 456.78,
    "total": 580.23
  }
}
```

#### energy_var
Variational energy progression:
```
# Iteration  Energy(Ha)      Error         N_det
1           -108.923456     1.2e-02       1
2           -108.945123     8.7e-03       156
3           -108.952341     2.1e-03       1234
...
```

#### energy_tot  
Total energy including PT corrections:
```
# eps_var     E_var(Ha)       E_PT(Ha)        E_total(Ha)     Error(Ha)
1.0e-04      -108.954356     -0.017035       -108.971391     0.000123
5.0e-05      -108.963245     -0.008156       -108.971401     0.000087
```

### Convergence Analysis

Monitor convergence using provided Python scripts:

```python
#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt

# Load results
with open('result.json') as f:
    data = json.load(f)

# Plot energy convergence
import numpy as np
energies = np.loadtxt('energy_var')

plt.figure(figsize=(10, 6))
plt.semilogy(energies[:, 0], np.abs(energies[:, 1] - energies[-1, 1]))
plt.xlabel('Davidson Iteration')
plt.ylabel('|Energy Error| (Ha)')
plt.title('Variational Energy Convergence')
plt.grid(True)
plt.savefig('convergence.png', dpi=300)
```

## Advanced Usage

### Custom Determinant Selection

Implement custom selection criteria:

```json
{
  "variational": {
    "selection_method": "custom",
    "custom_selection": {
      "energy_threshold": 1e-4,
      "amplitude_threshold": 1e-3,
      "max_excitation_level": 4,
      "include_t1_diagnostic": true
    }
  }
}
```

### Memory-Limited Calculations

For large systems with limited memory:

```json
{
  "memory": {
    "max_memory_gb": 8,
    "n_batches_pt": 5000,
    "streaming_io": true,
    "compress_determinants": true
  }
}
```

### Multi-State Calculations

Calculate excited states efficiently:

```json
{
  "davidson": {
    "n_states": 5,
    "state_averaging": false,
    "orthogonality_tolerance": 1e-8
  },
  "state_specific": {
    "target_state": 2,
    "shift_parameter": 0.1
  }
}
```

### Checkpoint and Restart

For long calculations, enable checkpointing:

```json
{
  "checkpoint": {
    "enabled": true,
    "frequency": 1000,
    "filename": "checkpoint.dat",
    "auto_restart": true
  }
}
```

## Performance Optimization

### Parallel Efficiency

Optimize MPI and OpenMP settings:

```bash
# Example for 64-core node
export OMP_NUM_THREADS=4          # 4 threads per MPI process
export OMP_PLACES=cores
export OMP_PROC_BIND=close
mpirun -np 16 --bind-to core --map-by socket ./shci
```

### Memory Optimization

```json
{
  "memory": {
    "determinant_storage": "compressed",
    "integral_caching": "adaptive",
    "garbage_collection_frequency": 100
  }
}
```

### Load Balancing

For heterogeneous systems:

```json
{
  "parallel": {
    "dynamic_load_balancing": true,
    "load_balance_frequency": 500,
    "work_stealing": true
  }
}
```

## Troubleshooting

### Common Issues

#### 1. Memory Errors
```
Error: Not enough memory for determinant storage
```

**Solutions:**
- Increase `n_batches_pt` parameter
- Reduce `eps_vars` values (use fewer determinants)
- Enable memory compression
- Use more MPI processes with less memory per process

#### 2. Convergence Problems
```
Warning: Davidson diagonalization not converged
```

**Solutions:**
- Increase `target_error` tolerance
- Increase `max_iterations` in Davidson section
- Check for near-linear dependencies in basis
- Verify FCIDUMP file integrity

#### 3. FCIDUMP Reading Errors
```
Error: Unable to parse FCIDUMP file
```

**Solutions:**
- Check file format (Fortran formatting required)
- Verify electron count matches `NELEC` parameter
- Ensure file is not corrupted or truncated
- Check orbital symmetries are correct

### Performance Issues

#### Slow Variational Phase
- Reduce initial `eps_vars` value
- Check determinant growth rate
- Optimize MPI communication pattern
- Profile memory access patterns

#### Inefficient PT Phase
- Increase `n_batches_pt` for better parallelization
- Enable streaming I/O for large systems
- Check load balancing efficiency
- Monitor memory usage patterns

### Debugging Tools

Enable detailed debugging output:

```json
{
  "debug": {
    "level": 2,
    "output_file": "debug.log",
    "memory_tracking": true,
    "timing_breakdown": true,
    "determinant_analysis": true
  }
}
```

## Examples

### Example 1: Water Molecule (H₂O)

Complete calculation setup for water:

```json
{
  "system": "chem",
  "n_up": 5,
  "n_dn": 5,
  "eps_vars": [1e-4, 5e-5, 2e-5],
  "eps_pt_dtm": 1e-6,
  "target_error": 1e-5,
  "time_sym": true,
  "chem": {
    "point_group": "c2v",
    "irrep": 1
  },
  "davidson": {
    "n_states": 1,
    "max_iterations": 100
  }
}
```

Expected output:
```
Variational energy: -76.234567 Ha
PT correction: -0.089123 Ha
Total energy: -76.323690 ± 0.000045 Ha
```

### Example 2: Benzene (C₆H₆) Excited States

```json
{
  "system": "chem",
  "n_up": 21,
  "n_dn": 21,
  "eps_vars": [5e-4, 2e-4, 1e-4],
  "eps_pt_dtm": 1e-5,
  "target_error": 1e-4,
  "davidson": {
    "n_states": 5,
    "state_specific": false
  },
  "chem": {
    "point_group": "d6h",
    "fcidump_filename": "benzene.fcidump"
  }
}
```

### Example 3: Hubbard Model

```json
{
  "system": "heg",
  "n_up": 4,
  "n_dn": 4,
  "eps_vars": [1e-3, 5e-4],
  "eps_pt_dtm": 1e-5,
  "heg": {
    "n_sites": 8,
    "hubbard_u": 4.0,
    "periodic_boundary": true,
    "k_points": [[0,0], [0.5,0], [0.5,0.5]]
  }
}
```

### Example 4: Convergence Study

Python script for systematic convergence analysis:

```python
#!/usr/bin/env python3
import json
import subprocess
import numpy as np
import matplotlib.pyplot as plt

# Convergence study parameters
eps_values = [1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5]
energies = []
errors = []

for eps in eps_values:
    # Create configuration
    config = {
        "system": "chem",
        "n_up": 5,
        "n_dn": 5,
        "eps_vars": [eps],
        "eps_pt_dtm": 1e-6,
        "target_error": 1e-5
    }
    
    # Write config and run
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    result = subprocess.run(['mpirun', '-np', '4', './shci'], 
                          capture_output=True, text=True)
    
    # Parse results
    with open('result.json') as f:
        data = json.load(f)
        energies.append(data['energies']['total'])
        errors.append(data['energies']['error_bar'])

# Plot convergence
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.semilogx(eps_values, energies, 'bo-')
plt.xlabel('ε_var')
plt.ylabel('Total Energy (Ha)')
plt.title('Energy vs. Variational Threshold')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.loglog(eps_values, errors, 'ro-')
plt.xlabel('ε_var')
plt.ylabel('Error Bar (Ha)')
plt.title('Statistical Error vs. Variational Threshold')
plt.grid(True)

plt.tight_layout()
plt.savefig('convergence_study.png', dpi=300)
print("Convergence study completed. See convergence_study.png")
```

This manual provides comprehensive guidance for using SHCI effectively. For additional support, consult the API documentation or contact the development team.