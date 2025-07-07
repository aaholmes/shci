# SHCI Configuration Reference

## Table of Contents
1. [Overview](#overview)
2. [Basic Parameters](#basic-parameters)
3. [System-Specific Parameters](#system-specific-parameters)
4. [Algorithm Parameters](#algorithm-parameters)
5. [Performance Parameters](#performance-parameters)
6. [Advanced Parameters](#advanced-parameters)
7. [Configuration Examples](#configuration-examples)
8. [Parameter Validation](#parameter-validation)
9. [Best Practices](#best-practices)

## Overview

SHCI uses JSON configuration files to control all aspects of the calculation. This document provides comprehensive documentation for all available parameters, their valid ranges, default values, and usage recommendations.

### Configuration File Structure

```json
{
  "system": "chem",
  "n_up": 5,
  "n_dn": 5,
  "eps_vars": [1e-4, 5e-5, 2e-5],
  "eps_pt_dtm": 1e-6,
  "target_error": 1e-4,
  "chem": { /* chemistry-specific parameters */ },
  "davidson": { /* Davidson solver parameters */ },
  "parallel": { /* parallel execution parameters */ },
  "memory": { /* memory management parameters */ }
}
```

## Basic Parameters

### Core System Configuration

#### `system` (string, required)
**Description**: Specifies the type of quantum system  
**Valid values**: `"chem"`, `"heg"`  
**Default**: No default (required)  
**Example**: `"system": "chem"`

- `"chem"`: Molecular quantum chemistry systems using molecular orbitals
- `"heg"`: Homogeneous Electron Gas / Hubbard model systems

#### `n_up` (integer, required)
**Description**: Number of α (spin-up) electrons  
**Valid range**: 1 to 100  
**Default**: No default (required)  
**Example**: `"n_up": 5`

**Notes**:
- Must be consistent with the molecular system being studied
- Total electron count affects computational complexity significantly

#### `n_dn` (integer, required)
**Description**: Number of β (spin-down) electrons  
**Valid range**: 1 to 100  
**Default**: No default (required)  
**Example**: `"n_dn": 5`

**Notes**:
- For closed-shell systems: `n_up = n_dn`
- For open-shell systems: `n_up ≠ n_dn`
- Must satisfy: `ms2 = n_up - n_dn`

#### `ms2` (integer, optional)
**Description**: 2 × total spin (S_z × 2)  
**Valid range**: Depends on electron count  
**Default**: `n_up - n_dn`  
**Example**: `"ms2": 0`

**Notes**:
- For singlet states: `ms2 = 0`
- For doublet states: `ms2 = 1`
- Automatically calculated if not specified

### Convergence Parameters

#### `eps_vars` (array of floats, required)
**Description**: Variational determinant selection thresholds  
**Valid range**: Each value 1e-12 to 1e-2  
**Default**: No default (required)  
**Example**: `"eps_vars": [1e-4, 5e-5, 2e-5]`

**Usage guidelines**:
- Values should be in decreasing order for optimal efficiency
- Start with moderate values (1e-3 to 1e-4) for initial exploration
- Use multiple stages for systematic convergence
- Final value determines variational space accuracy

**Performance impact**:
- Smaller values = larger determinant space = higher accuracy + more computation
- Each order of magnitude typically increases determinants by ~10x

#### `eps_pt_dtm` (float, required)
**Description**: Deterministic perturbation theory threshold  
**Valid range**: 1e-12 to 1e-2  
**Default**: `1e-6`  
**Example**: `"eps_pt_dtm": 1e-6"`

**Notes**:
- Should be smaller than final `eps_vars` value
- Controls transition between deterministic and stochastic PT
- Smaller values increase deterministic PT accuracy but cost more memory

#### `target_error` (float, optional)
**Description**: Davidson diagonalization convergence criterion  
**Valid range**: 1e-12 to 1e-2  
**Default**: `1e-4`  
**Example**: `"target_error": 1e-5"`

**Recommendations**:
- Should be 5-10× smaller than final `eps_vars` value
- Tighter convergence needed for high-accuracy calculations
- Balance between accuracy and computational cost

### Time-Reversal Symmetry

#### `time_sym` (boolean, optional)
**Description**: Enable time-reversal symmetry optimization  
**Valid values**: `true`, `false`  
**Default**: `true`  
**Example**: `"time_sym": true"`

**Benefits when enabled**:
- Reduces determinant space by ~factor of 2
- Exploits α↔β electron symmetry in closed-shell systems
- Recommended for most molecular calculations

**When to disable**:
- Open-shell systems with different α/β orbital occupations
- Systems with strong spin-orbit coupling
- Debugging or comparison with non-symmetric codes

## System-Specific Parameters

### Chemistry Systems (`"system": "chem"`)

#### `chem` (object, optional)
Container for chemistry-specific parameters.

##### `chem.point_group` (string, optional)
**Description**: Molecular point group symmetry  
**Valid values**: Standard point group symbols  
**Default**: `"c1"` (no symmetry)  
**Example**: `"point_group": "d2h"`

**Common point groups**:
- `"c1"`: No symmetry
- `"cs"`: Mirror plane
- `"c2v"`: Water, formaldehyde
- `"d2h"`: Ethylene, benzene
- `"d6h"`: Benzene
- `"oh"`: Octahedral (SF₆)
- `"td"`: Tetrahedral (methane)

##### `chem.irrep` (integer, optional)
**Description**: Target irreducible representation  
**Valid range**: 1 to 8 (depends on point group)  
**Default**: `1`  
**Example**: `"irrep": 1"`

**Notes**:
- Irrep 1 is typically the ground state
- Higher irreps correspond to excited states
- Must be valid for the specified point group

##### `chem.fcidump_filename` (string, optional)
**Description**: Path to FCIDUMP integral file  
**Default**: `"FCIDUMP"`  
**Example**: `"fcidump_filename": "molecule.fcidump"`

**File requirements**:
- Must be in standard FCIDUMP format
- Contains one- and two-electron integrals
- Generated by quantum chemistry packages (PySCF, Molpro, etc.)

##### `chem.load_integrals_cache` (boolean, optional)
**Description**: Enable integral caching to disk  
**Default**: `true`  
**Example**: `"load_integrals_cache": true"`

**Benefits**:
- Faster restart of calculations
- Reduced integral transformation time
- Useful for parameter studies

### Hubbard/HEG Systems (`"system": "heg"`)

#### `heg` (object, optional)
Container for Hubbard/HEG-specific parameters.

##### `heg.n_sites` (integer, required for HEG)
**Description**: Number of lattice sites  
**Valid range**: 4 to 1000  
**Example**: `"n_sites": 16"`

##### `heg.hubbard_u` (float, required for HEG)
**Description**: Hubbard U parameter  
**Valid range**: 0.0 to 20.0  
**Example**: `"hubbard_u": 4.0"`

##### `heg.periodic_boundary` (boolean, optional)
**Description**: Use periodic boundary conditions  
**Default**: `true`  
**Example**: `"periodic_boundary": true"`

##### `heg.k_points` (array, optional)
**Description**: k-point sampling for momentum space  
**Example**: `"k_points": [[0,0], [0.5,0], [0.5,0.5]]`

## Algorithm Parameters

### Davidson Diagonalization

#### `davidson` (object, optional)
Container for Davidson algorithm parameters.

##### `davidson.n_states` (integer, optional)
**Description**: Number of eigenvalues/eigenvectors to compute  
**Valid range**: 1 to 20  
**Default**: `1`  
**Example**: `"n_states": 3"`

**Usage**:
- `n_states = 1`: Ground state only
- `n_states > 1`: Ground + excited states
- Higher values increase memory and computation

##### `davidson.max_iterations` (integer, optional)
**Description**: Maximum Davidson iterations  
**Valid range**: 10 to 1000  
**Default**: `100`  
**Example**: `"max_iterations": 200"`

##### `davidson.max_subspace_size` (integer, optional)
**Description**: Maximum Davidson subspace dimension  
**Valid range**: `2 * n_states` to 200  
**Default**: `50`  
**Example**: `"max_subspace_size": 80"`

**Requirements**:
- Must be at least `2 * n_states`
- Larger values improve convergence but use more memory
- Automatically limited by available memory

##### `davidson.preconditioner` (string, optional)
**Description**: Preconditioning method  
**Valid values**: `"diagonal"`, `"none"`  
**Default**: `"diagonal"`  
**Example**: `"preconditioner": "diagonal"`

##### `davidson.state_specific` (boolean, optional)
**Description**: Use state-specific Davidson for excited states  
**Default**: `false`  
**Example**: `"state_specific": true"`

### Perturbation Theory

#### `max_pt_iterations` (integer, optional)
**Description**: Maximum perturbation theory iterations  
**Valid range**: 1 to 1000  
**Default**: `10`  
**Example**: `"max_pt_iterations": 20"`

#### `n_batches_pt` (integer, optional)
**Description**: Number of batches for PT calculations (memory management)  
**Valid range**: 1 to 1,000,000  
**Default**: `1000`  
**Example**: `"n_batches_pt": 5000"`

**Usage**:
- Higher values reduce memory usage per batch
- Lower values may improve cache efficiency
- Adjust based on available memory and system size

#### `pt_deterministic_threshold` (float, optional)
**Description**: Alternative name for `eps_pt_dtm`  
**See**: `eps_pt_dtm`

## Performance Parameters

### Parallel Execution

#### `parallel` (object, optional)
Container for parallel execution parameters.

##### `parallel.load_balance_frequency` (integer, optional)
**Description**: Load balancing check interval (iterations)  
**Valid range**: 10 to 100,000  
**Default**: `1000`  
**Example**: `"load_balance_frequency": 500"`

##### `parallel.dynamic_load_balancing` (boolean, optional)
**Description**: Enable dynamic load balancing  
**Default**: `true`  
**Example**: `"dynamic_load_balancing": true"`

##### `parallel.work_stealing` (boolean, optional)
**Description**: Enable work stealing for better load balance  
**Default**: `false`  
**Example**: `"work_stealing": true"`

##### `parallel.communication_scheme` (string, optional)
**Description**: MPI communication pattern  
**Valid values**: `"alltoall"`, `"tree"`, `"ring"`  
**Default**: `"alltoall"`  
**Example**: `"communication_scheme": "tree"`

### Memory Management

#### `memory` (object, optional)
Container for memory management parameters.

##### `memory.max_memory_gb` (float, optional)
**Description**: Maximum memory per MPI process (GB)  
**Valid range**: 0.1 to 1024.0  
**Default**: `16.0`  
**Example**: `"max_memory_gb": 32.0"`

**Usage**:
- Set based on available system memory
- Leave headroom for OS and other processes
- For N MPI processes: total_memory = N × max_memory_gb

##### `memory.determinant_storage` (string, optional)
**Description**: Determinant storage method  
**Valid values**: `"normal"`, `"compressed"`  
**Default**: `"normal"`  
**Example**: `"determinant_storage": "compressed"`

##### `memory.integral_storage` (string, optional)
**Description**: Integral storage strategy  
**Valid values**: `"memory"`, `"disk"`, `"distributed"`  
**Default**: `"distributed"`  
**Example**: `"integral_storage": "memory"`

##### `memory.garbage_collection_frequency` (integer, optional)
**Description**: Memory cleanup interval  
**Valid range**: 10 to 10,000  
**Default**: `100`  
**Example**: `"garbage_collection_frequency": 500"`

## Advanced Parameters

### Checkpointing and Restart

#### `checkpoint` (object, optional)
Container for checkpoint/restart parameters.

##### `checkpoint.enabled` (boolean, optional)
**Description**: Enable checkpoint writing  
**Default**: `false`  
**Example**: `"enabled": true"`

##### `checkpoint.frequency` (integer, optional)
**Description**: Checkpoint frequency (iterations)  
**Valid range**: 10 to 10,000  
**Default**: `1000`  
**Example**: `"frequency": 500"`

##### `checkpoint.filename` (string, optional)
**Description**: Checkpoint file name  
**Default**: `"checkpoint.dat"`  
**Example**: `"filename": "calc_checkpoint.dat"`

##### `checkpoint.auto_restart` (boolean, optional)
**Description**: Automatically restart from checkpoint  
**Default**: `true`  
**Example**: `"auto_restart": true"`

### Wavefunction Analysis

#### `save_wavefunction` (boolean, optional)
**Description**: Save final wavefunction to disk  
**Default**: `false`  
**Example**: `"save_wavefunction": true"`

#### `wavefunction_filename` (string, optional)
**Description**: Wavefunction output filename  
**Default**: `"wavefunction.dat"`  
**Example**: `"wavefunction_filename": "final_wf.dat"`

#### `load_wf_file` (string, optional)
**Description**: Load initial wavefunction from file  
**Default**: `""` (start from HF)  
**Example**: `"load_wf_file": "initial_wf.dat"`

#### `analyze_wavefunction` (object, optional)
Container for wavefunction analysis options.

##### `analyze_wavefunction.natural_orbitals` (boolean, optional)
**Description**: Compute natural orbitals  
**Default**: `false`  
**Example**: `"natural_orbitals": true"`

##### `analyze_wavefunction.ci_coefficients` (boolean, optional)
**Description**: Print dominant CI coefficients  
**Default**: `false`  
**Example**: `"ci_coefficients": true"`

### Debugging and Development

#### `debug` (object, optional)
Container for debugging parameters.

##### `debug.level` (integer, optional)
**Description**: Debug output verbosity level  
**Valid range**: 0 to 5  
**Default**: `0`  
**Example**: `"level": 2"`

**Levels**:
- `0`: No debug output
- `1`: Basic progress information
- `2`: Detailed algorithm steps
- `3`: Memory and timing information
- `4`: Mathematical details
- `5`: Full debugging (very verbose)

##### `debug.output_file` (string, optional)
**Description**: Debug output filename  
**Default**: `""` (console output)  
**Example**: `"output_file": "debug.log"`

##### `debug.memory_tracking` (boolean, optional)
**Description**: Enable detailed memory tracking  
**Default**: `false`  
**Example**: `"memory_tracking": true"`

##### `debug.timing_breakdown` (boolean, optional)
**Description**: Enable detailed timing analysis  
**Default**: `false`  
**Example**: `"timing_breakdown": true"`

## Configuration Examples

### Basic Ground State Calculation

```json
{
  "system": "chem",
  "n_up": 5,
  "n_dn": 5,
  "eps_vars": [1e-4],
  "eps_pt_dtm": 1e-6,
  "target_error": 1e-5,
  "time_sym": true
}
```

### Systematic Convergence Study

```json
{
  "system": "chem",
  "n_up": 10,
  "n_dn": 10,
  "eps_vars": [1e-3, 5e-4, 2e-4, 1e-4, 5e-5],
  "eps_pt_dtm": 1e-6,
  "target_error": 1e-6,
  "time_sym": true,
  "davidson": {
    "max_iterations": 200,
    "max_subspace_size": 100
  }
}
```

### High-Accuracy Calculation

```json
{
  "system": "chem",
  "n_up": 7,
  "n_dn": 7,
  "eps_vars": [5e-4, 2e-4, 1e-4, 5e-5, 2e-5],
  "eps_pt_dtm": 1e-7,
  "target_error": 1e-6,
  "time_sym": true,
  "n_batches_pt": 10000,
  "memory": {
    "max_memory_gb": 64.0,
    "determinant_storage": "compressed"
  }
}
```

### Excited States Calculation

```json
{
  "system": "chem",
  "n_up": 6,
  "n_dn": 6,
  "eps_vars": [2e-4, 1e-4, 5e-5],
  "eps_pt_dtm": 1e-6,
  "target_error": 1e-5,
  "davidson": {
    "n_states": 5,
    "max_iterations": 300,
    "max_subspace_size": 120,
    "state_specific": false
  },
  "chem": {
    "point_group": "d2h",
    "irrep": 1
  }
}
```

### Large System with Memory Optimization

```json
{
  "system": "chem",
  "n_up": 15,
  "n_dn": 15,
  "eps_vars": [1e-3, 5e-4, 2e-4],
  "eps_pt_dtm": 1e-5,
  "target_error": 1e-4,
  "n_batches_pt": 50000,
  "memory": {
    "max_memory_gb": 16.0,
    "determinant_storage": "compressed",
    "integral_storage": "distributed"
  },
  "parallel": {
    "load_balance_frequency": 100,
    "dynamic_load_balancing": true
  }
}
```

### Hubbard Model Calculation

```json
{
  "system": "heg",
  "n_up": 4,
  "n_dn": 4,
  "eps_vars": [1e-3, 5e-4],
  "eps_pt_dtm": 1e-5,
  "target_error": 1e-4,
  "heg": {
    "n_sites": 16,
    "hubbard_u": 4.0,
    "periodic_boundary": true,
    "k_points": [[0,0], [0.5,0], [0.5,0.5], [0.5,0.5]]
  }
}
```

### Production Calculation with All Features

```json
{
  "system": "chem",
  "n_up": 12,
  "n_dn": 12,
  "eps_vars": [1e-3, 5e-4, 2e-4, 1e-4],
  "eps_pt_dtm": 1e-6,
  "target_error": 1e-5,
  "time_sym": true,
  "max_pt_iterations": 20,
  "n_batches_pt": 20000,
  
  "chem": {
    "point_group": "c2v",
    "irrep": 1,
    "fcidump_filename": "molecule.fcidump",
    "load_integrals_cache": true
  },
  
  "davidson": {
    "n_states": 1,
    "max_iterations": 200,
    "max_subspace_size": 80,
    "preconditioner": "diagonal"
  },
  
  "memory": {
    "max_memory_gb": 32.0,
    "determinant_storage": "compressed",
    "integral_storage": "distributed",
    "garbage_collection_frequency": 200
  },
  
  "parallel": {
    "load_balance_frequency": 500,
    "dynamic_load_balancing": true,
    "work_stealing": false,
    "communication_scheme": "alltoall"
  },
  
  "checkpoint": {
    "enabled": true,
    "frequency": 1000,
    "filename": "production_checkpoint.dat",
    "auto_restart": true
  },
  
  "save_wavefunction": true,
  "wavefunction_filename": "final_wavefunction.dat",
  
  "analyze_wavefunction": {
    "natural_orbitals": true,
    "ci_coefficients": true
  },
  
  "debug": {
    "level": 1,
    "timing_breakdown": true
  }
}
```

## Parameter Validation

SHCI includes comprehensive parameter validation with helpful error messages and suggestions. The validation checks:

### Automatic Checks
- **Range validation**: All parameters checked against valid ranges
- **Type validation**: Correct data types (string, number, boolean, array)
- **Consistency checks**: Related parameters are consistent with each other
- **File existence**: Input files are accessible and correctly formatted
- **Memory estimation**: Warns about excessive memory requirements
- **System compatibility**: Checks against available hardware resources

### Common Validation Errors

#### Configuration Errors
```
Error: Parameter 'eps_vars' cannot be empty
Suggestion: Provide at least one variational threshold, e.g., [1e-4]
```

#### Range Errors
```
Error: Parameter 'n_up' value -1 is outside valid range [1, 100]
Suggestion: Set 'n_up' to a positive value
```

#### Consistency Errors
```
Warning: Final eps_vars value (2e-5) is smaller than eps_pt_dtm (1e-6)
Suggestion: Consider setting eps_pt_dtm smaller than the final eps_vars value
```

#### File Errors
```
Error: FCIDUMP file 'FCIDUMP' not found or not accessible
Suggestion: Ensure the FCIDUMP file exists in the current directory
```

## Best Practices

### Parameter Selection Strategy

1. **Start Simple**: Begin with single `eps_vars` value for initial testing
2. **Systematic Convergence**: Use decreasing sequence for production calculations
3. **Memory Awareness**: Monitor memory usage and adjust batch sizes accordingly
4. **Parallel Efficiency**: Balance MPI processes and OpenMP threads for your system

### Convergence Guidelines

1. **eps_vars sequence**: Use 3-5 values in geometric progression
   - Example: `[1e-3, 5e-4, 2e-4, 1e-4, 5e-5]`

2. **eps_pt_dtm**: Set 5-10× smaller than final `eps_vars`
   - If final `eps_vars = 2e-5`, use `eps_pt_dtm = 1e-6`

3. **target_error**: Set 5-10× smaller than final `eps_vars`
   - Ensures Davidson convergence doesn't limit accuracy

### Performance Optimization

1. **Memory Distribution**: 
   - Use more MPI processes for large systems
   - Keep memory per process under 80% of available RAM

2. **Batch Sizing**:
   - Increase `n_batches_pt` for memory-limited systems
   - Typical values: 1,000-50,000 depending on system size

3. **Checkpoint Strategy**:
   - Enable for calculations longer than 1 hour
   - Set frequency based on expected runtime

### System-Specific Recommendations

#### Small Molecules (< 10 electrons)
```json
{
  "eps_vars": [1e-4, 5e-5],
  "eps_pt_dtm": 1e-6,
  "n_batches_pt": 1000,
  "memory": {"max_memory_gb": 8.0}
}
```

#### Medium Molecules (10-20 electrons)
```json
{
  "eps_vars": [5e-4, 2e-4, 1e-4],
  "eps_pt_dtm": 1e-6,
  "n_batches_pt": 5000,
  "memory": {"max_memory_gb": 16.0}
}
```

#### Large Molecules (> 20 electrons)
```json
{
  "eps_vars": [1e-3, 5e-4, 2e-4],
  "eps_pt_dtm": 1e-5,
  "n_batches_pt": 20000,
  "memory": {
    "max_memory_gb": 32.0,
    "determinant_storage": "compressed"
  }
}
```

This comprehensive configuration reference provides detailed documentation for all SHCI parameters, enabling users to optimize their calculations for accuracy, performance, and resource usage.