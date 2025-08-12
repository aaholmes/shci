# Arrow

A high-performance implementation of the Semistochastic Heat-Bath Configuration Interaction (SHCI) method.

Arrow is a selected configuration interaction plus perturbation theory (SCI+PT) method for obtaining highly accurate, near-Full CI energies for challenging quantum chemistry problems. The underlying HCI and SHCI algorithms have innovations that make them among the fastest and most powerful SCI methods available. Arrow enables massively parallel computations using a hybrid MPI+OpenMP implementation.

[![Build Status](https://travis-ci.com/QMC-Cornell/shci.svg?token=Gy8pVWtUBHq57qdtpAKN&branch=master)](https://travis-ci.com/QMC-Cornell/shci)

## Installation

### Prerequisites
- MPI compiler (e.g., `mpic++`)
- OpenMP support
- C++11 compatible compiler

### Build Instructions
```bash
# Clone the repository
git clone https://github.com/aaholmes/shci.git
cd shci

# Initialize and update submodules
git submodule update --init --recursive

# Build the project
make -j
```

## How to Run
Instructions for running Arrow are documented in the [wiki](https://github.com/QMC-Cornell/shci/wiki).

## How to Contribute
Arrow is a research program rather than a fully tested catch-all software package. The efficiency and correctness of edge cases, or input values that differ greatly from the default or published values, are not guaranteed. We welcome help with extending the capabilities of Arrow. If interested, please contact Adam Holmes <adamaholmes@gmail.com> or Cyrus Umrigar <CyrusUmrigar@cornell.edu>.

## History and Authorship
The Heat-Bath Configuration Interaction (HCI) method was conceived, implemented, and first described by **Adam A. Holmes** while he was a student in the **Umrigar** group at Cornell. The subsequent semistochastic variant (SHCI) was developed through a collaborative effort with **Sandeep Sharma** and **Ali Alavi** at the University of Colorado and the Max Planck Institute. Since then the development of SHCI has been a joint effort between the Umrigar group and the Sharma group, where Holmes was a postdoc.

**Arrow** is the modern, high-performance C++ implementation of these methods. It was first implemented by **Junhao Li** and was later greatly extended by **Yuan Yao** and **Tyler Anderson**. It contains improved versions of the algorithms that were first realized in Holmes's original FORTRAN implementation, which was parallelized by **Matt Otten**. All authors, aside from Sharma and Alavi, were at Cornell when they first contributed.

## Citations
Any papers that use Arrow should cite the following foundational papers:

1.  **For the original HCI method:**
    "Heat-bath configuration interaction: An efficient selected configuration interaction algorithm inspired by heat-bath sampling", Adam A. Holmes, N. M. Tubman, and C. J. Umrigar, *J. Chem. Theory Comput.* 12, 3674 (2016).

2.  **For the semistochastic PT2 correction:**
    "Semistochastic heat-bath configuration interaction method...", Sandeep Sharma, Adam A. Holmes, Guillaume Jeanmairet, Ali Alavi, and C. J. Umrigar, *J. Chem. Theory Comput.* 13, 1595 (2017).

3.  **For the extrapolation to the Full CI or CAS limit and extension to excited states:**
    "Excited states using semistochastic heat-bath configuration interaction", Adam A. Holmes, C. J. Umrigar, and Sandeep Sharma, *J. Chem. Phys.* 147, 164111 (2017).

4.  **For the fast C++ implementation and algorithms in Arrow:**
    "Fast semistochastic heat-bath configuration interaction", Junhao Li, Matthew Otten, Adam A. Holmes, Sandeep Sharma, and C. J. Umrigar, *J. Chem. Phys.* 149, 214110 (2018).

For the **orbital optimization solver (including HCISCF)**, please cite:

5.  **First Implementation:** "Cheap and near exact CASSCF with large active spaces", J. E. T. Smith, B. Mussard, A. A. Holmes, and S. Sharma, *J. Chem. Theory Comput.* 13, 5468 (2017).
6.  **Landmark Application:** "Accurate many-body electronic structure near the basis set limit: Application to the chromium dimer", J. Li, Y. Yao, A. A. Holmes, et al., *Phys. Rev. Research* 2, 012015(R) (2020). This study used **HCISCF** to achieve its benchmark results.
7.  **Comprehensive Analysis:** "Orbital Optimization in Selected Configuration Interaction Methods", Y. Yao and C. J. Umrigar, *J. Chem. Theory Comput.* 17, 4183 (2021).

---
### BibTeX Entries

```bibtex
@article{HolTubUmr-JCTC-16,
  Author = {Adam A. Holmes and N. M. Tubman and C. J. Umrigar},
  Title = {Heat-bath Configuration Interaction: An efficient selected CI algorithm inspired by heat-bath sampling},
  Journal = {J. Chem. Theory Comput.},
  Volume = {12},
  Pages = {3674-3680},
  Year = {2016}
}

@article{ShaHolJeaAlaUmr-JCTC-17,
  Author = {Sandeep Sharma and Adam A. Holmes and Guillaume Jeanmairet and Ali Alavi and C. J. Umrigar},
  Title = {Semistochastic Heat-Bath Configuration Interaction Method: Selected Configuration Interaction with Semistochastic Perturbation Theory},
  Journal = {J. Chem. Theory Comput.},
  Year = {2017},
  Volume = {13},
  Pages = {1595-1604}
}

@article{HolUmrSha-JCP-17,
  Author = {Adam A. Holmes and C. J. Umrigar and Sandeep Sharma},
  Title = {Excited states using semistochastic heat-bath configuration interaction},
  Journal = {J. Chem. Phys.},
  Year = {2017},
  Volume = {147},
  Pages = {164111}
}

@article{LiOttHolShaUmr-JCP-18,
  Author = {Junhao Li and Matthew Otten and Adam A. Holmes and Sandeep Sharma and C. J. Umrigar},
  Title = {Fast Semistochastic Heat-Bath Configuration Interaction},
  Journal = {J. Chem. Phys.},
  Year = {2018},
  Volume = {149},
  Pages = {214110}
}

@article{SmithMusHolSha-JCTC-17,
  Author = {James E. T. Smith and Bastien Mussard and Adam A. Holmes and Sandeep Sharma},
  Title = {Cheap and Near Exact CASSCF with Large Active Spaces},
  Journal = {J. Chem. Theory Comput.},
  Year = {2017},
  Volume = {13},
  Pages = {5468-5478}
}

@article{LiYaoHol_PRR_20,
  Author = {Junhao Li and Yuan Yao and Adam A. Holmes and Matthew Otten and Qiming Sun and Sandeep Sharma and C. J. Umrigar},
  Title = {Accurate many-body electronic structure near the basis set limit: Application to the chromium dimer},
  Journal = {Phys. Rev. Research},
  Volume = {2},
  Issue = {1},
  Pages = {012015},
  Year = {2020}
}

@article{YaoUmr-JCTC-21,
  Author = {Yuan Yao and C. J. Umrigar},
  Title = {Orbital Optimization in Selected Configuration Interaction Methods},
  Journal = {J. Chem. Theory Comput.},
  Volume = {17},
  Pages = {4183-4194},
  Year = {2021}
}
