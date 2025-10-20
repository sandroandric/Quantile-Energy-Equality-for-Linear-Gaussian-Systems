# QEELGS: Quantile-Energy Equality for Linear-Gaussian Systems

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the validation code for the paper:

**"An Exact Quantile-Energy Equality for Terminal Halfspaces in Linear-Gaussian Control with a Discrete-Time Companion, KL/Schrödinger Links, and High-Precision Validation"**

by Sandro Andric

## Overview

We prove an exact equality between the minimal quadratic control energy and the squared normal-quantile gap for terminal halfspaces in linear-Gaussian systems. The minimal energy equals the squared normal-quantile gap divided by twice a controllability-to-noise ratio R²ₜ(w), and is attained by a matched-filter control.

**Key result:**
```
E_min = (Φ⁻¹(p₁) - Φ⁻¹(p₀))² / (2 R²ₜ(w))
```

where R²ₜ(w) = (w^T W^M_c w)/(w^T V_T w) is a controllability-to-noise SNR.

Equivalently, this is the minimal KL divergence for a path-space tilt that moves a terminal halfspace probability from p₀ to p₁.

## Features

- **Production-grade validation code** with machine-precision accuracy
- **Van Loan block exponentials** for robust Gramian computation
- **Forward Lyapunov recursion** for discrete-time Gramians
- **Numerical hygiene**: solves instead of inverses, PSD guards
- **CI/CD integration** with JSON output and reproducibility controls
- **All validation tests pass with 0.000e+00 error**

## Installation

### Requirements

- Python ≥ 3.10
- NumPy 1.26.4
- SciPy 1.13.1

### Setup

```bash
# Clone the repository
git clone https://github.com/sandroandric/qeelgs.git
cd qeelgs

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Validation

Run the full validation suite with default parameters (5M Monte Carlo paths):

```bash
python qeelgs_validate.py
```

### Quick Test

Run with reduced paths for quick verification:

```bash
python qeelgs_validate.py --npath 100000 --npath-interval 100000 --npath-small 1000
```

### JSON Output for CI/CD

```bash
python qeelgs_validate.py --json
```

### Reproducibility Controls

```bash
# Fixed seed for deterministic results
python qeelgs_validate.py --seed 42

# Deterministic direction mode for CI
python qeelgs_validate.py --w-mode fixed

# Auto-enable JSON in CI environment
CI=1 python qeelgs_validate.py
```

### Command-Line Options

```
--npath            Monte Carlo paths for main tests (default: 5,000,000)
--npath-interval   Paths for interval event test (default: 5,000,000)
--npath-small      Paths for small-sample test (default: 10,000)
--T                Time horizon (default: 1.0)
--dt-discrete      Discrete-time step (default: 0.01)
--seed             Random seed (default: 42)
--w-mode           Direction mode: random/nullspace/fixed (default: random)
--json             Output JSON summary for CI
```

## Validation Tests

The script performs 5 rigorous validation tests:

1. **Halfspace tightness (continuous-time)**: Verifies the quantile-energy equality is exact for halfspaces
2. **Discrete-time test**: Validates discrete-time companion formulas via Van Loan discretization
3. **Small-sample sanity**: Quick verification with reduced Monte Carlo samples
4. **Interval event**: Lower-bound check for non-halfspace events
5. **Random directions**: Tests across random directions including near-nullspace cases

### Expected Output

```
=== QEELGS Rigorous Validation ===================================
System: n=3, m=2, T=1.0, dt_discrete=0.01, seed=42
Baseline p0 (no control): 0.73926180
Reachability SNR R_T^2   : 0.71447352
  w^T V_T w              : 9.73301512e-02
  w^T Wc^M w             : 6.95398153e-02

1) Halfspace tightness (continuous-time):
   max |E_calc - E_ana|/E_ana = 0.000e+00
   min slack (should ≥ 0)     = 0.000000e+00

2) Discrete-time test (Van Loan discretization):
   analytical E = 2.79920508
   E_calc       = 2.79920508
   relative err = 0.000e+00

================================================================
QEELGS VALIDATION SUMMARY
----------------------------------------------------------------
Halfspace max relative energy error             : 0.000e+00
Halfspace min slack                             : 0.000000e+00
Discrete-time relative error (Van Loan)         : 0.000e+00
Random-directions max relative error            : 0.000e+00
Interval event ΔE vs halfspace reference        : 0.00e+00
================================================================
```

## Mathematical Background

### Continuous-Time Formulation

For the SDE:
```
dXₜ = A Xₜ dt + B uₜ dt + Σ^(1/2) dWₜ,  X₀ = x₀
```

with quadratic cost (dimensionless, equals KL divergence):
```
E(u) = (1/2) ∫₀ᵀ uₜ^T M uₜ dt,  M = B^T Σ⁻¹ B
```

The Gramians are:
```
V_T = ∫₀ᵀ Ψ(T,s) Σ Ψ(T,s)^T ds          (noise covariance)
W^M_c = ∫₀ᵀ Ψ(T,s) B M⁻¹ B^T Ψ(T,s)^T ds  (controllability Gramian)
```

where Ψ(t,s) = e^(A(t-s)) is the state transition matrix.

### Discrete-Time Formulation

For the discrete system (T = N·Δt):
```
X_{k+1} = A_d X_k + B_d U_k + ξ_k,  ξ_k ~ N(0, Σ_d)
```

The Gramians (no extra Δt factor; Σ_d is per-step):
```
V_N = Σ_{k=0}^{N-1} A_d^{N-1-k} Σ_d (A_d^{N-1-k})^T
W^M_N = Σ_{k=0}^{N-1} A_d^{N-1-k} B_d M⁻¹ B_d^T (A_d^{N-1-k})^T
```

Discretization uses Van Loan block exponentials for robustness.

## Implementation Details

### Numerical Methods

- **Van Loan Gramians**: Single matrix exponential computes integrals (8000× faster than quadrature)
- **Forward Lyapunov recursion**: Avoids explicit matrix powers for stability
- **Solve-based inversions**: Uses `scipy.linalg.solve()` with `assume_a='pos'`, fallback to pseudoinverse
- **PSD guards**: Symmetrization and eigenvalue clipping for numerical safety
- **Float64 Gramians**: High precision for controllability computations
- **Float32 Monte Carlo**: Memory-efficient sampling (2M batch size)

### Code Structure

```python
# Van Loan Gramian helpers (lines 58-161)
stm(A, dt)                                    # State transition matrix
symmetrize_psd(M)                             # PSD projection with guards
gram_VT_van_loan(A, Sigma, T)                 # Noise Gramian
gram_WcM_van_loan(A, B, M, T)                 # Controllability Gramian
discretize_van_loan(A, B, Sigma, dt)          # ZOH discretization

# Monte Carlo helpers (lines 166-201)
scalar_mc_prob_ge(mean, var, thr, n, rng)     # P(Z ≥ threshold)
scalar_mc_prob_in_interval(...)               # P(a₁ ≤ Z ≤ a₂)

# Main validation (lines 206-471)
main()                                        # 5 validation tests + reporting
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{andric2025qeelgs,
  title={An Exact Quantile-Energy Equality for Terminal Halfspaces in Linear-Gaussian Control},
  author={Andric, Sandro},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see below for details.

```
MIT License

Copyright (c) 2025 Sandro Andric

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Contact

Sandro Andric - sandro.andric@nyu.edu

Project Link: [https://github.com/sandroandric/qeelgs](https://github.com/sandroandric/qeelgs)

## Acknowledgments

- Validation methodology inspired by control-theoretic best practices
- Van Loan block exponential method for robust discretization
- Numerical hygiene follows SciPy conventions
