"""Unit test: the non-Limber early-exit tolerance, the FKEM hand-off
and the noise of the low-rank (SVD) engine.

The non-Limber real-space C_l switch a lens bin (gg) or a lens-source
pair (gs) to Limber once they agree with Limber within a tolerance; the
switch multipole moves with the parameters and puts steps into
finite-difference derivatives. init_nonlimber_tolerance sets the
tolerances (likelihood keys nonlimber_tolerance_gg = 0.002 and
nonlimber_tolerance_gs = 0.01 by default; 0 = no early exit). The
shared factory (cosmolike_core/cocoa_testing_suites.py,
nonlimber_tolerance_suite) documents the checks. On example2:

    This project's 6x2pt data vector is in real space: w(theta)
    (and gamma_t where adopt_limber_gs = 0) runs the non-Limber
    path with the early exit, so tests 3 and 4 measure what the
    tolerance does here.

The stencil noise is the whitened fourth difference of five evaluations
(cosmolike_notebook_utils.fisher.get_stencil_noise, step h = 0.025) of
the w0, wa and DES_B1_6 columns.

Measured on 2026-11-18 (OMP 4), then bounded:
  - test 3: delta^T C^-1 delta = 1.91e-4 (w(theta) only); bound 5e-4;
  - test 4: sigma (w0, wa, bias) = 1.58e-3, 8.5e-5, 6e-14 at the
    default tolerances and 2.77e-4, 8.5e-5, 5e-14 at 0; bound 2e-3;
  - test 5: FKEM caps 150/200 give w0 2.77e-4/2.81e-4 and
    wa 8.55e-5/8.55e-5; margin 0.1;
  - test 6: SVD/FKEM ratio at most 1.09 (w0, cap 200); margin 0.2.

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/desy1xplanck/tests/data_vector/test_nonlimber_tolerance.py -s
"""

import os

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load, so
# this must run before any cobaya/cosmolike import in the process.
os.environ["OMP_NUM_THREADS"] = "4"

import sys
import unittest

# The tests folder is not a package; put it on the import path so the
# shared harness resolves no matter where pytest was launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
import cocoa_test_utils as u

# The bounds below come from this project's measurement (the values
# are in the comment of the module docstring above).
TestNonLimberTolerance = u.nonlimber_tolerance_suite(
    u, name="TestNonLimberTolerance", example="example2",
    bias_parameter="DES_B1_6",
    expected_defaults=(0.002, 0.01),
    change_bound=0.0005,
    noise_bound=0.002,
    handoff_margin=0.1,
    svd_rule='not_larger',
    svd_margin=0.2)

# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
