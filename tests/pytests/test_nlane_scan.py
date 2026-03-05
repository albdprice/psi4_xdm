"""Tests for the nLanE-SCAN (non-linear non-empirical double hybrid) functional.

References:
    Khan, D. J. Chem. Phys. 163, 144115 (2025).
"""

import numpy as np
import pytest

import psi4

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.dft]


@pytest.fixture
def h2():
    """H2 molecule for quick tests."""
    return psi4.geometry("""
        0 1
        H  0.0  0.0  0.0
        H  0.0  0.0  0.74
        symmetry c1
    """)


@pytest.fixture
def he():
    """Helium atom — one-electron-pair system."""
    return psi4.geometry("""
        0 1
        He 0.0 0.0 0.0
        symmetry c1
    """)


@pytest.fixture(autouse=True)
def set_options():
    """Set default options for nLanE-SCAN tests."""
    psi4.set_options({
        'basis': 'cc-pvdz',
        'e_convergence': 8,
        'd_convergence': 8,
        'scf_type': 'df',
    })
    yield
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()


class TestNLanEMath:
    """Tests for the core mathematical functions."""

    def test_solve_abc_one_electron(self):
        """Test that one-electron systems give b=0, c=0."""
        from psi4.driver.procrouting.dft.nlane_scan import solve_abc

        a, b, c = solve_abc(Exx=-0.5, Ec_MP2=0.0, W1=-0.5)
        assert a == -0.5
        assert b == 0.0
        assert c == 0.0

    def test_solve_abc_constraints(self):
        """Test that solve_abc satisfies the three AC constraints."""
        from psi4.driver.procrouting.dft.nlane_scan import solve_abc, adiabatic_connection

        Exx = -5.0
        Ec_MP2 = -0.3
        W1 = -5.8

        a, b, c = solve_abc(Exx, Ec_MP2, W1)

        # Constraint 1: W(0) = E_x^HF
        W0 = adiabatic_connection(0.0, a, b, c)
        assert abs(W0 - Exx) < 1e-10, f"W(0)={W0} != Exx={Exx}"

        # Constraint 2: W'(0) = 2*E_c^MP2
        # From Eq. 8: b*(1/2 - c) = 2*E_c^MP2
        slope_constraint = b * (0.5 - c)
        assert abs(slope_constraint - 2 * Ec_MP2) < 1e-10

        # Constraint 3: W(1) = W1
        W1_check = adiabatic_connection(1.0, a, b, c)
        assert abs(W1_check - W1) < 1e-10, f"W(1)={W1_check} != W1={W1}"

    def test_compute_nlane_xc_one_electron(self):
        """Test that one-electron systems give E_xc = E_x^HF."""
        from psi4.driver.procrouting.dft.nlane_scan import compute_nlane_xc

        Exc, info = compute_nlane_xc(Exx=-0.5, Ec_MP2=0.0, W1=-0.5)
        assert abs(Exc - (-0.5)) < 1e-10

    def test_numerical_integration_consistency(self):
        """Test that compute_nlane_xc gives energy between E_x^HF and W1."""
        from psi4.driver.procrouting.dft.nlane_scan import compute_nlane_xc

        Exx = -5.0
        Ec_MP2 = -0.3
        W1 = -5.8

        Exc, info = compute_nlane_xc(Exx, Ec_MP2, W1)

        # The integral of W_lambda from 0 to 1 should be between W(0) and W(1)
        # since the interpolant is monotonic for well-behaved systems
        assert Exc < Exx, f"Exc={Exc} should be below W(0)={Exx}"
        assert Exc > W1, f"Exc={Exc} should be above W(1)={W1}"


class TestNLanEEnergy:
    """Tests for nLanE-SCAN energy calculations."""

    def test_energy_runs_h2(self, h2):
        """Test that energy('nLanE-SCAN') runs without error on H2."""
        e = psi4.energy('nLanE-SCAN', molecule=h2)
        assert isinstance(e, float)
        assert e < 0.0

    def test_energy_runs_he(self, he):
        """Test nLanE-SCAN on helium (one-electron pair, E_c^MP2 ~ 0)."""
        # He has very small MP2 correlation — tests the near-one-electron path
        e = psi4.energy('nLanE-SCAN', molecule=he)
        assert isinstance(e, float)
        assert e < 0.0

    def test_energy_differs_from_scan(self, h2):
        """Test that nLanE-SCAN gives different energy than pure r2SCAN."""
        e_nlane = psi4.energy('nLanE-SCAN', molecule=h2)

        psi4.core.clean()
        psi4.core.clean_variables()

        e_r2scan = psi4.energy('r2SCAN', molecule=h2)

        # nLanE-SCAN includes HF exchange and MP2, so should differ
        assert abs(e_nlane - e_r2scan) > 1e-4

    def test_psi_variables_set(self, h2):
        """Test that nLanE variables are set after calculation."""
        psi4.energy('nLanE-SCAN', molecule=h2)

        assert psi4.core.variable('NLANE-SCAN TOTAL ENERGY') != 0.0
        assert psi4.core.variable('NLANE XC ENERGY') != 0.0
        assert psi4.core.variable('NLANE HF EXCHANGE ENERGY') != 0.0
        assert psi4.core.variable('NLANE MP2 CORRELATION ENERGY') != 0.0
        assert psi4.core.variable('NLANE AC PARAMETER C') != 0.0

    def test_ac_parameters_reasonable(self, h2):
        """Test that the AC parameters are physically reasonable."""
        psi4.energy('nLanE-SCAN', molecule=h2)

        a = psi4.core.variable('NLANE AC PARAMETER A')
        c = psi4.core.variable('NLANE AC PARAMETER C')

        # a should be negative (exchange energy is negative)
        assert a < 0.0
        # c should be positive and finite
        assert 0 < abs(c) < 100
