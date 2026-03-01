"""Tests for the aPBE0 (adaptive PBE0) functional implementation."""

import numpy as np
import pytest

import psi4

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.dft, pytest.mark.apbe0]


@pytest.fixture
def water():
    """Water molecule for testing."""
    return psi4.geometry("""
        0 1
        O  0.000000  0.000000  0.117369
        H  0.000000  0.756950 -0.469476
        H  0.000000 -0.756950 -0.469476
        symmetry c1
    """)


@pytest.fixture
def helium():
    """Single helium atom — should trigger fallback to standard PBE0."""
    return psi4.geometry("""
        0 1
        He 0.0 0.0 0.0
        symmetry c1
    """)


@pytest.fixture
def h2():
    """H2 molecule — small system for quick tests."""
    return psi4.geometry("""
        0 1
        H  0.0  0.0  0.0
        H  0.0  0.0  0.74
        symmetry c1
    """)


@pytest.fixture
def methane_qm7b():
    """Methane (CH4) from QM7b dataset — produces non-default alpha (~0.297)."""
    return psi4.geometry("""
        0 1
        C   0.00000000   0.00000000   0.00000000
        H   0.63133900   0.63133900   0.63133900
        H  -0.63133900  -0.63133900   0.63133900
        H   0.63133900  -0.63133900  -0.63133900
        H  -0.63133900   0.63133900  -0.63133900
        symmetry c1
        units angstrom
    """)


@pytest.fixture
def ethane_qm7b():
    """Ethane (C2H6) from QM7b dataset — produces non-default alpha (~0.136)."""
    return psi4.geometry("""
        0 1
        C  -0.76544800  -0.00000300  -0.00000700
        C   0.76544800   0.00000300   0.00000700
        H  -1.16401600  -0.74984300  -0.69311400
        H  -1.16411700   0.97516700  -0.30277500
        H  -1.16405000  -0.22539200   0.99591200
        H   1.16405000   0.22514700  -0.99596800
        H   1.16411700  -0.97509200   0.30301500
        H   1.16401600   0.75001300   0.69293000
        symmetry c1
        units angstrom
    """)


@pytest.fixture(autouse=True)
def set_options():
    """Set default options for aPBE0 tests."""
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


class TestAPBE0Predictor:
    """Tests for the ML prediction module."""

    def test_predict_alpha_water(self, water):
        """Test that predict_alpha returns a value for water.

        Note: Water falls back to default PBE0 (alpha=0.25) because the
        W4-17 training set yields high model uncertainty for this geometry.
        """
        from psi4.driver.procrouting.dft.apbe0_predictor import predict_alpha

        alpha, is_fallback, info = predict_alpha(water)

        # Alpha should be a float in a reasonable range
        assert isinstance(alpha, float)
        assert 0.0 < alpha < 1.0
        # Water triggers fallback due to high model uncertainty
        assert is_fallback
        assert alpha == 0.25

    def test_predict_alpha_single_atom(self, helium):
        """Test that single atoms fall back to standard PBE0."""
        from psi4.driver.procrouting.dft.apbe0_predictor import predict_alpha

        alpha, is_fallback, info = predict_alpha(helium)

        assert alpha == 0.25
        assert is_fallback
        assert isinstance(info, str)

    def test_predict_alpha_deterministic(self, water):
        """Test that predictions are deterministic."""
        from psi4.driver.procrouting.dft.apbe0_predictor import predict_alpha

        alpha1, _, _ = predict_alpha(water)
        alpha2, _, _ = predict_alpha(water)

        assert alpha1 == alpha2

    def test_predict_alpha_methane_nondefault(self, methane_qm7b):
        """Test that methane (QM7b geometry) produces a non-default alpha.

        This molecule is within the model's training domain and should
        produce a confident, non-fallback prediction (~0.297).
        """
        from psi4.driver.procrouting.dft.apbe0_predictor import predict_alpha

        alpha, is_fallback, info = predict_alpha(methane_qm7b)

        assert not is_fallback, f"Expected confident prediction, got fallback: {info}"
        assert isinstance(info, dict)
        assert info['uncertainty'] < 0.7
        assert info['cutoff_factor'] > 0.99
        # Alpha should differ meaningfully from PBE0's 0.25
        assert abs(alpha - 0.25) > 0.01
        # Validate against reference value (from standalone validation)
        assert abs(alpha - 0.297) < 0.01

    def test_predict_alpha_ethane_nondefault(self, ethane_qm7b):
        """Test that ethane (QM7b geometry) produces a non-default alpha.

        Ethane predicts alpha < 0.25, demonstrating the model can predict
        both higher and lower exchange fractions than PBE0.
        """
        from psi4.driver.procrouting.dft.apbe0_predictor import predict_alpha

        alpha, is_fallback, info = predict_alpha(ethane_qm7b)

        assert not is_fallback, f"Expected confident prediction, got fallback: {info}"
        assert isinstance(info, dict)
        assert info['uncertainty'] < 0.7
        # Alpha should be less than PBE0's 0.25
        assert alpha < 0.25
        # Validate against reference value (from standalone validation)
        assert abs(alpha - 0.136) < 0.01


class TestAPBE0Energy:
    """Tests for aPBE0 energy calculations."""

    def test_energy_runs(self, h2):
        """Test that energy('aPBE0') runs without error on H2."""
        e = psi4.energy('aPBE0', molecule=h2)
        assert isinstance(e, float)
        assert e < 0.0  # energy should be negative

    def test_energy_nondefault_alpha(self, methane_qm7b):
        """Test aPBE0 energy with a molecule that gets non-default alpha.

        Methane (QM7b geometry) should produce a different energy than PBE0
        because the predicted alpha (~0.297) differs from 0.25.
        """
        e_apbe0 = psi4.energy('aPBE0', molecule=methane_qm7b)
        predicted_alpha = psi4.core.variable('APBE0 PREDICTED ALPHA')

        psi4.core.clean()
        psi4.core.clean_variables()

        e_pbe0 = psi4.energy('PBE0', molecule=methane_qm7b)

        # Alpha should differ from 0.25
        assert abs(predicted_alpha - 0.25) > 0.01
        # Energies should therefore differ
        assert abs(e_apbe0 - e_pbe0) > 1e-6

    def test_energy_differs_from_pbe0(self, h2):
        """Test that aPBE0 gives a different energy than standard PBE0 (unless alpha~0.25)."""
        e_apbe0 = psi4.energy('aPBE0', molecule=h2)
        predicted_alpha = psi4.core.variable('APBE0 PREDICTED ALPHA')

        psi4.core.clean()
        psi4.core.clean_variables()

        e_pbe0 = psi4.energy('PBE0', molecule=h2)

        # If predicted alpha != 0.25, energies should differ
        if abs(predicted_alpha - 0.25) > 1e-4:
            assert abs(e_apbe0 - e_pbe0) > 1e-8
        else:
            assert abs(e_apbe0 - e_pbe0) < 1e-6

    def test_psi_variable_set(self, h2):
        """Test that APBE0 PREDICTED ALPHA variable is set after calculation."""
        psi4.energy('aPBE0', molecule=h2)
        alpha = psi4.core.variable('APBE0 PREDICTED ALPHA')
        assert 0.0 < alpha < 1.0


class TestAPBE0Gradient:
    """Tests for aPBE0 gradient calculations."""

    def test_gradient_runs(self, h2):
        """Test that gradient('aPBE0') runs without error on H2."""
        g = psi4.gradient('aPBE0', molecule=h2)
        assert g is not None
        # Gradient matrix should have shape (natom, 3)
        gmat = g.np
        assert gmat.shape == (2, 3)
