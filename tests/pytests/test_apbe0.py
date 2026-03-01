"""Tests for the aPBE0 (adaptive PBE0) functional implementation."""

import numpy as np
import pytest

import psi4

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.dft]


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


class TestAPBE0Energy:
    """Tests for aPBE0 energy calculations."""

    def test_energy_runs(self, h2):
        """Test that energy('aPBE0') runs without error on H2."""
        e = psi4.energy('aPBE0', molecule=h2)
        assert isinstance(e, float)
        assert e < 0.0  # energy should be negative

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
