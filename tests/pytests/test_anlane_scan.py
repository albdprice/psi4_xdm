"""Tests for adaptive nLanE-SCAN (a-nLanE-SCAN)."""

import pytest
import psi4


@pytest.fixture(autouse=True)
def clean_psi4():
    """Reset Psi4 state before each test."""
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    yield
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()


def test_anlane_fallback_no_model():
    """Without a trained model, a-nLanE-SCAN should fall back to standard nLanE-SCAN."""
    psi4.set_options({
        'basis': 'cc-pvdz',
        'scf_type': 'df',
        'e_convergence': 1e-8,
        'd_convergence': 1e-7,
    })

    h2 = psi4.geometry("""
        0 1
        H 0 0 0
        H 0 0 0.74
        symmetry c1
    """)

    # a-nLanE-SCAN without trained model should equal nLanE-SCAN
    e_anlane = psi4.energy('anlane-scan', molecule=h2)
    dw1 = psi4.core.variable('NLANE DW1 ML')

    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    psi4.set_options({
        'basis': 'cc-pvdz',
        'scf_type': 'df',
        'e_convergence': 1e-8,
        'd_convergence': 1e-7,
    })

    e_nlane = psi4.energy('nlane-scan', molecule=h2)

    assert dw1 == 0.0, "ΔW1 should be 0 without trained model"
    assert abs(e_anlane - e_nlane) < 1e-10, \
        f"a-nLanE should equal nLanE without model: {e_anlane} vs {e_nlane}"


def test_anlane_alias():
    """Both 'anlane-scan' and 'a-nlane-scan' should work."""
    psi4.set_options({
        'basis': 'cc-pvdz',
        'scf_type': 'df',
    })

    he = psi4.geometry("He")

    e1 = psi4.energy('anlane-scan', molecule=he)

    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    psi4.set_options({
        'basis': 'cc-pvdz',
        'scf_type': 'df',
    })

    e2 = psi4.energy('a-nlane-scan', molecule=he)

    assert abs(e1 - e2) < 1e-12, "Both aliases should give same energy"


def test_anlane_stores_w1_variables():
    """a-nLanE should store W1 SCAN, W1 adaptive, and DW1 ML variables."""
    psi4.set_options({
        'basis': 'cc-pvdz',
        'scf_type': 'df',
    })

    h2 = psi4.geometry("""
        0 1
        H 0 0 0
        H 0 0 0.74
        symmetry c1
    """)

    psi4.energy('anlane-scan', molecule=h2)

    # Check all expected variables exist
    assert psi4.core.variable('NLANE W1 SCAN') != 0.0
    assert psi4.core.variable('NLANE DW1 ML') == 0.0  # No model
    assert psi4.core.variable('NLANE W1 ADAPTIVE') != 0.0
    # Without model, W1 adaptive should equal W1 SCAN
    assert abs(psi4.core.variable('NLANE W1 SCAN') - psi4.core.variable('NLANE W1 ADAPTIVE')) < 1e-12


def test_nlane_stores_w1_energy():
    """nLanE-SCAN should now store NLANE W1 ENERGY variable."""
    psi4.set_options({
        'basis': 'cc-pvdz',
        'scf_type': 'df',
    })

    h2 = psi4.geometry("""
        0 1
        H 0 0 0
        H 0 0 0.74
        symmetry c1
    """)

    psi4.energy('nlane-scan', molecule=h2)

    w1 = psi4.core.variable('NLANE W1 ENERGY')
    assert w1 != 0.0, "NLANE W1 ENERGY should be set"
    assert w1 < 0.0, "W1 should be negative (exchange + 2*correlation)"


def test_anlane_predictor_fallback_single_atom():
    """Predictor should fall back for single atoms (or no model)."""
    from psi4.driver.procrouting.dft.anlane_predictor import predict_dw1

    he = psi4.geometry("He")
    dw1, is_fallback, info = predict_dw1(he)

    assert is_fallback is True
    assert dw1 == 0.0
    # Without a trained model, falls back before checking atom count
    assert "Single atom" in info or "No trained" in info
