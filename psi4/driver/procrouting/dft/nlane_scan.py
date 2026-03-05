"""
nLanE-SCAN: Non-linear and non-empirical double hybrid density functional.

Derives exchange-correlation energy from an accurate interpolation of the
adiabatic connection (AC), bridging the weak-coupling perturbative limit with
the fully interacting limit from SCAN. Contains no fitted parameters.

The AC interpolant W_lambda = a + b*sqrt(lambda+1)/(c*lambda+1) is constrained by:
    W(0) = E_x^HF               (exact exchange)
    W'(0) = 2*E_c^MP2           (MP2 correlation slope)
    W(1) = E_x^SCAN + 2*E_c^SCAN (SCAN at physical interaction strength)

References:
    Khan, D. "SCAN based non-linear double hybrid density functional."
    J. Chem. Phys. 163, 144115 (2025).
"""

import numpy as np


def solve_abc(Exx, Ec_MP2, W1):
    """Solve for AC interpolant parameters a, b, c.

    Parameters
    ----------
    Exx : float
        HF exchange energy evaluated on KS orbitals.
    Ec_MP2 : float
        MP2 correlation energy evaluated on KS orbitals.
    W1 : float
        Estimate of W(lambda=1), typically E_x^SCAN + 2*E_c^SCAN.

    Returns
    -------
    a, b, c : float
        Parameters of the AC interpolant.
    """
    if np.abs(Ec_MP2) < 1e-6:
        # One-electron system: b=0, c=0
        return Exx, 0.0, 0.0

    alpha = (W1 - Exx) / (2 * Ec_MP2)

    c = np.sqrt(9 * alpha**2 - 16 * np.sqrt(2) * alpha + 12 * alpha + 4) - alpha + 2
    c = c / (4 * alpha)

    b = 2 * Ec_MP2 / (0.5 - c)
    a = Exx - b

    if a > W1:
        c = -c
        b = 2 * Ec_MP2 / (0.5 - c)
        a = Exx - b

    return a, b, c


def adiabatic_connection(lam, a, b, c):
    """Evaluate the AC interpolant W_lambda at a given lambda.

    W_lambda = a + b * sqrt(lambda + 1) / (c * lambda + 1)
    """
    return a + (b * np.sqrt(lam + 1)) / (c * lam + 1)


def compute_nlane_xc(Exx, Ec_MP2, W1):
    """Compute the nLanE exchange-correlation energy.

    E_xc = integral from 0 to 1 of W_lambda d(lambda), computed via
    numerical quadrature (scipy.integrate.quad) for robustness.

    Parameters
    ----------
    Exx : float
        HF exchange energy on KS orbitals.
    Ec_MP2 : float
        MP2 correlation energy on KS orbitals.
    W1 : float
        W(lambda=1) estimate, typically E_x^SCAN + 2*E_c^SCAN.

    Returns
    -------
    Exc_nlane : float
        nLanE exchange-correlation energy.
    info : dict
        Dictionary with intermediate values (a, b, c, alpha).
    """
    from scipy.integrate import quad

    a, b, c = solve_abc(Exx, Ec_MP2, W1)

    if np.abs(Ec_MP2) < 1e-6:
        # One-electron: E_xc = E_x^HF (no correlation)
        return Exx, {'a': a, 'b': b, 'c': c, 'alpha': 0.0}

    Exc, _ = quad(lambda lam: adiabatic_connection(lam, a, b, c), 0, 1, limit=1000)

    alpha_param = (W1 - Exx) / (2 * Ec_MP2)

    info = {
        'a': a,
        'b': b,
        'c': c,
        'alpha': alpha_param,
    }
    return Exc, info


def evaluate_xc_energy(scf_wfn, xc_dict, restricted=True):
    """Evaluate XC energy of a functional non-self-consistently on an existing density.

    Parameters
    ----------
    scf_wfn : psi4.core.Wavefunction
        Converged wavefunction whose density to use.
    xc_dict : dict
        Functional dictionary (e.g., {"x_functionals": {"MGGA_X_SCAN": {}}}).
    restricted : bool
        Whether the calculation is restricted (RKS) or unrestricted (UKS).

    Returns
    -------
    xc_energy : float
        The XC energy of the specified functional on the given density.
    """
    from psi4 import core
    from psi4.driver.procrouting.dft.dft_builder import build_superfunctional_from_dictionary

    npoints = core.get_option("SCF", "DFT_BLOCK_MAX_POINTS")
    ssuper, _ = build_superfunctional_from_dictionary(xc_dict, npoints, deriv=1, restricted=restricted)

    if restricted:
        vtype = "RV"
    else:
        vtype = "UV"

    vbase = core.VBase.build(scf_wfn.basisset(), ssuper, vtype)
    vbase.initialize()

    nbf = scf_wfn.basisset().nbf()

    if restricted:
        vbase.set_D([scf_wfn.Da()])
        V = core.Matrix(nbf, nbf)
        vbase.compute_V([V])
    else:
        vbase.set_D([scf_wfn.Da(), scf_wfn.Db()])
        Va = core.Matrix(nbf, nbf)
        Vb = core.Matrix(nbf, nbf)
        vbase.compute_V([Va, Vb])

    xc_energy = vbase.quadrature_values()['FUNCTIONAL']
    vbase.finalize()

    return xc_energy


# Functional dictionaries for non-self-consistent evaluation
SCAN_X_DICT = {
    "name": "SCAN_X_ONLY",
    "x_functionals": {"MGGA_X_SCAN": {}},
    "c_functionals": {},
}

SCAN_C_DICT = {
    "name": "SCAN_C_ONLY",
    "x_functionals": {},
    "c_functionals": {"MGGA_C_SCAN": {}},
}

R2SCAN_X_DICT = {
    "name": "R2SCAN_X_ONLY",
    "x_functionals": {"MGGA_X_R2SCAN": {}},
    "c_functionals": {},
}

R2SCAN_C_DICT = {
    "name": "R2SCAN_C_ONLY",
    "x_functionals": {},
    "c_functionals": {"MGGA_C_R2SCAN": {}},
}

HF_X_DICT = {
    "name": "HF_X_ONLY",
    "x_functionals": {},
    "x_hf": {"alpha": 1.0},
    "c_functionals": {},
}
