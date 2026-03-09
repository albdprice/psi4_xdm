"""
Adaptive nLanE (a-nLanE) predictor.

Predicts a correction ΔW1 to the SCAN estimate of the fully interacting
limit W(1) in the nLanE adiabatic connection framework, using kernel ridge
regression (KRR) with cMBDF molecular representations.

The adaptive correction improves upon the semi-local approximation
W1 ≈ E_x^SCAN + 2*E_c^SCAN, which is the dominant source of error
in the nLanE method.

The same KRR + cMBDF infrastructure as aPBE0 is used:
  - cMBDF molecular representation (compact many-body distribution functional)
  - Local Gaussian kernel (compares atoms of same element)
  - Bayesian uncertainty for fallback to standard nLanE-SCAN

Usage:
    from .anlane_predictor import predict_dw1
    dw1, is_fallback, info = predict_dw1(molecule)

    # Apply correction:
    W1_adaptive = W1_scan + dw1

References:
    Khan, D. J. Chem. Phys. 163, 144115 (2025).  [nLanE-SCAN]
    Khan, D., Rankine, C.D., von Lilienfeld, O.A. Sci. Adv. 10, eadt7769 (2024).  [aPBE0/cMBDF]
"""

import os
import numpy as np

# Default uncertainty threshold for fallback
DEFAULT_UNCERTAINTY_THRESHOLD = 0.7

# Supported elements (same as aPBE0 training set)
SUPPORTED_ELEMENTS = {1, 6, 7, 8, 9, 16, 17}  # H, C, N, O, F, S, Cl

# Lazy-loaded model cache
_model = None


def _load_model():
    """Load the trained a-nLanE KRR model from disk.

    Model file: data/anlane_model.npz

    Expected contents:
        xtrain : (N_train, pad, 40)  - cMBDF representations of training molecules
        qtrain : (N_train,)          - atomic charges per training molecule (object array)
        alpha  : (N_train,)          - KRR regression weights
        L      : (N_train, N_train)  - Cholesky factor of kernel matrix
        sigma  : float               - Gaussian kernel bandwidth
        rconvs : (2, 5, 4, ...)      - radial convolution parameters
        aconvs : (2, 5, 4, ...)      - angular convolution parameters
    """
    global _model

    if _model is not None:
        return _model

    model_dir = os.path.join(os.path.dirname(__file__), 'data')
    model_path = os.path.join(model_dir, 'anlane_model.npz')

    if not os.path.exists(model_path):
        return None  # No trained model available yet

    data = np.load(model_path, allow_pickle=True)

    _model = {
        'xtrain': data['xtrain'],
        'qtrain': data['qtrain'],
        'alpha': data['alpha'],
        'L': data['L'],
        'sigma': float(data['sigma']),
        'convs': (data['rconvs'], data['aconvs']),
    }

    return _model


def predict_dw1(molecule, uncertainty_threshold=DEFAULT_UNCERTAINTY_THRESHOLD):
    """Predict the ΔW1 correction for a molecule.

    Uses KRR with cMBDF molecular representations to predict a correction
    to the SCAN estimate of W(1), the fully interacting limit of the
    adiabatic connection.

    Parameters
    ----------
    molecule : psi4.core.Molecule
        The molecule to predict ΔW1 for.
    uncertainty_threshold : float, optional
        Maximum allowed uncertainty before falling back to ΔW1 = 0
        (i.e., standard nLanE-SCAN). Default: 0.7.

    Returns
    -------
    dw1 : float
        Predicted correction ΔW1 (in Hartree). Add to W1_SCAN.
    is_fallback : bool
        True if prediction was not made (fallback to standard nLanE).
    info : dict or str
        If predicted: dict with 'uncertainty', 'dw1', 'raw_dw1', 'cutoff_factor'
        If fallback: string explaining the reason.
    """
    from psi4.driver import constants

    model = _load_model()

    # Fallback: no trained model
    if model is None:
        return 0.0, True, "No trained a-nLanE model found (anlane_model.npz)"

    # Get atomic numbers and coordinates
    natom = molecule.natom()

    # Single atom: no ML correction needed
    if natom <= 1:
        return 0.0, True, "Single atom — no ΔW1 correction"

    charges = np.array([molecule.Z(i) for i in range(natom)], dtype=int)
    coords = molecule.geometry().np * constants.bohr2angstroms

    # Check for unsupported elements
    unique_z = set(charges.tolist())
    unsupported = unique_z - SUPPORTED_ELEMENTS
    if unsupported:
        elem_names = [_z_to_symbol(z) for z in unsupported]
        return 0.0, True, f"Unsupported elements: {', '.join(elem_names)}"

    # Generate cMBDF representation
    from .cmbdf import generate_mbdf

    rep = generate_mbdf(
        np.array([charges]),
        np.array([coords]),
        model['convs'],
        n_atm=2.0,
        pad=100
    )[0]  # shape: (100, 40)

    # Compute kernel vector between test molecule and training set
    k = _compute_kernel_vector(
        rep, charges,
        model['xtrain'], model['qtrain'],
        model['sigma']
    )

    # Compute self-kernel for uncertainty estimation
    sk = _compute_self_kernel(rep, charges, model['sigma'])

    # Bayesian uncertainty via Cholesky solve
    from scipy.linalg import cho_solve

    alpha2 = cho_solve((model['L'], True), k)
    variance = max(0.0, sk - np.dot(k, alpha2))  # Clamp to non-negative
    # raw_uncertainty = relative posterior variance (0 = confident, 1 = uncertain)
    raw_uncertainty = (variance / sk) if sk > 0 else 1.0

    # Smooth cutoff: suppress prediction when uncertain
    cutoff = 1.0 - 0.5 * (1.0 + np.tanh(5000.0 * (raw_uncertainty - uncertainty_threshold)))

    # Predict ΔW1
    Ne = sum(charges) - molecule.molecular_charge()
    raw_dw1 = np.dot(k, model['alpha']) / Ne / 100.0  # Normalize by electron count
    dw1 = cutoff * raw_dw1

    # High uncertainty fallback
    if cutoff < 1e-6:
        return 0.0, True, f"High uncertainty ({raw_uncertainty:.4f} > {uncertainty_threshold})"

    info = {
        'uncertainty': raw_uncertainty,
        'dw1': dw1,
        'raw_dw1': raw_dw1,
        'cutoff_factor': cutoff,
    }

    return dw1, False, info


def _compute_kernel_vector(x_test, q_test, x_train, q_train, sigma):
    """Compute local Gaussian kernel vector between test and training molecules.

    Uses local kernel: only compares atoms of matching nuclear charge.

    Parameters
    ----------
    x_test : ndarray (pad, 40)
        cMBDF representation of test molecule.
    q_test : ndarray (natom,)
        Nuclear charges of test molecule.
    x_train : ndarray (N_train, pad, 40)
        Training representations.
    q_train : ndarray (N_train,) of object arrays
        Training charges.
    sigma : float
        Kernel bandwidth.

    Returns
    -------
    k : ndarray (N_train,)
        Kernel values.
    """
    n_train = len(x_train)
    k = np.zeros(n_train)

    for i in range(n_train):
        k[i] = _local_kernel_pair(
            x_test, q_test,
            x_train[i], q_train[i],
            sigma
        )

    return k


def _compute_self_kernel(x, q, sigma):
    """Compute self-kernel k(x, x) for a single molecule."""
    return _local_kernel_pair(x, q, x, q, sigma)


def _local_kernel_pair(xa, qa, xb, qb, sigma):
    """Compute local Gaussian kernel between two molecules.

    K(A, B) = sum_i sum_j exp(-||x_i^A - x_j^B||^2 / (2*sigma^2))
    where sum is only over atoms i, j with same nuclear charge.
    """
    val = 0.0
    inv_2sig2 = 1.0 / (2.0 * sigma * sigma)

    for i, zi in enumerate(qa):
        if zi == 0:
            continue
        for j, zj in enumerate(qb):
            if zj == 0:
                continue
            if zi == zj:
                diff = xa[i] - xb[j]
                val += np.exp(-np.dot(diff, diff) * inv_2sig2)

    return val


def _z_to_symbol(z):
    """Convert atomic number to element symbol."""
    symbols = {1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N',
               8: 'O', 9: 'F', 10: 'Ne', 16: 'S', 17: 'Cl'}
    return symbols.get(z, f'Z={z}')
