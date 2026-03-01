"""
aPBE0 (adaptive PBE0) exchange fraction predictor.

Uses a kernel ridge regression (KRR) model trained on CCSD(T) atomization
energies to predict the optimal HF exact-exchange fraction for a given molecule.
The model uses cMBDF molecular representations and local Gaussian kernels.

For single atoms or high-uncertainty predictions, falls back to the standard
PBE0 value of alpha = 0.25.

References:
    Khan, D., Price, A. J. A., Ach, M. L., Trottier, O., & von Lilienfeld, O. A.
    "Adaptive hybrid density functionals." Sci. Adv. (2024).
"""

import os
import numpy as np

# Lazy-loaded model data (loaded on first call to predict_alpha)
_model = None
_FIRST_JIT_CALL = True

# Default PBE0 exchange fraction
DEFAULT_ALPHA = 0.25

# Uncertainty threshold for fallback to default PBE0
DEFAULT_UNCERTAINTY_THRESHOLD = 0.7


def _load_model():
    """Load the trained aPBE0 KRR model from the data directory."""
    global _model
    model_path = os.path.join(os.path.dirname(__file__), 'data', 'apbe0_model.npz')
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"aPBE0 model weights not found at {model_path}. "
            "Ensure the model file is installed with Psi4."
        )
    data = np.load(model_path, allow_pickle=True)
    _model = {
        'convs': (data['rconvs'], data['aconvs']),
        'xtrain': data['xtrain'],
        'qtrain': data['qtrain'],
        'alpha': data['alpha'],
        'sigma': float(data['sigma']),
        'L': data['L'],
    }


def _local_kernel(A, B, Q1, Q2, sigma):
    """Compute local Gaussian kernel between two molecular representations.

    Only compares atoms of the same nuclear charge.
    Uses numba JIT for performance.
    """
    from numba import njit

    @njit(parallel=True)
    def _kernel_impl(A, B, Q1, Q2, sigma):
        n1, n2 = len(Q1), len(Q2)
        K = 0.0
        for i in range(n1):
            k = 0.0
            for j in range(n2):
                if Q1[i] == Q2[j]:
                    dist = np.linalg.norm(A[i] - B[j])**2
                    k += np.exp(-dist / (2 * sigma**2))
            K += k
        return K

    return _kernel_impl(A, B, Q1, Q2, sigma)


def _compute_kernel_vector(xtrain, xtest, qtrain, qtest, sigma):
    """Compute kernel vector between test molecule and all training molecules."""
    n_train = len(qtrain)
    k = np.zeros(n_train)
    for i in range(n_train):
        k[i] = _local_kernel(xtrain[i], xtest, qtrain[i], qtest, sigma)
    return k


def predict_alpha(molecule, uncertainty_threshold=DEFAULT_UNCERTAINTY_THRESHOLD):
    """Predict the optimal HF exact-exchange fraction for a molecule.

    Parameters
    ----------
    molecule : psi4.core.Molecule
        Psi4 molecule object.
    uncertainty_threshold : float
        Uncertainty threshold (x0) for fallback to standard PBE0.
        Default is 0.7.

    Returns
    -------
    alpha : float
        Predicted HF exchange fraction.
    is_fallback : bool
        True if the prediction fell back to standard PBE0 (alpha=0.25).
    info : dict or str
        If not fallback: dict with 'uncertainty', 'delta', 'raw_uncertainty' keys.
        If fallback: string describing the reason.
    """
    global _model, _FIRST_JIT_CALL

    from psi4.driver import constants

    # Load model on first call
    if _model is None:
        _load_model()

    natom = molecule.natom()

    # Single atoms always get default PBE0
    if natom <= 1:
        return DEFAULT_ALPHA, True, "Single atom — using standard PBE0"

    # Extract atomic numbers and coordinates from Psi4 molecule
    charges = np.array([int(molecule.Z(i)) for i in range(natom)])
    # Psi4 stores geometry in Bohr; cMBDF expects Angstrom
    coords = molecule.geometry().np * constants.bohr2angstroms
    mol_charge = int(molecule.molecular_charge())

    # Print a message on first numba JIT compilation
    if _FIRST_JIT_CALL:
        try:
            from psi4 import core
            core.print_out("    aPBE0: Compiling ML kernels (first call only)...\n")
        except ImportError:
            pass
        _FIRST_JIT_CALL = False

    # Generate cMBDF representation
    from .cmbdf import generate_mbdf
    rep = generate_mbdf(
        np.array([charges]),
        np.array([coords]),
        _model['convs'],
        n_atm=2.0,
        pad=100
    )[0]

    # Total number of electrons
    Ne = np.sum(charges) - mol_charge

    sigma = _model['sigma']
    xtrain = _model['xtrain']
    qtrain = _model['qtrain']

    # Compute query kernel vector
    k = _compute_kernel_vector(xtrain, rep, qtrain, charges, sigma)

    # Compute self-kernel for uncertainty
    sk = _local_kernel(rep, rep, charges, charges, sigma)

    # Compute Bayesian uncertainty
    from scipy.linalg import cho_solve
    alpha2 = cho_solve((_model['L'], True), k)
    variance = sk - np.dot(k, alpha2)
    raw_uncertainty = 1.0 - (variance / sk) if sk > 0 else 0.0

    # Smooth cutoff: tanh sigmoid that suppresses prediction when uncertain
    cutoff = 1.0 - (0.5 * (1.0 + np.tanh(5000.0 * (raw_uncertainty - uncertainty_threshold))))

    # Predicted deviation from PBE0 fraction, normalized per electron
    delta = (np.dot(k, _model['alpha']) / Ne) / 100.0

    # Final alpha
    alpha = DEFAULT_ALPHA + (cutoff * delta)

    # Determine if we effectively fell back
    is_fallback = abs(cutoff) < 1e-6

    if is_fallback:
        return DEFAULT_ALPHA, True, f"High model uncertainty ({raw_uncertainty:.4f}) — using standard PBE0"
    else:
        info = {
            'uncertainty': raw_uncertainty,
            'delta': cutoff * delta,
            'raw_delta': delta,
            'cutoff_factor': cutoff,
        }
        return float(alpha), False, info
