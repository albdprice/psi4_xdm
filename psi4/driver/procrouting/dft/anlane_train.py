"""
Training utilities for adaptive nLanE (a-nLanE).

Generates training data and trains a KRR model to predict ΔW1 corrections
for the nLanE adiabatic connection framework.

Training workflow:
    1. Compute nLanE ingredients (Exx, Ec_MP2, W1_SCAN) for training molecules
    2. Obtain reference W1 from high-level theory (CCSD(T)/CBS or similar)
    3. Compute ΔW1 = W1_ref - W1_SCAN for each molecule
    4. Generate cMBDF representations
    5. Train KRR model: ΔW1 = f(cMBDF)
    6. Save model to anlane_model.npz

Usage:
    python -m psi4.driver.procrouting.dft.anlane_train \\
        --training_data training_data.npz \\
        --output anlane_model.npz \\
        --sigma 10.0 --lambda_reg 1e-10

Or from Python:
    from psi4.driver.procrouting.dft.anlane_train import train_model
    train_model(training_data_path, output_path, sigma=10.0, lambda_reg=1e-10)
"""

import numpy as np


def compute_training_targets(molecules_xyz, basis='def2-QZVPP'):
    """Compute nLanE ingredients for training molecules.

    For each molecule, runs nLanE-SCAN and extracts:
        - W1_SCAN = E_x^SCAN + 2*E_c^SCAN
        - Exx, Ec_MP2 (for reference)
        - Molecular geometry and charges

    The reference W1_ref must be obtained separately (e.g., from CCSD(T)).

    Parameters
    ----------
    molecules_xyz : list of str
        List of XYZ file paths or geometry strings.
    basis : str
        Basis set to use.

    Returns
    -------
    data : dict
        Dictionary with keys:
            'W1_scan': ndarray (N,) - SCAN W1 values
            'Exx': ndarray (N,) - HF exchange energies
            'Ec_MP2': ndarray (N,) - MP2 correlation energies
            'charges': list of ndarray - atomic charges per molecule
            'coords': list of ndarray - coordinates per molecule (Angstrom)
    """
    import psi4
    from psi4.driver import constants

    data = {
        'W1_scan': [], 'Exx': [], 'Ec_MP2': [],
        'charges': [], 'coords': [],
    }

    for i, xyz in enumerate(molecules_xyz):
        print(f"  [{i+1}/{len(molecules_xyz)}] Computing...", flush=True)

        psi4.core.clean()
        psi4.core.clean_options()
        psi4.core.clean_variables()

        mol = psi4.geometry(xyz)
        psi4.set_options({
            'basis': basis,
            'scf_type': 'df',
            'dft_spherical_points': 590,
            'dft_radial_points': 99,
        })

        e = psi4.energy('nlane-scan')

        Exx = psi4.core.variable('NLANE HF EXCHANGE ENERGY')
        Ec_MP2 = psi4.core.variable('NLANE MP2 CORRELATION ENERGY')
        a = psi4.core.variable('NLANE AC PARAMETER A')
        b = psi4.core.variable('NLANE AC PARAMETER B')
        c = psi4.core.variable('NLANE AC PARAMETER C')

        # Reconstruct W1 from parameters
        W1 = a + b * np.sqrt(2) / (c + 1)

        data['W1_scan'].append(W1)
        data['Exx'].append(Exx)
        data['Ec_MP2'].append(Ec_MP2)

        natom = mol.natom()
        charges = np.array([mol.Z(i) for i in range(natom)], dtype=int)
        coords = mol.geometry().np * constants.bohr2angstroms
        data['charges'].append(charges)
        data['coords'].append(coords)

    for key in ['W1_scan', 'Exx', 'Ec_MP2']:
        data[key] = np.array(data[key])

    return data


def generate_representations(charges_list, coords_list, convs, pad=100):
    """Generate cMBDF representations for a set of molecules.

    Parameters
    ----------
    charges_list : list of ndarray
        Atomic charges per molecule.
    coords_list : list of ndarray
        Coordinates per molecule (Angstrom).
    convs : ndarray
        Convolution parameters (from aPBE0 model or trained fresh).
    pad : int
        Padding size for representations.

    Returns
    -------
    reps : ndarray (N, pad, 40)
        cMBDF representations.
    """
    from .cmbdf import generate_mbdf

    reps = generate_mbdf(
        np.array(charges_list, dtype=object),
        np.array(coords_list, dtype=object),
        convs,
        n_atm=2.0,
        pad=pad,
    )

    return reps


def train_model(training_data, output_path, sigma=10.0, lambda_reg=1e-10,
                convs=None, pad=100):
    """Train a KRR model to predict ΔW1 corrections.

    Parameters
    ----------
    training_data : dict or str
        Training data dictionary or path to .npz file.
        Must contain:
            'charges': list of atomic charges per molecule
            'coords': list of coordinates per molecule (Angstrom)
            'dW1': ndarray (N,) - target ΔW1 values (Hartree)
        Optional:
            'Ne': ndarray (N,) - electron counts for normalization
    output_path : str
        Path to save the trained model (.npz).
    sigma : float
        Gaussian kernel bandwidth.
    lambda_reg : float
        Regularization parameter.
    convs : ndarray, optional
        Convolution parameters. If None, loads from aPBE0 model.
    pad : int
        Padding size for representations.

    Returns
    -------
    model : dict
        Trained model dictionary.
    """
    from scipy.linalg import cho_factor, cho_solve
    import os

    # Load training data
    if isinstance(training_data, str):
        data = np.load(training_data, allow_pickle=True)
        training_data = {k: data[k] for k in data.files}

    charges = training_data['charges']
    coords = training_data['coords']
    dW1_targets = training_data['dW1']
    N = len(dW1_targets)

    # Electron counts for normalization
    if 'Ne' in training_data:
        Ne = training_data['Ne']
    else:
        Ne = np.array([sum(q) for q in charges], dtype=float)

    # Normalized targets (like aPBE0)
    y = dW1_targets * Ne * 100.0

    # Load convolution parameters
    if convs is None:
        apbe0_model_path = os.path.join(os.path.dirname(__file__), 'data', 'apbe0_model.npz')
        if os.path.exists(apbe0_model_path):
            apbe0_data = np.load(apbe0_model_path, allow_pickle=True)
            convs = (apbe0_data['rconvs'], apbe0_data['aconvs'])
        else:
            raise FileNotFoundError("No convolution parameters found. Provide convs or apbe0_model.npz.")

    # Generate representations
    print(f"Generating cMBDF representations for {N} molecules...", flush=True)
    reps = generate_representations(charges, coords, convs, pad=pad)

    # Build kernel matrix
    print(f"Building {N}x{N} kernel matrix (sigma={sigma})...", flush=True)
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            kij = _local_kernel(reps[i], charges[i], reps[j], charges[j], sigma)
            K[i, j] = kij
            K[j, i] = kij

    # Regularize and solve
    K_reg = K + lambda_reg * np.eye(N)
    L = cho_factor(K_reg)
    alpha = cho_solve(L, y)

    # Cholesky factor for uncertainty estimation
    L_lower = np.linalg.cholesky(K_reg)

    # Extract convolution components
    n_conv = convs.shape[0] // 2
    rconvs = convs[:n_conv]
    aconvs = convs[n_conv:]

    # Save model
    model = {
        'xtrain': reps,
        'qtrain': np.array(charges, dtype=object),
        'alpha': alpha,
        'L': L_lower,
        'sigma': sigma,
        'rconvs': rconvs,
        'aconvs': aconvs,
    }

    np.savez(output_path, **model)
    print(f"Model saved to {output_path}", flush=True)
    print(f"  Training set: {N} molecules", flush=True)
    print(f"  Sigma: {sigma}", flush=True)
    print(f"  Lambda: {lambda_reg}", flush=True)

    return model


def _local_kernel(xa, qa, xb, qb, sigma):
    """Local Gaussian kernel between two molecules."""
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
                val += np.dot(diff, diff) * (-inv_2sig2)
    # Exp of sum for numerical stability
    return np.exp(val) if val > -500 else 0.0
