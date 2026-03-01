#!/usr/bin/env python3
"""
Benchmark aPBE0 predictions on the QM7b test set.

Compares predicted alpha values against the paper's findings:
- Khan et al. report optimal alpha typically increases for organic molecules,
  widening the HOMO-LUMO gap relative to PBE0.
- Average alpha for QM7b should be > 0.25 (organic molecules need more HF exchange).
- The model should produce confident (low-uncertainty) predictions for most
  QM7b molecules since they are organic molecules with H, C, N, O.

Usage (requires built Psi4 with aPBE0 or standalone with numpy/numba/scipy):
    python benchmark_apbe0_qm7b.py [--standalone]

With --standalone, runs without Psi4 using the validation code path.
Without --standalone, requires PYTHONPATH to include Psi4 stage/lib.
"""

import sys
import os
import argparse
import numpy as np

def run_standalone():
    """Run predictions without Psi4, using cMBDF directly."""
    from scipy.linalg import cho_solve
    from numba import njit

    WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    MODEL_PATH = os.environ.get('APBE0_MODEL_PATH',
        os.path.join(WORKSPACE, 'psi4', 'driver', 'procrouting', 'dft', 'data', 'apbe0_model.npz'))
    QM7B_PATH = os.environ.get('QM7B_DATA_PATH',
        os.path.join(WORKSPACE, '..', '14586555', 'qm7b_test_set.npz'))

    sys.path.insert(0, os.path.join(WORKSPACE, 'psi4', 'driver', 'procrouting', 'dft'))
    from cmbdf import generate_mbdf

    data = np.load(MODEL_PATH, allow_pickle=True)
    convs = (data['rconvs'], data['aconvs'])
    xtrain = data['xtrain']
    qtrain = data['qtrain']
    alpha_weights = data['alpha']
    sigma = float(data['sigma'])
    L = data['L']

    @njit
    def local_kernel(A, B, Q1, Q2, sigma):
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

    qm7b = np.load(QM7B_PATH, allow_pickle=True)
    n_mol = len(qm7b['charges'])

    print(f"Running aPBE0 predictions on {n_mol} QM7b molecules (standalone mode)...\n")
    print("Warming up numba JIT...")
    dummy = np.zeros((2, 40))
    dummy_q = np.array([1.0, 1.0])
    _ = local_kernel(dummy, dummy, dummy_q, dummy_q, 0.2)
    print("JIT ready.\n")

    alphas = np.zeros(n_mol)
    uncertainties = np.zeros(n_mol)
    cutoffs = np.zeros(n_mol)
    n_fallback = 0

    for i in range(n_mol):
        charges = qm7b['charges'][i].astype(np.float64)
        coords = qm7b['coordinates'][i].astype(np.float64)

        rep = generate_mbdf(np.array([charges]), np.array([coords]), convs, n_atm=2.0, pad=100)[0]
        Ne = np.sum(charges)

        k = np.zeros(len(qtrain))
        for j in range(len(qtrain)):
            k[j] = local_kernel(xtrain[j], rep, qtrain[j], charges, sigma)
        sk = local_kernel(rep, rep, charges, charges, sigma)

        alpha2 = cho_solve((L, True), k)
        variance = sk - np.dot(k, alpha2)
        unc = 1.0 - (variance / sk) if sk > 0 else 0.0
        cutoff = 1.0 - (0.5 * (1.0 + np.tanh(5000.0 * (unc - 0.7))))

        delta = (np.dot(k, alpha_weights) / Ne) / 100.0
        alpha = 0.25 + (cutoff * delta)

        alphas[i] = alpha
        uncertainties[i] = unc
        cutoffs[i] = cutoff

        if abs(cutoff) < 1e-6:
            n_fallback += 1

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_mol} done...")

    return alphas, uncertainties, cutoffs, n_fallback, qm7b


def run_with_psi4():
    """Run predictions using Psi4's predict_alpha."""
    import psi4
    from psi4.driver.procrouting.dft.apbe0_predictor import predict_alpha

    QM7B_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '14586555', 'qm7b_test_set.npz')
    qm7b = np.load(QM7B_PATH, allow_pickle=True)
    n_mol = len(qm7b['charges'])

    psi4.set_output_file("/dev/null", False)
    print(f"Running aPBE0 predictions on {n_mol} QM7b molecules (Psi4 mode)...\n")

    alphas = np.zeros(n_mol)
    uncertainties = np.zeros(n_mol)
    cutoffs = np.zeros(n_mol)
    n_fallback = 0

    elem_map = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 16: 'S', 17: 'Cl'}

    for i in range(n_mol):
        charges = qm7b['charges'][i]
        coords = qm7b['coordinates'][i]

        geom_str = "0 1\n"
        for j in range(len(charges)):
            sym = elem_map.get(int(charges[j]), f'X{charges[j]}')
            geom_str += f"{sym} {coords[j][0]:.8f} {coords[j][1]:.8f} {coords[j][2]:.8f}\n"
        geom_str += "symmetry c1\nunits angstrom\n"

        mol = psi4.geometry(geom_str)
        alpha, is_fallback, info = predict_alpha(mol)

        alphas[i] = alpha
        if is_fallback:
            n_fallback += 1
            uncertainties[i] = -1.0
            cutoffs[i] = 0.0
        else:
            uncertainties[i] = info['uncertainty']
            cutoffs[i] = info['cutoff_factor']

        psi4.core.clean()

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_mol} done...")

    return alphas, uncertainties, cutoffs, n_fallback, qm7b


def report(alphas, uncertainties, cutoffs, n_fallback, qm7b):
    """Print benchmark report."""
    n_mol = len(alphas)
    n_confident = n_mol - n_fallback
    confident_mask = cutoffs > 0.5

    print("\n" + "=" * 70)
    print("aPBE0 QM7b Benchmark Results")
    print("=" * 70)

    print(f"\nMolecules: {n_mol}")
    print(f"Confident predictions: {n_confident} ({100*n_confident/n_mol:.0f}%)")
    print(f"Fallbacks to PBE0: {n_fallback} ({100*n_fallback/n_mol:.0f}%)")

    print(f"\n--- Alpha statistics (all molecules) ---")
    print(f"  Mean alpha:   {alphas.mean():.4f}")
    print(f"  Median alpha: {np.median(alphas):.4f}")
    print(f"  Min alpha:    {alphas.min():.4f}")
    print(f"  Max alpha:    {alphas.max():.4f}")
    print(f"  Std alpha:    {alphas.std():.4f}")

    if n_confident > 0:
        conf_alphas = alphas[confident_mask]
        print(f"\n--- Alpha statistics (confident predictions only) ---")
        print(f"  Mean alpha:   {conf_alphas.mean():.4f}")
        print(f"  Median alpha: {np.median(conf_alphas):.4f}")
        print(f"  Min alpha:    {conf_alphas.min():.4f}")
        print(f"  Max alpha:    {conf_alphas.max():.4f}")
        print(f"  Std alpha:    {conf_alphas.std():.4f}")

        conf_unc = uncertainties[confident_mask]
        print(f"\n--- Uncertainty statistics (confident predictions) ---")
        print(f"  Mean uncertainty: {conf_unc.mean():.4f}")
        print(f"  Max uncertainty:  {conf_unc.max():.4f}")

    print(f"\n--- Alpha distribution ---")
    for lo, hi in [(0.0, 0.15), (0.15, 0.25), (0.25, 0.35), (0.35, 0.50), (0.50, 1.0)]:
        count = ((alphas >= lo) & (alphas < hi)).sum()
        bar = '#' * count
        print(f"  [{lo:.2f}, {hi:.2f}): {count:3d}  {bar}")

    print(f"\n--- Comparison to paper expectations ---")
    print(f"  Paper: optimal alpha for organic molecules typically > 0.25")
    if n_confident > 0:
        above = (conf_alphas > 0.25).sum()
        below = (conf_alphas < 0.25).sum()
        print(f"  Our results: {above} above 0.25, {below} below 0.25 (of {n_confident} confident)")
        print(f"  Mean confident alpha: {conf_alphas.mean():.4f} (paper reports ~0.42 average for training set)")

    # Per-molecule details
    print(f"\n--- Per-molecule details (first 20) ---")
    print(f"{'#':>3s} {'natom':>5s} {'alpha':>8s} {'unc':>8s} {'cutoff':>8s} {'elements'}")
    print("-" * 60)
    for i in range(min(20, n_mol)):
        elems = qm7b['elements'][i]
        elem_str = ''.join(elems)
        print(f"{i:3d} {len(elems):5d} {alphas[i]:8.4f} {uncertainties[i]:8.4f} {cutoffs[i]:8.4f}  {elem_str}")

    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark aPBE0 on QM7b test set')
    parser.add_argument('--standalone', action='store_true',
                        help='Run without Psi4 (uses cMBDF directly)')
    args = parser.parse_args()

    if args.standalone:
        alphas, uncertainties, cutoffs, n_fallback, qm7b = run_standalone()
    else:
        alphas, uncertainties, cutoffs, n_fallback, qm7b = run_with_psi4()

    report(alphas, uncertainties, cutoffs, n_fallback, qm7b)
