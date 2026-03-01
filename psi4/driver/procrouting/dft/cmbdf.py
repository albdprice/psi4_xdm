"""
cMBDF (compact Many-Body Distribution Functionals) molecular representation.

Ported from the aPBE0 reference implementation by Khan et al.
Generates local atomic descriptors for use in the aPBE0 adaptive hybrid functional.

References:
    Khan et al., J. Chem. Phys. 159, 034106 (2023) — MBDF representation
    Khan et al., Sci. Adv. (2024) — aPBE0 adaptive functional
"""

import numpy as np
import numba as nb
from numpy import einsum
from scipy.signal import fftconvolve
from scipy.fft import next_fast_len


@nb.jit(nopython=True)
def hermite_polynomial(x, degree, a=1):
    if degree == 1:
        return -2*a*x
    elif degree == 2:
        x1 = (a*x)**2
        return 4*x1 - 2*a
    elif degree == 3:
        x1 = (a*x)**3
        return -8*x1 + 12*a*x
    elif degree == 4:
        x1 = (a*x)**4
        x2 = (a*x)**2
        return 16*x1 - 48*x2 + 12*a**2
    elif degree == 5:
        x1 = (a*x)**5
        x2 = (a*x)**3
        return -32*x1 + 160*x2 - 120*(a*x)


@nb.jit(nopython=True)
def chebyshev_polynomial(x, degree):
    if degree == 1:
        return x
    elif degree == 2:
        return 2*(x**2) - 1
    elif degree == 3:
        return 4*(x**3) - 3*x
    elif degree == 4:
        return 8*((x**4) - (x**2)) + 1
    elif degree == 5:
        return 16*(x**5) - 20*(x**3) + 5*x


@nb.jit(nopython=True)
def fcut(Rij, rcut):
    return 0.5*(np.cos((np.pi*Rij)/rcut)+1)


@nb.jit(nopython=True)
def generate_data(size, charges, coods, rconvs, aconvs, cutoff_r=12.0, n_atm=2.0):
    rconvs, aconvs = rconvs[0], aconvs[0]
    m1, n1 = rconvs.shape[0], rconvs.shape[1]
    m2, n2 = aconvs.shape[0], aconvs.shape[1]
    nrs = m1*n1
    nAs = m2*n2
    rstep = cutoff_r/rconvs.shape[-1]
    astep = np.pi/aconvs.shape[-1]
    twob = np.zeros((size, size, nrs))
    threeb = np.zeros((size, size, size, nAs))

    for i in range(size):
        z, atom = charges[i], coods[i]

        for j in range(i+1, size):
            rij = atom - coods[j]
            rij_norm = np.linalg.norm(rij)

            if rij_norm != 0 and rij_norm < cutoff_r:
                z2 = charges[j]
                ind = int(rij_norm/rstep)
                pref = np.sqrt(z*z2)

                id2 = 0
                for i1 in range(m1):
                    for i2 in range(n1):
                        conv = pref*rconvs[i1][i2][ind]
                        twob[i][j][id2] = conv
                        twob[j][i][id2] = conv
                        id2 += 1
                if size > 2:
                    for k in range(j+1, size):
                        rik = atom - coods[k]
                        rik_norm = np.linalg.norm(rik)

                        if rik_norm != 0 and rik_norm < cutoff_r:
                            z3 = charges[k]

                            rkj = coods[k] - coods[j]

                            rkj_norm = np.linalg.norm(rkj)

                            cos1 = np.minimum(1.0, np.maximum(np.dot(rij, rik)/(rij_norm*rik_norm), -1.0))
                            cos2 = np.minimum(1.0, np.maximum(np.dot(rij, rkj)/(rij_norm*rkj_norm), -1.0))
                            cos3 = np.minimum(1.0, np.maximum(np.dot(-rkj, rik)/(rkj_norm*rik_norm), -1.0))
                            ang1 = np.arccos(cos1)
                            ang2 = np.arccos(cos2)
                            ang3 = np.arccos(cos3)

                            ind1 = int(ang1/astep)
                            ind2 = int(ang2/astep)
                            ind3 = int(ang3/astep)

                            atm = (rij_norm*rik_norm*rkj_norm)**n_atm

                            charge = np.cbrt(z*z2*z3)

                            pref = charge

                            id2 = 0
                            for i1 in range(m2):
                                for i2 in range(n2):
                                    if i2 == 0:
                                        conv1 = (pref*aconvs[i1][i2][ind1]*cos2*cos3)/atm
                                        conv2 = (pref*aconvs[i1][i2][ind2]*cos1*cos3)/atm
                                        conv3 = (pref*aconvs[i1][i2][ind3]*cos2*cos1)/atm
                                    else:
                                        conv1 = (pref*aconvs[i1][i2][ind1])/atm
                                        conv2 = (pref*aconvs[i1][i2][ind2])/atm
                                        conv3 = (pref*aconvs[i1][i2][ind3])/atm

                                    threeb[i][j][k][id2] = conv1
                                    threeb[i][k][j][id2] = conv1

                                    threeb[j][i][k][id2] = conv2
                                    threeb[j][k][i][id2] = conv2

                                    threeb[k][j][i][id2] = conv3
                                    threeb[k][i][j][id2] = conv3

                                    id2 += 1

    return twob, threeb


def get_cmbdf(charges, coods, convs, pad=None, rcut=10.0, n_atm=1.0, gradients=False):
    """Returns the local cMBDF representation for a molecule."""
    rconvs, aconvs = convs
    size = len(charges)
    if pad is None:
        pad = size

    m1, n1 = rconvs[0].shape[0], rconvs[0].shape[1]
    m2, n2 = aconvs[0].shape[0], aconvs[0].shape[1]
    nr = m1*n1
    na = m2*n2
    desc_size = nr + na
    mat = np.zeros((pad, desc_size))

    assert size > 1, "No implementation for mono-atomics yet"

    twob, threeb = generate_data(size, charges, coods, rconvs, aconvs, rcut, n_atm)
    mat[:size, :nr] = einsum('ij... -> i...', twob)
    mat[:size, nr:] = einsum('ijk... -> i...', threeb)
    return mat


def get_convolutions(rstep=0.0008, rcut=10.0, alpha_list=[1.5, 5.0], n_list=[3.0, 5.0],
                     order=4, a1=2.0, a2=2.0, astep=0.0002, nAs=4, gradients=False):
    """Returns cMBDF convolutions evaluated via Fast Fourier Transforms."""
    step_r = rcut/next_fast_len(int(rcut/rstep))
    astep = np.pi/next_fast_len(int(np.pi/astep))
    rgrid = np.arange(0.0, rcut, step_r)
    rgrid2 = np.arange(-rcut, rcut, step_r)
    agrid = np.arange(0.0, np.pi, astep)
    agrid2 = np.arange(-np.pi, np.pi, astep)

    size = len(rgrid)
    gaussian = np.exp(-a1*(rgrid2**2))

    m = order + 1

    temp1, temp2 = [], []
    dtemp1, dtemp2 = [], []

    fms = [gaussian, *[gaussian*hermite_polynomial(rgrid2, i, a1) for i in range(1, m+1)]]

    for i in range(m):
        fm = fms[i]

        temp, dtemp = [], []
        for alpha in alpha_list:
            gn = np.exp(-alpha*rgrid)
            arr = fftconvolve(gn, fm, mode='same')*step_r
            arr = arr/np.max(np.abs(arr))
            temp.append(arr)
            darr = np.gradient(arr, step_r)
            dtemp.append(darr)
        temp1.append(np.array(temp))
        dtemp1.append(np.array(dtemp))

        temp, dtemp = [], []
        for n in n_list:
            gn = 2.2508*((rgrid+1)**n)
            arr = fftconvolve(1/gn, fm, mode='same')[:size]*step_r
            arr = arr/np.max(np.abs(arr))
            temp.append(arr)
            darr = np.gradient(arr, step_r)
            dtemp.append(darr)
        temp2.append(np.array(temp))
        dtemp2.append(np.array(dtemp))

    rconvs = np.concatenate((np.asarray(temp1), np.asarray(temp2)), axis=1)
    drconvs = np.concatenate((np.asarray(dtemp1), np.asarray(dtemp2)), axis=1)

    size = len(agrid)
    gaussian = np.exp(-a2*(agrid2**2))

    m = order + 1

    temp1, dtemp1 = [], []

    fms = [gaussian, *[gaussian*hermite_polynomial(agrid2, i, a2) for i in range(1, m+1)]]

    for i in range(m):
        fm = fms[i]

        temp, dtemp = [], []
        for n in range(1, nAs+1):
            gn = np.cos(n*agrid)
            arr = fftconvolve(gn, fm, mode='same')*astep
            arr = arr/np.max(np.abs(arr))
            temp.append(arr)
            darr = np.gradient(arr, astep)
            dtemp.append(darr)
        temp1.append(np.array(temp))
        dtemp1.append(np.array(dtemp))

    aconvs, daconvs = np.asarray(temp1), np.asarray(dtemp1)

    return np.asarray([rconvs, drconvs]), np.asarray([aconvs, daconvs])


def generate_mbdf(nuclear_charges, coords, convs, alpha_list=[1.5, 5.0], n_list=[3.0, 5.0],
                  gradients=False, local=True, n_jobs=1, a1=2.0, pad=None, rstep=0.0008,
                  rcut=10.0, astep=0.0002, nAs=4, order=4, progress_bar=False, a2=2.0, n_atm=2.0):
    """Generate cMBDF representations for a set of molecules.

    Parameters
    ----------
    nuclear_charges : array-like
        Array of arrays of nuclear charges for each molecule.
    coords : array-like
        Array of arrays of atomic coordinates (in Angstrom) for each molecule.
    convs : tuple
        Precomputed (rconvs, aconvs) convolution arrays.
    n_jobs : int
        Number of parallel jobs. Default 1 (serial) to avoid conflicts with Psi4's parallelism.

    Returns
    -------
    np.ndarray
        cMBDF representations.
    """
    try:
        from joblib import Parallel, delayed
    except ImportError:
        raise ImportError(
            "aPBE0 requires the 'joblib' package. Install with: pip install joblib"
        )

    assert nuclear_charges.shape[0] == coords.shape[0], \
        "charges and coordinates array length mis-match"

    lengths, charges = [], []

    for i in range(len(nuclear_charges)):
        q, r = nuclear_charges[i], coords[i]
        assert q.shape[0] == r.shape[0], \
            "charges and coordinates array length mis-match for molecule at index " + str(i)
        lengths.append(len(q))
        charges.append(q.astype(np.float64))

    if pad is None:
        pad = max(lengths)

    if local:
        reps = Parallel(n_jobs=n_jobs)(
            delayed(get_cmbdf)(charge, cood, convs, pad, rcut, n_atm, gradients=False)
            for charge, cood in list(zip(charges, coords))
        )
        return np.asarray(reps)
    else:
        keys = np.unique(np.concatenate(charges))
        asize = {key: max([(mol == key).sum() for mol in charges]) for key in keys}
        rep_size = sum(asize.values())
        reps = Parallel(n_jobs=n_jobs)(
            delayed(get_cmbdf_global)(charge, cood, asize, rep_size, keys, convs, rcut, n_atm)
            for charge, cood in list(zip(charges, coords))
        )
        return np.asarray(reps)
