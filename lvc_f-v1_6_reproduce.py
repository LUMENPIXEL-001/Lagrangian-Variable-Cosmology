# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LUMENPIXEL
#
# Released under the MIT License.

"""
LVC final-v1.6 -- Single-file reproduction script
==================================================

Reproduces the numerical results in the final-v1.6 working paper:

    "Lagrangian Variable Cosmology final-v1.6:
     Correction of the Sound-Horizon Calibration and
     Re-evaluation of the v1.1 Modulation on the Corrected Pipeline"
                                LUMENPIXEL, October 2026

Single self-contained Python file. Required external dependencies:
numpy, pandas, scipy, camb (CAMB 2.0.4 was used for the paper).

CORRECTION NOTE
---------------
Two inconsistencies in the reproduction scripts of v0.1 through v1.5 are
corrected here (paper Section 2.1):

  (i)  distance-prior integrals (v0.1-v1.4): T(z) applied to the
       dark-energy term of H^2 only          -> T multiplies H everywhere
  (ii) BAO sound horizon (v0.1-v1.5):  r_d = theta_* * D_M(z_*)
       (this is r_s at photon decoupling, ~144.4 Mpc)
                                           -> r_drag from CAMB (~147.1 Mpc)

The flags CFG['de_only'] (error i) and CFG['rd_theta'] (error ii)
re-introduce each inconsistency; Section 8 uses them to print the 2x2
decomposition of paper Table 1.

Data set (paper Section 2.2), N = 1596:
    DESI DR2 BAO (13) + Pantheon+ without calibrators and z_HD <= 0.01
    (1580, full STAT+SYS covariance) + Planck 2018 distance priors (3).
    No local H0 prior, no DESI DR1 / BOSS / eBOSS.

Reproduces (key paper numbers, N = 1596):
----------------------------------------
Section 8:  Effect of the two inconsistencies (Table 1)
    LCDM, both present / (ii) present / both corrected:
        chi^2 = 1441.45 / 1441.45 / 1405.56
    v1.1 Om-locked, both corrected   dchi2 = +21.76   dBIC = +14.38
    B0 (Om free), both corrected     dchi2 = -10.69

Section 9:  Re-evaluation of the v1.1 family (Section 2.4)
    B0                               dchi2 = -10.69  (Om = 0.3116, H0 = 67.14)
    16 lock combinations (a, b)      (4,3) best: -10.69; (3,4) -10.10; (4,4) -9.71
    shapes at v1.1 locks             gauss -11.1, expl -10.6, lorentz -8.8,
                                     sech -6.1, sign-flipped +46.1
    ladder variants                  single rung -10.68; v1.5 3-rung lock -6.44; ...

Section 10: Baselines, LOO, SH0ES, sensitivity (Section 3, Tables 2-3, 5)
    LCDM chi^2 = 1405.56 (H0 = 68.52, Om = 0.2995)
    wCDM dchi2 = -1.51;   CPL dchi2 = -8.00
    A, theta_c, w free: dchi2 = -11.46 (theta_c = 0.4767)
    LOO LRG2-removed dchi2 = -4.11
    SH0ES term included: dchi2(B0 - LCDM) = +1.71
    sensitivity variants: B0 in [-12.26, -7.21]; Om-locked in [+11.12, +33.79]

Section 11: Position and amplitude profiles (Tables 4, aprof)
    theta_c profile (A,w locked / free), A profile

Section 12: Effective dark energy and growth (Section 4)
    B0 w_eff min -1.65 (z = 0.51), max -0.42 (z = 0.77), rho ratio 1.32
    fsigma8: profiled sigma_8 LCDM 0.824 / B0 0.837

Section 13: DESI DR3 tables (Section 6)
    prediction table, mock chi^2 (refit), frozen-parameter chi^2 (f = 1..2)

External data
-------------
Pantheon+ data files:
    Pantheon+SH0ES.dat
    Pantheon+SH0ES_STAT+SYS.cov
If absent, the script offers automatic git-based download.

Usage
-----
    python lvc_v16_reproduce.py                    # full run (hours)
    python lvc_v16_reproduce.py --quick            # coarse grids, fewer fits
    python lvc_v16_reproduce.py --only err,de      # selected sections
    python lvc_v16_reproduce.py --skip-profiles    # skip Section 11
    python lvc_v16_reproduce.py --help

Section keys for --only: err, family, baseline, profile, de, dr3.

License
-------
MIT License. Copyright (c) 2026 LUMENPIXEL.
"""

import os
import sys
import time
import argparse
import subprocess

import numpy as np
import pandas as pd
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import differential_evolution, minimize, brentq
from scipy.stats import norm


# ============================================================================
# SECTION 1.  CONSTANTS
# ============================================================================

C_KMS          = 299_792.458
THETA_STAR     = 0.010409            # used only to re-create inconsistency (ii)
H0_SHOES       = 73.04
SIG_H0_SHOES   = 1.04

OMEGA_GAMMA_H2 = 2.473e-5
NEFF_STD       = 3.046
OMEGA_R_H2     = OMEGA_GAMMA_H2 * (1 + 7 / 8 * (4 / 11) ** (4 / 3) * NEFF_STD)

PHI         = (1 + np.sqrt(5)) / 2.0
LNPHI       = np.log(PHI)
LNPHI_3     = LNPHI ** 3
LNPHI_4     = LNPHI ** 4
OM_RECURSIVE = 1.0 / (1.0 + np.exp(3.0 * LNPHI / PHI))       # 0.290653

# Planck 2018 distance priors (Chen, Huang, Wang 2019)
R_PLANCK  = 1.7493
LA_PLANCK = 301.462
WB_PLANCK = 0.02236
SIG_DP    = np.array([0.0049, 0.090, 0.00015])
CORR_DP   = np.array([[ 1.0000,  0.4720, -0.6536],
                      [ 0.4720,  1.0000, -0.3392],
                      [-0.6536, -0.3392,  1.0000]])
COV_DP    = CORR_DP * SIG_DP[:, None] * SIG_DP[None, :]
COVINV_DP = np.linalg.inv(COV_DP)

# Planck 2018 fiducial cosmology used for the R, l_A calibration constants
FID = (67.36, 0.3153, 0.02237)

SIGMA8_PLANCK = 0.811
N_TOTAL = 1596

# Pipeline configuration (flags re-introduce the v0.1-v1.5 inconsistencies)
CFG = dict(
    de_only=False,      # inconsistency (i)
    rd_theta=False,     # inconsistency (ii)
    rd_scale=1.0,       # sensitivity test (Section 10)
    drop_la=False,      # sensitivity test: remove l_A from distance priors
    cal_r=1.0,          # set below
    cal_la=1.0,         # set below
    skip_bao=set(),     # leave-one-out
    shoes=False,        # add Gaussian SH0ES term
)

# Best-fit values of the paper (Table 2); used as starting points / frozen models
BEST = dict(
    LCDM = dict(H0=68.52, Om=0.2995, ob=0.02252),
    B0   = dict(H0=67.14, Om=0.3116, ob=0.02254),
    FREE = dict(H0=67.4578, Om=0.3088, ob=0.0225, A=0.0593, tc=0.4767, w=0.0777),
    CPL  = dict(H0=67.5968, Om=0.3098, ob=0.0225, w0=-0.8489, wa=-0.511),
)


# ============================================================================
# SECTION 2.  PANTHEON+ DATA LOADER (auto-download)
# ============================================================================

PP_DIR = os.environ.get(
    "PANTHEONPLUS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "PP_data"))
PP_DAT = os.path.join(PP_DIR, "Pantheon+SH0ES.dat")
PP_COV = os.path.join(PP_DIR, "Pantheon+SH0ES_STAT+SYS.cov")


def auto_download_pp():
    print(f"\nPantheon+ data not found at {PP_DIR}.")
    print("Attempting automatic download from GitHub...")
    response = input("\nDownload now? [Y/n] ").strip().lower()
    if response and response != 'y':
        print("Aborted. Set PANTHEONPLUS_DIR or sparse-clone manually:")
        print("  git clone --depth=1 --filter=blob:none --no-checkout \\")
        print("      https://github.com/PantheonPlusSH0ES/DataRelease.git")
        print("  cd DataRelease")
        print("  git sparse-checkout init --cone")
        print("  git sparse-checkout set 'Pantheon+_Data/4_DISTANCES_AND_COVAR'")
        print("  git checkout main")
        sys.exit(1)
    os.makedirs(PP_DIR, exist_ok=True)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        cmds = [
            ['git', 'clone', '--depth=1', '--filter=blob:none', '--no-checkout',
             'https://github.com/PantheonPlusSH0ES/DataRelease.git', 'DataRelease'],
            ['git', '-C', 'DataRelease', 'sparse-checkout', 'init', '--cone'],
            ['git', '-C', 'DataRelease', 'sparse-checkout', 'set',
             'Pantheon+_Data/4_DISTANCES_AND_COVAR'],
            ['git', '-C', 'DataRelease', 'checkout', 'main'],
        ]
        for cmd in cmds:
            print("  Running:", " ".join(cmd))
            r = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True,
                               timeout=180)
            if r.returncode != 0:
                print("ERROR:", r.stderr)
                sys.exit(1)
        src = os.path.join(tmp, 'DataRelease', 'Pantheon+_Data',
                           '4_DISTANCES_AND_COVAR')
        for fname in ['Pantheon+SH0ES.dat', 'Pantheon+SH0ES_STAT+SYS.cov']:
            with open(os.path.join(src, fname), 'rb') as fi, \
                 open(os.path.join(PP_DIR, fname), 'wb') as fo:
                fo.write(fi.read())
            print(f"  Copied {fname}")
    print("Download complete.\n")


def load_pantheonplus():
    """Non-calibrator rows with z_HD > 0.01 (paper Section 2.2): 1580 SNe."""
    if not os.path.exists(PP_DAT) or not os.path.exists(PP_COV):
        auto_download_pp()
    df = pd.read_csv(PP_DAT, sep=r'\s+')
    with open(PP_COV) as f:
        ncov = int(f.readline().strip())
    cov = np.loadtxt(PP_COV, skiprows=1).reshape((ncov, ncov))
    sel = (df['IS_CALIBRATOR'].values == 0) & (df['zHD'].values > 0.01)
    idx = np.where(sel)[0]
    pp = dict(zHD=df['zHD'].values[idx], zHEL=df['zHEL'].values[idx],
              mB=df['m_b_corr'].values[idx], N=len(idx))
    pp['Cinv'] = np.linalg.inv(cov[np.ix_(idx, idx)])
    pp['one'] = np.ones(pp['N'])
    pp['s11'] = pp['one'] @ pp['Cinv'] @ pp['one']
    return pp


# ============================================================================
# SECTION 3.  CAMB TABLE FOR r_drag, z_*, r_s(z_*)
# ============================================================================
#
# r_drag, z_* and r_s(z_*) depend on (omega_b, omega_m) only (T = 1 for z > 10).
# They are tabulated with CAMB (neutrino mass sum 0.06 eV) on a grid and
# interpolated with bicubic splines. Agreement with direct CAMB evaluation
# is better than 2e-6 (relative) at test points (paper Section 2.2).

CAMB_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "camb_table_v16.npz")
_OB_GRID = np.linspace(0.0200, 0.0250, 11)
_OM_GRID = np.linspace(0.10, 0.18, 17)
_SPL = {}


def build_camb_table():
    import camb
    print("  Building CAMB table for r_drag, z_*, r_*  (one-off, ~1-3 min) ...")
    keys = ['rdrag', 'zdrag', 'zstar', 'rstar']
    tab = {k: np.zeros((len(_OB_GRID), len(_OM_GRID))) for k in keys}
    for i, b in enumerate(_OB_GRID):
        for j, m in enumerate(_OM_GRID):
            p = camb.CAMBparams()
            p.set_cosmology(H0=67.5, ombh2=b, omch2=m - b, mnu=0.06, omk=0)
            p.set_for_lmax(2000)
            d = camb.get_background(p).get_derived_params()
            for k in keys:
                tab[k][i, j] = d[k]
    np.savez(CAMB_CACHE, **tab)
    return tab


def _load_camb_table():
    if os.path.exists(CAMB_CACHE):
        z = np.load(CAMB_CACHE)
        tab = {k: z[k] for k in z.files}
    else:
        tab = build_camb_table()
    for k, v in tab.items():
        _SPL[k] = RectBivariateSpline(_OB_GRID, _OM_GRID, v, kx=3, ky=3)


def emu(key, ob, om_h2):
    if not _SPL:
        _load_camb_table()
    return float(_SPL[key](ob, om_h2)[0, 0])


# ============================================================================
# SECTION 4.  BACKGROUND COSMOLOGY AND MODULATION T(z)
# ============================================================================

_ZG  = np.concatenate([[0.0], np.geomspace(1e-4, 5.0, 1500)])
_GLX, _GLW = np.polynomial.legendre.leggauss(80)
_ZGL = 0.5 * 30 * (_GLX + 1)          # Gauss-Legendre nodes on [0, 30]
_WGL = 0.5 * 30 * _GLW


def _sech2(x):
    return 1.0 / np.cosh(np.clip(x, -40.0, 40.0)) ** 2


def T_sech2(A, tc, w):
    """T(z) = 1 + A sech^2((theta - tc)/w),  theta = ln(1+z)."""
    return lambda z: 1.0 + A * _sech2((np.log1p(np.asarray(z, float)) - tc) / w)


def T_B0():
    """B0: v1.1 profile, theta_c = ln phi, w = (ln phi)^3, A = (ln phi)^4."""
    return T_sech2(LNPHI_4, LNPHI, LNPHI_3)


def T_shape(name, A=LNPHI_4, tc=LNPHI, w=LNPHI_3):
    th = lambda z: np.log1p(np.asarray(z, float))
    clip = lambda x: np.clip(x, -40.0, 40.0)
    if name == 'sech2':
        return T_sech2(A, tc, w)
    if name == 'gauss':
        return lambda z: 1.0 + A * np.exp(-clip((th(z) - tc) / w) ** 2)
    if name == 'lorentz':
        return lambda z: 1.0 + A / (1.0 + ((th(z) - tc) / w) ** 2)
    if name == 'sech1':
        return lambda z: 1.0 + A / np.cosh(clip((th(z) - tc) / w))
    if name == 'explift':
        return lambda z: np.exp(A * _sech2((th(z) - tc) / w))
    raise ValueError(name)


def T_ladder(amps, w=LNPHI_3):
    """T - 1 = sum_n A_n sech^2((theta - n ln phi)/w)  (v1.3 / v1.5 ladder)."""
    def f(z):
        t = np.log1p(np.asarray(z, float))
        out = np.ones_like(t)
        for n, a in enumerate(amps, 1):
            out = out + a * _sech2((t - n * LNPHI) / w)
        return out
    return f


def T_minus_1_exact(z, A=LNPHI_4, tc=LNPHI, w=LNPHI_3):
    """T - 1 without underflow: 4A exp(-2|x|)/(1+exp(-2|x|))^2."""
    x = np.abs(np.log1p(z) - tc) / w
    e = np.exp(-2.0 * x)
    return A * 4.0 * e / (1.0 + e) ** 2


def E_background(Om, Tf=None, wde=None, z=None):
    """E(z) = H/H0 on z (default grid): LCDM (or w0wa) times T."""
    z = _ZG if z is None else np.asarray(z, float)
    if wde is None:
        e2 = Om * (1 + z) ** 3 + (1 - Om)
    else:
        w0, wa = wde
        fde = (1 + z) ** (3 * (1 + w0 + wa)) * np.exp(-3 * wa * z / (1 + z))
        e2 = Om * (1 + z) ** 3 + (1 - Om) * fde
    E = np.sqrt(e2)
    return E if Tf is None else E * Tf(z)


def comoving_cumint(E, H0):
    inv = C_KMS / (H0 * E)
    dz = np.diff(_ZG)
    return np.concatenate([[0.0], np.cumsum(0.5 * (inv[:-1] + inv[1:]) * dz)])


# ============================================================================
# SECTION 5.  BAO DATA (DESI DR2)
# ============================================================================

DESI_DR2_DV = [(0.295, 7.942, 0.075)]
DESI_DR2_PAIR = [
    ('LRG1', 0.510, 13.587, 0.169, 21.863, 0.427, -0.475),
    ('LRG2', 0.706, 17.347, 0.180, 19.458, 0.332, -0.423),
    ('LRG3', 0.934, 21.574, 0.153, 17.641, 0.193, -0.425),
    ('QSO',  1.321, 27.605, 0.320, 14.178, 0.217, -0.437),
    ('ELG2', 1.484, 30.519, 0.758, 12.816, 0.513, -0.489),
    ('Lya',  2.330, 38.988, 0.531,  8.632, 0.101, -0.431),
]

# 16 fsigma8 measurements (Table 10 of v1.5): (z, fsigma8, sigma)
FSIGMA8_DATA = np.array([
    (0.067, 0.423, 0.055), (0.150, 0.530, 0.085), (0.380, 0.497, 0.045),
    (0.510, 0.458, 0.038), (0.440, 0.413, 0.080), (0.600, 0.390, 0.063),
    (0.730, 0.437, 0.072), (0.698, 0.471, 0.045), (0.850, 0.315, 0.095),
    (1.480, 0.462, 0.045), (0.295, 0.439, 0.045), (0.510, 0.476, 0.030),
    (0.706, 0.453, 0.022), (0.930, 0.476, 0.029), (1.317, 0.434, 0.035),
    (1.491, 0.395, 0.045)])


# ============================================================================
# SECTION 6.  LIKELIHOOD BLOCKS
# ============================================================================

def chi2_sn(DC, pp):
    DMz = np.interp(pp['zHD'], _ZG, DC)
    mu = 5 * np.log10((1 + pp['zHEL']) * DMz) + 25
    d = pp['mB'] - mu
    Cd = pp['Cinv'] @ d
    M = (pp['one'] @ Cd) / pp['s11']                     # profiled magnitude
    return float(d @ Cd - M * (pp['one'] @ Cd))


def chi2_bao(DC, E, H0, rd):
    skip = CFG['skip_bao']
    c2 = 0.0
    if 'BGS' not in skip:
        for z, DVo, s in DESI_DR2_DV:
            DM = np.interp(z, _ZG, DC)
            Hz = H0 * np.interp(z, _ZG, E)
            DV = (z * DM ** 2 * C_KMS / Hz) ** (1 / 3)
            c2 += ((DVo - DV / rd) / s) ** 2
    for lab, z, DMo, sM, DHo, sH, rho in DESI_DR2_PAIR:
        if lab in skip:
            continue
        DMp = np.interp(z, _ZG, DC) / rd
        DHp = C_KMS / (H0 * np.interp(z, _ZG, E)) / rd
        cov = np.array([[sM ** 2, rho * sM * sH], [rho * sM * sH, sH ** 2]])
        d = np.array([DMo - DMp, DHo - DHp])
        c2 += float(d @ np.linalg.solve(cov, d))
    return c2


def DM_at_zstar(H0, Om, zs, Tf=None, wde=None):
    """D_M(z_*) with radiation; T multiplies H (paper-intent), or the DE term
    only if CFG['de_only'] (inconsistency (i))."""
    h = H0 / 100.0
    Orf = OMEGA_R_H2 / h ** 2
    if wde is None:
        fde = lambda z: 1.0
    else:
        w0, wa = wde
        fde = lambda z: (1 + z) ** (3 * (1 + w0 + wa)) * np.exp(-3 * wa * z / (1 + z))
    Hf = lambda z: np.sqrt(Om * (1 + z) ** 3 + Orf * (1 + z) ** 4
                           + (1 - Om - Orf) * fde(z))
    I0 = quad(lambda z: 1.0 / Hf(z), 0, zs, limit=300, epsrel=1e-10)[0]
    if Tf is None:
        return C_KMS / H0 * I0
    z = _ZGL
    Hz = Hf(z)
    T = Tf(z)
    if CFG['de_only']:
        Hm = np.sqrt(Om * (1 + z) ** 3 + Orf * (1 + z) ** 4
                     + (1 - Om - Orf) * fde(z) * T ** 2)
        corr = np.sum(_WGL * (1.0 / Hm - 1.0 / Hz))
    else:
        corr = np.sum(_WGL * (1.0 / (Hz * T) - 1.0 / Hz))
    return C_KMS / H0 * (I0 + corr)


def chi2_dp(H0, Om, ob, Tf=None, wde=None):
    om = Om * (H0 / 100.0) ** 2
    zs = emu('zstar', ob, om)
    rs = emu('rstar', ob, om)
    DM = DM_at_zstar(H0, Om, zs, Tf, wde)
    R = np.sqrt(Om) * H0 * DM / C_KMS * CFG['cal_r']
    lA = np.pi * DM / rs * CFG['cal_la']
    d = np.array([R - R_PLANCK, lA - LA_PLANCK, ob - WB_PLANCK])
    if CFG['drop_la']:
        Cm = COV_DP[np.ix_([0, 2], [0, 2])]
        dd = d[[0, 2]]
        return float(dd @ np.linalg.solve(Cm, dd)), R, lA
    return float(d @ COVINV_DP @ d), R, lA


def _calibrate_dp():
    """Multiplicative R, l_A constants fixed at the Planck 2018 fiducial."""
    CFG['cal_r'], CFG['cal_la'] = 1.0, 1.0
    _, R, lA = chi2_dp(*FID, Tf=None)
    CFG['cal_r'] = R_PLANCK / R
    CFG['cal_la'] = LA_PLANCK / lA


def chi2_parts(H0, Om, ob, Tf=None, pp=None, wde=None):
    """All blocks. Returns dict or None if outside the validity domain."""
    om = Om * (H0 / 100.0) ** 2
    if not (0.10 <= om <= 0.18 and 0.0200 <= ob <= 0.0250 and 0.05 < Om < 0.7):
        return None
    E = E_background(Om, Tf, wde)
    if Tf is not None:
        zz = np.concatenate([_ZG, _ZGL])
        if E.min() <= 0 or np.min(Tf(zz)) < 0.3:
            return None
    DC = comoving_cumint(E, H0)
    rd = emu('rdrag', ob, om) * CFG['rd_scale']
    if CFG['rd_theta']:                                  # inconsistency (ii)
        rd = THETA_STAR * DM_at_zstar(H0, Om, emu('zstar', ob, om), Tf, wde)
    out = dict(rd=rd, bao=chi2_bao(DC, E, H0, rd))
    out['sn'] = chi2_sn(DC, pp) if pp is not None else 0.0
    out['dp'], out['R'], out['lA'] = chi2_dp(H0, Om, ob, Tf, wde)
    out['shoes'] = ((H0 - H0_SHOES) / SIG_H0_SHOES) ** 2 if CFG['shoes'] else 0.0
    out['tot'] = out['bao'] + out['sn'] + out['dp'] + out['shoes']
    return out


# ============================================================================
# SECTION 7.  FIT HELPERS
# ============================================================================

def _nm(cost, x0, iters=3000):
    r = minimize(cost, np.asarray(x0, float), method='Nelder-Mead',
                 options=dict(xatol=1e-8, fatol=1e-8, maxiter=iters, adaptive=True))
    r = minimize(cost, r.x, method='Nelder-Mead',
                 options=dict(xatol=1e-9, fatol=1e-9, maxiter=iters, adaptive=True))
    return float(r.fun), r.x


def _de_nm(cost, bounds, seed=42, maxiter=60, popsize=10, nm_iter=4000):
    """Differential evolution + Nelder-Mead polish."""
    de = differential_evolution(cost, bounds, seed=seed, tol=1e-8,
                                maxiter=maxiter, polish=False, popsize=popsize)
    r = minimize(cost, de.x, method='Nelder-Mead',
                 options=dict(xatol=1e-9, fatol=1e-9, maxiter=nm_iter, adaptive=True))
    return float(r.fun), r.x


_STARTS3 = [(67.0, 0.313, 0.0224), (68.5, 0.300, 0.0225),
            (70.0, 0.285, 0.0226), (66.0, 0.330, 0.0222)]


def _mk_cost(pp, make_T=None, Om_lock=None, wde_from=None):
    """Cost over (H0, Om, ob) [+extra]. make_T(extra) -> T function."""
    def cost(x):
        if Om_lock is None:
            H0, Om, ob, ex = x[0], x[1], x[2], x[3:]
        else:
            H0, Om, ob, ex = x[0], Om_lock, x[1], x[2:]
        Tf = None if make_T is None else make_T(ex)
        wde = None if wde_from is None else wde_from(ex)
        p = chi2_parts(H0, Om, ob, Tf, pp, wde)
        return 1e10 if p is None else p['tot']
    return cost


def fit3(pp, Tf=None, Om_lock=None, starts=None):
    """Multi-start Nelder-Mead over (H0, Om, ob) or (H0, ob) if Om_lock."""
    cost = _mk_cost(pp, (lambda ex: Tf) if Tf is not None else None, Om_lock)
    best = None
    st = starts or ([(68.5, 0.0225), (69.5, 0.0228)] if Om_lock is not None
                    else _STARTS3)
    for s in st:
        f, x = _nm(cost, s)
        if best is None or f < best[0]:
            best = (f, x)
    return best


def dbic(chi2, chi2_ref, k, k_ref=3, N=N_TOTAL):
    return chi2 - chi2_ref + (k - k_ref) * np.log(N)


# ============================================================================
# SECTION 8.  EFFECT OF THE TWO INCONSISTENCIES  (paper Table 1)
# ============================================================================

def run_error_section(pp):
    print("\n" + "=" * 76)
    print(" SECTION 8.  EFFECT OF THE TWO INCONSISTENCIES (v1.6 paper Table 1)")
    print("=" * 76)
    print(" v1.1 profile (theta_c = ln phi, w = (ln phi)^3, A = (ln phi)^4);")
    print(" Om locked to 0.290653 (k=2) and Om free (k=3), N = 1596.")
    print()
    cfgs = [("(i) and (ii) present",          True,  True),
            ("(ii) present, (i) corrected",   False, True),
            ("(i) present, (ii) corrected",   True,  False),
            ("both corrected",                False, False)]
    Tb = T_B0()
    print(f" {'configuration':32s} {'chi2(LCDM)':>11s} {'dchi2 Om-locked':>16s}"
          f" {'dBIC':>8s} {'dchi2 Om-free':>14s}")
    for name, de_only, rd_theta in cfgs:
        CFG['de_only'], CFG['rd_theta'] = de_only, rd_theta
        fl, xl = fit3(pp)
        fv, _ = fit3(pp, Tb, Om_lock=OM_RECURSIVE)
        fb, _ = fit3(pp, Tb)
        print(f" {name:32s} {fl:11.2f} {fv - fl:+16.2f} "
              f"{dbic(fv, fl, 2):+8.2f} {fb - fl:+14.2f}", flush=True)
    CFG['de_only'], CFG['rd_theta'] = False, False
    print("\n Paper values: 1441.45/-31.40/-38.77/-32.41 ; 1441.45/-19.40/-26.78/"
          "-19.72 ;\n               1405.56/+22.90/+15.53/-13.96 ; 1405.56/+21.76/"
          "+14.38/-10.69")


# ============================================================================
# SECTION 9.  RE-EVALUATION OF THE v1.1 FAMILY  (paper Section 2.4)
# ============================================================================

def run_family_section(pp, quick=False):
    print("\n" + "=" * 76)
    print(" SECTION 9.  RE-EVALUATION OF THE v1.1 FAMILY (corrected pipeline)")
    print("=" * 76)
    fl, xl = fit3(pp)
    print(f"\n LCDM: chi2 = {fl:.3f}  H0 = {xl[0]:.2f}  Om = {xl[1]:.4f}  ob = {xl[2]:.5f}")

    Tb = T_B0()
    fv, xv = fit3(pp, Tb, Om_lock=OM_RECURSIVE)
    print(f" v1.1 Om locked (k=2): dchi2 = {fv - fl:+.2f}  dBIC = {dbic(fv, fl, 2):+.2f}")
    fb, xb = fit3(pp, Tb)
    pb = chi2_parts(xb[0], xb[1], xb[2], Tb, pp)
    print(f" B0 (k=3):             dchi2 = {fb - fl:+.2f}  Om = {xb[1]:.4f}  "
          f"H0 = {xb[0]:.2f}  ob = {xb[2]:.5f}  r_drag = {pb['rd']:.2f}  "
          f"chi2_BAO = {pb['bao']:.1f}")

    # ---- 16 lock combinations
    if not quick:
        print("\n -- A = (ln phi)^a, w = (ln phi)^b, theta_c = ln phi, Om free (k=3) --")
        print("      b=2       b=3       b=4       b=5")
        for a in [2, 3, 4, 5]:
            row = []
            for b in [2, 3, 4, 5]:
                f, _ = fit3(pp, T_sech2(LNPHI ** a, LNPHI, LNPHI ** b))
                row.append(f - fl)
            print(f" a={a} " + " ".join(f"{v:+9.2f}" for v in row), flush=True)
        print(" (number of combinations evaluated: 16; 2 ln 16 = 5.5)")

    # ---- shapes at the v1.1 locks
    print("\n -- profile shapes at the v1.1 locks (Om free, k=3) --")
    for nm_ in ['gauss', 'explift', 'lorentz', 'sech1']:
        f, _ = fit3(pp, T_shape(nm_))
        print(f"   {nm_:8s}  dchi2 = {f - fl:+7.2f}", flush=True)
    f, _ = fit3(pp, T_sech2(-LNPHI_4, LNPHI, LNPHI_3))
    print(f"   sign-flipped A  dchi2 = {f - fl:+7.2f}")

    # ---- parameter release (theta_c = ln phi unless free)
    print("\n -- parameter release --")
    cA = _mk_cost(pp, lambda ex: T_sech2(ex[0], LNPHI, LNPHI_3))
    f, x = _nm(cA, [67.4, 0.3096, 0.0225, 0.0446])
    print(f"   A free            k=4  dchi2 = {f - fl:+7.2f}  dBIC = {dbic(f, fl, 4):+6.2f}  A = {x[3]:.4f}")
    cAw = _mk_cost(pp, lambda ex: T_sech2(ex[0], LNPHI, ex[1]))
    f, x = _nm(cAw, [67.46, 0.3087, 0.0225, 0.0569, 0.0806], 5000)
    print(f"   A, w free         k=5  dchi2 = {f - fl:+7.2f}  dBIC = {dbic(f, fl, 5):+6.2f}  "
          f"A = {x[3]:.4f}  w = {x[4]:.4f}")
    c6 = _mk_cost(pp, lambda ex: T_sech2(ex[0], ex[1], ex[2]))
    b6 = BEST['FREE']
    if quick:
        f, x = _nm(c6, [b6['H0'], b6['Om'], b6['ob'], b6['A'], b6['tc'], b6['w']], 6000)
    else:
        lo = [62, 0.25, 0.0215, 0.0, 0.15, 0.03]
        hi = [76, 0.36, 0.0235, 0.25, 1.2, 0.5]
        def c6b(x):
            if any(v < l or v > h for v, l, h in zip(x, lo, hi)):
                return 1e10
            return c6(x)
        f, x = min((_de_nm(c6b, list(zip(lo, hi)), seed=s) for s in (1, 2, 3)),
                   key=lambda t: t[0])
    print(f"   A, theta_c, w free k=6 dchi2 = {f - fl:+7.2f}  dBIC = {dbic(f, fl, 6):+6.2f}  "
          f"A = {x[3]:.4f}  theta_c = {x[4]:.4f}  w = {x[5]:.4f}")

    # ---- ladder variants (Om free)
    print("\n -- multi-rung ladder (Om free) --")
    A1 = LNPHI_4
    def run_lad(name, k, amp_fn, x_extra):
        def cost(x):
            Tf = T_ladder(amp_fn(x[3:]))
            if np.min(Tf(np.concatenate([_ZG, _ZGL]))) <= 0.3:
                return 1e10
            p = chi2_parts(x[0], x[1], x[2], Tf, pp)
            return 1e10 if p is None else p['tot']
        best = None
        for s0 in [(67.0, 0.313, 0.0224), (68.5, 0.300, 0.0225)]:
            for e in x_extra:
                f_, x_ = _nm(cost, list(s0) + list(e), 4000)
                if best is None or f_ < best[0]:
                    best = (f_, x_)
        amps = amp_fn(best[1][3:])
        print(f"   {name:44s} k={k} dchi2 = {best[0] - fl:+7.2f}  "
              f"amps = {np.round(amps, 4)}", flush=True)
    run_lad('1 rung (B0)', 3, lambda y: [A1], [()])
    run_lad('2 rung alt r=phi  (A2 = -phi A1)', 3, lambda y: [A1, -A1 * PHI], [()])
    run_lad('2 rung alt r=1    (A2 = -A1)', 3, lambda y: [A1, -A1], [()])
    run_lad('2 rung same-sign r=1/phi', 3, lambda y: [A1, A1 / PHI], [()])
    run_lad('3 rung same-sign r=1/phi', 3, lambda y: [A1, A1 / PHI, A1 / PHI ** 2], [()])
    run_lad('3 rung v1.5 lock (-1, -phi)', 3, lambda y: [A1, -A1, -PHI * A1], [()])
    run_lad('3 rung alt r=phi', 3,
            lambda y: [A1 * (-1) ** (n + 1) * PHI ** (n - 1) for n in range(1, 4)], [()])
    run_lad('4 rung alt r=phi', 3,
            lambda y: [A1 * (-1) ** (n + 1) * PHI ** (n - 1) for n in range(1, 5)], [()])
    run_lad('2 rung A2 free', 4, lambda y: [A1, y[0]], [(0.03,), (-0.05,)])
    run_lad('3 rung A2, A3 free', 5, lambda y: [A1, y[0], y[1]],
            [(0.03, 0.0), (-0.05, -0.05)])
    if not quick:
        run_lad('4 rung A2, A3, A4 free', 6, lambda y: [A1, y[0], y[1], y[2]],
                [(0.03, 0.0, 0.0), (-0.05, -0.1, 0.1)])
    print("\n alternating-sign family: A_8 =", f"{A1 * (-1) ** 9 * PHI ** 7:+.3f}",
          " (T crosses zero at the eighth rung)")


# ============================================================================
# SECTION 10.  BASELINES, LOO, SH0ES, SENSITIVITY  (paper Section 3, Tables 2-3, 5)
# ============================================================================

def run_baseline_section(pp, quick=False):
    print("\n" + "=" * 76)
    print(" SECTION 10.  BASELINES, LEAVE-ONE-OUT, LOCAL H0, SENSITIVITY")
    print("=" * 76)
    Tb = T_B0()
    fl, xl = fit3(pp)
    print(f"\n LCDM chi2 = {fl:.3f}")

    # ---- wCDM and CPL
    cw = _mk_cost(pp, None, None, lambda ex: (ex[0], 0.0))
    f, x = _nm(cw, [67.9, 0.3037, 0.0226, -0.97], 4000)
    print(f" wCDM (k=4): dchi2 = {f - fl:+.2f}  dBIC = {dbic(f, fl, 4):+.2f}  w0 = {x[3]:.4f}")
    cc = _mk_cost(pp, None, None, lambda ex: (ex[0], ex[1]))
    f, x = min((_nm(cc, s, 5000) for s in ([67.6, 0.31, 0.0225, -0.85, -0.5],
                                           [68.0, 0.30, 0.0225, -1.0, 0.0])),
               key=lambda t: t[0])
    print(f" CPL  (k=5): dchi2 = {f - fl:+.2f}  dBIC = {dbic(f, fl, 5):+.2f}  "
          f"w0 = {x[3]:.4f}  wa = {x[4]:.4f}")

    # ---- BAO leave-one-out
    print("\n -- BAO leave-one-out: dchi2 (B0 - LCDM), both refitted --")
    for lab in [None, 'BGS', 'LRG1', 'LRG2', 'LRG3', 'QSO', 'ELG2', 'Lya']:
        CFG['skip_bao'] = set() if lab is None else {lab}
        a, _ = fit3(pp)
        b, _ = fit3(pp, Tb)
        print(f"   removed {str(lab):5s}  dchi2 = {b - a:+7.2f}", flush=True)
    CFG['skip_bao'] = set()

    # ---- SH0ES sensitivity
    print("\n -- Gaussian SH0ES term added (73.04 +- 1.04) --")
    CFG['shoes'] = True
    a, xa = fit3(pp)
    b, xb = fit3(pp, Tb)
    print(f"   LCDM chi2 = {a:.2f} (H0 = {xa[0]:.2f}, Om = {xa[1]:.4f}) | "
          f"B0 chi2 = {b:.2f} (H0 = {xb[0]:.2f}, Om = {xb[1]:.4f}) | "
          f"dchi2 (B0 - LCDM) = {b - a:+.2f}")
    CFG['shoes'] = False

    # ---- sensitivity to pipeline choices (Table 5)
    print("\n -- sensitivity to pipeline choices (paper Table 5) --")
    cal0 = (CFG['cal_r'], CFG['cal_la'])
    variants = [
        ("baseline", {}),
        ("no distance-prior calibration", dict(cal=(1.0, 1.0))),
        ("l_A calibration doubled", dict(cal=(cal0[0], 1 - 2 * (1 - cal0[1])))),
        ("l_A removed from distance priors", dict(drop=True)),
        ("r_drag x 1.003", dict(rds=1.003)),
        ("r_drag x 0.997", dict(rds=0.997)),
    ]
    print(f"   {'variant':34s} {'chi2(LCDM)':>10s} {'Om(LCDM)':>9s} "
          f"{'dchi2 B0':>9s} {'dchi2 Om-locked':>16s}")
    for name, o in variants:
        CFG['cal_r'], CFG['cal_la'] = o.get('cal', cal0)
        CFG['drop_la'] = o.get('drop', False)
        CFG['rd_scale'] = o.get('rds', 1.0)
        a, xa = fit3(pp)
        b, _ = fit3(pp, Tb)
        v_, _ = fit3(pp, Tb, Om_lock=OM_RECURSIVE)
        print(f"   {name:34s} {a:10.2f} {xa[1]:9.4f} {b - a:+9.2f} {v_ - a:+16.2f}",
              flush=True)
    CFG['cal_r'], CFG['cal_la'] = cal0
    CFG['drop_la'], CFG['rd_scale'] = False, 1.0


# ============================================================================
# SECTION 11.  POSITION AND AMPLITUDE PROFILES  (paper Section 3.4)
# ============================================================================

def run_profile_section(pp, quick=False):
    print("\n" + "=" * 76)
    print(" SECTION 11.  PROFILES IN theta_c AND A")
    print("=" * 76)
    fl, _ = fit3(pp)
    print(f" LCDM chi2 = {fl:.3f}\n")

    # ---- theta_c profile
    grid = [0.35, 0.45, 0.475, 0.50, 0.55, 0.60] if quick else \
           [0.300 + 0.025 * i for i in range(16)]
    print(" -- theta_c profile: dchi2 vs LCDM; (A, w) locked | (A, w) free --")
    x3 = [67.4, 0.309, 0.0225]
    x5 = [67.4, 0.309, 0.0225, 0.045, 0.09]
    for tc in grid:
        Tb = T_sech2(LNPHI_4, tc, LNPHI_3)
        c3 = _mk_cost(pp, lambda ex, Tb=Tb: Tb)
        f3, x3 = _nm(c3, x3, 3000)
        def c5(x, tc=tc):
            if x[3] <= 0 or x[4] <= 0.02:
                return 1e10
            p = chi2_parts(x[0], x[1], x[2], T_sech2(x[3], tc, x[4]), pp)
            return 1e10 if p is None else p['tot']
        best = None
        for s in (x5, [67.4, 0.309, 0.0225, 0.05, 0.10], [67.4, 0.309, 0.0225, 0.07, 0.06]):
            f5, xx = _nm(c5, s, 5000)
            if best is None or f5 < best[0]:
                best = (f5, xx)
        x5 = list(best[1])
        print(f"   theta_c = {tc:.3f}   {f3 - fl:+7.2f} | {best[0] - fl:+7.2f}   "
              f"A = {best[1][3]:.4f}  w = {best[1][4]:.4f}", flush=True)
    print("   (entries with w at the lower bound 0.02 are not interpreted)")

    # ---- A profile
    print("\n -- A profile: theta_c, w (w >= 0.03) free --")
    for A in ([0.0, 0.02, 0.06, 0.10] if quick else
              [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]):
        if A == 0.0:
            print("   A = 0.000   dchi2 = +0.00 (LCDM)")
            continue
        def cA(x, A=A):
            tc, w = x[3], x[4]
            if not (0.2 <= tc <= 1.0 and 0.03 <= w <= 0.5):
                return 1e10
            p = chi2_parts(x[0], x[1], x[2], T_sech2(A, tc, w), pp)
            return 1e10 if p is None else p['tot']
        best = None
        for tc, w in [(0.475, 0.08), (0.45, 0.12), (0.40, 0.17), (0.50, 0.09), (0.35, 0.27)]:
            f, x = _nm(cA, [67.4, 0.309, 0.0225, tc, w], 4000)
            if best is None or f < best[0]:
                best = (f, x)
        print(f"   A = {A:.3f}   dchi2 = {best[0] - fl:+7.2f}   theta_c = {best[1][3]:.4f}  "
              f"w = {best[1][4]:.4f}", flush=True)


# ============================================================================
# SECTION 12.  EFFECTIVE DARK ENERGY AND GROWTH  (paper Section 4)
# ============================================================================

def _model_E2(m, z):
    """E^2(z) with radiation; T multiplies H; optional CPL dark energy."""
    Om = m['Om']
    z = np.asarray(z, float)
    h = m['H0'] / 100.0
    Orf = OMEGA_R_H2 / h ** 2
    if m.get('cpl') is None:
        de = (1 - Om - Orf) * np.ones_like(z)
    else:
        w0, wa = m['cpl']
        de = (1 - Om - Orf) * (1 + z) ** (3 * (1 + w0 + wa)) * np.exp(-3 * wa * z / (1 + z))
    e2 = Om * (1 + z) ** 3 + Orf * (1 + z) ** 4 + de
    if m.get('T') is not None:
        e2 = e2 * m['T'](z) ** 2
    return e2


def _models():
    b = BEST
    return {
        'LCDM': dict(H0=b['LCDM']['H0'], Om=b['LCDM']['Om'], T=None, cpl=None),
        'B0':   dict(H0=b['B0']['H0'], Om=b['B0']['Om'], T=T_B0(), cpl=None),
        'free sech2': dict(H0=b['FREE']['H0'], Om=b['FREE']['Om'],
                           T=T_sech2(b['FREE']['A'], b['FREE']['tc'], b['FREE']['w']), cpl=None),
        'CPL':  dict(H0=b['CPL']['H0'], Om=b['CPL']['Om'], T=None,
                     cpl=(b['CPL']['w0'], b['CPL']['wa'])),
    }


def solve_growth(m):
    """D'' + (2 + dlnH/dN) D' = 1.5 Om(a) D;  D(a_i) = D'(a_i) = a_i, a_i = e^-6."""
    Om = m['Om']
    ai = np.exp(-6.0)

    def rhs(N, y):
        a = np.exp(N)
        z = 1.0 / a - 1.0
        e2 = _model_E2(m, z)
        dz = 1e-5 * max(1.0, z)
        zm = max(z - dz, 0.0)
        dlnE_dz = (np.log(_model_E2(m, z + dz)) - np.log(_model_E2(m, zm))) / (z + dz - zm) / 2.0
        dlnH_dN = -(1 + z) * dlnE_dz
        Oma = Om * a ** -3 / e2
        return [y[1], -(2 + dlnH_dN) * y[1] + 1.5 * Oma * y[0]]

    sol = solve_ivp(rhs, [np.log(ai), 0.0], [ai, ai], dense_output=True,
                    rtol=1e-9, atol=1e-12)
    D = lambda z: sol.sol(-np.log1p(z))[0]
    f = lambda z: sol.sol(-np.log1p(z))[1] / sol.sol(-np.log1p(z))[0]
    return D, f


def run_de_section():
    print("\n" + "=" * 76)
    print(" SECTION 12.  EFFECTIVE DARK ENERGY AND GROWTH (paper Section 4)")
    print("=" * 76)
    models = _models()
    zs = np.linspace(0, 3, 3001)
    print("\n -- effective dark energy: rho_DE,eff = E^2 - Om(1+z)^3 - Or(1+z)^4 --")
    for name in ('B0', 'free sech2'):
        m = models[name]
        h = m['H0'] / 100.0
        Orf = OMEGA_R_H2 / h ** 2
        rho = _model_E2(m, zs) - m['Om'] * (1 + zs) ** 3 - Orf * (1 + zs) ** 4
        lnrho = np.log(np.abs(rho) / abs(rho[0]))
        weff = -1 - np.gradient(lnrho, np.log1p(zs)) / 3
        imin, imax = np.argmin(weff), np.argmax(weff)
        cross = [zs[i] for i in range(1, len(zs))
                 if (weff[i] + 1) * (weff[i - 1] + 1) < 0 and zs[i] > 0.1]
        print(f"   {name:11s} rho>0 on [0,3]: {bool((rho > 0).all())}  w_eff(0) = {weff[0]:.3f}  "
              f"min {weff[imin]:.2f} (z={zs[imin]:.2f})  max {weff[imax]:.2f} (z={zs[imax]:.2f})  "
              f"crossing z = {cross[0]:.2f}  rho/rho0(0.62) = {np.interp(0.62, zs, rho / rho[0]):.2f}")
    # CPL emulation of B0 on [0, 2.5]
    m = models['B0']
    zz = np.linspace(0, 2.5, 200)
    def f(p):
        mm = dict(H0=m['H0'], Om=p[2], T=None, cpl=(p[0], p[1]))
        return np.sum((np.sqrt(_model_E2(mm, zz)) - np.sqrt(_model_E2(m, zz))) ** 2)
    r = minimize(f, [-0.9, -0.3, m['Om']], method='Nelder-Mead',
                 options=dict(xatol=1e-9, fatol=1e-14, maxiter=4000))
    print(f"   CPL emulation of B0 E(z) on [0,2.5]: w0 = {r.x[0]:.3f}  wa = {r.x[1]:.3f}  "
          f"rms dE/E = {np.sqrt(r.fun / 200):.4f}")

    print("\n -- growth (G_eff = G), 16 fsigma8 points, diagonal errors --")
    z, y, s = FSIGMA8_DATA.T
    for name, m in models.items():
        D, ff = solve_growth(m)
        shape = np.array([ff(zi) * D(zi) / D(0.0) for zi in z])
        A = np.sum(shape * y / s ** 2) / np.sum(shape ** 2 / s ** 2)
        chi2 = np.sum(((y - A * shape) / s) ** 2)
        chi2_811 = np.sum(((y - SIGMA8_PLANCK * shape) / s) ** 2)
        sig = np.sqrt(1.0 / np.sum(shape ** 2 / s ** 2))
        s8 = A * np.sqrt(m['Om'] / 0.3)
        print(f"   {name:11s} profiled sigma_8(0) = {A:.3f} +- {sig:.3f}  chi2 = {chi2:.2f}  "
              f"| chi2(sigma_8 = 0.811) = {chi2_811:.2f}  | S8 = {s8:.3f}")


# ============================================================================
# SECTION 13.  DESI DR3 TABLES  (paper Section 6)
# ============================================================================

def _bao_preds(m, rd=None):
    """Predicted (label, z, D/r_d) at the DR2 points for model m."""
    H0, Om = m['H0'], m['Om']
    ob = m.get('ob', 0.0225)
    E = E_background(Om, m.get('Tf'), m.get('wde'))
    DC = comoving_cumint(E, H0)
    if rd is None:
        rd = emu('rdrag', ob, Om * (H0 / 100.0) ** 2)
    out = []
    for z, DVo, s in DESI_DR2_DV:
        DM = np.interp(z, _ZG, DC)
        Hz = H0 * np.interp(z, _ZG, E)
        out.append(('BGS(DV)', z, (z * DM ** 2 * C_KMS / Hz) ** (1 / 3) / rd))
    for lab, z, DMo, sM, DHo, sH, rho in DESI_DR2_PAIR:
        out.append((lab + ' DM', z, np.interp(z, _ZG, DC) / rd))
        out.append((lab + ' DH', z, C_KMS / (H0 * np.interp(z, _ZG, E)) / rd))
    return out


def _chi2_mock(Xmock, Xtest, f):
    """chi^2 of Xtest against noiseless Xmock with sigma(DR3) = sigma(DR2)/f."""
    c, i = 0.0, 0
    for z, DVo, s in DESI_DR2_DV:
        c += ((Xmock[i][2] - Xtest[i][2]) / (Xmock[i][2] * s / DVo / f)) ** 2
        i += 1
    for lab, z, DMo, sM, DHo, sH, rho in DESI_DR2_PAIR:
        sMr = sM / DMo * Xmock[i][2] / f
        sHr = sH / DHo * Xmock[i + 1][2] / f
        cov = np.array([[sMr ** 2, rho * sMr * sHr], [rho * sMr * sHr, sHr ** 2]])
        d = np.array([Xmock[i][2] - Xtest[i][2], Xmock[i + 1][2] - Xtest[i + 1][2]])
        c += float(d @ np.linalg.solve(cov, d))
        i += 2
    return c


def run_dr3_section(quick=False):
    print("\n" + "=" * 76)
    print(" SECTION 13.  DESI DR3 TABLES (paper Section 6)")
    print("=" * 76)
    b = BEST
    mods = {
        'LCDM': dict(H0=b['LCDM']['H0'], Om=b['LCDM']['Om'], ob=b['LCDM']['ob']),
        'B0':   dict(H0=b['B0']['H0'], Om=b['B0']['Om'], ob=b['B0']['ob'], Tf=T_B0()),
        'free': dict(H0=b['FREE']['H0'], Om=b['FREE']['Om'], ob=b['FREE']['ob'],
                     Tf=T_sech2(b['FREE']['A'], b['FREE']['tc'], b['FREE']['w'])),
        'CPL':  dict(H0=b['CPL']['H0'], Om=b['CPL']['Om'], ob=b['CPL']['ob'],
                     wde=(b['CPL']['w0'], b['CPL']['wa'])),
    }
    pr = {k: _bao_preds(m) for k, m in mods.items()}

    # ---- Table: predicted offsets (percent) relative to LCDM
    sig = {('BGS(DV)', 0.295): DESI_DR2_DV[0][2] / DESI_DR2_DV[0][1] * 100}
    for lab, z, DMo, sM, DHo, sH, rho in DESI_DR2_PAIR:
        sig[(lab + ' DM', z)] = sM / DMo * 100
        sig[(lab + ' DH', z)] = sH / DHo * 100
    print("\n -- predicted D/r_d relative to LCDM (percent); sigma_DR3 = sigma_DR2 / 2 --")
    print(f"   {'observable':11s} {'z':>6s} {'sDR2':>6s} {'sDR3':>6s} {'B0':>7s} {'free':>7s}"
          f" {'CPL':>7s} {'(B0-CPL)/sDR3':>14s}")
    for i, (lab, z, x0) in enumerate(pr['LCDM']):
        d = {k: (pr[k][i][2] / x0 - 1) * 100 for k in ('B0', 'free', 'CPL')}
        s2 = sig[(lab, z)]
        print(f"   {lab:11s} {z:6.3f} {s2:6.2f} {s2 / 2:6.2f} {d['B0']:+7.2f} {d['free']:+7.2f}"
              f" {d['CPL']:+7.2f} {(d['B0'] - d['CPL']) / (s2 / 2):+14.2f}")

    # ---- Table: expected chi^2, refit (BAO only, f = 2, r_d = 147.45)
    print("\n -- expected BAO-only chi^2, refit (data = noiseless prediction; f = 2) --")
    RD = 147.45
    refit_mods = {k: dict(m) for k, m in mods.items()}
    def mock_of(k):
        return _bao_preds(mods[k], rd=RD)
    def refit(mock, kind):
        def cost(x):
            if not (0.15 < x[0] < 0.5 and 55 < x[1] < 85):
                return 1e9
            m = dict(H0=x[1], Om=x[0], ob=0.0225)
            if kind == 'B0':
                m['Tf'] = T_B0()
            elif kind == 'CPL':
                m['wde'] = (x[2], x[3])
            elif kind == 'free':
                if not (x[2] > 0 and x[4] > 0.03):
                    return 1e9
                m['Tf'] = T_sech2(x[2], x[3], x[4])
            return _chi2_mock(mock, _bao_preds(m, rd=RD), 2.0)
        starts = {'LCDM': [(0.30, 68), (0.31, 67)],
                  'B0': [(0.31, 67.5), (0.30, 68), (0.33, 66)],
                  'CPL': [(0.31, 67.5, -0.85, -0.5), (0.30, 68, -1, 0)],
                  'free': [(0.31, 67.5, 0.05, 0.48, 0.09), (0.30, 68, 0.04, 0.45, 0.12)]}[kind]
        return min((minimize(cost, s, method='Nelder-Mead',
                             options=dict(xatol=1e-7, fatol=1e-9, maxiter=4000)).fun
                    for s in starts))
    print(f"   {'data generated by':18s} {'B0 (locked)':>12s} {'free':>8s} {'LCDM':>8s} {'CPL':>8s}")
    for truth in ('LCDM', 'CPL', 'B0'):
        mk = mock_of(truth)
        print(f"   {truth:18s} {refit(mk, 'B0'):12.2f} {refit(mk, 'free'):8.2f} "
              f"{refit(mk, 'LCDM'):8.2f} {refit(mk, 'CPL'):8.2f}", flush=True)

    # ---- Table: frozen-parameter chi^2 and error rates
    print("\n -- expected frozen-parameter chi^2 (own r_d); scales as (f/2)^2 --")
    rows = [('B0', 'LCDM'), ('B0', 'CPL'), ('LCDM', 'B0'), ('CPL', 'B0'), ('CPL', 'LCDM')]
    base = {r: _chi2_mock(pr[r[0]], pr[r[1]], 2.0) for r in rows}
    print(f"   {'f':>4s} {'B0 data,LCDM':>13s} {'B0 data,CPL':>12s} {'LCDM data,B0':>13s} "
          f"{'CPL data,B0':>12s} {'CPL data,LCDM':>14s}")
    for f in (1.0, 1.3, 1.5, 2.0):
        sc = (f / 2.0) ** 2
        print(f"   {f:4.1f} " + " ".join(f"{base[r] * sc:{w}.2f}"
              for r, w in zip(rows, (13, 12, 13, 12, 14))))
    print("\n -- error probabilities of the rule 'reject B0 if chi2_frozen(B0) > chi2_frozen(LCDM)' --")
    mu0, mup0 = base[('B0', 'LCDM')], base[('LCDM', 'B0')]
    for f in (1.0, 1.3, 1.5, 2.0):
        sc = (f / 2.0) ** 2
        mu, mup = mu0 * sc, mup0 * sc
        print(f"   f = {f:.1f}: mu = {mu:6.2f}  P(reject B0 | B0) = {norm.sf(np.sqrt(mu) / 2):.3f}"
              f"   mu' = {mup:6.2f}  P(retain B0 | LCDM) = {norm.sf(np.sqrt(mup) / 2):.3f}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LVC v1.6 reproduction script "
                    "(r_drag and distance-prior inconsistencies corrected).")
    parser.add_argument('--quick', action='store_true',
                        help="Coarser grids and fewer fits (minutes instead of hours).")
    parser.add_argument('--only', type=str, default='',
                        help="Comma-separated sections: err,family,baseline,profile,de,dr3.")
    parser.add_argument('--skip-profiles', action='store_true',
                        help="Skip Section 11 (theta_c and A profiles).")
    parser.add_argument('--skip-dr3', action='store_true',
                        help="Skip Section 13 (DR3 tables).")
    parser.add_argument('--skip-error-section', action='store_true',
                        help="Skip Section 8 (effect of the two inconsistencies).")
    args = parser.parse_args()

    print("=" * 76)
    print(" LVC final-v1.6 reproduction script")
    print(" r_drag from CAMB; T multiplies H in all blocks; N = 1596.")
    print("=" * 76)
    print()
    print("  Constants:")
    print(f"    phi             = {PHI:.10f}")
    print(f"    ln phi          = {LNPHI:.10f}")
    print(f"    (ln phi)^3 = w  = {LNPHI_3:.10f}")
    print(f"    (ln phi)^4 = A  = {LNPHI_4:.10f}")
    print(f"    Omega_m lock    = {OM_RECURSIVE:.10f}   (v1.1; not carried forward)")

    wanted = set(s.strip() for s in args.only.split(',') if s.strip())
    run = lambda key: (not wanted) or (key in wanted)

    # CAMB table and distance-prior calibration
    _load_camb_table()
    _calibrate_dp()
    print(f"    CAL_R, CAL_lA   = {CFG['cal_r']:.6f}, {CFG['cal_la']:.6f}"
          f"   (fixed at the Planck 2018 fiducial)")

    pp = None
    if any(run(k) for k in ('err', 'family', 'baseline', 'profile')):
        print(f"\n  Loading Pantheon+ data from {PP_DIR} ...")
        pp = load_pantheonplus()
        print(f"  Loaded.  N_SN = {pp['N']}  (N_total = {pp['N'] + 13 + 3})")

    t0 = time.time()
    if run('err') and not args.skip_error_section:
        run_error_section(pp)
    if run('family'):
        run_family_section(pp, quick=args.quick)
    if run('baseline'):
        run_baseline_section(pp, quick=args.quick)
    if run('profile') and not args.skip_profiles:
        run_profile_section(pp, quick=args.quick)
    if run('de'):
        run_de_section()
    if run('dr3') and not args.skip_dr3:
        run_dr3_section(quick=args.quick)

    print("\n" + "=" * 76)
    print(f" Reproduction complete in {time.time() - t0:.1f} s")
    print("=" * 76)


if __name__ == "__main__":
    main()
