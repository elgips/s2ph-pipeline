"""
seq_coldstart.py -- generate an epsilon-consistent cold-start dataset (piece 2a).

Writes {name}_noisy.dat in the pipeline's own format so your UNCHANGED stage2
(smooth invert) and stage3 (CV) consume it directly. Datum error is
    eps_i = floor + |dU * k_i / rhoa_i|   (pwhg_wrapper.noise_std, floor=0.005)
applied as log-normal multiplicative noise AND written into the container's
`err` field, so the smooth inversion is itself eps-weighted -- one noise model
shared by cold-start seed, Sigma_0, and greedy Fisher/EIG.

STAGED NOISE PLAN (current phase = LOW NOISE):
  dU=1e-6 (default here): the floor dominates, eps ~ flat 0.5% (spread ~1).
    EIG selection has ~no noise-weighting structure, so any adaptive gain is
    from the mean moving + re-linearizing J at theta_hat -- the mechanism we
    test first against the deep-feature (depth-limit) problem.
  dU=1e-3 (later phase): strongly heteroscedastic (eps spread ~90x). Advance
    to it ONLY after the low-noise loop shows improvement. It is a one-
    parameter change, not a code change.

Boundary respected: this does NOT touch inversion or CV. It only produces the
.dat. Run your stage2/stage3 on it as usual to get the seed in parameters.xlsx.

Parameterization: 7-param C_1,1  [world_rho, L0_y, L0_rho, C0_x, C0_y, C0_r, C0_rho]
with the two resistivity/radius entries in log10 (matching AnomalyWorld.get_x0).
"""
import json
import numpy as np
from pygimli.physics import ert
from Anandlyn_log import AnomalyWorld, Layer, CircleAnom
import pwhg_wrapper as W
from seq_log import Log

# ordering of the 7-param C_1,1 theta, for reference / validation
THETA_LABELS = [('world', 'rho'), ('L0', 'y'), ('L0', 'rho'),
                ('C0', 'x'), ('C0', 'y'), ('C0', 'r'), ('C0', 'rho')]


def make_scheme(n_elec=21, x0=-25.0, x1=25.0, scheme_name='dd'):
    return ert.createData(elecs=np.linspace(x0, x1, n_elec), schemeName=scheme_name)


def world_from_theta(theta, scheme):
    """Build an AnomalyWorld at a 7-param C_1,1 theta.

    theta = [world_log10rho, L0_y, L0_log10rho, C0_x, C0_y, C0_log10r, C0_log10rho]
    NOTE: _rho_world_varflag=True is REQUIRED (with False, get_forward_solution
    returns an empty container -- a quirk in AnomalyWorld). rho values are set
    in linear space in the constructor; theta carries log10 for rho/r.
    """
    th = np.asarray(theta, float)
    w_rho = 10.0 ** th[0]
    L0_y, L0_rho = th[1], 10.0 ** th[2]
    C0_x, C0_y = th[3], th[4]
    C0_r, C0_rho = 10.0 ** th[5], 10.0 ** th[6]
    world = AnomalyWorld(
        _start=[-25, 0], _end=[25, -20], _scheme=scheme,
        _layers=[Layer(name='layer1', y=float(L0_y), rho=float(L0_rho), _cnum=1,
                       _varflag=[True, True], _varlims=((-10, -1), (1, 1e5)))],
        _circles=[CircleAnom(name='c1', x=float(C0_x), y=float(C0_y),
                             r=float(C0_r), rho=float(C0_rho), _c_num=1,
                             _varflag=[True, True, True, True],
                             _varlims=((-25, 25), (-10, 0), (0.5, 5), (1, 1e5)))],
        _rho_world=float(w_rho), _rho_world_varflag=True,
        _rho_world_varlim=[1, 1e5])
    return world


def clean_rhoa(world):
    """Reliable clean apparent resistivity via the wrapper forward (row 1),
    sidestepping the flaky get_forward_solution. Returns (rhoa_lin, theta)."""
    theta = np.asarray(world.get_x0(), float)
    y = np.asarray(W.forward(world, theta), float)   # (2, m): [ln rhoa, rhoa]
    if y.ndim == 2 and y.shape[0] == 2:
        return y[1], theta
    # fallback: single row -> assume linear if large, ln otherwise
    row = y.ravel()
    return (row if np.median(row) > 10 else np.exp(row)), theta


def eps_noisy(world, rng, dU=1e-6, floor=5e-3):
    """Return (rhoa_clean, rhoa_noisy, eps) with log-normal eps noise."""
    rhoa_c, _ = clean_rhoa(world)
    eps = np.asarray(W.noise_std(world, rhoa_c, dU=dU, floor=floor), float)
    z = rng.standard_normal(rhoa_c.size)
    rhoa_n = rhoa_c * np.exp(eps * z)          # ln(rhoa_n) = ln(rhoa_c) + eps*z
    return rhoa_c, rhoa_n, eps


def write_coldstart_dat(path, world, rng, dU=1e-6, floor=5e-3,
                        also_clean=None, geom_path=None):
    """Write an eps-noisy .dat consumable by stage2/stage3.

    Takes the clean data container (has a,b,m,n,k,valid), overrides `rhoa` with
    the eps-noisy values and `err` with eps, saves. Returns dict with the eps,
    the clean/noisy rhoa, and the achieved relative rms.
    """
    theta = np.asarray(world.get_x0(), float)
    d = world.get_forward_solution(_noise=False)   # container to carry fields
    rhoa_c = np.asarray(d['rhoa'], float)
    if rhoa_c.size == 0:
        # flaky get_forward_solution -> rebuild via simulate, then re-fetch
        world.parse_all(force_regenerate=True)
        d = ert.simulate(world.mesh, scheme=world.scheme,
                         res=world.resistivity_map, verbose=False)
        rhoa_c = np.asarray(d['rhoa'], float)
    eps = np.asarray(W.noise_std(world, rhoa_c, dU=dU, floor=floor), float)
    z = rng.standard_normal(rhoa_c.size)
    rhoa_n = rhoa_c * np.exp(eps * z)

    d.set('rhoa', rhoa_n)
    d.set('err', eps)                              # eps-weighted inversion
    d.save(str(path))

    if also_clean is not None:
        dc = world.get_forward_solution(_noise=False)
        dc.save(str(also_clean))
    if geom_path is not None:
        geom_path.write_text(json.dumps(dict(
            theta=theta.tolist(), labels=[list(t) for t in THETA_LABELS],
            dU=dU, floor=floor, m=int(rhoa_c.size)), indent=2))

    rel_rms = float(np.sqrt(np.mean(((rhoa_n - rhoa_c) / rhoa_c) ** 2)))
    _lg = Log("coldstart")
    _lg.info("wrote .dat", str(path))
    _lg.info(f"m={rhoa_c.size} dU={dU} floor={floor}")
    _lg.check("eps range", float(eps.max()), lo=floor*0.9, hi=1.0)
    _lg.check("noise rel_rms ~ mean(eps)", rel_rms/float(np.mean(eps)),
              lo=0.3, hi=3.0)
    return dict(theta=theta, eps=eps, rhoa_clean=rhoa_c, rhoa_noisy=rhoa_n,
                rel_rms=rel_rms, m=int(rhoa_c.size), path=str(path))


def _selftest():
    rng = np.random.default_rng(0)
    scheme = make_scheme()
    # nominal C_1,1: world 100, layer y=-3 rho=50, circle (5,-4) r=1.2 rho=500
    theta = np.array([2.0, -3.0, np.log10(50), 5.0, -4.0, np.log10(1.2),
                      np.log10(500)])
    world = world_from_theta(theta, scheme)
    got = np.asarray(world.get_x0(), float)
    print("theta set   :", np.round(theta, 3))
    print("world get_x0:", np.round(got, 3))
    ok_theta = np.allclose(got, theta, atol=1e-6)

    import tempfile, os
    from pathlib import Path
    tmp = Path(tempfile.mkdtemp())
    res = write_coldstart_dat(tmp / "T_noisy.dat", world, rng,
                              also_clean=tmp / "T_clean.dat",
                              geom_path=tmp / "T_geom.json")
    print(f"m={res['m']}  eps range [{res['eps'].min():.4f},{res['eps'].max():.4f}]"
          f"  rel_rms={res['rel_rms']:.4f}")

    # re-read and check rhoa/err round-trip
    d2 = ert.load(str(tmp / "T_noisy.dat"))
    rhoa2 = np.asarray(d2['rhoa'], float)
    err2 = np.asarray(d2['err'], float)
    rt_rhoa = np.allclose(rhoa2, res['rhoa_noisy'], rtol=1e-4)
    rt_err = np.allclose(err2, res['eps'], rtol=1e-4)
    # achieved noise rms should sit near the mean eps (within a factor)
    near = 0.3 < res['rel_rms'] / np.mean(res['eps']) < 3.0
    print(f"round-trip rhoa/err: {rt_rhoa}/{rt_err}   rms~mean(eps): {near}")

    ok = ok_theta and rt_rhoa and rt_err and near
    print(f"\nSELF-TEST {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
