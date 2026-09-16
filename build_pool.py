# -*- coding: utf-8 -*-
"""
build_pool.py
=============
Enrich the OED candidate pool beyond the basic dipole-dipole set. The forward
solve scales with the number of electrodes (per-source FE solves + superposition),
NOT with the number of quadrupoles, so a comprehensive pool costs essentially the
same forward time as the 171-quad dd set. The greedy loop is O(n_quad * p^2 *
budget), trivial for p=7. So the only reason to prune is physics: drop configs
whose expected voltage is below the noise floor.

Pool = all non-reciprocal 4-electrode configurations (3 * C(N,4)), minus configs
with |k| too large (expected u = rho_bg/|k| below u_floor). Reciprocal pairs
(AB,MN) and (MN,AB) carry identical information and identical modeled noise, so
they are deduplicated.

Returns a pygimli DataContainerERT with 'k' set, usable directly as the scheme in
SoftTriForward(world, scheme=pool).

Run in ERT_GUI. [PYGIMLI?] geometric-factor constructor name may vary by version.
"""

from itertools import combinations

import numpy as np
import pygimli as pg
import pygimli.physics.ert as ert


def _geometric_factors(data):
    for fn in ("createGeometricFactors", "geometricFactors"):
        f = getattr(ert, fn, None)
        if f is not None:
            try:
                return np.asarray(f(data), float)
            except Exception:
                continue
    raise RuntimeError("no geometric-factor constructor found for this pygimli.")


def _make_container(elec_x, abmn):
    data = pg.DataContainerERT()
    for x in elec_x:
        data.createSensor([float(x), 0.0])
    a, b, m, n = abmn
    data.resize(len(a))
    data["a"] = np.asarray(a, float)
    data["b"] = np.asarray(b, float)
    data["m"] = np.asarray(m, float)
    data["n"] = np.asarray(n, float)
    data["valid"] = np.ones(len(a))
    return data


def build_comprehensive_pool(world, rho_bg=None, dU=1e-3, u_floor=5e-3,
                             k_max=None, verbose=True):
    """
    world    : AnomalyWorld (for electrode positions and background rho).
    u_floor  : minimum expected voltage (V) at I=1A on a rho_bg half-space; configs
               with smaller expected voltage are dropped. Sets the SNR filter.
    k_max    : optional hard cap on |k|; overrides u_floor if given.
    Returns  : (DataContainerERT pool, info dict).
    """
    elec_x = np.array(world.scheme.sensorPositions())[:, 0]
    N = len(elec_x)
    if rho_bg is None:
        rho_bg = float(world.rho_world)

    # enumerate non-reciprocal 4-electrode configs
    seen = set()
    a_l, b_l, m_l, n_l = [], [], [], []
    for four in combinations(range(N), 4):
        p, q, r, s = four
        for cur, pot in (((p, q), (r, s)), ((p, r), (q, s)), ((p, s), (q, r))):
            key = frozenset({cur, pot})            # dedups the reciprocal
            if key in seen:
                continue
            seen.add(key)
            a_l.append(cur[0]); b_l.append(cur[1])
            m_l.append(pot[0]); n_l.append(pot[1])

    full = _make_container(elec_x, (a_l, b_l, m_l, n_l))
    k = _geometric_factors(full)

    if k_max is None:
        k_max = rho_bg / u_floor                    # |u| = rho_bg/|k| >= u_floor
    keep = np.isfinite(k) & (np.abs(k) <= k_max)

    pool = _make_container(elec_x, (np.array(a_l)[keep], np.array(b_l)[keep],
                                    np.array(m_l)[keep], np.array(n_l)[keep]))
    pool["k"] = k[keep]

    info = dict(n_full=len(a_l), n_pool=int(keep.sum()), k_max=float(k_max),
                eps_floor=0.005,
                eps_at_kmax=0.005 + dU * k_max / rho_bg)
    if verbose:
        print(f"comprehensive pool: {info['n_full']} configs -> "
              f"{info['n_pool']} after |k|<= {k_max:.0f} "
              f"(eps <= {info['eps_at_kmax']:.2f}); dd baseline ~171")
    return pool, info


def main():
    from validate_sensitivity import build_c11_world
    world = build_c11_world()
    for uf in (1e-2, 5e-3, 2e-3):
        _, info = build_comprehensive_pool(world, u_floor=uf)
        print(f"  u_floor={uf:.0e} -> pool size {info['n_pool']}")


if __name__ == "__main__":
    main()
