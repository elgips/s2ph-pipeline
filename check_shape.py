"""Diagnose why forward returns 342 = 2*171, and read the base of each half."""
import numpy as np
from pygimli.physics import ert
from Anandlyn_log import AnomalyWorld, Layer, CircleAnom
from pwhg_forward_soft import SoftTriForward
import pwhg_wrapper as W

scheme = ert.createData(elecs=np.linspace(-25, 25, 21), schemeName='dd')
world = AnomalyWorld(
    _start=[-25, 0], _end=[25, -20], _scheme=scheme,
    _layers=[Layer(name='layer1', y=-3.0, rho=50.0, _cnum=1,
                   _varflag=[True, True], _varlims=((-10, -1), (1, 1e5)))],
    _circles=[CircleAnom(name='c1', x=5.0, y=-4.0, r=1.2, rho=500.0, _c_num=1,
                         _varflag=[True, True, True, True],
                         _varlims=((-25, 25), (-10, 0), (0.5, 5), (1, 1e5)))],
    _rho_world=100.0, _rho_world_varflag=True, _rho_world_varlim=[50, 1000])
world.parse_all(force_regenerate=True)
rhoa = np.asarray(world.get_forward_solution(_noise=False)['rhoa'], float)
m = rhoa.size
print(f"reference m = {m}")

soft = SoftTriForward(world, width=0.2, tri_area=0.25, scheme=scheme)
th = np.asarray(soft.get_x0(), float)

for tag, fwd in [("wrap", lambda t: W.forward(world, t)),
                 ("soft", lambda t: soft.forward(t))]:
    y = np.asarray(fwd(th), float)
    print(f"\n{tag}: shape={y.shape}")
    print(f"  head[:3]      = {np.round(y[:3], 4)}")
    print(f"  around m      = {np.round(y[m-2:m+2], 4)}")
    print(f"  tail[-3:]     = {np.round(y[-3:], 4)}")
    if y.size == 2 * m:
        a, b = y[:m], y[m:]
        print(f"  half A range  = [{a.min():.4f},{a.max():.4f}]")
        print(f"  half B range  = [{b.min():.4f},{b.max():.4f}]")
        print(f"  A == B?       = {np.allclose(a, b)}")
        for nm, half in [("A", a), ("B", b)]:
            e_ln  = np.median(np.abs(np.exp(half) - rhoa) / rhoa)
            e_log = np.median(np.abs(10.0**half - rhoa) / rhoa)
            e_lin = np.median(np.abs(half - rhoa) / rhoa)
            best = min([("ln", e_ln), ("log10", e_log), ("linear", e_lin)],
                       key=lambda t: t[1])
            print(f"  half {nm} base -> {best[0]:7s} "
                  f"(ln={e_ln:.2e} log10={e_log:.2e} lin={e_lin:.2e})")

eps = np.asarray(W.noise_std(world, rhoa), float)
print(f"\nnoise_std shape = {eps.shape} (m={m})")
