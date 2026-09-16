"""
check_base.py -- determine the log base of pwhg_wrapper.forward / SoftTriForward.
Run in ERT_GUI. Fully wired to the real API; no hooks to fill.

Everything downstream (jacobian_generic, noise_std, fisher, posterior_cov,
gamma) is self-consistent WITH the forward's base by construction. So the only
question is: is `forward` natural-log? seq_local_update requires natural-log.

Decisive readouts:
  [A] base of forward: exp(y) vs 10**y vs y, each compared to LINEAR rhoa.
      - wrap (remesh) should match its base to ~1e-12 (it IS the reference fwd).
      - soft should match the SAME base to ~2% (the SoftTri blur, not a base err).
  [B] soft vs wrap agree on base (surrogate consistent with faithful).
  [C] noise_std: min eps should equal the floor (0.005) => eps is a relative /
      natural-log std, which is what fisher() assumes.
"""
import numpy as np
from pygimli.physics import ert
from Anandlyn_log import AnomalyWorld, Layer, CircleAnom
from pwhg_forward_soft import SoftTriForward
import pwhg_wrapper as W

LN10 = np.log(10.0)

# C11 scene WITH varflags. rho_world_varflag=True is REQUIRED: with it
# False, AnomalyWorld.get_forward_solution returns an empty container.
# So theta0 leads with world_log10rho (p=7, the C_1,1 case SoftTriForward expects).
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
rhoa_lin = np.asarray(world.get_forward_solution(_noise=False)['rhoa'], float)
if rhoa_lin.size == 0:  # defensive fallback
    d = ert.simulate(world.mesh, scheme=scheme, res=world.resistivity_map, verbose=False)
    rhoa_lin = np.asarray(d['rhoa'], float)

# SoftTriForward carries its OWN parameterization; use it as the source of truth.
soft = SoftTriForward(world, width=0.2, tri_area=0.25, scheme=scheme)
theta0 = np.asarray(soft.get_x0(), float)
labels = soft.param_labels()
print(f"soft labels ({len(labels)}): {labels}")
print(f"theta0 (from soft): {np.round(theta0, 3)}")
print(f"linear rhoa: m={rhoa_lin.size}, range [{rhoa_lin.min():.2f},{rhoa_lin.max():.2f}]")


def base_of(y, ref):
    e_ln  = np.median(np.abs(np.exp(y)   - ref) / ref)
    e_log = np.median(np.abs(10.0 ** y   - ref) / ref)
    e_lin = np.median(np.abs(y           - ref) / ref)
    name, err = min([("natural-log", e_ln), ("log10", e_log), ("linear", e_lin)],
                    key=lambda t: t[1])
    return name, err, (e_ln, e_log, e_lin)


# --- forwards -------------------------------------------------------------
y_wrap = np.asarray(W.forward(world, theta0), float)
y_soft = np.asarray(soft.forward(theta0), float)

print("\n[A] base of forward (compared to LINEAR rhoa)")
for tag, y in [("wrap", y_wrap), ("soft", y_soft)]:
    if y.size != rhoa_lin.size:
        print(f"    {tag}: size {y.size} != {rhoa_lin.size}; skipping")
        continue
    name, err, (eln, elog, elin) = base_of(y, rhoa_lin)
    print(f"    {tag}: exp={eln:.2e}  10**={elog:.2e}  id={elin:.2e}  "
          f"-> {name.upper()} (resid {err:.2e}"
          f"{' = blur, expected ~2e-2' if tag=='soft' else ''})")

# --- [B] soft vs wrap same base ------------------------------------------
if y_soft.size == y_wrap.size:
    same = base_of(y_soft, np.exp(y_wrap))[1]   # if both ln, exp(y_wrap)=rhoa
    print(f"\n[B] soft vs wrap: soft matches exp(wrap) at {same:.2e} "
          f"(~2e-2 blur if both natural-log)")

# --- [C] noise_std base ---------------------------------------------------
eps = np.asarray(W.noise_std(world, rhoa_lin), float)
print("\n[C] noise_std")
print(f"    eps range [{eps.min():.5f},{eps.max():.5f}]  floor should be 0.005")
print(f"    min eps == floor(0.005)? {np.isclose(eps.min(), 0.005, atol=1e-4)}"
      f"  -> eps is a relative / natural-log std (what fisher assumes)")
print(f"    heteroscedastic spread max/min = {eps.max()/eps.min():.2f} "
      f"(~1 => flat, OED collapses toward geometric)")

# --- verdict --------------------------------------------------------------
nm_wrap = base_of(y_wrap, rhoa_lin)[0]
print("\nVERDICT:")
print(f"  forward base = {nm_wrap.upper()}. "
      + ("Consistent with seq_local_update; jacobian_generic + noise_std + "
         "fisher all natural-log by construction."
         if nm_wrap == "natural-log"
         else "NOT natural-log -- fisher() and seq_local_update need a "
              f"correction of ln(10)^k. Do NOT proceed until resolved."))
