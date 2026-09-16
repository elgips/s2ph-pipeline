"""Diagnose the C0_y (depth) Jacobian column: is the FD step too small for the
soft-boundary depth direction, giving a near-zero / step-dependent derivative?"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np, logging
import pygimli as pg
try: pg.setDebug(False)
except Exception: pass
try:
    from pygimli.utils import cache as _c; _c.CacheManager().cachingActive=False
except Exception: pass
logging.getLogger("pyGIMLi").setLevel(logging.ERROR)
logging.getLogger("Core").setLevel(logging.CRITICAL)

import pwhg_wrapper as W
from pwhg_forward_soft import SoftTriForward
from seq_coldstart import make_scheme, world_from_theta

NAMES=['world_lrho','L0_y','L0_lrho','C0_x','C0_y','C0_lr','C0_lrho']
scheme=make_scheme()
theta=np.array([2.0,-3.0,np.log10(50),5.0,-4.0,np.log10(1.2),np.log10(500)])
world=world_from_theta(theta,scheme)
soft=SoftTriForward(world,width=0.2,tri_area=0.25,scheme=scheme)
f=lambda th: np.asarray(soft.forward(th))[0]     # ln rhoa

steps_default=np.asarray(W.default_steps(world),float)
print("default FD steps:", dict(zip(NAMES, np.round(steps_default,4))))

# central-difference column for a given param at several step sizes
def col(p, h):
    tp=theta.copy(); tp[p]+=h
    tm=theta.copy(); tm[p]-=h
    return (f(tp)-f(tm))/(2*h)

print(f"\n{'param':<10}{'step':>8}{'||col||':>12}{'max|col|':>12}")
for p in [3,4,5,6]:  # geometric params
    for h in [0.005, 0.02, 0.05, 0.1, 0.2, 0.4]:
        c=col(p,h)
        print(f"{NAMES[p]:<10}{h:>8.3f}{np.linalg.norm(c):>12.5f}{np.max(np.abs(c)):>12.5f}")
    print()

# convergence check: does the C0_y column stabilize as h shrinks, or is it noise?
print("C0_y column direction-consistency across steps (cos vs h=0.05 ref):")
ref=col(4,0.05); ref/= (np.linalg.norm(ref)+1e-30)
for h in [0.005,0.01,0.02,0.05,0.1,0.2]:
    c=col(4,h); c/=(np.linalg.norm(c)+1e-30)
    print(f"  h={h:.3f}  cos={float(ref@c):+.4f}  ||col||={np.linalg.norm(col(4,h)):.5f}")

# compare depth sensitivity to lateral: is dy genuinely weaker than dx here?
print(f"\nrelative geometric sensitivity at theta (step=0.05):")
for p in [3,4,5,6]:
    print(f"  {NAMES[p]:<8} ||col||={np.linalg.norm(col(p,0.05)):.5f}")
