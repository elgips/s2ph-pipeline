"""Decisive check: what is the whitened residual AT theta_true?
If ~1: data is fittable, the loop stalls (optimizer/seed). 
If ~2+: irreducible data-model inconsistency (the whit-floor is structural,
        independent of the estimate) -> then dy 'freeze' is the model, not GN.
Also: does the layer-depth offset in the seed drive it?"""
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

scheme=make_scheme()
theta_true=np.array([2.0,-3.0,np.log10(50),5.0,-4.0,np.log10(1.2),np.log10(500)])
rng=np.random.default_rng(0)

# generate self-consistent data with SOFT at truth (exactly as the loop does when data_forward='soft')
world_t=world_from_theta(theta_true,scheme)
soft_t=SoftTriForward(world_t,width=0.2,tri_area=0.25,scheme=scheme)
ln_true=np.asarray(soft_t.forward(theta_true))[0]
rhoa_true=np.exp(ln_true)
eps=np.asarray(W.noise_std(world_t,rhoa_true,dU=1e-6,floor=5e-3),float)
noise=eps*rng.standard_normal(eps.size)
d=ln_true+noise                          # noisy self-consistent data

# whit at theta_true with the SAME soft forward, various sigma_model
soft_inf=SoftTriForward(world_from_theta(theta_true,scheme),width=0.2,tri_area=0.25,scheme=scheme)
g_true=np.asarray(soft_inf.forward(theta_true))[0]
r=d-g_true
print("data-gen vs inference forward identical? ", np.allclose(ln_true, g_true))
print(f"||ln_true - g_true(theta_true)|| = {np.linalg.norm(ln_true-g_true):.2e}  (should be ~0)")
for sm in [0.0, 0.005, 0.01, 0.02]:
    nv=eps**2+sm**2
    whit=np.sqrt(np.mean(r**2/nv))
    print(f"  sigma_model={sm:.3f}: whit(theta_true) = {whit:.3f}")

# now the loop's actual quirk: it re-linearizes soft at MOVING theta but the
# data d_all is soft(theta_true). Is there a fresh-SoftTriForward inconsistency?
# build TWO soft objects at DIFFERENT theta and compare their forward at theta_true
soft_a=SoftTriForward(world_from_theta(theta_true,scheme),width=0.2,tri_area=0.25,scheme=scheme)
theta_off=theta_true.copy(); theta_off[4]=-4.3   # a nearby depth (like theta_hat0)
soft_b=SoftTriForward(world_from_theta(theta_off,scheme),width=0.2,tri_area=0.25,scheme=scheme)
ga=np.asarray(soft_a.forward(theta_true))[0]
gb=np.asarray(soft_b.forward(theta_true))[0]
print(f"\nSAME theta_true through soft built at DIFFERENT theta:")
print(f"  ||g_a - g_b|| = {np.linalg.norm(ga-gb):.4e}   max={np.max(np.abs(ga-gb)):.4e}")
print("  (if >0, SoftTriForward is NOT stateless in theta -> the moving-theta")
print("   re-linearization corrupts the data match; that is the whit floor.)")
