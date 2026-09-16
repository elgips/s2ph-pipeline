"""
check_par_vs_serial_jac.py -- does the PARALLEL Jacobian equal the loop's own
serial one (pool_forward_jac -> W.jacobian_generic)?

If these differ at all, turning parallelism on would silently change the greedy
loop's behaviour. They must match to ~machine precision (same central-difference
formula, same steps, same quads).
"""
import numpy as np
from seq_greedy import build_pool, pool_forward_jac
from seq_coldstart import world_from_theta
from pwhg_forward_soft import SoftTriForward
from seq_parallel import ParallelSoft
import pwhg_wrapper as W


def main():
    sch, quads, xe = build_pool(n_elec=21, n_max=300)
    quads = np.column_stack([np.asarray(sch[t], int) for t in "abmn"])
    k = np.asarray(sch["k"], float)
    xe = np.asarray([sch.sensor(i)[0] for i in range(sch.sensorCount())], float)

    th = np.array([2.0, -3.0, np.log10(50), 5.0, -4.0, np.log10(1.2),
                   np.log10(500)])
    world = world_from_theta(th, sch)
    soft = SoftTriForward(world, width=0.2, tri_area=0.25, scheme=sch)
    steps = np.asarray(W.default_steps(world), float)

    print(f"pool m={len(quads)}  steps={np.round(steps,4)}")

    # serial path exactly as the greedy loop computes it
    Js, lns, _ = pool_forward_jac(soft, th, steps)
    Js = np.asarray(Js, float); lns = np.asarray(lns, float)

    # parallel path
    with ParallelSoft(xe, quads, k, th, n_workers=8, verbose=True) as ps:
        Jp, lnp = ps.jacobian(th, steps)

    dJ = np.max(np.abs(Jp - Js))
    dl = np.max(np.abs(lnp - lns))
    relJ = dJ / (np.max(np.abs(Js)) + 1e-30)
    print(f"\nmax |J_par - J_ser|   = {dJ:.3e}   (relative {relJ:.3e})")
    print(f"max |ln_par - ln_ser| = {dl:.3e}")
    ok = np.allclose(Jp, Js, rtol=1e-8, atol=1e-10) and np.allclose(lnp, lns)
    print(f"\n{'PASS - parallel is a faithful drop-in' if ok else 'FAIL - do NOT enable parallelism'}")
    if not ok:
        bad = np.unravel_index(np.argmax(np.abs(Jp - Js)), Js.shape)
        print(f"  worst element {bad}: ser={Js[bad]:.6e} par={Jp[bad]:.6e}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
