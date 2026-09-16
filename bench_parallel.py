"""
bench_parallel.py -- where does the parallel time actually go?

The one-Jacobian self-test showed only 1.4x. Hypothesis: per-worker startup
(pygimli import + mesh + SoftTriForward build) dominates a single 15-job
Jacobian. This measures:
  (a) pool STARTUP cost alone,
  (b) per-Jacobian cost once the pool is warm (the number that matters for the
      greedy loop, which reuses one pool across many acquisitions),
  (c) scaling vs worker count, to find the useful worker number.

Run:  python bench_parallel.py
"""
import time
import numpy as np
from seq_parallel import ParallelSoft, geometric_factors


def main():
    from seq_greedy import build_pool
    sch, quads, xe = build_pool(n_elec=21, n_max=400)
    quads = np.column_stack([np.asarray(sch[t], int) for t in "abmn"])
    k = np.asarray(sch["k"], float)
    th = np.array([2.0, -3.0, np.log10(50), 5.0, -4.0, np.log10(1.2),
                   np.log10(500)])
    steps = np.array([0.005, 0.01, 0.005, 0.01, 0.01, 0.005, 0.005])
    N_REPEAT = 5

    # --- serial baseline: cost of ONE jacobian, warm ---
    ser = ParallelSoft(xe, quads, k, th, n_workers=1, verbose=False)
    t0 = time.time(); ser._init_serial(); t_ser_init = time.time() - t0
    t0 = time.time()
    for _ in range(N_REPEAT):
        ser.jacobian(th, steps)
    t_ser_jac = (time.time() - t0) / N_REPEAT
    print(f"serial : init {t_ser_init:5.2f}s   per-Jacobian {t_ser_jac:5.2f}s")

    print(f"\n{'workers':>8}{'startup':>10}{'per-Jac':>10}{'speedup':>9}"
          f"{'amort@20':>10}")
    for nw in [2, 4, 8, 12, 16]:
        t0 = time.time()
        ps = ParallelSoft(xe, quads, k, th, n_workers=nw, verbose=False)
        ps.start()
        t_start = time.time() - t0
        # warm pool: time repeated Jacobians
        t0 = time.time()
        for _ in range(N_REPEAT):
            ps.jacobian(th, steps)
        t_jac = (time.time() - t0) / N_REPEAT
        ps.close()
        spd = t_ser_jac / max(t_jac, 1e-9)
        # effective speedup if the pool is reused for 20 Jacobians (greedy loop)
        amort = (20 * t_ser_jac) / max(t_start + 20 * t_jac, 1e-9)
        print(f"{nw:>8}{t_start:>10.2f}{t_jac:>10.2f}{spd:>9.1f}x{amort:>9.1f}x")

    print("\nread: 'per-Jac' speedup is the real gain once the pool is warm;")
    print("'amort@20' is what the greedy loop sees (one pool, 20 acquisitions).")


if __name__ == "__main__":
    main()
