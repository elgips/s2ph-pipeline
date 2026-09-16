# s2ph-pipeline

S2PH ERT inversion pipeline: config, forward models, sequential OED, CV stages, validation.

## Dependencies
- Python 3.10+
- pyGIMLi 1.6.0
- numpy, scipy, openpyxl, matplotlib, shapely

## Environment
Run in the `ERT_GUI` conda environment (pygimli_env for sequential modules).

## Structure

### S2PH pipeline (Article 1)
- `config.py` — single source of truth for geometry, noise, CV thresholds, paths
- `pipelib.py` — checkpointing, manifests, validation gates
- `stage1_forward.py` — build worlds, compute forward responses
- `stage2_invert.py` — smooth L2 inversion + deterministic raster export
- `stage3_cv.py` — CV segmentation sweep, component audit, initializer extraction
- `run_cij_opti.py` — PWHG optimisation (ls_opti_cs DE optimizer)
- `Anandlyn_log.py` — AnomalyWorld class (PWHG parameterization)
- `ert_interpreter.py` — patched CV/segmentation GUI functions

### Sequential OED (Article 3)
- `pwhg_wrapper.py` — forward, Jacobian, noise model, Fisher/posterior
- `pwhg_forward_soft.py` — SoftTriForward (stateless, smooth sigmoid boundaries)
- `seq_local_update.py` — GN/Laplace update with eigenvalue floor + containment
- `seq_coldstart.py` — epsilon-consistent cold-start dataset generation
- `seq_handoff.py` — CV seed → handoff state (theta_hat0, Sigma0)
- `seq_greedy.py` — greedy D-optimal EIG acquisition loop
- `seq_sweep.py` — characterize greedy loop across prior-drawn scenes
- `seq_parallel.py` — process-parallel SoftTri forward/Jacobian
- `seq_log.py` — logging
- `seq_sentinel_softde.py`, `seq_sentinel_ls.py` — sentinel re-globalization
- `seq_debias_compare.py` — SoftTri vs remesh polish comparison
- `seq_scene_score.py` — Article 1 Q_scene metric for sequential results
- `seq_radius_sweep.py` — controlled 1D radius-threshold experiment
- `build_pool.py` — comprehensive candidate pool construction
- `oed_linearized.py`, `oed_c1.py` — linearized OED gates

### Validation / diagnostics
- `validate_sensitivity.py` — deep check of PWHG sensitivity/Fisher
- `validate_geometry.py` — geometry validation
- `validate_soft.py` — soft forward validation
- `check_base.py`, `check_cov.py`, `check_fix_rho.py`, `check_saturation.py`, `check_shape.py`, `check_jac_depth.py`, `check_par_vs_serial_jac.py`, `check_whit_true.py`
- `compare_arrays.py`, `compare_forward_meshes.py`
- `scene_metric.py` — sensitivity/contrast-weighted scene quality score
- `wk_report.py` — w_k analysis for over-specified scenes

### Benchmarks
- `bench_forward_cost.py`, `bench_parallel.py`, `bench_tri_area.py`
