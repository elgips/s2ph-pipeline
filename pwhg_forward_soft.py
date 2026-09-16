# -*- coding: utf-8 -*-
"""
pwhg_forward_soft.py
====================
The forward model chosen after the mesh comparison: a single uniform triangular
mesh (pygimli refines near electrodes -> accurate solve) with a SOFT sigmoid
boundary of width 0.2 m. This is the model we will actually invert, so its
Jacobian is self-consistent and -- unlike the remesh path -- smooth in the
geometric parameters (validated: all of C0/r, C0/x, L0/y show FD convergence
valleys at width 0.2, roughness ~1e-2, blur cost ~1.9% << data noise floor).

SoftTriForward is stateless: every call rebuilds the resistivity field on the
same fixed mesh, so there is no object mutation to restore and it is safe to call
from parallel workers. It exposes the same (ln_rhoa, rhoa) forward contract and a
.scheme attribute, so pwhg_wrapper.jacobian_generic / .noise_std / .fisher work
on it directly.

Specialized to the C_1,1 scene (1 layer + 1 circle + background, 7 params).
Run in ERT_GUI. Needs compare_forward_meshes.py, pwhg_wrapper.py, Anandlyn_log.py.
"""

import numpy as np
import pygimli.physics.ert as ert

import compare_forward_meshes as cfm
import pwhg_wrapper as W


class SoftTriForward:
    def __init__(self, world, width=0.2, tri_area=0.25, scheme=None):
        self.width = float(width)
        self.tri_area = float(tri_area)
        self.labels = W.param_labels(world)
        if len(self.labels) != 7:
            raise ValueError("SoftTriForward is specialized to the 7-param C_1,1 case.")
        self._theta0 = np.asarray(world.get_x0(), float)

        L, C = world.layers[0], world.Circles[0]
        self.base = dict(bg=world.rho_world, ly=L.y, lrho=L.rho,
                         cx=C.x, cy=C.y, cr=C.r, crho=C.rho)

        elec_x = np.array(world.scheme.sensorPositions())[:, 0]
        # scheme=None -> basic dd; pass an enriched pool (build_pool) for OED.
        self.scheme = scheme if scheme is not None else \
            ert.createData(elecs=elec_x, schemeName="dd")
        self.mesh = cfm.build_uniform_tri(self.tri_area, list(world.start),
                                          list(world.end), elec_x)

    def get_x0(self):
        return list(self._theta0)

    def param_labels(self):
        return list(self.labels)

    def _scene(self, theta):
        scene = dict(self.base)
        for val, (nm, kind) in zip(np.asarray(theta, float), self.labels):
            if nm == "world" and kind == "rho": scene["bg"] = 10.0 ** val
            elif nm == "L0" and kind == "y":    scene["ly"] = val
            elif nm == "L0" and kind == "rho":  scene["lrho"] = 10.0 ** val
            elif nm == "C0" and kind == "x":    scene["cx"] = val
            elif nm == "C0" and kind == "y":    scene["cy"] = val
            elif nm == "C0" and kind == "r":    scene["cr"] = 10.0 ** val
            elif nm == "C0" and kind == "rho":  scene["crho"] = 10.0 ** val
        return scene

    def forward(self, theta):
        """theta -> (ln_rhoa, rhoa). No noise; deterministic; stateless."""
        res = cfm.assign_soft(self.mesh, self._scene(theta), self.width)
        data = ert.simulate(self.mesh, scheme=self.scheme, res=res, verbose=False)
        rhoa = np.asarray(data["rhoa"], float)
        if np.any(rhoa <= 0):
            raise ValueError("Non-positive rhoa from soft forward; check theta/mesh.")
        return np.log(rhoa), rhoa
