"""V9 — Operability (Theory Manual sections 6.6, 6.7, 9 item 9; ARCH-5.4,
ARCH-8 M5).

A thin operability harness over the continuation driver: build a case
(topology + inlet + rows), traverse an operating line, and read back the two
operability behaviours V9 asks for — **stable BC-switching across the
choke-proximal region** and **surge-flag behaviour with a recorded
criterion**.

Status split (honest, and the same boundary V5 already sits on):

  * **Surge flagging is demonstrated on the V5 rotor** (``v5_rotor``): the
    meanline operating line rises in pressure ratio and the traversal reports
    a stall onset at the peak, section 6.7 "report, don't solve through". The
    recorded criterion is ``validity_saturated``: the Lieblein correlation's
    validity collapses as incidence climbs toward stall before the (correctly
    low, post-omega_bar-fix) loss turns PR over — see the continuation
    ``_classify_stall`` note. Point-by-point agreement with a *reported* surge
    line is **[VERIFY]**, blocked on the reference data (same as V5).

  * **Stable BC-switching is demonstrated on a well-posed testbed**
    (``bc_switch_testbed``, a swirling duct with a clean annulus choke
    capacity): the traversal enters the choke-proximal region, switches to the
    back-pressure branch, throttles, and switches back on recovery with no
    limit-cycling. The *V5* meanline cannot be driven onto its own choke knee:
    the single-node continuity Jacobian is singular at the capacity peak
    (``dF/dVm = 0`` at the compressible mass-flux maximum; the retuned V5
    meanline annulus chokes at mdot ~ 118 kg/s), so the mdot-parameterized
    problem
    is singular there by construction, and the supersonic-mdot branch has no
    physical entropy without a *compressor* shock-loss closure — that
    traversal is **[VERIFY]**.

    **M6-4 correction to the earlier diagnosis:** the blocker is NOT "M6
    shock-loss closures". M6 delivered the *turbine* Kacker-Okapuu shock
    term, which by AD-5 does not apply to the Lieblein *compressor* set that
    V5 uses. Two distinct things are needed and neither is turbine work:
    (i) mdot as a state unknown so the traversal is not mdot-parameterized at
    the singular peak — that already exists (the M5-3 back-pressure mode);
    (ii) a *compressor* shock-loss closure (Koch-Smith / Aungier shock term,
    already a recorded deferral on the axial-compressor set) so the
    supersonic branch carries real entropy. So the V5 choke-knee is a
    compressor-set + continuation matter (V5 calibration / M8 revisit), not
    the turbine milestone. The BC-switch machinery itself is case-independent
    and bound here on the testbed.

Provenance: M5 sub-step 4; V5-diagnosis correction M6 sub-step 4.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np  # verification layer: case definitions  # ad6: allow

from ..closures.axial_compressor import LIEBLEIN_NACA65
from ..drivers import MapResult, SpeedlineConfig, solve_speedline
from ..drivers.classical import RowSpec
from ..fluid.perfectgas import PerfectGas
from ..geometry.bladerow import ParamRowGeometry
from ..grid import GridTopology
from ..transport import TransportFields
from ..types import FidelityConfig
from .v1_analytic_ree import V1FreeVortex, annulus_topology
from .v5_axial_compressor import V5AxialRotor

__all__ = ["V9Operability"]

_DEG = np.pi / 180.0


@dataclass(frozen=True)
class V9Operability:
    """An operability case: a grid, inlet, rows, and fidelity the speedline
    driver traverses. Use the named constructors for the two V9 behaviours."""

    topology: GridTopology
    inlet: TransportFields
    rows: tuple
    gas: PerfectGas
    fidelity: FidelityConfig

    # --- named cases ----------------------------------------------------
    @classmethod
    def v5_rotor(cls) -> "V9Operability":
        """The V5 axial-compressor rotor at Tier-1 meanline (the surge-flag
        demonstration)."""
        case = V5AxialRotor()
        topo = GridTopology(case._flowpath(), n_sl=1)
        inlet = TransportFields(h0=np.full(1, case.h0_in),
                                s=np.full(1, case.s_in), rvt=np.zeros(1))
        geom = ParamRowGeometry(
            blade_count=case.blade_count, beta1=case.beta1_blade_deg * _DEG,
            beta2=case.beta2_blade_deg * _DEG, chord_len=case.chord,
            solidity_val=case.solidity, thickness=case.thickness)
        row = RowSpec(row_id="r1", omega=case.omega,
                      swirl=LIEBLEIN_NACA65.swirl, loss=LIEBLEIN_NACA65.loss,
                      blade_count=case.blade_count, geometry=geom)
        return cls(topology=topo, inlet=inlet, rows=(row,), gas=case.gas,
                   fidelity=FidelityConfig.tier1())

    @classmethod
    def bc_switch_testbed(cls, n_sl: int = 9) -> "V9Operability":
        """A swirling duct with a clean annulus choke capacity — the well-posed
        case for the stable-BC-switching demonstration (no closure stiffness at
        the knee)."""
        case = V1FreeVortex.compressible()
        topo = annulus_topology(case.r0, case.r1, case.length, n_sl,
                                case.n_stations)
        inlet = TransportFields(h0=np.full(n_sl, case.h0),
                                s=np.full(n_sl, case.s),
                                rvt=np.full(n_sl, case.rvt))
        return cls(topology=topo, inlet=inlet, rows=(), gas=case.gas,
                   fidelity=FidelityConfig.tier2())

    # --- traversal ------------------------------------------------------
    def operating_line(self, *, mdot_start: float, mdot_min: float,
                       mdot_step: float,
                       config: SpeedlineConfig = None) -> MapResult:
        """Traverse the operating line choke -> stall (sections 6.6/6.7)."""
        if config is None:
            config = SpeedlineConfig()
        return solve_speedline(self.topology, self.gas, self.fidelity,
                               self.inlet, rows=self.rows,
                               mdot_start=mdot_start, mdot_min=mdot_min,
                               mdot_step=mdot_step, config=config)
