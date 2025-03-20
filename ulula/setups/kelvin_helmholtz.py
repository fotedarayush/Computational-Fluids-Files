###################################################################################################
#
# Ulula -- kelvin_helmholtz.py
#
# Initial conditions for Kelvin-Helmholtz instability
#
# by Benedikt Diemer
# Slightly adapted by Paul McMillan (2025)
#
###################################################################################################

import numpy as np

import ulula.simulation as ulula_sim
import ulula.setup_base as setup

###################################################################################################

# Shorthand for frequently used index constants
DN = ulula_sim.DN
VX = ulula_sim.VX
VY = ulula_sim.VY
PR = ulula_sim.PR

###################################################################################################


class SetupKelvinHelmholtz(setup.Setup):
    """
    Kelvin-Helmholtz instability

    The KH instability forms at the interface between two fluids that are moving past each other.
    The user can choose between a setup where the interface is infinitely sharp and the smooth
    ICs of Robertson et al. 2010. In the sharp case, the instability is seeded by grid noise, but
    we still add a small velocity perturbation to obtain more well-defined behavior. The smooth
    version is recommended as it leads to a more physical test case.

    Parameters
    -----------------------------------------------------------------------------------------------
    sharp_ics: bool, default False
            Use sharp boundary between fluids instead of smoothed ICs.
    n_waves: int, default 1
            Number of wave periods in the domain. The number of periods that can be resolved depends on the resolution.
    rho1: float, default 2.0
            Density of the central fluid.
    velocity_difference: float, default 1.0
            Difference in velocity between the two fluids.
    """

    def __init__(self, sharp_ics=False, n_waves=1, rho1=2.0, velocity_difference=1.0):

        setup.Setup.__init__(self)
        print("Setting up Kelvin-Helmholtz instability")
        self.gamma = 5.0 / 3.0
        self.rho1 = rho1
        self.rho2 = 1.0
        self.v1 = 0.5 * velocity_difference
        self.v2 = -0.5 * velocity_difference
        self.P0 = 2.5
        self.lamb = 1.0 / n_waves

        if sharp_ics:
            self.delta_y = 0.0
            self.delta_vy = 0.001
        else:
            self.delta_y = 0.05
            self.delta_vy = 0.1

        return

    # ---------------------------------------------------------------------------------------------

    def shortName(self):

        return "kh"

    # ---------------------------------------------------------------------------------------------

    def setInitialData(self, sim, nx):

        sim.setDomain(nx, nx, xmin=0.0, xmax=1.0, ymin=0.0, bc_type="periodic")
        sim.setFluidProperties(self.gamma)

        x, y = sim.xyGrid()

        if self.delta_y > 0.0:

            # Softened initial conditions from Robertson et al (2010), Abel (2010)
            y1 = (y - 0.25) / self.delta_y
            y2 = (0.75 - y) / self.delta_y
            R = (1.0 + np.exp(0.5 / self.delta_y)) ** 2 / (
                (1.0 + np.exp(2 * y1)) * (1.0 + np.exp(2 * y2))
            )
            vy = self.delta_y * np.sin(2.0 * np.pi * x / self.lamb)

        else:

            # Instability seeded by grid-noise, e.g. Springel (2010)
            R = np.zeros_like(x)
            R[np.abs(y - 0.5) < 0.25] = 1.0
            vy = (
                self.delta_vy
                * np.sin(2.0 * np.pi * x / self.lamb)
                * (np.exp(-0.5 * (y - 0.25) ** 2) + np.exp(-0.5 * (0.75 - y) ** 2))
            )

        sim.V[DN] = self.rho2 + R * (self.rho1 - self.rho2)
        sim.V[VX] = self.v2 + R * (self.v1 - self.v2)
        sim.V[VY] = vy
        sim.V[PR] = self.P0

        return

    # ---------------------------------------------------------------------------------------------

    def plotLimits(self, q_plot):

        vmin = []
        vmax = []

        for q in q_plot:
            if q == "DN":
                vmin.append(self.rho2 * 0.85)
                vmax.append(self.rho1 * 1.05)
            elif q in ["VX", "VY"]:
                vmin.append(self.v2 * 1.2)
                vmax.append(self.v1 * 1.2)
            elif q == "PR":
                vmin.append(self.P0 * 0.9)
                vmax.append(self.P0 * 1.15)
            else:
                vmin.append(None)
                vmax.append(None)

        return vmin, vmax


###################################################################################################
