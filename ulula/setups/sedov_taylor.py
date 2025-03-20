###################################################################################################
#
# Ulula -- sedov_taylor.py
#
# Initial conditions and analytical solution for 2D Sedov-Taylor explosion
#
# by Benedikt Diemer
#
###################################################################################################

import numpy as np
from scipy.interpolate import interp1d

import ulula.simulation as ulula_sim
import ulula.setup_base as setup

###################################################################################################

# Shorthand for frequently used index constants
DN = ulula_sim.DN
VX = ulula_sim.VX
VY = ulula_sim.VY
PR = ulula_sim.PR

###################################################################################################

class SedovSolution(object):
	"""
	Analytical solution to the Sedov problem
	
	This code is adapted from the Gandalf code by Hubber et al. The solution goes back to 
	Book (1991) and Korobeynikov et al. (1961).

	Parameters
	-----------------------------------------------------------------------------------------------
	E: float
		Explosion energy
	rho: float
		Density of the ambient
	gamma: float
		Adiabatic index
	nu: int, default=3
		Number of dimensions
	w: float, default=0
		Exponent of a power-law density profile, rho=rho0 r^-w
	"""
	def __init__(self, E, rho, gamma = 5.0 / 3.0, nu = 3, w = 0.0):
		
		if not nu in [1, 2, 3]:
			raise Exception('Number of dimensions must be 1, 2, or 3. Found %d.' % (nu))
		
		self._tiny = 1e-50
		self._E = E
		self._gamma = gamma
		self._rho0 = rho
		self._rho1 = ((gamma + 1.)/(gamma - 1.))*rho
		self._nDim = nu
		self._w = w
		
		# Constants for the parametic equations:
		w1 = (3.0 * nu - 2 + gamma * (2 - nu)) / (gamma + 1.0)
		w2 = (2.0 * (gamma - 1) + nu) / gamma
		w3 = nu * (2.0 - gamma)
		
		b0 = 1.0 / (nu * gamma - nu + 2)
		b2 = (gamma - 1.0) / (gamma * (w2 - w))
		b3 = (nu - w) / (float(gamma) * (w2 - w))
		b5 = (2.0 * nu - w * (gamma + 1)) / (w3 - w)
		b6 = 2.0 / (nu + 2 - w)
		b1 = b2 + (gamma + 1.0) * b0 - b6
		b4 = b1 * (nu - w) * (nu + 2.0 - w) / (w3 - w)
		b7 = w * b6
		b8 = nu * b6
		
		# C0 is surface area function in ndim. Use approx:
		C0 = 2 * (nu - 1) * np.pi + (nu - 2) * (nu - 3)
		C5 = 2.0 / (gamma - 1.0)
		C6 = (gamma + 1) / 2.0
		C1 = C5 * gamma
		C2 = C6 / gamma
		C3 = (nu * gamma - nu + 2.0) / ((w1 - w) * C6)
		C4 = (nu + 2.0 - w) * b0 * C6
		
		# Lambdas for setting up the interpolating functions:
		ETA = lambda F: (F**-b6) * ((C1 * (F - C2))**b2) * ((C3 * (C4 - F))**(-b1))
		D   = lambda F: (F**-b7) * ((C1 * (F - C2))**(b3 - w * b2)) * ((C3 * (C4 - F))**(b4 + w * b1)) * ((C5 * (C6 - F))**-b5)
		P   = lambda F: (F** b8) * ((C3 * (C4 - F))**(b4 + (w - 2) * b1)) * ((C5 * (C6 - F))**(1 - b5))
		V   = lambda F: ETA(F) * F
		
		# Characterize the solution
		if w1 > w:
			Fmin = C2
		else:
			Fmin = C6
		
		F = np.logspace(np.log10(Fmin), 0, 100000)
		
		# Sort the etas for our interpolation function
		eta = ETA(F)
		F = F[eta.argsort()]
		eta.sort()
		
		d = D(F)
		p = P(F)
		v = V(F)
		
		# If min(eta) != 0 then all values for eta < min(eta) = 0
		if eta[0] > 0:
			e01 = [0.0, eta[0] * (1 - 1E-10)]
			d01 = [0.0, 0]
			p01 = [0.0, 0]
			v01 = [0.0, 0]
			
			eta = np.concatenate([np.array(e01), eta])
			d   = np.concatenate([np.array(d01), d])
			p   = np.concatenate([np.array(p01), p])
			v   = np.concatenate([np.array(v01), v])
		
		# Set up our interpolation functions
		self._d = interp1d(eta, d, bounds_error = False, fill_value = 1.0 / self._rho1)
		self._p = interp1d(eta, p, bounds_error = False, fill_value = 0.0)
		self._v = interp1d(eta, v, bounds_error = False, fill_value = 0.0)
		
		# Finally Calculate the normalization of R_s:
		I = eta**(nu  -1) * (d * v**2 + p)
		I = 0.5 * (I[1: ] + I[:-1])
		deta = (eta[1:] - eta[:-1])
		alpha = (I * deta).sum() * (8 * C0) / ((gamma**2 - 1.0) * (nu + 2.0 - w)**2)
		self._C = (1.0 / alpha)**(1.0 / (nu + 2 - w))

		return

	def R_s(self, t):
		"""
		Shock radius at time t
		"""

		t = np.maximum(t, self._tiny)
		
		return self._C *(self.E * t**2 / self.rho0)**(1.0 / (self._nDim + 2 - self._w))
	
	def V_s(self, t):
		"""
		Velocity of the shockwave
		"""
		
		t = np.maximum(t, self._tiny)
		
		return (2.0 / (self._nDim + 2 - self._w)) * self.R_s(t) / t
	
	def P_s(self, t):
		"""Post shock pressure"""

		return (2./(self.gamma+1))*self.rho0*self.V_s(t)**2
	
	@property
	def Rho_s(self):
		"""
		Post-shock density
		"""
		
		return self._rho1
	
	def rho(self, r, t):
		"""
		Density at radius r and time t
		"""

		eta = r/self.R_s(t)
		
		return self.Rho_s * self._d(eta)
	
	def P(self, r, t):
		"""
		Pressure at radius r and time t
		"""

		eta = r / self.R_s(t)
		
		return self.P_s(t) * self._p(eta)
	
	def v(self, r, t):
		"""
		Velocity at radius r and time t
		"""

		eta = r / self.R_s(t)
		
		return self._v(eta) * (2 / (self.gamma + 1)) * self.V_s(t)
	
	def u(self, r, t):
		"""
		Internal energy per unit density at radius r and time t
		"""
		
		return self.P(r, t) / (self.rho(r, t) * (self.gamma - 1))
	
	def e_int(self, r, t):
		"""
		Internal energy at radius r and time t
		"""
		
		return self.P(r, t) / (self.gamma - 1)
	
	def Entropy(self,r,t):
		"""
		Entropy at radius r and time t
		"""

		return self.P(r, t) / self.rho(r, t)**self.gamma

	@property
	def E(self):
		"""
		Total energy
		"""

		return self._E
	
	@property
	def gamma(self):
		"""
		Ratio of specific heats
		"""

		return self._gamma
	
	@property
	def rho0(self):
		"""
		Background density
		"""

		return self._rho0

###################################################################################################

class SetupSedov(setup.Setup):
	"""
	Sedov-Taylor explosion

	The Sedov-Taylor solution represents a blastwave created by a large amount of energy that is 
	injected at the center of the domain. If the energy is much larger than the thermal energy
	of the surrounding gas, the solution is self-similar, i.e., does not depend on any physical 
	scale and evolves in time as a power-law. 

	In this setup, the energy is distributed in a Gaussian with a radius of ~1 cell. This avoids
	grid artifacts that can arise to the square nature of the grid cells. The exact solution code 
	is adapted from the Gandalf code by Hubber et al. 

	Parameters
	-----------------------------------------------------------------------------------------------
	E: float
		Explosion energy in code units; the thermal energy in the domain is about unity in these 
		units, so ``E`` should be significantly larger than unity
	rho0: float
		The density of the surrounding medium in code units
	gamma: float
		The adiabatic index of the gas
	"""
	
	def __init__(self, E = 50.0, rho0 = 1.0, gamma = 5.0 / 3.0):

		setup.Setup.__init__(self)
		
		self.E = E
		self.rho0 = rho0
		self.gamma = gamma
		self.P0 = 1.0

		# Initial radius of the Gaussian in cell sizes. Experimentally, a radius close to 1 gives
		# better results than larger radii.
		self.r_ini = 0.75
		
		return 

	# ---------------------------------------------------------------------------------------------

	def shortName(self):
		
		return 'sedov'

	# ---------------------------------------------------------------------------------------------
	
	def setInitialData(self, sim, nx):
		
		sim.setDomain(nx, nx, xmin = 0.0, xmax = 1.0, ymin = 0.0, bc_type = 'outflow')
		sim.setFluidProperties(self.gamma)

		# Set Gaussian with extra energy into the center of the domain. We compute the Gaussian
		# profile first and normalize it such that the extra pressure corresponds to the total
		# explosion energy given by the user.
		x, y = sim.xyGrid()
		r = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)
		r_gaussian = self.r_ini / nx
		E_extra = np.exp(-0.5 * r**2 / r_gaussian**2)
		E_tot = np.sum(E_extra) / float(nx**2)
		E_extra *= self.E / E_tot
		P_extra = E_extra * (self.gamma - 1.0)

		sim.V[DN] = self.rho0
		sim.V[PR] = self.P0
		sim.V[PR] += P_extra
		
		return
	
	# ---------------------------------------------------------------------------------------------
	
	def trueSolution(self, sim, x, q_plot):

		sedov_sol = SedovSolution(self.E, self.rho0, gamma = self.gamma, nu = 2, w = 0.0)

		t = sim.t
		nq = len(q_plot)
		V_sol = np.zeros((nq, len(x)), np.float)

		for i in range(nq):
			if q_plot[i] == 'DN':
				V_sol[i] = sedov_sol.rho(x, t)
			elif q_plot[i] == 'VT':
				V_sol[i] = sedov_sol.v(x, t)
			elif q_plot[i] == 'PR':
				V_sol[i] = sedov_sol.P(x, t)
			else:
				raise Exception('Cannot evaluate Sedov-Taylor solution for quantity %s (only for DN, VT, PR).' \
							% (q_plot[i]))
		
		return V_sol
	
###################################################################################################
