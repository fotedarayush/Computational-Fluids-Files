###################################################################################################
#
# Ulula -- simulation.py
#
# This is the main file containing the Ulula hydro solver.
#
# by Benedikt Diemer
#
###################################################################################################

import numpy as np
import h5py

import ulula.utils as ulula_utils

###################################################################################################

# Primitive variable vector indices
DN = 0
VX = 1
VY = 2
PR = 3

# Conserved variable indices; density is the same as primitive
MS = 0
MX = 1
MY = 2
ET = 3

slc_none = (slice(None), slice(None), slice(None))

# The current file version is written to each Ulula file. If the code tries to open a file that is
# old enough to be incompatible, an error will be thrown.
file_version_current = '0.2.0'
file_version_oldest = '0.2.0'

###################################################################################################

class HydroScheme():
	"""
	Container class for hydro algorithms

	Parameters
	-----------------------------------------------------------------------------------------------
	reconstruction: string
		Reconstruction algorithm; see listing for valid choices
	limiter: string
		Slope limiter algorithm; see listing for valid choices
	riemann: string
		Riemann solver; see listing for valid choices
	time_integration: string
		Time integration scheme; see listing for valid choices
	cfl: float
		CFL number (must be between 0 and 1); determines the timestep as CFL number times cell size
		divided by the maximum signal speed in the domain
	"""
	
	def __init__(self, reconstruction = 'const', limiter = 'minmod', riemann = 'hll', 
				time_integration = 'euler', cfl = 0.8):

		self.reconstruction = reconstruction
		self.limiter = limiter
		self.riemann = riemann
		self.time_integration = time_integration
		self.cfl = cfl
		
		return

###################################################################################################

class Simulation():
	"""
	Main class for the Ulula hydro solver
	
	This class contains all simulation data and routines. The internal fields have the following 
	meaning (after the hydro scheme, domain, fluid properties, and initial conditions have been set):
	
	============  ======
	Field         Meaning
	============  ======
	Scalars
	--------------------
	``dx``        Width of cells (same in both x and y directions)
	``nq``        Number of fluid variables (typically 4 for rho, vx, vy, P)
	``nx``        Number of cells in the x-direction
	``ny``        Number of cells in the y-direction
	``ng``        Number of ghost cells around each edge
	``xlo``       First index of physical grid in x-direction (without left ghost zone)
	``xhi``       Last index of physical grid in x-direction (without right ghost zone)
	``ylo``       First index of physical grid in y-direction (without bottom ghost zone)
	``yhi``       Last index of physical grid in y-direction (without top ghost zone)
	``t``         Current time of the simulation
	``step``      Current step counter
	``last_dir``  Direction of last sweep in previous timestep (x=0, y=1)
	``gamma``     Adiabatic index 
	``gm1``       gamma - 1
	``gm1_inv``   1 / (gamma - 1)
	------------  ------
	Settings
	--------------------
	``bc_type``   Type of boundary condition ('periodic' or 'outflow')
	``hs``        HydroScheme object
	------------  ------
	1D vectors
	--------------------
	``x``         Array of x values at cell centers (dimensions [nx + 2 ng])
	``y``         Array of y values at cell centers (dimensions [ny + 2 ng])
	------------  ------
	3D vectors
	--------------------
	``U``         Vector of conserved fluid variables (dimensions [nq, nx + 2 ng, ny + 2 ng])
	``V``         Vector of primitive fluid variables (dimensions [nq, nx + 2 ng, ny + 2 ng])
	``V_im12``    Cell-edge states at left side (same dimensions as V)
	``V_ip12``    Cell-edge states at right side (same dimensions as V)
	------------  ------
	Slices
	--------------------
	``slc1dL``    1D slices for idir [0, 1], physical domain, shifted one cell left
	``slc1dR``    1D slices for idir [0, 1], physical domain, shifted one cell right
	``slc1dC``    1D slices for idir [0, 1], physical domain
	``slc3dL``    3D slices for idir [0, 1], physical domain, shifted one cell left
	``slc3dR``    3D slices for idir [0, 1], physical domain, shifted one cell right
	``slc3dC``    3D slices for idir [0, 1], physical domain
	``slc3aL``    3D slices for idir [0, 1], total domain, shifted one cell left
	``slc3aR``    3D slices for idir [0, 1], total domain, shifted one cell right
	``slc3aC``    3D slices for idir [0, 1], total domain
	``slc3fL``    3D slice of flux vector from left interface
	``slc3fR``    3D slice of flux vector from right interface	
	------------  ------
	Dictionaries for fluid variables
	--------------------
	``q_prim``    Dictionary containing the indices of the primitive fluid variables in the ``V`` array
	``q_cons``    Dictionary containing the indices of the conserved fluid variables in the ``U`` array
	============  ======
	
	The constructor takes the following parameters:
	
	Parameters
	-----------------------------------------------------------------------------------------------
	hydro_scheme: HydroScheme
		Container class for algorithmic choices
	"""

	def __init__(self, hydro_scheme):
	
		# Create a dictionary of the primitive and conserved fields and their indices. This helps
		# routines using the simulation to know where a quantity is stored.
		self.q_prim = {}
		self.q_prim['DN'] = DN
		self.q_prim['VX'] = VX
		self.q_prim['VY'] = VY
		self.q_prim['PR'] = PR

		self.q_cons = {}
		self.q_cons['MS'] = DN
		self.q_cons['MX'] = MX
		self.q_cons['MY'] = MY
		self.q_cons['ET'] = ET
		
		self.nq = len(self.q_prim)

		self.hs = hydro_scheme
	
		# Set functions based on reconstruction scheme. If we are reconstructing, we need two
		# ghost zones instead of one due to slope calculations.
		if self.hs.reconstruction == 'const':
			self.reconstruction = self.reconstructionConst
			self.nghost = 1
		elif self.hs.reconstruction == 'linear':
			self.reconstruction = self.reconstructionLinear
			self.nghost = 2
		else:
			raise Exception('Unknown reconstruction scheme, %s.' % (self.hs.reconstruction))

		# Set limiter
		if self.hs.limiter == 'none':
			self.limiter = self.limiterNone
		elif self.hs.limiter == 'minmod':
			self.limiter = self.limiterMinMod
		elif self.hs.limiter == 'vanleer':
			self.limiter = self.limiterVanLeer
		elif self.hs.limiter == 'mc':
			self.limiter = self.limiterMC
		else:
			raise Exception('Unknown limiter, %s.' % (self.hs.limiter))
		
		# Set functions related to Riemann solver		
		if self.hs.riemann == 'hll':
			self.riemannSolver = self.riemannSolverHLL
		else:
			raise Exception('Unknown Riemann solver, %s.' % self.hs.riemann)

		# Check the time integration scheme for invalid values	
		if not self.hs.time_integration in ['euler', 'hancock', 'hancock_cons']:
			raise Exception('Unknown time integration scheme, %s.' % self.hs.time_integration)

		# Variables that need to be set
		self.gamma = None

		return
	
	# ---------------------------------------------------------------------------------------------

	def setDomain(self, nx, ny, xmin = 0.0, xmax = 1.0, ymin = 0.0, bc_type = 'periodic'):
		"""
		Set the physical and numerical size of the domain
		
		This function creates the memory structure for the simulation as well as pre-computed 
		slices that index the arrays.

		Parameters
		-------------------------------------------------------------------------------------------
		nx: int
			Number of grid points in x-direction
		ny: int
			Number of grid points in y-direction
		xmin: float
			Left edge in physical coordinates (code units)
		xmax: float
			Right edge in physical coordinates (code units)
		ymin: float
			Bottom edge in physical coordinates (code units)
		ymax: float
			Top edge in physical coordinates (code units)
		bc_type: string
			Type of boundary conditions; can be ``periodic`` or ``outflow``
		"""

		self.nx = nx
		self.xmin = xmin
		self.xmax = xmax
		self.dx = (xmax - xmin) / float(nx)
		self.bc_type = bc_type
		
		self.ny = ny
		self.ymin = ymin
		self.ymax = self.ymin + ny * self.dx
		
		ng = self.nghost

		self.xlo = ng
		self.xhi = ng + self.nx - 1
		self.ylo = ng
		self.yhi = ng + self.ny - 1

		self.x = xmin + (np.arange(self.nx + 2 * ng) - ng) * self.dx + 0.5 * self.dx
		self.y = ymin + (np.arange(self.ny + 2 * ng) - ng) * self.dx + 0.5 * self.dx
		
		print('Grid setup %d x %d, dimensions x = [%.2e .. %.2e] y =  [%.2e .. %.2e]' \
			% (self.nx, self.ny, self.xmin, self.xmax, self.ymin, self.ymax))

		# Set up slices that can be reused. The names are slc<dims><daf><LRC> which means the
		# dimensionality of the slice (1, 2, 3), whether it covers the physical domain (d), the
		# entire domain including ghost cells (a), or the flux vector (f), and whether it selects
		# the cells (C), there left (L) or right (R) neighbors.
		self.slc1dL = []
		self.slc1dR = []
		self.slc1dC = []

		self.slc3dL = []
		self.slc3dR = []
		self.slc3dC = []

		self.slc3aL = []
		self.slc3aR = []
		self.slc3aC = []

		self.slc3fL = []
		self.slc3fR = []
			
		for idir in range(2):
			
			if idir == 0:
				lo = self.xlo
				hi = self.xhi
			else:
				lo = self.ylo
				hi = self.yhi
				
			slc1dL = slice(lo - 1, hi + 1)
			slc1dR = slice(lo, hi + 2)
			slc1dC = slice(lo, hi + 1)
			
			if idir == 0:
				slc3dL = (slice(None), slc1dL,         slice(None))
				slc3dR = (slice(None), slc1dR,         slice(None))
				slc3dC = (slice(None), slc1dC,         slice(None))

				slc3aL = (slice(None), slice(0, -2),   slice(None))
				slc3aR = (slice(None), slice(2, None), slice(None))
				slc3aC = (slice(None), slice(1, -1),   slice(None))
				
				slc3fL = (slice(None), slice(0, -1),   slice(None))
				slc3fR = (slice(None), slice(1, None), slice(None))
			else:
				slc3dL = (slice(None), slice(None), slc1dL)
				slc3dR = (slice(None), slice(None), slc1dR)
				slc3dC = (slice(None), slice(None), slc1dC)

				slc3aL = (slice(None), slice(None), slice(0, -2))
				slc3aR = (slice(None), slice(None), slice(2, None))
				slc3aC = (slice(None), slice(None), slice(1, -1))
				
				slc3fL = (slice(None), slice(None), slice(0, -1))
				slc3fR = (slice(None), slice(None), slice(1, None))
			
			self.slc1dL.append(slc1dL)
			self.slc1dR.append(slc1dR)
			self.slc1dC.append(slc1dC)
			
			self.slc3dL.append(slc3dL)
			self.slc3dR.append(slc3dR)
			self.slc3dC.append(slc3dC)

			self.slc3aL.append(slc3aL)
			self.slc3aR.append(slc3aR)
			self.slc3aC.append(slc3aC)
			
			self.slc3fL.append(slc3fL)
			self.slc3fR.append(slc3fR)
		
		# Time
		self.t = 0.0
		self.step = 0
		self.last_dir = -1
		
		# Storage for the primitive and conserved fluid variables and other arrays. The initial
		# conditions can be useful for plotting.
		self.U = self.emptyArray()
		self.V = self.emptyArray()
		
		# Storage for the cell-edge states and conservative fluxes. If we are using 
		# piecewise-constant, both states are the same (and the same as the cell centers).
		if self.hs.reconstruction == 'const':
			self.V_im12 = self.V
			self.V_ip12 = self.V
		elif self.hs.reconstruction == 'linear':
			self.V_im12 = self.emptyArray()
			self.V_ip12 = self.emptyArray()
		else:
			raise Exception('Unknown reconstruction scheme, %s.' % (self.hs.reconstruction))
		
		return
	
	# ---------------------------------------------------------------------------------------------

	def setFluidProperties(self, gamma = 5.0 / 3.0):
		"""
		Set the physical properties of the fluid

		Parameters
		-------------------------------------------------------------------------------------------
		gamma: float
			Adiabatic index of the ideal gas to be simulated; should be 5/3 for atomic gases or
			7/5 for diatomic molecular gases.
		"""
	
		self.gamma = gamma
		self.gm1 = gamma - 1.0
		self.gm1_inv = 1.0 / self.gm1
	
		return
	
	# ---------------------------------------------------------------------------------------------

	# Return an empty array with the dimensions of the solution
	
	def emptyArray(self, nq = None):
		"""
		Get an empty array for fluid variables

		Parameters
		-------------------------------------------------------------------------------------------
		nq: int
			The number of quantities for which the array should contain space. If ``None``, the
			number of fluid quantities is used (4 in two dimensions).

		Returns
		-------------------------------------------------------------------------------------------
		ret: array_like
			Float array of size nq times the size of the domain including ghost cells. If 
			``nq == 1``, the first dimension is omitted.
		"""
			
		if nq is None:
			nq = self.nq
			
		if nq == 1:
			ret = np.zeros((self.nx + 2 * self.nghost, self.ny + 2 * self.nghost), float)
		else:
			ret = np.zeros((nq, self.nx + 2 * self.nghost, self.ny + 2 * self.nghost), float)
			
		return ret 

	# ---------------------------------------------------------------------------------------------

	def xyGrid(self):
		"""
		Get a grid of the x and y cell center positions
		
		This function returns two arrays with the x and y positions at each grid point. These 
		arrays can be convenient when setting the initial conditions.

		Returns
		-------------------------------------------------------------------------------------------
		x: array_like
			2D array with x positions of all cells (including ghost cells)
		y: array_like
			2D array with x positions of all cells (including ghost cells)
		"""
	
		return np.meshgrid(self.x, self.y, indexing = 'ij')

	# ---------------------------------------------------------------------------------------------
	
	def enforceBoundaryConditions(self):
		"""
		Enforce boundary conditions after changes
		
		This function fills the ghost cells with values from the physical domain. For periodic BCs,
		those originate from the other side; for outflow BCs, they are copied from the edge of the
		physical domain. This function must be executed at each timestep.
		"""
			
		if self.bc_type is None:
			raise Exception('Type of boundary condition must be set.')
		
		xlo = self.xlo
		xhi = self.xhi
		ylo = self.ylo
		yhi = self.yhi
		ng = self.nghost
		slc_x = self.slc1dC[0]
		slc_y = self.slc1dC[1]
		
		for v in [self.V, self.U]:
			
			if self.bc_type == 'periodic':
				# Left/right ghost
				v[:, 0:ng, slc_y] = v[:, xhi-ng+1:xhi+1, slc_y]		
				v[:, -ng:, slc_y] = v[:, xlo:xlo+ng,     slc_y]
				# Bottom/top ghost
				v[:, slc_x, 0:ng] = v[:, slc_x, yhi-ng+1:yhi+1]		
				v[:, slc_x, -ng:] = v[:, slc_x,     ylo:ylo+ng]
				# Corners
				v[:, 0:ng,  0:ng] = v[:, xhi-ng+1:xhi+1, yhi-ng+1:yhi+1]
				v[:, 0:ng,  -ng:] = v[:, xhi-ng+1:xhi+1, ylo:ylo+ng]
				v[:, -ng:,  0:ng] = v[:, xlo:xlo+ng,     yhi-ng+1:yhi+1]
				v[:, -ng:,  -ng:] = v[:, xlo:xlo+ng,     ylo:ylo+ng]
			
			elif self.bc_type == 'outflow':
				# Left/right ghost
				v[:, 0:ng, slc_y] = v[:, xlo, slc_y][:, None, :]
				v[:, -ng:, slc_y] = v[:, xhi, slc_y][:, None, :]
				# Bottom/top ghost
				v[:, slc_x, 0:ng] = v[:, slc_x, ylo][:, :, None]
				v[:, slc_x, -ng:] = v[:, slc_x, yhi][:, :, None]
				# Corners
				v[:, 0:ng, 0:ng]  = v[:, xlo, ylo][:, None, None]
				v[:, 0:ng, -ng:]  = v[:, xlo, yhi][:, None, None]
				v[:, -ng:, 0:ng]  = v[:, xhi, ylo][:, None, None]
				v[:, -ng:, -ng:]  = v[:, xhi, yhi][:, None, None]
			
			else:
				raise Exception('Unknown type of boundary condition, %s.' % (self.bc_type))
			
		return

	# ---------------------------------------------------------------------------------------------
	
	def primitiveToConserved(self, V, U):
		"""
		Convert primitive to conserved variables
		
		This function takes the input and output arrays as parameters instead of assuming that it
		should use the main V and U arrays. In some cases, conversions need to be performed on 
		other fluid states.

		Parameters
		-------------------------------------------------------------------------------------------
		V: array_like
			Input array of primitive fluid variables with first dimension nq (rho, vx, vy, P...)
		U: array_like
			Output array of fluid variables with first dimension nq (rho, u * vx...)
		"""
				
		rho = V[DN]
		ux = V[VX]
		uy = V[VY]
		
		U[DN] = rho
		U[MX] = ux * rho
		U[MY] = uy * rho
		U[ET] = 0.5 * (ux**2 + uy**2) * rho + V[PR] * self.gm1_inv

		return

	# ---------------------------------------------------------------------------------------------
	
	def primitiveToConservedRet(self, V):
		"""
		Convert primitive to new conserved array
		
		Same as :func:`primitiveToConserved`, but creating the conserved output array.

		Parameters
		-------------------------------------------------------------------------------------------
		V: array_like
			Input array of primitive fluid variables with first dimension nq (rho, vx, vy, P...)

		Returns
		-------------------------------------------------------------------------------------------
		U: array_like
			Array of fluid variables with first dimension nq (rho, u * vx...) and same dimensions as
			input array.
		"""
		
		U = np.zeros_like(V)
		self.primitiveToConserved(V, U)
		
		return U

	# ---------------------------------------------------------------------------------------------
	
	def conservedToPrimitive(self, U, V):
		"""
		Convert conserved to primitive variables
		
		This function takes the input and output arrays as parameters instead of assuming that it
		should use the main U and V arrays. In some cases, conversions need to be performed on 
		other fluid states.

		Parameters
		-------------------------------------------------------------------------------------------
		U: array_like
			Input array of conserved fluid variables with first dimension nq (rho, u * vx...)
		V: array_like
			Output array of primitive fluid variables with first dimension nq (rho, vx, vy, P...)
		"""
		
		rho = U[DN]
		ux = U[MX] / rho
		uy = U[MY] / rho
		
		V[DN] = rho
		V[VX] = ux
		V[VY] = uy
		V[PR] = (U[ET] - 0.5 * rho * (ux**2 + uy**2)) * self.gm1

		if np.min(V[PR]) <= 0.0:
			raise Exception('Zero or negative pressure found. Aborting.')
		
		return

	# ---------------------------------------------------------------------------------------------

	def fluxVector(self, idir, V):
		"""
		Convert the flux vector F(V)
		
		The flux of the conserved quantities density, momentum, and total energy as a function of 
		a primitive fluid state.

		Parameters
		-------------------------------------------------------------------------------------------
		idir: int
			Direction of sweep (0 = x, 1 = y)
		V: array_like
			Input array of primitive fluid variables with first dimension nq (rho, vx, vy, P...)

		Returns
		-------------------------------------------------------------------------------------------
		F: array_like
			Array of fluxes with first dimension nq and same dimensions as input array.
		"""
		
		idir2 = (idir + 1) % 2

		rho = V[DN]
		u1 = V[VX + idir]
		u2 = V[VX + idir2]
		prs = V[PR]
		
		rho_u1 = rho * u1
		etot = 0.5 * rho * (u1**2 + u2**2) + prs * self.gm1_inv

		F = np.zeros_like(V)
		F[DN] = rho_u1
		F[MX + idir] = rho_u1 * u1 + prs
		F[MX + idir2] = rho_u1 * u2
		F[ET] = (etot + prs) * u1
		
		return F

	# ---------------------------------------------------------------------------------------------


	def primitiveEvolution(self, idir, V, dV_dx):
		"""
		Linear approximation of the Euler equations
		
		Instead of the conservation-law form, we can also think of the Euler equations as 
		:math:`dV/dt + A(V) dV/dx = S`. This function returns :math:`\Delta V/ \Delta t` given an
		input state and a vector of spatial derivatives :math:`\Delta V/ \Delta x`. The result is
		used in the Hancock step.

		Parameters
		-------------------------------------------------------------------------------------------
		idir: int
			Direction of sweep (0 = x, 1 = y)
		V: array_like
			Array of primitive fluid variables with first dimension nq (rho, vx, vy, P...)
		dV_dx: array_like
			Array of derivative of fluid variables with first dimension nq

		Returns
		-------------------------------------------------------------------------------------------
		dV_dt: array_like
			Array of linear approximation to time evolution of fluid variables, with same dimensions
			as input arrays.
		"""
		
		idir2 = (idir + 1) % 2
		V1 = VX + idir
		V2 = VX + idir2
		
		dV_dt = np.zeros_like(dV_dx)
		
		dV_dt[DN] = -(V[V1] * dV_dx[DN] + dV_dx[V1] * V[DN])
		dV_dt[V1] = -(V[V1] * dV_dx[V1] + dV_dx[PR] / V[DN])
		dV_dt[V2] = -(V[V1] * dV_dx[V2])
		dV_dt[PR] = -(V[V1] * dV_dx[PR] + dV_dx[V1] * V[PR] * self.gamma)

		return dV_dt

	# ---------------------------------------------------------------------------------------------

	def soundSpeed(self, V):
		"""
		Sound speed
		
		Parameters
		-------------------------------------------------------------------------------------------
		V: array_like
			Input array of primitive fluid variables with first dimension nq (rho, vx, vy, P...)

		Returns
		-------------------------------------------------------------------------------------------
		cs: array_like
			Array of sound speed with first dimension nq and same dimensions as input array.
		"""
		
		cs = np.sqrt(self.gamma * V[PR] / V[DN])

		if np.any(np.isnan(cs)):
			print(np.min(V[PR]), np.min(V[DN]))
			raise Exception('Could not compute sound speed. Aborting.')
		
		return cs

	# ---------------------------------------------------------------------------------------------

	def maxSpeedInDomain(self):
		"""
		Largest signal speed in domain
		
		This function returns the largest possible signal speed anywhere in the domain. It
		evaluates the sound speed and adds it to the absolute x and y velocities. We do not need to
		add those velocities in quadrature since we are taking separate sweeps in the x and y 
		directions. Thus, the largest allowed timestep is determined by the largest speed in 
		either direction.
		
		Parameters
		-------------------------------------------------------------------------------------------
		V: array_like
			Input array of primitive fluid variables with first dimension nq (rho, vx, vy, P...)

		Returns
		-------------------------------------------------------------------------------------------
		c_max: float
			Largest possible signal speed in the domain.
		"""
		
		cs = self.soundSpeed(self.V)
		c_max = np.max(np.maximum(np.abs(self.V[VX]), np.abs(self.V[VY])) + cs)
		
		if np.isnan(c_max):
			raise Exception('Could not compute fastest speed in domain. Aborting.')
		
		return c_max

	# ---------------------------------------------------------------------------------------------

	def reconstructionConst(self, idir, dt):
		"""
		Piecewise-constant reconstruction
		
		Piecewise-constant means no reconstruction. The left/right cell edge value arrays are already
		set to the cell-centered values so that this function does nothing at all. It serves as a 
		placeholder to which the reconstruction function pointer can be set.
		
		Parameters
		-------------------------------------------------------------------------------------------
		idir: int
			Direction of sweep (0 = x, 1 = y)
		dt: float
			Timestep
		"""
		
		return

	# ---------------------------------------------------------------------------------------------

	def reconstructionLinear(self, idir, dt):
		"""
		Piecewise-linear reconstruction
		
		This function creates left and right cell-edge states based on the cell-centered states. It
		first computes the left and right slopes, uses a slope limiter to determine the limited
		slope to use, and interpolates linearly within each cell.
		
		If the time integration scheme is Hancock, the reconstructed edge states are also advanced
		by half a timestep to get 2nd-order convergence in the flux calculation. There are two ways 
		to perform the Hancock step. The more conventionally described way is to take the fluxes 
		according to the L/R states as an approximation for the flux differential across the cell 
		(the ``hancock_cons`` integration scheme). The differential is then used to updated the 
		conserved cell-edge states. However, this method necessitates a 
		primitive->conserved->primitive conversion and a flux calculation. By contrast, the 
		so-called primitive Hancock method uses the Euler equations in primitive variables to 
		estimate the change in time from the change across the cell (see 
		:func:`primitiveEvolution`). The two methods should give almost identical results, but the 
		primitive version is noticeably faster.
		
		Parameters
		-------------------------------------------------------------------------------------------
		idir: int
			Direction of sweep (0 = x, 1 = y)
		dt: float
			Timestep
		"""
		
		V = self.V
		slc3aL = self.slc3aL[idir]
		slc3aR = self.slc3aR[idir]
		slc3aC = self.slc3aC[idir]

		# Compute undivided derivatives
		sL = (V[slc3aC] - V[slc3aL])
		sR = (V[slc3aR] - V[slc3aC])
		
		# Apply slope limiter. 
		slim = np.zeros_like(sL)
		self.limiter(sL, sR, slim)
	
		# Set left and right edge states in each cell (except one layer of ghost cells)
		self.V_im12[slc3aC] = self.V[slc3aC] - slim * 0.5
		self.V_ip12[slc3aC] = self.V[slc3aC] + slim * 0.5		
		
		# Hancock step, if that time integration scheme is selected
		if self.hs.time_integration == 'hancock':
			fac = 0.5 * dt / self.dx
			self.V_im12[slc3aC] += fac * self.primitiveEvolution(idir, self.V_im12[slc3aC], slim)
			self.V_ip12[slc3aC] += fac * self.primitiveEvolution(idir, self.V_ip12[slc3aC], slim)
	
		elif self.hs.time_integration == 'hancock_cons':
			fac = 0.5 * dt / self.dx
			U_im12 = self.primitiveToConservedRet(self.V_im12[slc3aC])
			U_ip12 = self.primitiveToConservedRet(self.V_ip12[slc3aC])
			F_im12 = self.fluxVector(idir, self.V_im12[slc3aC])
			F_ip12 = self.fluxVector(idir, self.V_ip12[slc3aC])
			Fdiff = (F_im12 - F_ip12)
			U_im12 += fac * Fdiff
			U_ip12 += fac * Fdiff
			self.conservedToPrimitive(U_im12, self.V_im12[slc3aC])
			self.conservedToPrimitive(U_ip12, self.V_ip12[slc3aC])

		return

	# ---------------------------------------------------------------------------------------------

	def limiterNone(self, sL, sR, slim):
		"""
		Non-limiter (central derivative)
		
		This limiter is the absence thereof: it does not limit the left and right slopes but 
		returns their average (the central derivative). This generally produces unstable schemes
		but is implemented for testing and demonstration purposes.
		
		Parameters
		-------------------------------------------------------------------------------------------
		sL: array_like
			Array of left slopes
		sR: array_like
			Array of right slopes
		slim: array_like
			Output array of limited slope; must have same dimensions as sL and sR.
		"""
			
		slim[:] = 0.5 * (sL + sR)	
		
		return

	# ---------------------------------------------------------------------------------------------

	def limiterMinMod(self, sL, sR, slim):
		"""
		Minimum-modulus limiter
		
		The most conservative limiter, which always chooses the shallower out of the left and 
		right slopes.
		
		Parameters
		-------------------------------------------------------------------------------------------
		sL: array_like
			Array of left slopes
		sR: array_like
			Array of right slopes
		slim: array_like
			Output array of limited slope; must have same dimensions as sL and sR.
		"""
		
		sL_abs = np.abs(sL)
		sR_abs = np.abs(sR)
		mask = (sL * sR > 0.0) & (sL_abs <= sR_abs)
		slim[mask] = sL[mask]
		mask = (sL * sR > 0.0) & (sL_abs > sR_abs)
		slim[mask] = sR[mask]		
		
		return

	# ---------------------------------------------------------------------------------------------

	def limiterVanLeer(self, sL, sR, slim):
		"""
		The limiter of van Leer
		
		An intermediate limiter that is less conservative than minimum modulus but more 
		conservative than monotonized central.
		
		Parameters
		-------------------------------------------------------------------------------------------
		sL: array_like
			Array of left slopes
		sR: array_like
			Array of right slopes
		slim: array_like
			Output array of limited slope; must have same dimensions as sL and sR.
		"""
		
		mask = (sL * sR > 0.0)
		slim[mask] = 2.0 * sL[mask]* sR[mask] / (sL[mask] + sR[mask])	
		
		return

	# ---------------------------------------------------------------------------------------------

	def limiterMC(self, sL, sR, slim):
		"""
		Monotonized-central limiter
		
		As the name suggests, this limiter chooses the central derivative wherever possible, but 
		reduces its slope where it would cause negative cell-edge values. This limiter leads to the
		sharpest solutions but is also the least stable.
		
		Parameters
		-------------------------------------------------------------------------------------------
		sL: array_like
			Array of left slopes
		sR: array_like
			Array of right slopes
		slim: array_like
			Output array of limited slope; must have same dimensions as sL and sR.
		"""
		
		sC = (sL + sR) * 0.5
		sL_abs = np.abs(sL)
		sR_abs = np.abs(sR)
		mask = (sL * sR > 0.0) & (sL_abs <= sR_abs)
		slim[mask] = 2.0 * sL[mask]
		mask = (sL * sR > 0.0) & (sL_abs > sR_abs)
		slim[mask] = 2.0 * sR[mask]
		mask = np.abs(slim) > np.abs(sC)
		slim[mask] = sC[mask]
		
		return

	# ---------------------------------------------------------------------------------------------
	
	def riemannSolverHLL(self, idir, VL, VR):
		"""
		The HLL Riemann solver
		
		The Riemann solver computes the fluxes across cell interfaces given two discontinuous 
		states on the left and right sides of each interface. The Harten-Lax-van Leer (HLL) 
		Riemann solver is one of the simplest such algorithms. It takes into account the 
		fastest waves traveling left and right, but it computes only one intermediate state that
		ignores contact discontinuities.
		
		Parameters
		-------------------------------------------------------------------------------------------
		idir: int
			Direction of sweep (0 = x, 1 = y)
		VL: array_like
			Array of primitive state vectors on the left sides of the interfaces
		VR: array_like
			Array of primitive state vectors on the right sides of the interfaces
	
		Returns
		-------------------------------------------------------------------------------------------
		flux: array_like
			Array of conservative fluxes across interfaces; has the same dimensions as VL and VR.
		"""
	
		# Sound speed to the left and right of the interface
		csL = self.soundSpeed(VL)
		csR = self.soundSpeed(VR)
		
		# Maximum negative velocity to the left and positive velocity to the right
		SL = VL[VX + idir] - csL
		SR = VR[VX + idir] + csR
		
		# Get conserved states for left and right states
		UL = self.primitiveToConservedRet(VL)
		UR = self.primitiveToConservedRet(VR)
		
		# F(V) on the left and right
		FL = self.fluxVector(idir, VL)
		FR = self.fluxVector(idir, VR)
		
		# Formula for the HLL Riemann solver. We first set all fields to the so-called HLL flux, i.e.,
		# the flux in the intermediate state between the two fastest waves SL and SR. If even SL is 
		# positive (going to the right), then we take the flux from the left cell, FL. If even SR is
		# going to the left, we take the right flux. Since these cases can be rare in some problems,
		# we first do a quick check whether there are any cells that match the condition before setting
		# them to the correct fluxes.
		flux = (SR * FL - SL * FR + SL * SR * (UR - UL)) / (SR - SL)

		# Check for cases where all speeds are on one side of the fan		
		if np.any(SL >= 0.0):
			mask_L = (SL >= 0.0)
			flux[:, mask_L] = FL[:, mask_L]
		if np.any(SR <= 0.0):
			mask_R = (SR <= 0.0)
			flux[:, mask_R] = FR[:, mask_R]
		
		return flux

	# ---------------------------------------------------------------------------------------------

	def cflCondition(self):
		"""
		Compute the size of the next timestep
		
		This function computes the maximum signal speed anywhere in the domain and sets a timestep
		based on the CFL condition. This routine 
		
		Returns
		-------------------------------------------------------------------------------------------
		dt: float
			Size of the next timestep
		"""
		
		u_max = self.maxSpeedInDomain()
		dt = self.hs.cfl * self.dx / u_max
		
		return dt
		
	# ---------------------------------------------------------------------------------------------
	
	def timestep(self, dt = None):
		"""
		Advance the fluid state by one timestep
		
		This timestepping routine implements a dimensionally split scheme, where we alternate the
		order of execution with each timestep to maintain a 2nd-order scheme in time (xy-yx-xy
		and so on). In each direction, we reconstruct the cell-edge states, compute the 
		conservative Godunov fluxes with the Riemann solver, and add the flux difference to the
		converved fluid variables. Technically, we should make sure that the same timestep is 
		taken for both the xy and yx steps because otherwise the first-order error terms do not 
		perfectly cancel. However, we omit this complication, given that the timestep should not
		change drastically between steps.
		
		Parameters
		-------------------------------------------------------------------------------------------
		dt: float
			Size of the timestep to be taken; if ``None`` the timestep is computed from the CFL
			condition using the :func:`~ulula.simulation.Simulation.cflCondition` function. This 
			timestep should be used in most circumstances, but sometimes we wish to take a manually 
			set timestep, for example, to output a file or plot. Thus, the two functions are 
			separated.
		
		Returns
		-------------------------------------------------------------------------------------------
		dt: float
			The timestep taken
		"""
			
		# If the timestep is not given, compute it from the CFL condition
		if dt is None:
			dt = self.cflCondition()
	
		# Use Strang splitting to maintain 2nd order accuracy; we go xy-yx-xy-yx and so on
		if self.last_dir == 0:
			dirs = [0, 1]
		else:
			dirs = [1, 0]
			
		for idir in dirs:
	
			# Load slices for this dimension
			slc3dL = self.slc3dL[idir]
			slc3dR = self.slc3dR[idir]
			slc3dC = self.slc3dC[idir]
			slc3fL = self.slc3fL[idir]
			slc3fR = self.slc3fR[idir]
			
			# Reconstruct states at left and right cell edges
			self.reconstruction(idir, dt)
			
			# Use states at cell edges (right edges in left cells, left edges in right cells) as 
			# input for the Riemann solver, which computes the Godunov fluxes at the interface 
			# walls. Here, we call interface i the interface between cells i-1 and i.
			flux = self.riemannSolver(idir, self.V_ip12[slc3dL], self.V_im12[slc3dR])
		
			# Update conserved fluid state. We are using Godunov's scheme, as in, we difference the 
			# fluxes taken from the Riemann solver. Note the convention that index i in the flux array
			# means the left interface of cell i, and i+1 the right interface of cell i.
			self.U[slc3dC] = self.U[slc3dC] + dt / self.dx * (flux[slc3fL] - flux[slc3fR])
			
			# Convert U -> V; this way, we are sure that plotting functions etc find both the correct
			# conserved and primitive variables.
			self.conservedToPrimitive(self.U, self.V)
		
			# Impose boundary conditions. This needs to happen after each dimensional sweep rather
			# than after each timestep, otherwise the second sweep will encounter some less 
			# advanced cells near the boundaries.
			self.enforceBoundaryConditions()
		
		# Increase timestep
		self.t += dt
		self.step += 1
		self.last_dir = idir
	
		return dt

	# ---------------------------------------------------------------------------------------------
	
	def save(self, filename = None):
		"""
		Save the current state of a simulation
		
		Parameters
		-------------------------------------------------------------------------------------------
		filename: str
			Output filename; auto-generated if ``None``
		"""
		
		if filename is None:
			filename = 'ulula_%04d.hdf5' % (self.step)
	
		print('Saving to file %s' % (filename))
	
		f = h5py.File(filename, 'w')

		f.create_group('code')
		f['code'].attrs['file_version'] = file_version_current
		
		f.create_group('hydro_scheme')
		f['hydro_scheme'].attrs['reconstruction'] = self.hs.reconstruction
		f['hydro_scheme'].attrs['limiter'] = self.hs.limiter
		f['hydro_scheme'].attrs['riemann'] = self.hs.riemann
		f['hydro_scheme'].attrs['time_integration'] = self.hs.time_integration
		f['hydro_scheme'].attrs['cfl'] = self.hs.cfl
	
		f.create_group('domain')
		f['domain'].attrs['xmin'] = self.xmin
		f['domain'].attrs['xmax'] = self.xmax
		f['domain'].attrs['ymin'] = self.ymin
		f['domain'].attrs['ymax'] = self.ymax
		f['domain'].attrs['dx'] = self.dx
		f['domain'].attrs['nq'] = self.nq
		f['domain'].attrs['nx'] = self.nx
		f['domain'].attrs['ny'] = self.ny
		f['domain'].attrs['nghost'] = self.nghost
		f['domain'].attrs['bc_type'] = self.bc_type
	
		f.create_group('physics')
		f['physics'].attrs['gamma'] = self.gamma
	
		f.create_group('run')
		f['run'].attrs['t'] = self.t
		f['run'].attrs['step'] = self.step
		f['run'].attrs['last_dir'] = self.last_dir
		
		f.create_group('grid')
		for q in self.q_prim:
			f['grid'][q] = self.V[self.q_prim[q], self.xlo:self.xhi+1, self.ylo:self.yhi+1]
		
		f.close()
		
		return

###################################################################################################

def load(filename):
	"""
	Load a snapshot file into a simulation object
	
	Parameters
	-----------------------------------------------------------------------------------------------
	filename: str
		Input filename

	Returns
	-----------------------------------------------------------------------------------------------
	sim: Simulation
		Object of type :data:`~ulula.simulation.Simulation`
	"""

	print('Loading simulation from file %s' % (filename))

	f = h5py.File(filename, 'r')

	# Check file version for compatibility
	file_version = f['code'].attrs['file_version']
	
	if ulula_utils.versionIsOlder(file_version_oldest, file_version):
		raise Exception('Cannot load simulation from file %s because version %s is too old (allowed %s).' \
					% (filename, file_version, file_version_oldest))

	# Load hydro scheme parameters
	hs_pars = {}
	hs_pars['reconstruction'] = f['hydro_scheme'].attrs['reconstruction']
	hs_pars['limiter'] = f['hydro_scheme'].attrs['limiter']
	hs_pars['riemann'] = f['hydro_scheme'].attrs['riemann']
	hs_pars['time_integration'] = f['hydro_scheme'].attrs['time_integration']
	hs_pars['cfl'] = f['hydro_scheme'].attrs['cfl']
	
	# Create hydro scheme and simulation objects
	hs = HydroScheme(**hs_pars)	
	sim = Simulation(hs)
	
	# Load domain parameters and initialize domain
	nx = f['domain'].attrs['nx']
	ny = f['domain'].attrs['ny']
	xmin = f['domain'].attrs['xmin']
	xmax = f['domain'].attrs['xmax']
	ymin = f['domain'].attrs['ymin']
	bc_type = f['domain'].attrs['bc_type']
	sim.setDomain(nx, ny, xmin = xmin, xmax = xmax, ymin = ymin, bc_type = bc_type)

	# Load fluid parameters
	gamma = f['physics'].attrs['gamma']
	sim.setFluidProperties(gamma)
	
	# Load and reset time and step
	sim.t = f['run'].attrs['t']
	sim.step = f['run'].attrs['step']
	sim.last_dir = f['run'].attrs['last_dir']
	
	# Set grid variables
	for q in sim.q_prim:
		sim.V[sim.q_prim[q], sim.xlo:sim.xhi+1, sim.ylo:sim.yhi+1] = f['grid'][q]
		
	# Initialize the conserved variables and ghost cells
	sim.primitiveToConserved(sim.V, sim.U)
	sim.enforceBoundaryConditions()

	f.close()
	
	return sim

###################################################################################################
