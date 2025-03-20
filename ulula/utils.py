###################################################################################################
#
# Ulula -- utils.py
#
# A collection of utility functions used in Ulula.
#
# by Benedikt Diemer
#
###################################################################################################

import numpy as np

###################################################################################################

def parseVersionString(version_str):
	"""
	Parse a version string into numbers.
	
	There are more official functions to parse version numbers that use regular expressions and 
	can handle more formats according to PEP 440. Since Ulula versions are always composed of 
	three numbers and no letters, we implement a comparison manually to avoid needing to include 
	non-standard libraries.

	Parameters
	---------------------------
	version_str: str
		The version string to be converted.
	
	Returns
	-------
	nums: array_like
		A list of the three version numbers.
	"""
	
	w = version_str.split('.')
	if len(w) != 3:
		raise Exception('Version string invalid (%s), expected three numbers separated by dots.' \
					% version_str)

	nums = []
	for i in range(3):
		try:
			n = int(w[i])
		except Exception:
			raise Exception('Could not parse version element %s, expected a number.' % w[i])
		nums.append(n)
		
	return nums

###################################################################################################

def versionIsOlder(v1, v2):
	"""
	Compare two version strings.

	Parameters
	---------------------------
	v1: str
		A version string.
	v2: str
		A second version string.
	
	Returns
	-------
	is_older: bool
		``True`` if v2 is older than v1, ``False`` otherwise.
	"""

	n1 = parseVersionString(v1)
	n2 = parseVersionString(v2)

	is_older = False
	for i in range(3):
		if n2[i] < n1[i]:
			is_older = True
			break
	
	return is_older

###################################################################################################

def circleSquareOverlap(circle_x, circle_y, circle_r, square_x, square_y, square_size):
	"""
	Overlap between a circle and squares.
	
	Compute the overlapping area between a circle and squares with given left bottom corners. The
	squares can be arrays. This method is useful when averaging a 2D grid within circular annuli.
	
	This routine is based on a code by Philip Mansfield.
	
	Parameters
	-----------------------------------------------------------------------------------------------
	circle_x: float
		X-coordinate of the circle's center
	circle_y: float
		Y-coordinate of the circle's center
	circle_r: float
		Radius of the circle
	square_x: array_like
		Lower x-coordinates of the squares
	square_y: array_like
		Lower y-coordinates of the squares; must have same dimensions as ``square_x``
	square_size: array_like
		Sizes of the squares (can also be a number if all squares are equally large)

	Returns
	-----------------------------------------------------------------------------------------------
	overlap: array_like
		Area of each square that overlaps with the circle; has dimensions of (n, n) where n is the
		size of the ``square_x`` and ``square_y`` arrays
	"""

	def inside(x, y):
		return x**2 + y**2 < 1
	
	def intersect(y):
		return np.sqrt(1 - y**2)
	
	def integral(lim):
		return 0.5 * (np.sqrt(1 - lim**2) * lim + np.arcsin(lim))

	# Overlap between square with south west corner of (x, y) and side length of a with a unit 
	# circle.
	def vec_normalized_overlap(x, y, a):
		
		n = y + a
		e = x + a
		s = y
		w = x

		olp = (vec_quad_overlap(+n, +e, +s, +w, a) + # quadrant I
				vec_quad_overlap(+n, -w, +s, -e, a) + # quadrant II
				vec_quad_overlap(-s, -w, -n, -e, a) + # quadrand III
				vec_quad_overlap(-s, +e, -n, +w, a))  # quadrant IV
		
		return olp

	# Square overlap with quadrant I of a unit circle.
	def vec_quad_overlap(n, e, s, w, a):
		
		# Bound all variables to be at least zero.
		n = np.maximum(n, 0.0)
		e = np.maximum(e, 0.0)
		s = np.maximum(s, 0.0)
		w = np.maximum(w, 0.0)
		
		# There are two easy cases that often cover almost the entire domain: either the quadrant
		# is outside the circle in which case we leave it at zero. Or it is completely inside in 
		# which case we give it the full area of the quadrant.
		ret = np.zeros_like(n)
		mask_not_set = inside(w, s)
		mask_inside = inside(e, n)
		ret[mask_inside] = (n[mask_inside] - s[mask_inside]) * (e[mask_inside] - w[mask_inside])
		mask_not_set = mask_not_set & np.logical_not(mask_inside)
		
		# For the rest of the quadrants, we need to integrate the area of the circle as 
		# \int sqrt(1-x^2).
		n_ = n[mask_not_set]
		w_ = w[mask_not_set]
		e_ = e[mask_not_set]
		s_ = s[mask_not_set]
		start = np.array(w_)
		end = np.array(e_)
		mask = np.logical_not(inside(e_, s_))
		end[mask] = intersect(s_[mask])
		mask = inside(w_, n_)
		start[mask] = intersect(n_[mask])
	
		ret[mask_not_set] = (integral(end) - integral(start) - # classical integral
		        s_ * (end - start) + # area to the south of rectangle
		        (n_ - s_) * (start - w_)) # unintegrated area to the west of rectangle

		return ret
		
	# ---------------------------------------------------------------------------------------------

	dx = square_x - circle_x
	dy = square_y - circle_y
	olp = vec_normalized_overlap(dx / circle_r, dy / circle_r, square_size / circle_r) * circle_r**2

	return olp

###################################################################################################
