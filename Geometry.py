import numpy as np


class Geometry:
	def __init__(self):
		# The following use of 'N' denotes an integer, it does not mean that 'N' is the same value everywhere.
		self.control_points = None		# np.ndarray of size (N, 2) with columns [x, y]
		self.bound_vortices = None		# np.ndarray of size (N, 2) with columns [x, y]
		self.trailing_vortices = None	# np.ndarray of size (N, 2) with columns [x, y]

		self.dt = None 					# time step duration
		self.inflow_speed = None		# inflow speed to the flat plate

	def set(self, dt, inflow_speed):
		self._set(**{param: value for param, value in locals().items() if param != "self"})

	def get_positions(self):
		return {"bound_vortices": self.bound_vortices,
				"trailing_vortices": self.trailing_vortices,
				"control_points": self.control_points}

	def set_blade(self, plate_length: float, plate_res: int, angle_of_attack: float) -> None:
		"""
		  - Update "bound_vortices" and "control_points"
		  - Uses: self.rotate()
		:param plate_length:
		:param plate_res:
		:param angle_of_attack: angle of attack in radians
		:return:
		"""
		quarter_distance = plate_length/(4*plate_res)
		bound_vortices = np.zeros((plate_res, 2))
		control_points = np.zeros((plate_res, 2))
		bound_vortices[:, 0] = np.linspace(0, plate_length, plate_res)+quarter_distance
		control_points[:, 0] = bound_vortices[:, 0]+2*quarter_distance
		self.bound_vortices = self.rotate(bound_vortices, angle_of_attack)
		self.control_points = self.rotate(control_points, angle_of_attack)
		return None

	def rotate(self, to_rotate, angle):
		"""
		Update the "parameter" (could be either "bound_vortices" or "control_points"). This function rotates the
		coordinates from the input variable "to_rotate".
		:param to_rotate: np.ndarray of size (N, 3). The first two columns are the x and y coordinates.
		:param angle: rotation angle in radians
		:return:
		"""
		rot_matrix = np.asarray([[np.cos(angle), np.sin(angle)],
								 [-np.sin(angle), np.cos(angle)]])
		return np.dot(to_rotate, rot_matrix)

	def advect_trailing_vortices(self, velocities) -> None:
		"""

		:param velocities:
		:return:
		"""
		self.trailing_vortices += velocities+self.inflow_speed
		return None

	def _set(self, **kwargs) -> None:
		"""
		Sets any parameters of the instance. Raises an error if a parameter is trying to be set that doesn't exist.
		:param kwargs:
		:return:
		"""
		existing_parameters = [*self.__dict__]
		for parameter, value in kwargs.items():  # puts the tuples of parameters and values
			if parameter not in existing_parameters:
				raise ValueError(f"Parameter {parameter} cannot be set. Settable parameters are {existing_parameters}.")
			self.__dict__[parameter] = value
		return None
