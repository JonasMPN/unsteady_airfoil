import numpy as np
from copy import copy


class Geometry:
	def __init__(self):
		# The following use of 'N' denotes an integer, it does not mean that 'N' is the same value everywhere.
		self.control_points = None		# np.ndarray of size (N, 2) with columns [x, y]
		self.bound_vortices = None		# np.ndarray of size (N, 2) with columns [x, y]
		self.trailing_vortices = None	# np.ndarray of size (N, 2) with columns [x, y]
		
		self.control_points_base = None
		self.bound_vortices_base = None
		
		self.dt = None 					# time step duration
		self.inflow_speed = None		# inflow speed to the flat plate
		self.n_time_steps = None
		self.time_step_counter = 0
		self.plate_res = None

	def set(self, dt: float, n_time_steps: int):
		self._set(**{param: value for param, value in locals().items() if param != "self"})
		self.trailing_vortices = np.zeros((n_time_steps, 2))
		
	def update_aoa(self, angle_of_attack: float) -> None:
		"""
		
		:param angle_of_attack: in radians
		:return:
		"""
		self.bound_vortices = self._rotate(self.bound_vortices_base, angle_of_attack)
		self.control_points = self._rotate(self.control_points_base, angle_of_attack)
		return None
		
	def get_positions(self):
		return {"bound_vortices": copy(self.bound_vortices),
				"trailing_vortices": copy(self.trailing_vortices[:self.time_step_counter, :][::-1]),
				"control_points": copy(self.control_points)}

	def set_blade(self, plate_length: float, plate_res: int, angle_of_attack: float) -> None:
		"""
		  - Update "bound_vortices" and "control_points"
		  - Uses: self.rotate()
		:param plate_length:
		:param plate_res:
		:param angle_of_attack: angle of attack in radians
		:return:
		"""
		self.plate_res = plate_res
		quarter_distance = plate_length/(4*plate_res)
		bound_vortices_x = np.linspace(0, plate_length, plate_res)+quarter_distance
		bound_vortices = np.append([bound_vortices_x], np.zeros((1, plate_res)), axis=0).T
		control_points = np.append([bound_vortices_x+2*quarter_distance], np.zeros((1, plate_res)), axis=0).T
		self.bound_vortices_base = bound_vortices
		self.control_points_base = control_points
		self.bound_vortices = self._rotate(bound_vortices, angle_of_attack)
		self.control_points = self._rotate(control_points, angle_of_attack)
		return None

	def advect_trailing_vortices(self, induced_velocities: np.ndarray, inflow_speed: float) -> None:
		"""
		
		:param induced_velocities:
		:param inflow_speed:
		:return:
		"""
		self.trailing_vortices[self.time_step_counter, :] = 5*self.bound_vortices[0, :]*self.plate_res
		inflow_speed = np.append(inflow_speed*np.ones((self.time_step_counter, 1)),
								 np.zeros((self.time_step_counter, 1)), axis=1)
		self.trailing_vortices[:self.time_step_counter, :] += (induced_velocities+inflow_speed)*self.dt
		self.time_step_counter += 1
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
	
	@staticmethod
	def _rotate(to_rotate, angle):
		"""
		Update the "parameter" (could be either "bound_vortices" or "control_points"). This function rotates the
		coordinates from the input variable "to_rotate".
		:param to_rotate: np.ndarray of size (N, 3). The first two columns are the x and y coordinates.
		:param angle: rotation angle in radians
		:return:
		"""
		angle = -angle
		rot_matrix = np.asarray([[np.cos(angle), np.sin(angle)],
								 [-np.sin(angle), np.cos(angle)]])
		return np.dot(to_rotate, rot_matrix)
