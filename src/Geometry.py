import matplotlib.pyplot as plt
import numpy as np
from copy import copy


class Geometry:
	def __init__(self):
		# The following use of 'N' denotes an integer, it does not mean that 'N' is the same value everywhere.
		self.control_points = None  # np.ndarray of size (N, 2) with columns [x, y]
		self.bound_vortices = None  # np.ndarray of size (N, 2) with columns [x, y]
		self.trailing_vortices = None  # np.ndarray of size (N, 2) with columns [x, y]
		self.plate_normal = None
		
		self.control_points_base = None
		self.bound_vortices_base = None
		self.plate_elem_length = None
		
		self.dt = None  # time step duration
		self.inflow_speed = None  # inflow speed to the flat plate
		self.aoa = None
		self.n_time_steps = None
		self.trailing_counter = 0
		self.plate_res = None
		self.lhs = None
		self.trailing_induction_matrix = None
		self.bound_distances_v_0 = None
	
	def set_constants(self, dt: float, n_time_steps: int):
		self._set(**{param: value for param, value in locals().items() if param != "self"})
		self.trailing_vortices = np.zeros((n_time_steps, 2))
	
	def set_blade(self, plate_length: float, plate_res: int) -> None:
		"""
		  - Update "bound_vortices" and "control_points"
		  - Uses: self.rotate()
		:param plate_length:
		:param plate_res:
		:param angle_of_attack: angle of attack in radians
		:return:
		"""
		self.plate_res = plate_res
		self.plate_elem_length = plate_length/plate_res
		quarter_distance = self.plate_elem_length/4
		bound_vortices_x = np.asarray([quarter_distance+i*self.plate_elem_length for i in range(plate_res)])
		bound_vortices = np.append([bound_vortices_x], np.zeros((1, plate_res)), axis=0).T
		control_points = np.append([bound_vortices_x+2*quarter_distance], np.zeros((1, plate_res)), axis=0).T
		self.bound_vortices_base = bound_vortices
		self.control_points_base = control_points
		self._init_lhs_matrix()
		return None
	
	def update_flow(self, angle_of_attack: float, inflow_speed: float) -> None:
		"""
		
		:param inflow_speed:
		:param angle_of_attack: in radians
		:return:
		"""
		self.aoa = -angle_of_attack  # positive aoas are defined for negative angles in the coordinate system
		self.inflow_speed = inflow_speed
		self.bound_vortices = self._rotate(self.bound_vortices_base, self.aoa)
		self.control_points = self._rotate(self.control_points_base, self.aoa)
		self.plate_normal, _ = self._unit_normal_and_length(self.bound_vortices[0, :])
		return None
	
	def get_positions(self):
		return {"bound_vortices"   : copy(self.bound_vortices),
				"trailing_vortices": copy(self.trailing_vortices[:self.trailing_counter, :]),
				"control_points"   : copy(self.control_points)}
	
	def get_normals(self):
		return {"plate": copy(self.plate_normal)}
	
	def get_trailing_induction_matrix(self):
		if self.trailing_counter == 0:
			pass
		else:
			if self.trailing_induction_matrix is not None:
				if self.trailing_induction_matrix.shape[0] == self.trailing_counter:
					copy(self.trailing_induction_matrix)
			mat = np.zeros((self.plate_res, self.trailing_counter))
			for i_cp, cp in enumerate(self.control_points):
				for i_trailing, trailing_vortex in enumerate(self.trailing_vortices[:self.trailing_counter]):
					vortex_to_cp = cp-trailing_vortex
					induction_direction, distance = self._unit_normal_and_length(vortex_to_cp)
					mat[i_cp, i_trailing] = self.plate_normal@induction_direction/(2*np.pi*distance)
			self.trailing_induction_matrix = mat
		return copy(self.trailing_induction_matrix)
	
	def get_lhs_matrix(self):
		if self.inflow_speed is None:
			print("Initialising a LHS matrix for an inflow velocity of 0m/s")
			self._init_lhs_matrix()
		else:
			tim = self.get_trailing_induction_matrix()
			if tim is None:
				print("Calculating for steady case. If the unsteady case was intended, add trailing vortices using "
					  "'add_trailing_vortex' before calculating the lhs matrix.")
				self.lhs = self.lhs[:-1, :-1]
			else:
				self.lhs[:-1, -1] = tim[:, -1]
		return copy(self.lhs)
	
	def get_trailing_displacement_matrices(self, additional_vortices: np.ndarray = None):
		mat_bound_x = np.zeros((self.trailing_counter, self.plate_res))
		mat_bound_y = np.zeros((self.trailing_counter, self.plate_res))
		mat_trailing_x = np.zeros((self.trailing_counter, self.trailing_counter))
		mat_trailing_y = np.zeros((self.trailing_counter, self.trailing_counter))
		mat_additional_x = None if additional_vortices is None else np.zeros((additional_vortices.shape[0],
																			  self.trailing_counter))
		mat_additional_y = None if additional_vortices is None else np.zeros((additional_vortices.shape[0],
																			  self.trailing_counter))
		if additional_vortices is not None:
			raise NotImplementedError("Additional vortices are not implemented yet.")
		
		for i_trailing, trailing_vortex in enumerate(self.trailing_vortices[:self.trailing_counter]):
			for i_bound, bound_vortex in enumerate(self.bound_vortices):
				bound_to_trailing = trailing_vortex-bound_vortex
				induction_direction, distance = self._unit_normal_and_length(bound_to_trailing)
				mat_bound_x[i_trailing, i_bound] = induction_direction[0]/distance
				mat_bound_y[i_trailing, i_bound] = induction_direction[1]/distance
		
		for i_inducing, v_inducing in enumerate(self.trailing_vortices[:self.trailing_counter]):
			vortices_induced = self.trailing_vortices[i_inducing+1:self.trailing_counter]
			for j, v_induced in enumerate(vortices_induced):
				inducing_to_induced = v_induced-v_inducing
				induction_direction, distance = self._unit_normal_and_length(inducing_to_induced)
				mat_trailing_x[i_inducing, j+i_inducing+1] = induction_direction[0]/distance
				mat_trailing_y[i_inducing, j+i_inducing+1] = induction_direction[1]/distance
		
		correction = 1/(2*np.pi)
		return {"x": mat_bound_x*correction, "y": mat_bound_y*correction}, \
			   {"x": (-mat_trailing_x+mat_trailing_x.T)*correction, "y": (-mat_trailing_y+mat_trailing_y.T)*correction}, \
			   {"x": mat_additional_x, "y": mat_additional_y}
	
	def add_trailing_vortex(self, new_trailing_fac: float = 0.5):
		trailing_edge = self.control_points[-1, :]+self.bound_vortices[0, :]
		distance_traveled = self.inflow_speed*self.dt
		new_trailing_pos = trailing_edge+new_trailing_fac*np.asarray([distance_traveled, 0])
		self.trailing_vortices[self.trailing_counter, :] = new_trailing_pos
		self.trailing_counter += 1
	
	def displace_trailing_vortices(self, induced_velocities: np.ndarray) -> None:
		"""S
		
		:param new_trailing_fac:
		:param induced_velocities:
		:return:
		"""
		v_inflow_speed = np.c_[self.inflow_speed*np.ones((self.trailing_counter, 1)),
							   np.zeros((self.trailing_counter, 1))]
		self.trailing_vortices[:self.trailing_counter, :] += (induced_velocities+v_inflow_speed)*self.dt
		return None
	
	def plot_vortices(self, show: bool = True):
		fig, ax = plt.subplots()
		positions = self.get_positions()
		bound, trailing, cp = positions["bound_vortices"], positions["trailing_vortices"], positions["control_points"]
		ax.plot(bound[:, 0], bound[:, 1], "xr")
		ax.plot(trailing[:, 0], trailing[:, 1], "xg")
		ax.plot(cp[:, 0], cp[:, 1], "xk")
		if show:
			plt.show()
		else:
			return fig, ax
	
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
	
	def _init_lhs_matrix(self):
		self.bound_distances_v_0 = np.asarray([self.plate_elem_length/2+self.plate_elem_length*i for i in
											   range(self.plate_res)])
		self.bound_distances_v_0 = np.append(self.bound_distances_v_0[::-1], -self.bound_distances_v_0)
		inductions = 1/(2*np.pi*self.bound_distances_v_0)
		self.lhs = np.ones((self.plate_res+1, self.plate_res+1))
		for i in range(self.plate_res):
			self.lhs[i, :] = inductions[self.plate_res-i-1:2*self.plate_res-i]
		return None
	
	@staticmethod
	def _rotate(to_rotate: np.ndarray, angle: float):
		"""
		Update the "parameter" (could be either "bound_vortices" or "control_points"). This function rotates the
		coordinates from the input variable "to_rotate".
		:param to_rotate: np.ndarray of size (N, 3). The first two columns are the x and y coordinates.
		:param angle: rotation angle in radians
		:return:
		"""
		rot_matrix = np.asarray([[np.cos(angle), -np.sin(angle)],
								 [np.sin(angle), np.cos(angle)]])
		return to_rotate@rot_matrix.T
	
	@staticmethod
	def _unit_normal_and_length(unit_normal_for: np.ndarray):
		vector_length = np.linalg.norm(unit_normal_for)
		normalised = unit_normal_for/vector_length
		return np.asarray([-normalised[1], normalised[0]]), vector_length
