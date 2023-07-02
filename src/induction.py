import numpy as np
from numpy.linalg import norm
import numba as nb


class Induction:
	def __init__(self, plate_res: int, flap_res: int = 0):
		self.plate_res = plate_res
		self.flap_res = flap_res
		self.n_cp = self.plate_res+self.flap_res
			
	def lhs_matrix(self,
				   bound_vortices: np.ndarray,
				   shed_vortex: np.ndarray,
				   plate_control_points: np.ndarray,
				   flap_control_points: np.ndarray = None,
				   precision: np.typecodes = np.float64):
		plate_normal, _ = self._unit_normal_and_length(bound_vortices[0, :])
		flap_normal = None if self.flap_res == 0 else self._flap_normal(plate_control_points, flap_control_points)

		control_points = plate_control_points if self.n_cp == 0 else np.r_[plate_control_points, flap_control_points]
		bound_vortices = np.r_[bound_vortices, [shed_vortex]].astype(precision)
		control_points.astype(precision)
		x_normals = np.r_[plate_normal[0]*np.ones((self.plate_res, self.n_cp+1)),
						  flap_normal[0]*np.ones((self.flap_res, self.n_cp+1)),
						  np.ones((1, self.n_cp+1))].astype(precision)
		y_normals = np.r_[plate_normal[1]*np.ones((self.plate_res, self.n_cp+1)),
						  flap_normal[1]*np.ones((self.flap_res, self.n_cp+1)),
						  np.zeros((1, self.n_cp+1))].astype(precision)
		pre_allocated = np.ones((2, self.n_cp+1, self.n_cp+1), dtype=precision)
		induction = self.induction_matrices(bound_vortices, control_points, pre_allocated, (x_normals, y_normals))
		return induction[0, :, :]+induction[1, :, :]
	
	def lhs_steady_matrix(self,
					      bound_vortices: np.ndarray,
					      plate_control_points: np.ndarray,
					      flap_control_points: np.ndarray = None,
					      precision: np.typecodes = np.float64):
		plate_normal, _ = self._unit_normal_and_length(bound_vortices[0, :])
		flap_normal = None if self.flap_res == 0 else self._flap_normal(plate_control_points, flap_control_points)

		control_points = plate_control_points if self.flap_res == 0 else np.r_[plate_control_points, flap_control_points]
		bound_vortices = bound_vortices.astype(precision)
		control_points.astype(precision)
		if self.flap_res == 0:
			x_normals = plate_normal[0]*np.ones((self.plate_res, self.n_cp), dtype=precision)
			y_normals = plate_normal[1]*np.ones((self.plate_res, self.n_cp), dtype=precision)
		else:
			x_normals = np.r_[plate_normal[0]*np.ones((self.plate_res, self.n_cp)),
							  flap_normal[0]*np.ones((self.flap_res, self.n_cp))].astype(precision)
			y_normals = np.r_[plate_normal[1]*np.ones((self.plate_res, self.n_cp)),
							  flap_normal[1]*np.ones((self.flap_res, self.n_cp))].astype(precision)
		pre_allocated = np.empty((2, self.n_cp, self.n_cp), dtype=precision)
		induction = self.induction_matrices(bound_vortices, control_points, pre_allocated, (x_normals, y_normals))
		return induction[0, :, :]+induction[1, :, :]

	def control_point_induction(self,
								plate_control_points: np.ndarray,
								vortices: np.ndarray,
								flap_control_points: np.ndarray = None,
								precision: np.typecodes = np.float64):
		plate_normal, _ = self._unit_normal_and_length(plate_control_points[0, :])
		flap_normal = None if self.flap_res == 0 else self._flap_normal(plate_control_points, flap_control_points)
		control_points = plate_control_points if self.flap_res == 0 else np.r_[
			plate_control_points, flap_control_points]

		x_normals = np.r_[plate_normal[0]*np.ones((self.plate_res, vortices.shape[0])),
						  flap_normal[0]**np.ones((self.flap_res, vortices.shape[0]))].astype(precision)
		y_normals = np.r_[plate_normal[1]*np.ones((self.plate_res, vortices.shape[0])),
						  flap_normal[1]**np.ones((self.flap_res, vortices.shape[0]))].astype(precision)
		pre_allocated = np.zeros((2, self.n_cp, vortices.shape[0]), dtype=precision)
		induction = self.induction_matrices(vortices.astype(precision), control_points.astype(precision),
											pre_allocated, (x_normals, y_normals))
		return induction[0, :, :]+induction[1, :, :]

	def free_vortex_induction(self,
							  free_vortices: np.ndarray,
							  bound_vortices: np.ndarray,
							  precision: np.typecodes = np.float64):
		free_vortices.astype(precision), bound_vortices.astype(precision)
		pre_allocated = np.zeros((2, free_vortices.shape[0], free_vortices.shape[0]), dtype=precision)
		fvi_free_mat_x, fvi_free_mat_y = self.wake_induction_matrices(free_vortices, pre_allocated)

		pre_allocated = np.zeros((2, free_vortices.shape[0], bound_vortices.shape[0]), dtype=precision)
		fvi_bound_mat_x, fvi_bound_mat_y = self.induction_matrices(bound_vortices, free_vortices, pre_allocated)
		return {"x": fvi_free_mat_x, "y": fvi_free_mat_y}, {"x": fvi_bound_mat_x, "y": fvi_bound_mat_y}

	@staticmethod
	@nb.njit(fastmath=True, error_model="numpy", parallel=True)
	def induction_matrices(vortices: np.ndarray, induction_points: np.ndarray, save_to: np.ndarray,
						   normals: tuple = None):
		fac = 2*np.pi
		n_vortices = vortices.shape[0]
		for ip_i in range(induction_points.shape[0]):
			for v_i in nb.prange(n_vortices):
				vortex_to_ip = induction_points[ip_i]-vortices[v_i]
				distance = norm(vortex_to_ip)
				save_to[0, ip_i, v_i] = -vortex_to_ip[1]/(fac*distance**2)
				save_to[1, ip_i, v_i] = vortex_to_ip[0]/(fac*distance**2)
		if normals is not None:
			save_to[0, :, :] *= normals[0]
			save_to[1, :, :] *= normals[1]
		return save_to

	@staticmethod
	@nb.njit(fastmath=True, error_model="numpy", parallel=True)
	def wake_induction_matrices(vortices: np.ndarray, save_to: np.ndarray):
		fac = 2*np.pi
		n_vortices = vortices.shape[0]
		for ip_i in range(n_vortices):
			for v_i in nb.prange(ip_i+1, n_vortices):
				vortex_to_ip = vortices[ip_i]-vortices[v_i]
				distance = norm(vortex_to_ip)
				save_to[0, ip_i, v_i] = -vortex_to_ip[1]/(fac*distance**2)
				save_to[1, ip_i, v_i] = vortex_to_ip[0]/(fac*distance**2)
		return save_to[0, :, :]-save_to[0, :, :].T, save_to[1, :, :]-save_to[1, :, :].T

	def _flap_normal(self, plate_control_points: np.ndarray, flap_control_points: np.ndarray):
		if self.flap_res > 1:
			return self._unit_normal_and_length(flap_control_points[1, :]-flap_control_points[0, :])[0]
		else:
			vec_from = plate_control_points[-1, :]+plate_control_points[0, :]/3
			return self._unit_normal_and_length(flap_control_points[0, :]-vec_from)[0]

	@staticmethod
	def _unit_normal_and_length(unit_normal_for: np.ndarray):
		vector_length = np.linalg.norm(unit_normal_for)
		normalised = unit_normal_for/vector_length
		return np.r_[-normalised[1], normalised[0]], vector_length
