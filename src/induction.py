import numpy as np
from copy import copy


class Induction:
	def __init__(self, plate_res: int, flap_res: int):
		self.plate_res = plate_res
		self.flap_res = flap_res
		self.n_cp = self.plate_res+self.flap_res
	
	def lhs_matrix(self,
				   bound_vortices: np.ndarray,
				   shed_vortex: np.ndarray,
				   plate_control_points: np.ndarray,
				   flap_control_points: np.ndarray=None):
		plate_normal, _ = self._unit_normal_and_length(bound_vortices[0, :])
		flap_normal = None if self.n_cp == 0 else self._flap_normal(plate_control_points, flap_control_points)
		
		lhs = np.ones((self.n_cp+1, self.n_cp+1))
		control_points = plate_control_points if self.n_cp == 0 else np.r_[plate_control_points, flap_control_points]
		normals = self.plate_res*[plate_normal] + self.flap_res*[flap_normal]
		for (i_cp, cp), normal in zip(enumerate(control_points), normals):
			for i_v, vortex in enumerate(bound_vortices):
				lhs[i_cp, i_v] = normal@self._induction(vortex, cp)
			lhs[i_cp, -1] = normal@self._induction(shed_vortex, cp)
		return lhs
	
	def control_point_induction(self,
							    plate_control_points: np.ndarray,
							    vortices: np.ndarray,
							    flap_control_points: np.ndarray = None):
		plate_normal, _ = self._unit_normal_and_length(plate_control_points[0, :])
		flap_normal = None if self.n_cp == 0 else self._flap_normal(plate_control_points, flap_control_points)
		
		cpi_mat = np.zeros((self.n_cp, vortices.shape[0]))
		control_points = plate_control_points if self.n_cp == 0 else np.r_[plate_control_points, flap_control_points]
		normals = self.plate_res*[plate_normal] + self.flap_res*[flap_normal]
		for (i_cp, cp), normal in zip(enumerate(control_points), normals):
			for i_v, vortex in enumerate(vortices):
				cpi_mat[i_cp, i_v] = normal@self._induction(vortex, cp)
		return cpi_mat
	
	def free_vortex_induction(self,
							  free_vortices: np.ndarray,
							  bound_vortices: np.ndarray):
		fvi_free_mat_x = np.zeros((free_vortices.shape[0], free_vortices.shape[0]))
		fvi_free_mat_y = np.zeros((free_vortices.shape[0], free_vortices.shape[0]))
		for i_fv, free_vortex in enumerate(free_vortices):
			vortices_inducing = free_vortices[i_fv+1:]
			for i_inducing, inducing_vortex in enumerate(vortices_inducing):
				induced = self._induction(inducing_vortex, free_vortex)
				fvi_free_mat_x[i_fv, i_inducing+i_fv+1] = induced[0]
				fvi_free_mat_y[i_fv, i_inducing+i_fv+1] = induced[1]
		fvi_free_mat_x = fvi_free_mat_x-fvi_free_mat_x.T
		fvi_free_mat_y = fvi_free_mat_y-fvi_free_mat_y.T
		
		fvi_bound_mat_x = np.zeros((free_vortices.shape[0], bound_vortices.shape[0]))
		fvi_bound_mat_y = np.zeros((free_vortices.shape[0], bound_vortices.shape[0]))
		for i_fv, free_vortex in enumerate(free_vortices):
			for i_inducing, inducing_vortex in enumerate(bound_vortices):
				induced = self._induction(inducing_vortex, free_vortex)
				fvi_bound_mat_x[i_fv, i_inducing] = induced[0]
				fvi_bound_mat_y[i_fv, i_inducing] = induced[1]
		return {"x": fvi_free_mat_x, "y": fvi_free_mat_y}, {"x": fvi_bound_mat_x, "y": fvi_bound_mat_y}
	
	def _induction(self, vortex: np.ndarray, induction_point: np.ndarray):
		direction, distance = self._unit_normal_and_length(induction_point-vortex)
		return 1/(2*np.pi*distance)*direction
	
	def _flap_normal(self, plate_control_points: np.ndarray, flap_control_points: np.ndarray):
		if self.flap_res > 1:
			return self._unit_normal_and_length(flap_control_points[1, :]-flap_control_points[0, :])
		else:
			vec_from = plate_control_points[-1, :]+plate_control_points[0, :]/3
			return self._unit_normal_and_length(flap_control_points[0, :]-vec_from)[0]
		
	@staticmethod
	def _unit_normal_and_length(unit_normal_for: np.ndarray):
		vector_length = np.linalg.norm(unit_normal_for)
		normalised = unit_normal_for/vector_length
		return np.r_[-normalised[1], normalised[0]], vector_length

		
