from typing import Tuple, Dict

from numpy import ndarray

from geometry import Geometry
from induction import Induction
import numpy as np
import timeit


class UnsteadyAirfoil:
	def __init__(self, max_time_steps: int,
	             plate_res: int, plate_length: float,
	             flap_res: int = 0, flap_length: float = None):
		self.plate_res = plate_res
		self.flap_res = flap_res
		self.structure_res = plate_res+flap_res
		self.n_time_steps = max_time_steps
		self.free_vortices_circulation = np.zeros((0, 1))
		self.use_flap = False
		self.geometry = Geometry(max_time_steps)
		self.induction = Induction(plate_res, flap_res)
		self.n_free_vortices = 0
		
		self.geometry.set_plate(plate_length, plate_res)
		if flap_res != 0:
			self.geometry.set_flap(flap_length, flap_res)
			self.use_flap = True
	
	def toggle_flap(self, use_flap: bool) -> None:
		self.use_flap = use_flap
		return None
	
	def add_free_vortices(self, coordinates: np.ndarray, circulations: list or np.ndarray) -> None:
		self.geometry.add_free_vortices(coordinates)
		self.free_vortices_circulation = circulations if type(circulations) == np.ndarray else np.asarray(circulations)
		self.n_free_vortices = self.free_vortices_circulation.size
		return None
	
	def solve(self,
	          dt: float,
	          plate_angles: float or list or np.ndarray,
	          inflows: tuple or list[list] or np.ndarray,
	          flap_angles: float or list or np.ndarray = None,
	          shed_trailing_distance: float = 0.5,
	          precision: np.dtype = np.float32) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
		plate_angles = plate_angles if type(plate_angles) != float else [plate_angles for _ in range(self.n_time_steps)]
		inflows = inflows if type(inflows) != tuple else [[inflows[0], inflows[1]] for _ in range(self.n_time_steps)]
		if flap_angles is None:
			flap_angles = np.zeros(self.n_time_steps)
		else:
			flap_angles = flap_angles if type(flap_angles) != float else [flap_angles for _ in range(self.n_time_steps)]
		self.geometry.toggle_flap(self.use_flap)
		
		surrounding_vortices = np.zeros((self.n_free_vortices+self.n_time_steps, 2))
		surrounding_vortices[:self.n_free_vortices] = self.geometry.get_positions()[3]
		surrounding_circulation = np.zeros((self.n_free_vortices+self.n_time_steps, 1))
		if self.free_vortices_circulation.size != 0:
			surrounding_circulation[:self.n_free_vortices] = self.free_vortices_circulation
		plate_circulation = np.zeros((self.n_time_steps, self.plate_res), dtype=precision)
		flap_circulation = np.zeros((self.n_time_steps, self.flap_res), dtype=precision)
		
		old_plate_angle, old_flap_angle, old_inflow = 0, 0, [0, 0]
		normal_inflow = [0, 0]
		start_time = timeit.default_timer()
		for dt_i, (plate_angle, flap_angle, inflow) in enumerate(zip(plate_angles, flap_angles, inflows)):
			dt_i_free = dt_i+self.n_free_vortices
			dt_i += self.n_free_vortices
			if dt_i == 0 or (old_plate_angle != plate_angle or old_flap_angle != flap_angle or old_inflow != inflow):
				self.geometry.update_rotation(plate_angle=plate_angle, flap_angle=flap_angle)
				self.geometry.shed_vortex(inflow=inflow, time_step=dt, new_trailing_fac=shed_trailing_distance)
				bound, control, trailing, free = self.geometry.get_positions()
				combined_bound = np.r_[bound["plate"], bound["flap"]] if self.flap_res != 0 else bound["plate"]
				lhs = self.induction.lhs_matrix(bound_vortices=combined_bound, shed_vortex=trailing[-1, :],
				                                plate_control_points=control["plate"],
				                                flap_control_points=control["flap"])
				inv_lhs = np.linalg.inv(lhs)
				plate_normal, flap_normal = self.geometry.get_normals()
				normal_inflow = np.asarray([inflow[0], inflow[1]])@plate_normal*np.ones((self.plate_res, 1))
				if self.flap_res != 0:
					flap_inflow = np.asarray([inflow[0], inflow[1]])@flap_normal
					normal_inflow = np.r_[normal_inflow, flap_inflow*np.ones((self.flap_res, 1))]
				old_plate_angle, old_flap_angle, old_inflow = plate_angle, flap_angle, inflow
			else:
				self.geometry.shed_vortex(inflow=inflow, time_step=dt, new_trailing_fac=shed_trailing_distance)
				bound, control, trailing, free = self.geometry.get_positions()
				combined_bound = np.r_[bound["plate"], bound["flap"]] if self.flap_res != 0 else bound["plate"]
			
			cpi = self.induction.control_point_induction(plate_control_points=control["plate"],
			                                             flap_control_points=control["flap"],
			                                             vortices=surrounding_vortices[:dt_i_free])
			induction_from_surrounding = cpi@surrounding_circulation[:dt_i_free]
			bound_circulation = -inv_lhs@(np.r_[normal_inflow+induction_from_surrounding,
			                                    [[np.sum(surrounding_circulation[self.n_free_vortices:dt_i_free]), ]]])
			surrounding_circulation[dt_i_free, 0] = bound_circulation[-1]
			
			surrounding_vortices = np.r_[free, trailing]
			from_surrounding, from_bound = self.induction.free_vortex_induction(free_vortices=surrounding_vortices,
			                                                                    bound_vortices=combined_bound)
			
			from_wake_velocities_x = from_surrounding["x"]@surrounding_circulation[:dt_i_free+1]
			from_wake_velocities_y = from_surrounding["y"]@surrounding_circulation[:dt_i_free+1]
			from_bound_velocities_x = from_bound["x"]@bound_circulation[:-1]
			from_bound_velocities_y = from_bound["y"]@bound_circulation[:-1]
			
			x_vel = from_wake_velocities_x+from_bound_velocities_x
			y_vel = from_wake_velocities_y+from_bound_velocities_y
			background_flow = np.asarray((dt_i_free+1)*[[inflow[0], inflow[1]]])
			self.geometry.displace_vortices(velocities=np.c_[x_vel, y_vel]+background_flow, time_step=dt)
			
			plate_circulation[dt_i] = bound_circulation[:self.plate_res].flatten()
			if self.flap_res != 0:
				flap_circulation[dt_i] = bound_circulation[self.plate_res:-1].flatten()
			if (dt_i-self.n_free_vortices)%20 == 0:
				end_time = timeit.default_timer()
				calc_time = np.round(end_time-start_time, 4)
				print("iteration: ", dt_i-self.n_free_vortices, "calculation time: ", calc_time, "s")
				start_time = timeit.default_timer()
		circulations = {"plate": plate_circulation, "flap": flap_circulation,
		                "trailing": surrounding_circulation[self.n_free_vortices:],
		                "free": surrounding_circulation[:self.n_free_vortices]}
		vortex_positions = {"plate": bound["plate"], "flap": bound["flap"], "trailing": trailing, "free": free}
		return circulations, vortex_positions
	
	def solve_for_process(self,
	                      dt: float,
	                      plate_angles: float or list or np.ndarray,
	                      inflows: tuple or list[list] or np.ndarray,
	                      flap_angles: float or list or np.ndarray = None,
	                      shed_trailing_distance: float = 0.5,
	                      precision: np.dtype = np.float32) -> tuple[dict[str, ndarray], dict[str, ndarray]]:
		plate_angles = plate_angles if type(plate_angles) != float else [plate_angles for _ in range(self.n_time_steps)]
		inflows = inflows if type(inflows) != tuple else [[inflows[0], inflows[1]] for _ in range(self.n_time_steps)]
		if flap_angles is None:
			flap_angles = np.zeros(self.n_time_steps)
		else:
			flap_angles = flap_angles if type(flap_angles) != float else [flap_angles for _ in range(self.n_time_steps)]
		self.geometry.toggle_flap(self.use_flap)
		
		surrounding_vortices = np.zeros((self.n_free_vortices+self.n_time_steps, 2))
		surrounding_vortices[:self.n_free_vortices] = self.geometry.get_positions()[3]
		surrounding_circulation = np.zeros((self.n_free_vortices+self.n_time_steps, 1))
		if self.free_vortices_circulation.size != 0:
			surrounding_circulation[:self.n_free_vortices] = self.free_vortices_circulation
		
		all_bound = np.empty((self.n_time_steps, self.structure_res, 2), dtype=precision)
		all_cp = np.empty((self.n_time_steps, self.structure_res, 2), dtype=precision)
		all_free = np.empty((self.n_time_steps, self.n_free_vortices, 2), dtype=precision)
		all_trailing = np.empty(self.n_time_steps, dtype=object)
		for dt_i in range(self.n_time_steps):
			all_trailing[dt_i] = np.empty((dt_i, 2), dtype=precision)
		plate_circulation = np.zeros((self.n_time_steps, self.plate_res), dtype=precision)
		flap_circulation = np.zeros((self.n_time_steps, self.flap_res), dtype=precision)
		
		old_plate_angle, old_flap_angle, old_inflow = 0, 0, [0, 0]
		normal_inflow = [0, 0]
		start_time = timeit.default_timer()
		for dt_i, (plate_angle, flap_angle, inflow) in enumerate(zip(plate_angles, flap_angles, inflows)):
			dt_i_free = dt_i+self.n_free_vortices
			if dt_i == 0 or (old_plate_angle != plate_angle or old_flap_angle != flap_angle or old_inflow != inflow):
				self.geometry.update_rotation(plate_angle=plate_angle, flap_angle=flap_angle)
				self.geometry.shed_vortex(inflow=inflow, time_step=dt, new_trailing_fac=shed_trailing_distance)
				bound, control, trailing, free = self.geometry.get_positions()
				combined_bound = np.r_[bound["plate"], bound["flap"]] if self.flap_res != 0 else bound["plate"]
				lhs = self.induction.lhs_matrix(bound_vortices=combined_bound, shed_vortex=trailing[-1, :],
				                                plate_control_points=control["plate"],
				                                flap_control_points=control["flap"], precision=precision)
				inv_lhs = np.linalg.inv(lhs)
				plate_normal, flap_normal = self.geometry.get_normals()
				normal_inflow = np.asarray([inflow[0], inflow[1]])@plate_normal*np.ones((self.plate_res, 1))
				if self.flap_res != 0:
					flap_inflow = np.asarray([inflow[0], inflow[1]])@flap_normal
					normal_inflow = np.r_[normal_inflow, flap_inflow*np.ones((self.flap_res, 1))]
				old_plate_angle, old_flap_angle, old_inflow = plate_angle, flap_angle, inflow
			else:
				self.geometry.shed_vortex(inflow=inflow, time_step=dt, new_trailing_fac=shed_trailing_distance)
				bound, control, trailing, free = self.geometry.get_positions()
				combined_bound = np.r_[bound["plate"], bound["flap"]] if self.flap_res != 0 else bound["plate"]
			
			all_bound[dt_i], all_trailing[dt_i], all_free[dt_i] = combined_bound, trailing, free
			all_cp[dt_i] = np.r_[control["plate"], control["flap"]] if self.flap_res != 0 else control["plate"]
			
			cpi = self.induction.control_point_induction(plate_control_points=control["plate"],
			                                             flap_control_points=control["flap"],
			                                             vortices=surrounding_vortices[:dt_i_free], precision=precision)
			induction_from_surrounding = cpi@surrounding_circulation[:dt_i_free]
			bound_circulation = -inv_lhs@(np.r_[normal_inflow+induction_from_surrounding,
			                                    [[np.sum(surrounding_circulation[self.n_free_vortices:dt_i_free]), ]]])
			surrounding_circulation[dt_i_free, 0] = bound_circulation[-1]
			
			surrounding_vortices = np.r_[free, trailing]
			from_surrounding, from_bound = self.induction.free_vortex_induction(free_vortices=surrounding_vortices,
			                                                                    bound_vortices=combined_bound,
			                                                                    precision=precision)
			
			from_wake_velocities_x = from_surrounding["x"]@surrounding_circulation[:dt_i_free+1]
			from_wake_velocities_y = from_surrounding["y"]@surrounding_circulation[:dt_i_free+1]
			from_bound_velocities_x = from_bound["x"]@bound_circulation[:-1]
			from_bound_velocities_y = from_bound["y"]@bound_circulation[:-1]
			
			x_vel = from_wake_velocities_x+from_bound_velocities_x
			y_vel = from_wake_velocities_y+from_bound_velocities_y
			background_flow = np.asarray((dt_i_free+1)*[[inflow[0], inflow[1]]])
			self.geometry.displace_vortices(velocities=np.c_[x_vel, y_vel]+background_flow, time_step=dt)
			
			plate_circulation[dt_i] = bound_circulation[:self.plate_res].flatten()
			if self.flap_res != 0:
				flap_circulation[dt_i] = bound_circulation[self.plate_res:-1].flatten()
			if dt_i%20 == 0:
				end_time = timeit.default_timer()
				calc_time = np.round(end_time-start_time, 4)
				print("iteration: ", dt_i-self.n_free_vortices, "calculation time: ", calc_time, "s")
				start_time = timeit.default_timer()
		
		circulations = {"plate": plate_circulation, "flap": flap_circulation, "free": self.free_vortices_circulation}
		return (circulations,
		        {"bound_vortices": all_bound, "control_points": all_cp, "trailing_vortices": all_trailing,
		         "free_vortices" : all_free})
	
	def solve_steady(self,
			         plate_angle: float,
			         inflow: tuple,
			         flap_angle: float = None,
			         precision: np.dtype = np.float32) -> Tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
		self.geometry.update_rotation(plate_angle=plate_angle, flap_angle=flap_angle)
		bound, control, _, _ = self.geometry.get_positions()
		combined_bound = np.r_[bound["plate"], bound["flap"]] if bound["flap"] is not None else bound["plate"]
		lhs = self.induction.lhs_steady_matrix(bound_vortices=combined_bound, plate_control_points=control["plate"],
		                                flap_control_points=control["flap"], precision=precision)
		inv_lhs = np.linalg.inv(lhs)
		plate_normal, flap_normal = self.geometry.get_normals()
		normal_inflow = np.asarray([inflow[0], inflow[1]])@plate_normal*np.ones((self.plate_res, 1))
		if self.flap_res != 0:
			flap_inflow = np.asarray([inflow[0], inflow[1]])@flap_normal
			normal_inflow = np.r_[normal_inflow, flap_inflow*np.ones((self.flap_res, 1))]
		bound_circulation = -inv_lhs@normal_inflow
		return ({"plate": bound_circulation[:self.plate_res], "flap": bound_circulation[self.plate_res:]},
		        {"bound_vortices": bound, "control_points": control})
		
	def plot_final_state(self, show = True, plot_structure: bool = True,
	                     ls_bound: str = "o",
	                     ls_control: str = "x",
	                     ls_trailing: str = "-",
	                     ls_free: str = "x") -> None or tuple:
		return self.geometry.plot_final_state(show, plot_structure, ls_bound, ls_control, ls_trailing, ls_free)
