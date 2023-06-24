from geometry import Geometry
from induction import Induction
import numpy as np


class UnsteadyAirfoil:
	def __init__(self, max_time_steps: int,
	             plate_res: int, plate_length: float,
	             flap_res: int = None, flap_length: float = None):
		self.plate_res = plate_res
		self.flap_res = flap_res
		self.n_time_steps = max_time_steps
		self.free_vortices_circulation = np.zeros((0, 1))
		self.use_flap = False
		self.geometry = Geometry(max_time_steps)
		self.induction = Induction(plate_res, flap_res)
		self.n_free_vortices = 0
		
		self.geometry.set_plate(plate_length, plate_res)
		if flap_res is not None:
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
	          shed_trailing_distance: float = 0.5):
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

		old_plate_angle, old_flap_angle, old_inflow = 0, 0, [0, 0]
		normal_inflow = [0, 0]
		for dt_i, (plate_angle, flap_angle, inflow) in enumerate(zip(plate_angles, flap_angles, inflows)):
			dt_i += self.n_free_vortices
			if dt_i == 0 or (old_plate_angle != plate_angle or old_flap_angle != flap_angle or old_inflow != inflow):
				self.geometry.update_rotation(plate_angle=plate_angle, flap_angle=flap_angle)
				self.geometry.shed_vortex(inflow=inflow, time_step=dt, new_trailing_fac=shed_trailing_distance)
				bound, control, trailing, free = self.geometry.get_positions()
				combined_bound = np.r_[bound["plate"], bound["flap"]]
				lhs = self.induction.lhs_matrix(bound_vortices=combined_bound, shed_vortex=trailing[-1, :],
				                                plate_control_points=control["plate"],
				                                flap_control_points=control["flap"])
				inv_lhs = np.linalg.inv(lhs)
				plate_normal, flap_normal = self.geometry.get_normals()
				plate_inflow = np.asarray([inflow[0], inflow[1]])@plate_normal
				flap_inflow = np.asarray([inflow[0], inflow[1]])@flap_normal
				normal_inflow = np.r_[plate_inflow, flap_inflow].reshape(self.plate_res+self.flap_res, 1)
				old_plate_angle, old_flap_angle, old_inflow = plate_angle, flap_angle, inflow
			else:
				self.geometry.shed_vortex(inflow=inflow, time_step=dt, new_trailing_fac=shed_trailing_distance)
				bound, control, trailing, free = self.geometry.get_positions()
			
			cpi = self.induction.control_point_induction(plate_control_points=control["plate"],
			                                             flap_control_points=control["flap"],
			                                             vortices=surrounding_vortices[:dt_i])
			induction_from_surrounding = cpi@surrounding_circulation[:dt_i]
			bound_circulation = -inv_lhs@(np.r_[normal_inflow+induction_from_surrounding,
			                                    [[np.sum(surrounding_circulation[self.n_free_vortices:dt_i]), ]]])
			surrounding_circulation[dt_i, 0] = bound_circulation[-1]

			surrounding_vortices = np.r_[free, trailing]
			from_surrounding, from_bound = self.induction.free_vortex_induction(free_vortices=surrounding_vortices,
			                                                                    bound_vortices=combined_bound)
			
			from_wake_velocities_x = from_surrounding["x"]@surrounding_circulation[:dt_i+1]
			from_wake_velocities_y = from_surrounding["y"]@surrounding_circulation[:dt_i+1]
			from_bound_velocities_x = from_bound["x"]@bound_circulation[:-1]
			from_bound_velocities_y = from_bound["y"]@bound_circulation[:-1]
			
			x_vel = from_wake_velocities_x+from_bound_velocities_x
			y_vel = from_wake_velocities_y+from_bound_velocities_y
			background_flow = np.asarray((dt_i+1)*[[inflow[0], inflow[1]]])
			self.geometry.displace_vortices(velocities=np.c_[x_vel, y_vel]+background_flow, time_step=dt)
			
			if (dt_i-self.n_free_vortices) % 20 == 0:
				print("iteration: ", dt_i-self.n_free_vortices)
		self.geometry.plot()
			
			
			
		
			
			
			
			
			
		
		
		
		
		



		
		
		
