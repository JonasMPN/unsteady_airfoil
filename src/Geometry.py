import matplotlib.pyplot as plt
import numpy as np
from copy import copy
from helper_functions import Helper
helper = Helper()


class Geometry:
    def __init__(self, max_time_steps: int):
        # The following use of 'N' denotes an integer, it does not mean that 'N' is the same value everywhere.
        self.control_points = None  # np.ndarray of size (N, 2) with columns [x, y]
        self.control_points_base = None
        self.bound_vortices = None  # np.ndarray of size (N, 2) with columns [x, y]
        self.bound_vortices_base = None
        self.plate_res = None
        self.plate_elem_length = None
        
        self.control_points_flap = None
        self.control_points_base_flap = None
        self.bound_vortices_flap = None
        self.bound_vortices_base_flap = None
        self.flap_res = None
        self.flap_elem_length = None
        
        self.trailing_vortices = np.zeros((max_time_steps, 2))  # np.ndarray of size (N, 2) with columns [x, y]
        self.use_flap = False
        self.trailing_counter = 0
        
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
        return None
    
    def set_flap(self, flap_length: float, flap_res: int):
        self.flap_res = flap_length
        self.flap_elem_length = flap_length/flap_res
        quarter_distance = self.flap_elem_length/4
        bound_vortices_x = np.asarray([quarter_distance+i*self.flap_elem_length for i in range(flap_res)])
        bound_vortices = np.append([bound_vortices_x], np.zeros((1, flap_res)), axis=0).T
        control_points = np.append([bound_vortices_x+2*quarter_distance], np.zeros((1, flap_res)), axis=0).T
        self.bound_vortices_base_flap = bound_vortices
        self.control_points_base_flap = control_points
        self.use_flap = True
        return None
    
    def get_positions(self):
        return ({"plate": copy(self.bound_vortices), "flap": copy(self.bound_vortices_flap)},
                {"plate": copy(self.control_points), "flap": copy(self.control_points_flap)},
                copy(self.trailing_vortices[:self.trailing_counter, :]))
    
    def update_aoa(self, plate_angle_of_attack: float, flap_angle_of_attack: float = None):
        if self.use_flap and flap_angle_of_attack is None:
            raise ValueError("When using a flap the angle of attack for the flap has to be specified.")
        
        self.bound_vortices = self._rotate(self.bound_vortices_base, -plate_angle_of_attack)
        self.control_points = self._rotate(self.control_points_base, -plate_angle_of_attack)
        
        if self.use_flap:
            self.bound_vortices_flap = self._rotate(self.bound_vortices_base_flap, -flap_angle_of_attack)
            self.control_points_flap = self._rotate(self.control_points_base_flap, -flap_angle_of_attack)
            plate_trailing_edge = self.bound_vortices[0, :]+self.control_points[-1, :]
            self.bound_vortices_flap += plate_trailing_edge
            self.control_points_flap += plate_trailing_edge
        return None
        
    def toggle_flap(self, use_flap: bool):
        self.use_flap = use_flap
        return None
    
    def add_trailing_vortex(self, inflow: tuple, time_step: float, new_trailing_fac: float = 0.5):
        if not self.use_flap:
            trailing_edge = self.bound_vortices[0, :]+self.control_points[-1, :]
        else:
            quarter_flap_element = (self.control_points_flap[0, :] - self.bound_vortices_flap[0, :])/2
            trailing_edge = self.control_points_flap[-1, :]+quarter_flap_element
            
        distance_traveled = np.asarray([inflow[0], inflow[1]])*time_step
        new_trailing_pos = trailing_edge+new_trailing_fac*distance_traveled
        self.trailing_vortices[self.trailing_counter, :] = new_trailing_pos
        self.trailing_counter += 1
        
    def displace_trailing_vortices(self, velocities: tuple, time_step: float) -> None:
        """
        :param velocities:
        :return:
        """
        self.trailing_vortices[:self.trailing_counter, :] += np.asarray(velocities)*time_step
        return None
    
    def plot(self, until_time_step: int = None, show=True):
        fig, ax = plt.subplots()
        until_time_step = until_time_step if until_time_step is not None else self.trailing_counter
        ax.plot(self.bound_vortices[:, 0], self.bound_vortices[:, 1], "x")
        ax.plot(self.control_points[:, 0], self.control_points[:, 1], "x")
        if self.use_flap:
            ax.plot(self.bound_vortices_flap[:, 0], self.bound_vortices_flap[:, 1], "x")
            ax.plot(self.control_points_flap[:, 0], self.control_points_flap[:, 1], "x")
        if self.trailing_counter != 0:
            ax.plot(self.trailing_vortices[:until_time_step, 0], self.trailing_vortices[:until_time_step, 1], "x")
        if show:
            plt.show()
        return fig, ax
    
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
    
