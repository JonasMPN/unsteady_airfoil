import matplotlib.animation as animation
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import numpy as np
from copy import copy
from helper_functions import Helper
helper = Helper()


class Geometry:
    def __init__(self, max_time_steps: int):
        # The following use of 'N' denotes an integer, it does not mean that 'N' is the same value everywhere.
        # The "base" coordinates have 0 rotation and are used as a basis to calculate the rotated coordinates.
        self.control_points = None  # np.ndarray of size (N, 2) with columns [x, y].
        self.control_points_base = None  #
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
        self.free_vortices = np.zeros((0, 2))
        self.use_flap = False
        self.trailing_counter = 0
        
    def set_plate(self, plate_length: float, plate_res: int) -> None:
        """
        Set plate properties. Each plate element will have the same length with a bound vortex positioned at the
        element's quarter chord and a control point positioned at the plate's 3-quarter chord.
        
        :param plate_length: length of the plate
        :param plate_res: number of plate elements
        :return: None
        """
        self.plate_res = plate_res
        self.plate_elem_length = plate_length/plate_res
        quarter_distance = self.plate_elem_length/4
        bound_vortices_x = np.asarray([quarter_distance+i*self.plate_elem_length for i in range(plate_res)])
        bound_vortices = np.c_[bound_vortices_x, np.zeros(plate_res)]
        control_points = np.c_[bound_vortices_x+2*quarter_distance, np.zeros(plate_res)]
        self.bound_vortices_base = bound_vortices  # These "base" values will be rotated to get the actual coordinates
        self.control_points_base = control_points  # when the plate is at an angle of attack
        return None
    
    def set_flap(self, flap_length: float, flap_res: int) -> None:
        """
        Set flap properties. Each plate element will have the same length with a bound vortex positioned at the
        element's quarter chord and a control point positioned at the plate's 3-quarter chord.

        :param flap_length: length of the flap
        :param flap_res: number of plate elements
        :return: None
        """
        self.flap_res = flap_length
        self.flap_elem_length = flap_length/flap_res
        quarter_distance = self.flap_elem_length/4
        bound_vortices_x = np.asarray([quarter_distance+i*self.flap_elem_length for i in range(flap_res)])
        bound_vortices = np.c_[bound_vortices_x, np.zeros(flap_res)]
        control_points = np.c_[bound_vortices_x+2*quarter_distance, np.zeros(flap_res)]
        self.bound_vortices_base_flap = bound_vortices
        self.control_points_base_flap = control_points
        # These "base" values will be rotated to get the actual coordinates when the plate is at an angle of attack
        self.use_flap = True
        return None
    
    def add_free_vortices(self, coordinates: np.ndarray) -> None:
        """
        Sets the initial coordinates of free vortices.
        
        :param coordinates: np.ndarray of size (N,2) with (x,y) columns.
        :return: None
        """
        self.free_vortices = coordinates
        return None
    
    def get_positions(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """
        Returns the coordinates of (bound vortices, control points, trailing vortices, free vortices) as a tuple in
        this order. The bound and trailing vortices will come in a dict that differentiates them between plate and
        flap. Hence, to get the control points of the flap:
        
        bound_vortices, control_points, trailing_vortices, free_vortices = .get_positions()
        
        flap_cp = control_points["flap"]
        
        :return: tuple of two dicts and a np.ndarray. Dicts have str as keys and np.ndarrays as values
        """
        return ({"plate": copy(self.bound_vortices), "flap": copy(self.bound_vortices_flap)},
                {"plate": copy(self.control_points), "flap": copy(self.control_points_flap)},
                copy(self.trailing_vortices[:self.trailing_counter, :]),
                copy(self.free_vortices))
    
    def get_normals(self):
        return (self._unit_normal_and_length(self.control_points[0, :])[0],
               self._flap_normal(self.control_points, self.control_points_flap))
    
    def update_rotation(self, plate_angle: float, flap_angle: float = None) -> None:
        """
        This function needs to be called to build the plate or plate-flap structure. The function updates the
        rotation of the plate and the flap around the axis coming out of the plane. The final flap angle is the sum
        of the plate and flap angle. The angles are only equal to the angle of attack for an inflow parallel to the
        x-axis.
        
        :param plate_angle: Rotation in degrees for the plate.
        :param flap_angle: Rotation in degrees for the flap.
        :return: None
        """
        if self.use_flap and flap_angle is None:
            raise ValueError("When using a flap the angle of attack for the flap has to be specified.")
        plate_angle = np.deg2rad(plate_angle)
        # The actual coordinates of the plate are the rotated base values. The leading edge is connected to the
        # coordinate systems' origin.
        self.bound_vortices = self._rotate(self.bound_vortices_base, plate_angle)
        self.control_points = self._rotate(self.control_points_base, plate_angle)
        
        if self.use_flap:
            flap_angle = np.deg2rad(np.rad2deg(plate_angle)+flap_angle)
            self.bound_vortices_flap = self._rotate(self.bound_vortices_base_flap, flap_angle)
            self.control_points_flap = self._rotate(self.control_points_base_flap, flap_angle)
            plate_trailing_edge = self.bound_vortices[0, :]+self.control_points[-1, :]
            # The "leading edge" of the flap is connected to the trailing edge of the plate.
            self.bound_vortices_flap += plate_trailing_edge
            self.control_points_flap += plate_trailing_edge
        return None
        
    def toggle_flap(self, use_flap: bool) -> None:
        """
        Define whether a flap is being used or not
        
        :param use_flap: Is there a flap or is there not.
        :return:None
        """
        self.use_flap = use_flap
        return None
    
    def shed_vortex(self, inflow: tuple or list[list], time_step: float, new_trailing_fac: float = 0.5):
        """
        Adds a newly shed vortex to the trailing vortices. The location of the shed vortex is dependent on the
        inflow, the time step and "new_trailing_fac". It is assumed that the trailing edge moved in a straight line
        opposite to the inflow for the time "time_step". The shed vortex will be placed along this line.
        "new_trailing_fac" determines at which position along that line: 0 is directly at the trailing edge,
        1 is where the trailing edge was before the time step.
        
        :param inflow: (x_velocity, y_velocity)
        :param time_step: float of time step duration
        :param new_trailing_fac: float between 0 and 1
        :return:
        """
        if new_trailing_fac < 0 or new_trailing_fac > 1:
            raise ValueError("'new_trailing_fac' must be in the interval [0; 1].")
        if not self.use_flap:  # then the trailing edge is at the end of the plate
            trailing_edge = self.bound_vortices[0, :]+self.control_points[-1, :]
        else:  # then the trailing edge is at the end of the flap
            quarter_flap_element = (self.control_points_flap[0, :] - self.bound_vortices_flap[0, :])/2
            trailing_edge = self.control_points_flap[-1, :]+quarter_flap_element
            
        distance_traveled = np.asarray([inflow[0], inflow[1]])*time_step
        new_trailing_pos = trailing_edge+new_trailing_fac*distance_traveled
        self.trailing_vortices[self.trailing_counter, :] = new_trailing_pos  # new vortices are appended to the list
        self.trailing_counter += 1  # keep track how many vortices have been shed.
        
    def displace_vortices(self, velocities: np.ndarray, time_step: float) -> None:
        """
        Displaces all (trailing and free) vortices based on the induced velocities on them and a time step duration.
        
        :param velocities: np.ndarray of size (number of trailing vortices + number of free vortices, 2). The columns
        are (u, v) with u velocity along x, v velocity along y. The induced velocities at the free vortices need to
        be appended to the induced velocities at the trailing vortices.
        :param time_step: float, time duration of the time step
        :return:
        """
        self.trailing_vortices[:self.trailing_counter, :] += velocities[:self.trailing_counter]*time_step
        if self.free_vortices.shape[0] > 0:
            self.free_vortices += velocities[self.trailing_counter:, :]*time_step
        return None
    
    def plot_final_state(self,
                         show=True,
                         plot_structure: bool = True,
                         ls_bound: str = "o",
                         ls_control: str = "x",
                         ls_trailing: str = "-",
                         ls_free: str = "x") -> None or tuple:
        """
        Plots the bound vortices, the control points, the trailing vortices, and the free vortices (if those are
        present).
        
        :param show: show the plot immediately. If false, returns the figure and axis instead.
        :param plot_structure: Plot the structure of the plate & flap as a black line
        :param ls_bound: Line style for the bound vortices
        :param ls_control: Line style for the control points
        :param ls_trailing: Line style for the trailing vortices
        :param ls_free: Line style for the free vortices
        :return:
        """
        fig, ax = plt.subplots()
        ax.plot(self.bound_vortices[:, 0], self.bound_vortices[:, 1], ls_bound)
        ax.plot(self.control_points[:, 0], self.control_points[:, 1], ls_control)
        if plot_structure:
            plate_trailing_edge = self.control_points[-1, :]+self.bound_vortices[0, :]
            coordinates_to_plot = np.asarray([[0, 0], plate_trailing_edge])
            if self.use_flap:
                flap_qc = (self.control_points_flap[0, :]-self.bound_vortices_flap[0, :])/2
                flap_trailing_edge = self.control_points_flap[-1, :]+flap_qc
                coordinates_to_plot = np.r_[coordinates_to_plot, [flap_trailing_edge]]
            ax.plot(coordinates_to_plot[:, 0], coordinates_to_plot[:, 1], "k")
        if self.use_flap:
            ax.plot(self.bound_vortices_flap[:, 0], self.bound_vortices_flap[:, 1], ls_bound)
            ax.plot(self.control_points_flap[:, 0], self.control_points_flap[:, 1], ls_control)
        if self.trailing_counter != 0:
            ax.plot(self.trailing_vortices[:self.trailing_counter, 0],
                    self.trailing_vortices[:self.trailing_counter, 1], ls_trailing)
        if self.free_vortices.shape[0] != 0:
            ax.plot(self.free_vortices[:, 0], self.free_vortices[:, 1], ls_free)
        if show:
            plt.show()
            return None
        else:
            return fig, ax

    def _flap_normal(self, plate_control_points: np.ndarray, flap_control_points: np.ndarray):
        if self.flap_res is None:
            return None
        if self.flap_res > 1:
            return self._unit_normal_and_length(flap_control_points[1, :]-flap_control_points[0, :])
        else:
            vec_from = plate_control_points[-1, :]+plate_control_points[0, :]/3
            return self._unit_normal_and_length(flap_control_points[0, :]-vec_from)[0]

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
        return np.r_[-normalised[1], normalised[0]], vector_length


def plot_process(bound_vortices: np.ndarray, control_points: np.ndarray,
                 trailing_vortices: np.ndarray, free_vortices: np.ndarray,
                 plate_res: int=0, flap_res: int=0,
                 ls_bound: str = "o", ls_control: str = "x", ls_trailing: str = "-", ls_free: str = "x",
                 show: bool = True, size: tuple=(15,5), dpi: int=250) -> None or animation.Animation:
    fig, ax = plt.subplots()
    fig.set_size_inches(size)
    fig.set_dpi(dpi)
    n_free_vortices = free_vortices.size
    min_x_free = 0 if n_free_vortices == 0 else np.min(free_vortices[:, :, 0])
    max_x_free = 0 if n_free_vortices == 0 else np.max(free_vortices[:, :, 0])
    min_y_free = 0 if n_free_vortices == 0 else np.min(free_vortices[:, :, 1])
    max_y_free = 0 if n_free_vortices == 0 else np.max(free_vortices[:, :, 1])

    min_x_trailing, max_x_trailing, min_y_trailing, max_y_trailing = 0, 0, 0, 0
    for i in range(trailing_vortices.shape[0]):
        min_x_trailing = np.min(np.r_[min_x_free, trailing_vortices[i][:, 0]])
        max_x_trailing = np.max(np.r_[max_x_free, trailing_vortices[i][:, 0]])
        min_y_trailing = np.min(np.r_[min_y_free, trailing_vortices[i][:, 1]])
        max_y_trailing = np.max(np.r_[max_y_free, trailing_vortices[i][:, 1]])

    min_x = 1.1*min(-0.1, min_x_free, min_x_trailing)
    max_x = 1.1*max(max_x_free, max_x_trailing)

    min_y = 1.1*min(min_y_free, min_y_trailing, np.min(bound_vortices[:, :, 1]))
    max_y = 1.1*max(max_y_free, max_y_trailing, np.max(bound_vortices[:, :, 1]))

    ax.set(xlim=[min_x, max_x], ylim=[min_y, max_y])
    line_bound, = ax.plot(bound_vortices[0, :, 0], bound_vortices[0, :, 1], ls_bound)
    line_cp, = ax.plot(control_points[0, :, 0], control_points[0, :, 1], ls_control)
    line_trailing, = ax.plot(trailing_vortices[0][:, 0], trailing_vortices[0][:, 1], ls_trailing)
    line_free, = ax.plot(free_vortices[0, :, 0], free_vortices[0, :, 1], ls_free)
    title = ax.text(0.5, 0, "0")

    def update(frame: int):
        if len(trailing_vortices[frame][:, 0]) > 5:
            tck, u = splprep([trailing_vortices[frame][:, 0], trailing_vortices[frame][:, 1]], k=4, s=0)
            u_new = np.linspace(0, 1, num=10000)
            x_spline, y_spline = splev(u_new, tck)
            line_trailing.set_data(x_spline, y_spline)
        else:
            line_trailing.set_data(trailing_vortices[frame][:, 0], trailing_vortices[frame][:, 1])
        line_bound.set_data(bound_vortices[frame, :, 0], bound_vortices[frame, :, 1])
        line_cp.set_data(control_points[frame, :, 0], control_points[frame, :, 1])
        line_free.set_data(free_vortices[frame, :, 0], free_vortices[frame, :, 1])
        title.set_text(frame)
        return line_bound, line_cp, line_trailing, line_free, title

    ani = animation.FuncAnimation(fig=fig, func=update, frames=bound_vortices.shape[0], interval=30, blit=True)
    if show:
        plt.show()
    else:
        return ani
