import numpy as np
import matplotlib.pyplot as plt
from geometry import Geometry, plot_process
from induction import Induction
from unsteady_aero import UnsteadyAirfoil
from matplotlib import ticker, cm


test = {
	"unsteady_sinusoidal": False,
	"step_response": False,
	"gust_response": True,
}

def unsteady_sinusoidal(k, A):
    """
    Compute the unsteady airfoil pitching motion at a reduced frequency k with amplitude A
    A : amplitude in degrees
    k: reduced freq
    """
    # -----------------------------_#
    # Parameters
    # -----------------------------_#

    time_steps = 1000
    dt = 0.2
    time_array = np.arange(0, dt*time_steps, dt)

    plate_res = 5
    flap_res = 0
    plate_length = 1
    flap_length = 0
    
    inflow = (1, 0)
    omega = k * 2 * np.linalg.norm(inflow)/ (plate_length)
    
    # -----------------------------_#
    # Unsteady computation
    # -----------------------------_#
     
    unsteady_airfoil = UnsteadyAirfoil(time_steps, plate_res, plate_length, flap_res, flap_length)
    plate_angles = A * np.sin(omega * time_array)     # np.linspace(0, t_sim, dt))
    
    # plate_angles = -10*np.sin(5*np.linspace(0, 2*np.pi, time_steps))


    # do normal calculation
    circulation, positions = unsteady_airfoil.solve(dt=dt, plate_angles=plate_angles, inflows=inflow)
    # combine all calculated circulations into one row array
    
    # -----------------------------_#
    # Postprocessing
    # -----------------------------_#

    # This is where the time step is selected
    all_circulations = np.r_[circulation["plate"][time_steps-1], circulation["flap"][time_steps-1],
                             circulation["trailing"].flatten(), circulation["free"].flatten()]
    
    # convert row array into column array
    all_circulations = all_circulations.reshape((all_circulations.shape[0], 1))
    
    # combine all vortices into an array
    all_vortices = np.r_[positions["plate"], positions["trailing"], positions["free"]]
    
    # ---------------------------------------------------------- #
    # Plotting
    # ---------------------------------------------------------- #

    # resolution per axis of the flow field (equi-distant and same for all axes)
    flow_field_axis_res = 90
    x_flow_field_axis_res = 120
    y_flow_field_axis_res = 90
    coords = np.linspace(-1.5*plate_length, 1.5*plate_length, flow_field_axis_res)  # calculate coordinates
    
    x_coords = np.linspace(-1.5*plate_length, 5*plate_length, x_flow_field_axis_res)  # calculate coordinates
    y_coords = np.linspace(-1.5*plate_length, 1.5*plate_length, y_flow_field_axis_res)  # calculate coordinates
    X, Y = np.meshgrid(x_coords, y_coords)  # use same coordinates for x and y. Get meshgrid
    induction_points = np.vstack([X.ravel(), Y.ravel()]).T  # convert meshgrid into vector of all mesh points
    
    # pre-allocate memory
    # pre_allocate = np.empty((2, flow_field_axis_res**2, all_circulations.shape[0]), dtype=np.float32)
    pre_allocate = np.empty((2, x_flow_field_axis_res* y_flow_field_axis_res, all_circulations.shape[0]), dtype=np.float32)
    
    # calculate induction matrices in x and y direction
    induction_xy = Induction(plate_res, flap_res).induction_matrices(vortices=all_vortices,
                                                                     induction_points=induction_points,
                                                                     save_to=pre_allocate)

    induction_x = induction_xy[0, :, :]  # induction matrix in x direction
    induction_y = induction_xy[1, :, :]  # induction matrix in y direction
    # u_ind = (induction_x@all_circulations).reshape((flow_field_axis_res, flow_field_axis_res))  # induced velocity in x
    # v_ind = (induction_y@all_circulations).reshape((flow_field_axis_res, flow_field_axis_res))  # induced velocity in y
    
    u_ind = (induction_x@all_circulations).reshape((y_flow_field_axis_res, x_flow_field_axis_res))  # induced velocity in x
    v_ind = (induction_y@all_circulations).reshape((y_flow_field_axis_res, x_flow_field_axis_res))  # induced velocity in y
    plt.figure(1)
    levels = np.linspace(-0.1, 0.1, 10)
    plt.contourf(X, Y, u_ind, levels=levels)
    plt.colorbar()
    plt.figure(2)
    plt.quiver(X, Y, u_ind+inflow[0], v_ind+inflow[1])  # don't forget to add inflow

    # ------------------------------------------------ #
    # Lift coefficient plot
    # ------------------------------------------------ #
    
    circulation_combined = np.sum(circulation["plate"], 1)  # sum of all circulations, for each individual timestep
    # cl = rhu * u * circulation_combined / (0.5 * rho * u **2 )

    cl = circulation_combined / (0.5 * np.linalg.norm(inflow) * plate_length)
    cl_analytical = 2 * np.pi * np.sin(np.deg2rad(np.linspace(-A, A, 20)))
    # cl_analytical[i] = 2 * np.pi * np.sin(np.deg2rad(angle))

    plt.figure(3)
    plt.plot(plate_angles, cl, label="unsteady")
    plt.plot(np.linspace(-A, A, 20), cl_analytical, "--r", label="steady")
    plt.arrow(plate_angles[0], cl[0], plate_angles[5], cl[5], color="C0",  # not so nice but it works ...
              shape="full", lw=2, length_includes_head=True, head_width=.1)
    # , shape='full', lw=5,
    #   length_includes_head=True, head_width=.02)
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$C_l$")
    plt.legend()
    plt.grid()
    plt.savefig(f"../results/unsteady_k{k}_cl.pdf", bbox_inches="tight")
    
    plt.show()

def step_response(A):
    
    
    # -----------------------------_#
    # Parameters
    # -----------------------------_#

    time_steps = 50
    dt = 4
    time_array = np.arange(0, dt*time_steps, dt)
    plate_res = 5
    flap_res = 0
    plate_length = 10
    flap_length = 0
    inflow = (1, 0)
    # omega = k * 2 * np.linalg.norm(inflow)/ (plate_length)
    
    # -----------------------------_#
    # Unsteady computation
    # -----------------------------_#
     
    unsteady_airfoil = UnsteadyAirfoil(time_steps, plate_res, plate_length, flap_res, flap_length)
    plate_angles = np.zeros(len(time_array))
    plate_angles[:] = A
    circulation, positions = unsteady_airfoil.solve(dt=dt, plate_angles=plate_angles, inflows=inflow)
    
    circulation_combined = np.sum(circulation["plate"], 1)  # sum of all circulations, for each individual timestep
    cl = circulation_combined / (0.5 * np.linalg.norm(inflow) * plate_length)
    cl_analytical = 2 * np.pi * np.sin(np.deg2rad(A))
    
    s = (np.linalg.norm(inflow)*2/plate_length)*time_array
    
    C1, C2, eps1, eps2 = 0.165, 0.335, 0.0455, 0.3
    cl_wagner = (1-C1*np.exp(-eps1*s)-C2*np.exp(-eps2*s))*cl_analytical
    
    cl_kussner = (1-0.5*(np.exp(-0.13*s)+np.exp(-s)))*cl_analytical
    
    # print(cl/cl_analytical)
    plt.plot(s, cl/cl_analytical, label = "Panel code", color = "blue", ls = "dashed")
    # plt.plot(time_array, cl_kussner/cl_analytical, label = "Küssner")
    plt.plot(s, cl_wagner/cl_analytical, label = "Wagner", color = "red", ls = "solid")
    plt.xlabel(r"$s = \dfrac{U_\infty t}{b}$")
    plt.ylabel(r"$C_L/C_{L_{st}}$")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    
    
def gust_response():
        
    
    # -----------------------------_#
    # Parameters
    # -----------------------------_#

    time_steps = 200
    dt = 2
    time_array = np.arange(0, dt*time_steps, dt)
    plate_res = 5
    flap_res = 0
    plate_length = 10
    flap_length = 0
    A = 0

    inflow = [(1, 0) if i < len(time_array)//2 else (1, 0.01) for i in range(len(time_array))]
    A_rel = np.zeros(len(time_array)) # airfoil angle relative to flow
    for i, u in enumerate(inflow):
        A_rel[i] = np.rad2deg(np.arctan(u[1]/u[0]))+A
        
    inflow = (1, 0)
    # -----------------------------_#
    # Unsteady computation
    # -----------------------------_#
    s = (np.linalg.norm(inflow)*2/plate_length)*time_array
    cl_analytical = 2 * np.pi * np.sin(np.deg2rad(A_rel))
    
    
    unsteady_airfoil = UnsteadyAirfoil(time_steps, plate_res, plate_length, flap_res, flap_length)
    plate_angles = np.ones(len(time_array))*A_rel
    circulation, positions = unsteady_airfoil.solve(dt=dt, plate_angles=plate_angles, inflows=inflow)
    
    circulation_combined = np.sum(circulation["plate"], 1)  # sum of all circulations, for each individual timestep
    cl = circulation_combined / (0.5 * np.linalg.norm(inflow) * plate_length)


    
    C1, C2, eps1, eps2 = 0.165, 0.335, 0.0455, 0.3
    cl_wagner = (1-C1*np.exp(-eps1*s)-C2*np.exp(-eps2*s))
    cl_kussner = (1-0.5*(np.exp(-0.13*s)+np.exp(-s)))
    
    plt.plot(s, cl/cl_analytical, label = "Panel code", color = "blue", ls = "dashed")
    plt.plot(s, cl_kussner, label = "Küssner")
    plt.plot(s, cl_wagner, label = "Wagner", color = "red", ls = "solid")
    plt.xlabel(r"$s = \dfrac{U_\infty t}{b}$")
    plt.ylabel(r"$C_L/C_{L_{st}}$")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    


if __name__=="__main__":
    if test["unsteady_sinusoidal"]:
        A = 10  # max angle 
        k_vals = [0.02, 0.05, 0.1]  # different reduced frequencies
        k = k_vals[2]
        unsteady_sinusoidal(k, A)
    if test["step_response"]:
        A = 1  # angle
        step_response(A)
    if test[	"gust_response"]:
        gust_response()

