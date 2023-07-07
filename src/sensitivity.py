import numpy as np
from induction import Induction
from unsteady_aero import UnsteadyAirfoil
import matplotlib.pyplot as plt

# --------------------------------------------------- #
#  Switches
# --------------------------------------------------- #

lift_polar = True
velocity_field = False
pressure_field = False


# --------------------------------------------------- #
#  Steady Lift coefficient polar
# --------------------------------------------------- #


angles = np.linspace(0, 20, 5)
resolutions = [1, 10, 100]
markers = ["x", "o", "+", "v", "^"]
cl_list = []
cl = np.zeros(angles.shape)
cl_analytical = np.zeros(angles.shape)
for plate_res in resolutions:
    for i, angle in enumerate(angles):
        #plate_res = 100
        plate_length = 1
        unsteady_airfoil = UnsteadyAirfoil(0, plate_res, plate_length)
        #plate_angle = -10
        inflow = (1, 0)
        circulations, positions = unsteady_airfoil.solve_steady(angle, inflow)
        # analytical_sol = inflow[0]*np.pi*plate_length*np.deg2rad(plate_angle)
        # print("analytical solution: ", analytical_sol, ". Model solution: ",  np.sum(circulations["plate"]))
        # print("analytical minus model: ", analytical_sol-np.sum(circulations["plate"]))
        # unsteady_airfoil.plot_final_state()

        circulation_combined = sum(circulations["plate"])
        # cl = rhu * u * circulation_combined / (0.5 * rho * u **2 )

        cl[i] = circulation_combined / (0.5 * inflow[0] * plate_length)
        cl_analytical[i] = 2 * np.pi * np.sin(np.deg2rad(angle))
    #breakpoint() 
    cl_list.append(cl)
    cl_rms = np.sqrt(np.mean(cl-cl_analytical)**2)
    cl_rms_round = np.round(cl_rms, 2 - int(np.floor(np.log10(abs(cl_rms)))))


plt.figure(1)
for i, cl in enumerate(cl_list):
    plt.scatter(angles, cl, label=f"Panel code - n={resolutions[i]}", marker=markers[i])
plt.plot(angles, cl_analytical, label="Analytical")

#plt.text(0.02, 0.95, f'RMS: {cl_rms_round}', verticalalignment='top', horizontalalignment='left',transform=plt.gca().transAxes)

# make it look nice
plt.grid()
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$C_l$")
plt.legend(loc="lower right")
plt.savefig("../results/cl_polar_sensitivity.pdf", bbox_inches="tight")
#plt.show()

# --------------------------------------------------- #
#  Steady Lift coefficient polar
# --------------------------------------------------- #

def unsteady_sinusoidal(k, A, dt=0.2, plate_res=5, time_steps = 1000):
    """
    Compute the unsteady airfoil pitching motion at a reduced frequency k with amplitude A
    A : amplitude in degrees
    k: reduced freq
    """
    # -----------------------------_#
    # Parameters
    # -----------------------------_#

    #time_steps = 3000
    # dt = 0.2
    time_array = np.arange(0, dt*time_steps, dt)

    # plate_res = 5
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
    if False:
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
    return cl, plate_angles, cl_analytical


# -----------------------------------------------#
# Variation of the panel density
# -----------------------------------------------#
panel_density = False
if panel_density:
    linestyles = ["--", "-.", ":"]
    A = 10  # max angle
    k = 0.1
    dt = 0.2

    cl_unsteady = []
    plt.figure(3)
    for i, plate_res in enumerate(resolutions):
        cl, plate_angles, cl_analytical = unsteady_sinusoidal(k, A, dt=dt, plate_res=plate_res)
        cl_unsteady.append(cl)
        plt.plot(plate_angles, cl, label=f"unsteady n:{plate_res}", linewidth=1.5, linestyle=linestyles[i])

    plt.plot(np.linspace(-A, A, 20), cl_analytical, "--r", label="steady")
    plt.arrow(plate_angles[0], cl[0], plate_angles[5], cl[5], color="C0",  # not so nice but it works ...
            shape="full", lw=2, length_includes_head=True, head_width=.1)
    # , shape='full', lw=5,
    #   length_includes_head=True, head_width=.02)
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$C_l$")
    plt.legend()
    plt.grid()
    plt.savefig(f"../results/sensitivity_unsteady_k{k}_cl_panel_density.pdf", bbox_inches="tight")


# -----------------------------------------------#
# Variation of the time step size
# -----------------------------------------------#


cl_unsteady_dt = []
plt.figure(4)
A = 10  # max angle
linestyles = ["--", "-.", ":"]
k = 0.1
plate_res = 10  # should be sufficient
dt_steps = [np.pi *1/2, np.pi *1/20, np.pi * 1/100]
time_steps = [1000, 1500, 3000]
for i, dt in enumerate(dt_steps):
    cl, plate_angles, cl_analytical = unsteady_sinusoidal(k, A, dt=dt, plate_res=plate_res, time_steps=time_steps[i])
    cl_unsteady_dt.append(cl)
    plt.plot(plate_angles, cl, label=f"unsteady dt={np.around(dt, 4)}", linewidth=1.5, linestyle=linestyles[i])

plt.plot(np.linspace(-A, A, 20), cl_analytical, "--r", label="steady")
plt.arrow(plate_angles[0], cl[0], plate_angles[5], cl[5], color="C0",  # not so nice but it works ...
          shape="full", lw=2, length_includes_head=True, head_width=.1)
# , shape='full', lw=5,
#   length_includes_head=True, head_width=.02)
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$C_l$")
plt.legend()
plt.grid()
plt.savefig(f"../results/sensitivity_unsteady_k{k}_cl_timestep.pdf", bbox_inches="tight")
plt.show()




