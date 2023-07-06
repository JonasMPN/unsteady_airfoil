import numpy as np
from geometry import Geometry, plot_process
from induction import Induction
from unsteady_aero import UnsteadyAirfoil
import matplotlib.pyplot as plt
from matplotlib import animation

# --------------------------------------------------- #
#  Switches
# --------------------------------------------------- #

lift_polar = True
velocity_field = True
pressure_field = True

# --------------------------------------------------- #
#  Functions
# --------------------------------------------------- #



# --------------------------------------------------- #
#  Lift coefficient polar
# --------------------------------------------------- #

angles = np.linspace(0, 20, 5)
cl = np.zeros(angles.shape)
cl_analytical = np.zeros(angles.shape)

for i, angle in enumerate(angles):
    plate_res = 100
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
cl_rms = np.sqrt(np.mean(cl-cl_analytical)**2)
cl_rms_round = np.round(cl_rms, 2 - int(np.floor(np.log10(abs(cl_rms)))))



plt.figure(1)
plt.plot(angles, cl_analytical, label="Analytical")
plt.scatter(angles, cl, label="Panel code", marker = "x", color ="black")
plt.text(0.02, 0.95, f'RMS: {cl_rms_round}', verticalalignment='top', horizontalalignment='left',transform=plt.gca().transAxes)
plt.grid()
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$C_l$")
plt.legend(loc="lower right")
plt.savefig("../results/cl_polar.pdf", bbox_inches="tight")
#plt.show()

# --------------------------------------------------- #
#  Velocity plot
# --------------------------------------------------- #

if velocity_field:
    plate_res = 100
    flap_res = 0
    plate_length = 1
    unsteady_airfoil = UnsteadyAirfoil(0, plate_res, plate_length)
    plate_angle = -15
    inflow = (1, 0)
    rho = 1.225
    circulations, positions = unsteady_airfoil.solve_steady(plate_angle, inflow)
    # analytical_sol = inflow[0]*np.pi*plate_length*np.deg2rad(plate_angle)
    # print("analytical solution: ", analytical_sol, ". Model solution: ",  np.sum(circulations["plate"]))
    # print("analytical minus model: ", analytical_sol-np.sum(circulations["plate"]))
    # unsteady_airfoil.plot_final_state()

    circulation_combined = sum(circulations["plate"])

    all_vortices = np.r_[positions["bound_vortices"]["plate"]]

    flow_field_axis_res = 30
    coords = np.linspace(-1.5*plate_length, 1.5*plate_length, flow_field_axis_res)  # calculate coordinates
    X, Y = np.meshgrid(coords, coords)  # use same coordinates for x and y. Get meshgrid
    induction_points = np.vstack([X.ravel(), Y.ravel()]).T  # convert meshgrid into vector of all mesh points

    # pre-allocate memory --- whyyyyyyy
    # why does this have a different shape ???
    pre_allocate = np.empty((2, flow_field_axis_res**2, circulations["plate"].shape[0]), dtype=np.float32)
    # calculate induction matrices in x and y direction
    induction_xy = Induction(plate_res, flap_res).induction_matrices(vortices=all_vortices,
                                                                     induction_points=induction_points,
                                                                     save_to=pre_allocate)
    induction_x = induction_xy[0, :, :]  # induction matrix in x direction
    induction_y = induction_xy[1, :, :]  # induction matrix in y direction
    u_ind = (induction_x@circulations["plate"]).reshape((flow_field_axis_res, flow_field_axis_res))  # induced velocity in x
    v_ind = (induction_y@circulations["plate"]).reshape((flow_field_axis_res, flow_field_axis_res))  # induced velocity in y


    
    plt.figure(2)
    plt.quiver(X, Y, u_ind+inflow[0], v_ind+inflow[1])  # don't forget to add inflow
    plt.xlabel(r"x/c")
    plt.ylabel(r"y/c")
    plt.savefig("../results/steady_aoa15_velocityplot.pdf", bbox_inches="tight")

    # --------------------------------------------------- #
    #  Pressure plot
    # --------------------------------------------------- #
    if pressure_field:
        u_mag = np.sqrt((u_ind+inflow[0])**2 + (v_ind+inflow[1])**2)  # velocity magnitude
        # p = 0.5 * rho * (u_mag**2 - np.linalg.norm(inflow))
        # q_inf = 0.5 * rho * np.linalg.norm(inflow)
        p_over_q = (- u_mag**2 + np.linalg.norm(inflow)) / np.linalg.norm(inflow)

        plt.figure(3)
        h = plt.contourf(X, Y, p_over_q, 100, cmap='plasma')  # gives a nice 90s-ish look
        plt.axis('scaled')
        plt.colorbar(label=r'$p/Q_\infty$')  # plot ratio to stagnation pressure u_inf
        plt.xlabel(r"x/c")
        plt.ylabel(r"y/c")
        plt.savefig("../results/steady_aoa15_pressureplot.pdf", bbox_inches="tight")
plt.show()
