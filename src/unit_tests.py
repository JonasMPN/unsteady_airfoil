import numpy as np
from geometry import Geometry, plot_process
from induction import Induction
from unsteady_aero import UnsteadyAirfoil
import matplotlib.pyplot as plt
from matplotlib import animation

test = {
	"displacing": False,
	"cp_induction": False,
	"free_vortex_induction": False,
	"lhs_matrix": False,
	"steady_solution": True,
	"without_free": False,
	"with_free": False,
	"velocity_field": False,
}

if test["displacing"]:
	time_steps = 10
	dt = 0.1
	geom = Geometry(time_steps)
	geom.set_plate(1, 5)
	geom.set_flap(0.5, 3)
	geom.update_rotation(-20, -40)
	geom.add_free_vortices(np.asarray([[0, 0], [-0.1, -0.1]]))
	for i in range(time_steps):
		geom.shed_vortex((1, 0), dt)
		if i == time_steps-1:
			continue
		geom.displace_vortices(np.c_[np.ones(i+3), np.zeros(i+3)], dt)
	geom.plot_final_state()

if test["cp_induction"]:
	time_steps = 10
	dt = 0.1
	plate_res = 1
	flap_res = 1
	geom = Geometry(time_steps)
	geom.set_plate(1, plate_res)
	geom.set_flap(0.5, flap_res)
	geom.update_rotation(0, 0)
	geom.shed_vortex((1, 0), time_step=0.5)
	geom.add_free_vortices(np.asarray([[3, 1]]))

	ind = Induction(plate_res, flap_res)

	bound, control, trailing, free = geom.get_positions()
	print("Inversely calculated distances from the free vortices to the control points:")
	print(1/(2*np.pi*ind.control_point_induction(plate_control_points=control["plate"],
												 vortices=np.r_[trailing, free],
												 flap_control_points=control["flap"])))
	geom.plot_final_state(ls_trailing="x")

if test["free_vortex_induction"]:
	time_steps = 10
	dt = 0.1
	plate_res = 1
	flap_res = 1
	geom = Geometry(time_steps)
	geom.set_plate(1, plate_res)
	geom.set_flap(1, flap_res)
	geom.update_rotation(0, 0)
	geom.shed_vortex((1, 0), time_step=0.5)
	geom.displace_vortices(np.asarray([[1, 0]]), time_step=0.5, )
	geom.shed_vortex((1, 0), time_step=0.5)
	geom.add_free_vortices(np.asarray([[2., 0.]]))

	ind = Induction(plate_res, flap_res)

	bound, control, trailing, free = geom.get_positions()
	bound = np.r_[bound["plate"], bound["flap"]]

	cpi = ind.control_point_induction(control["plate"], np.r_[free, trailing], control["flap"])
	print(1/(2*np.pi*cpi))

	from_trailing, from_bound = ind.free_vortex_induction(np.r_[free, trailing], bound)

	print("x from free vortices onto each other")
	print(from_trailing["x"])
	print("y")
	print(from_trailing["y"])

	print("x from the bound vortices to the free vortices:")
	print(from_bound["x"])
	print("y")
	print(from_bound["y"])
	geom.plot_final_state()
	
if test["steady_solution"]:
	plate_res = 1
	plate_length = 1
	unsteady_airfoil = UnsteadyAirfoil(0, plate_res, plate_length)
	plate_angle = -10
	inflow = (1, 0)
	circulations, positions = unsteady_airfoil.solve_steady(plate_angle, inflow)
	analytical_sol = inflow[0]*np.pi*plate_length*np.deg2rad(plate_angle)
	print("analytical solution: ", analytical_sol, ". Model solution: ",  np.sum(circulations["plate"]))
	print("analytical minus model: ", analytical_sol-np.sum(circulations["plate"]))
	unsteady_airfoil.plot_final_state()

if test["lhs_matrix"]:
	time_steps = 10
	dt = 0.05
	plate_res = 1
	flap_res = 1
	geom = Geometry(time_steps)
	geom.set_plate(1, plate_res)
	geom.set_flap(1, flap_res)
	geom.update_rotation(0, 0)
	geom.shed_vortex((1, 0), time_step=0.5)

	ind = Induction(plate_res, flap_res)
	bound, control, trailing, free = geom.get_positions()
	bound = np.r_[bound["plate"], bound["flap"]]
	print(ind.lhs_matrix(bound, trailing[-1], control["plate"], control["flap"]))
	geom.plot_final_state(ls_trailing="x")

if test["without_free"]:
	time_steps = 500
	dt = 0.1
	plate_res = 5
	flap_res = 3
	plate_length = 1
	flap_length = 1
	unsteady_airfoil = UnsteadyAirfoil(time_steps, plate_res, plate_length, flap_res, flap_length)
	plate_angles = np.zeros(time_steps)
	plate_angles[10:15] = -np.linspace(0, 10, 5)
	plate_angles[15:200] = -10
	plate_angles[200:205] = -np.linspace(10, 0, 5)
	plate_angles[300:330] = 15*np.sin(np.linspace(0, 2*np.pi, 30))
	plate_angles[390:400] = -np.linspace(0, 30, 10)
	plate_angles[400:] = -30
	# plate_angles = -10*np.sin(5*np.linspace(0, 2*np.pi, time_steps))
	flap_angles = np.zeros(time_steps)
	flap_angles[100:107] = -np.linspace(0, 20, 7)
	flap_angles[107:200] = -20
	flap_angles[200:205] = -np.linspace(20, 0, 5)
	# flap_angles = -10*np.sin(5*np.linspace(0, 2*np.pi, time_steps))
	inflow = (1, 0)
	circulation, coordinates = unsteady_airfoil.solve_for_process(dt=dt, plate_angles=plate_angles, inflows=inflow,
													  			  flap_angles=flap_angles)
	# unsteady_airfoil.plot_final_state()
	ani = plot_process(**coordinates, show=False)
	if False:
		ani.save("../results/unit_tests/without_free.mp4", writer=animation.FFMpegWriter(fps=30))


if test["with_free"]:
	time_steps = 200
	dt = 0.08
	plate_res = 1
	flap_res = 1
	plate_length = 1
	flap_length = 1
	unsteady_airfoil = UnsteadyAirfoil(time_steps, plate_res, plate_length, flap_res, flap_length)
	# unsteady_airfoil.add_free_vortices(np.asarray([[-0., 3.]]), 0.)
	plate_angles = np.zeros(time_steps)
	plate_angles[:] = -10
	flap_angles = np.zeros(time_steps)
	# flap_angles[50:] = -10
	inflow = (1, 0)
	circulation, coordinates = unsteady_airfoil.solve_for_process(dt=dt, plate_angles=plate_angles, inflows=inflow,
													  			  flap_angles=flap_angles)
	unsteady_airfoil.plot_final_state()
	plot_process(**coordinates)

if test["velocity_field"]:
	time_steps = 200
	dt = 0.1
	plate_res = 5
	flap_res = 0
	plate_length = 1
	flap_length = 1
	unsteady_airfoil = UnsteadyAirfoil(time_steps, plate_res, plate_length, flap_res, flap_length)
	plate_angles = np.zeros(time_steps)
	plate_angles[10:] = -45
	# plate_angles = -10*np.sin(5*np.linspace(0, 2*np.pi, time_steps))
	flap_angles = np.zeros(time_steps)
	flap_angles[50:] = -40
	# flap_angles = -10*np.sin(5*np.linspace(0, 2*np.pi, time_steps))
	inflow = (1, 0)
	
	# do normal calculation
	circulation, positions = unsteady_airfoil.solve(dt=dt, plate_angles=plate_angles, inflows=inflow,
	                                                flap_angles=flap_angles)
	# combine all calculated circulations into one row array
	all_circulations = np.r_[circulation["plate"][time_steps-1], circulation["flap"][time_steps-1],
	                         circulation["trailing"].flatten(), circulation["free"].flatten()]
	# convert row array into column array
	all_circulations = all_circulations.reshape((all_circulations.shape[0], 1))
	# combine all vortices into an array
	if flap_res != 0:
		all_vortices = np.r_[positions["plate"], positions["flap"], positions["trailing"], positions["free"]]
	else:
		all_vortices = np.r_[positions["plate"], positions["trailing"], positions["free"]]

	# resolution per axis of the flow field (equi-distant and same for all axes)
	flow_field_axis_res = 30
	coords = np.linspace(-1.5*plate_length, 1.5*plate_length, flow_field_axis_res)  # calculate coordinates
	X, Y = np.meshgrid(coords, coords)  # use same coordinates for x and y. Get meshgrid
	induction_points = np.vstack([X.ravel(), Y.ravel()]).T  # convert meshgrid into vector of all mesh points
	
	# pre-allocate memory
	pre_allocate = np.empty((2, flow_field_axis_res**2, all_circulations.shape[0]), dtype=np.float32)
	# calculate induction matrices in x and y direction
	induction_xy = Induction(plate_res, flap_res).induction_matrices(vortices=all_vortices,
                                                                     induction_points=induction_points,
                                                                     save_to=pre_allocate)
	
	induction_x = induction_xy[0, :, :]  # induction matrix in x direction
	induction_y = induction_xy[1, :, :]  # induction matrix in y direction
	u_ind = (induction_x@all_circulations).reshape((flow_field_axis_res, flow_field_axis_res))  # induced velocity in x
	v_ind = (induction_y@all_circulations).reshape((flow_field_axis_res, flow_field_axis_res))  # induced velocity in y
	plt.quiver(X, Y, u_ind+inflow[0], v_ind+inflow[1])  # don't forget to add inflow
	plt.show()
	
	
