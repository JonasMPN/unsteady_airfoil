import numpy as np
from geometry import Geometry, plot_process
from induction import Induction
from unsteady_aero import UnsteadyAirfoil

test = {
	"displacing": False,
	"cp_induction": False,
	"free_vortex_induction": False,
	"lhs_matrix": False,
	"without_free": True,
	"with_free": False
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
	blade_res = 1
	flap_res = 1
	geom = Geometry(time_steps)
	geom.set_plate(1, blade_res)
	geom.set_flap(0.5, flap_res)
	geom.update_rotation(0, 0)
	geom.shed_vortex((1, 0), time_step=0.5)
	geom.add_free_vortices(np.asarray([[3, 1]]))

	ind = Induction(blade_res, flap_res)

	bound, control, trailing, free = geom.get_positions()
	print("Inversely calculated distances from the free vortices to the control points:")
	print(1/(2*np.pi*ind.control_point_induction(plate_control_points=control["plate"],
												 vortices=np.r_[trailing, free],
												 flap_control_points=control["flap"])))
	geom.plot_final_state(ls_trailing="x")

if test["free_vortex_induction"]:
	time_steps = 10
	dt = 0.1
	blade_res = 1
	flap_res = 1
	geom = Geometry(time_steps)
	geom.set_plate(1, blade_res)
	geom.set_flap(1, flap_res)
	geom.update_rotation(0, 0)
	geom.shed_vortex((1, 0), time_step=0.5)
	geom.displace_vortices(np.asarray([[1, 0]]), time_step=0.5, )
	geom.shed_vortex((1, 0), time_step=0.5)
	geom.add_free_vortices(np.asarray([[2., 0.]]))

	ind = Induction(blade_res, flap_res)

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

if test["lhs_matrix"]:
	time_steps = 10
	dt = 0.05
	blade_res = 1
	flap_res = 1
	geom = Geometry(time_steps)
	geom.set_plate(1, blade_res)
	geom.set_flap(1, flap_res)
	geom.update_rotation(0, 0)
	geom.shed_vortex((1, 0), time_step=0.5)

	ind = Induction(blade_res, flap_res)
	bound, control, trailing, free = geom.get_positions()
	bound = np.r_[bound["plate"], bound["flap"]]
	print(ind.lhs_matrix(bound, trailing[-1], control["plate"], control["flap"]))
	geom.plot_final_state(ls_trailing="x")

if test["without_free"]:
	time_steps = 600
	dt = 0.08
	plate_res = 1
	flap_res = 1
	plate_length = 1
	flap_length = 1
	unsteady_airfoil = UnsteadyAirfoil(time_steps, plate_res, plate_length, flap_res, flap_length)
	plate_angles = np.zeros(time_steps)
	plate_angles[:] = -10
	flap_angles = np.zeros(time_steps)
	flap_angles[50:] = -10
	inflow = (1, 0)
	circulation, coordinates = unsteady_airfoil.solve_for_process(dt=dt, plate_angles=plate_angles, inflows=inflow,
													  			  flap_angles=flap_angles)
	unsteady_airfoil.plot_final_state()
	plot_process(**coordinates)


if test["with_free"]:
	time_steps = 200
	dt = 0.08
	plate_res = 1
	flap_res = 1
	plate_length = 1
	flap_length = 1
	unsteady_airfoil = UnsteadyAirfoil(time_steps, plate_res, plate_length, flap_res, flap_length)
	unsteady_airfoil.add_free_vortices(np.asarray([[5., 0.]]), 0.5)
	plate_angles = np.zeros(time_steps)
	# plate_angles[:] = -10
	flap_angles = np.zeros(time_steps)
	# flap_angles[50:] = -10
	inflow = (1, 0)
	circulation, coordinates = unsteady_airfoil.solve_for_process(dt=dt, plate_angles=plate_angles, inflows=inflow,
													  			  flap_angles=flap_angles)
	unsteady_airfoil.plot_final_state()
	plot_process(**coordinates)
