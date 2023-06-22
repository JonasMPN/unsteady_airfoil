import numpy as np
from geometry import Geometry
from induction import Induction

test = {
	"geometry": False,
	"cp_induction": False,
	"free_vortex_induction": False,
	"lhs_matrix": False
}


if test["geometry"]:
	time_steps = 10
	dt = 0.1
	geom = Geometry(time_steps)
	geom.set_blade(1, 5)
	geom.set_flap(0.5, 3)
	geom.update_aoa(np.deg2rad(20), np.deg2rad(40))
	for i in range(time_steps):
		geom.add_trailing_vortex((1, 0), dt)
		if i == time_steps-1:
			continue
		geom.displace_trailing_vortices(np.c_[np.ones(i+1), np.zeros(i+1)], dt)
	geom.plot()
	
	
if test["cp_induction"]:
	time_steps = 10
	dt = 0.1
	blade_res = 2
	flap_res = 1
	geom = Geometry(time_steps)
	geom.set_blade(1, blade_res)
	geom.set_flap(0.5, flap_res)
	geom.update_aoa(0, 0)
	geom.add_trailing_vortex((1, 0), time_step=0.5)
	
	ind = Induction(blade_res, flap_res)
	
	bound, control, trailing = geom.get_positions()
	print("Inversely calculated distances from the free vortices to the control points:")
	print(1/(2*np.pi*ind.control_point_induction(plate_control_points=control["plate"], vortices=trailing,
	                                             flap_control_points=control["flap"])))
	geom.plot()
	
if test["free_vortex_induction"]:
	time_steps = 10
	dt = 0.1
	blade_res = 1
	flap_res = 1
	geom = Geometry(time_steps)
	geom.set_blade(1, blade_res)
	geom.set_flap(1, flap_res)
	geom.update_aoa(0, 0)
	geom.add_trailing_vortex((1, 0), time_step=0.5)
	geom.displace_trailing_vortices((1, 0), time_step=0.5,)
	geom.add_trailing_vortex((1, 0), time_step=0.5)
	
	ind = Induction(blade_res, flap_res)
	
	bound, control, trailing = geom.get_positions()
	bound = np.r_[bound["plate"], bound["flap"]]
	from_trailing, from_bound = ind.free_vortex_induction(trailing, bound)

	print("Inversely calculated distances from the free vortices onto one another:")
	print("x")
	# print(1/(2*np.pi*from_trailing["x"]))
	print(from_trailing["x"])
	print("y")
	print(1/(2*np.pi*from_trailing["y"]))

	print("Inversely calculated distances from the bound vortices to the free vortices:")
	print("x")
	print(1/(2*np.pi*from_bound["x"]))
	print("y")
	print(1/(2*np.pi*from_bound["y"]))
	geom.plot()

if test["lhs_matrix"]:
	time_steps = 10
	dt = 0.1
	blade_res = 2
	flap_res = 2
	geom = Geometry(time_steps)
	geom.set_blade(1, blade_res)
	geom.set_flap(1, flap_res)
	geom.update_aoa(0, 0)
	geom.add_trailing_vortex((1, 0), time_step=0.5)
	
	ind = Induction(blade_res, flap_res)
	bound, control, trailing = geom.get_positions()
	bound = np.r_[bound["plate"], bound["flap"]]
	print(ind.lhs_matrix(bound, trailing[-1], control["plate"], control["flap"]))
	geom.plot()
