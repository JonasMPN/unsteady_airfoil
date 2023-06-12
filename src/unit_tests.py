from src.Geometry import Geometry
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

test = {
		"convection": True,
		
}

if test["convection"]:
	plate_length = 1
	plate_res = 5
	cycles = 4
	cycles_res = 100
	max_angle = 80  # degree
	angle_of_attacks = np.deg2rad(max_angle)*np.sin(np.linspace(0, 2*np.pi*cycles, cycles_res*cycles))
	
	geom = Geometry()
	geom.set(dt=0.1, n_time_steps=cycles_res*cycles)
	geom.set_blade(plate_length=plate_length, plate_res=plate_res, angle_of_attack=angle_of_attacks[0])
	bound_x = list()
	bound_y = list()
	trailing_x = list()
	trailing_y = list()
	
	for i, aoa in enumerate(angle_of_attacks):
		geom.update_aoa(aoa)
		geom.advect_trailing_vortices(0.5*(np.random.random((i, 2))-0.5), inflow_speed=2)
		positions = geom.get_positions()
		bound, trailing, _ = positions["bound_vortices"], positions["trailing_vortices"], positions["control_points"]
		bound_x.append(bound[:, 0])
		bound_y.append(bound[:, 1])
		trailing_x.append(trailing[:, 0])
		trailing_y.append(trailing[:, 1])
		
	fig, ax = plt.subplots()
	line_bound, = ax.plot(bound_x[0], bound_y[0], "xr")
	line_trailing, = ax.plot(trailing_x[0], trailing_y[0], "xg")
	ax.set(xlim=[0, 30], ylim=[-1.5, 1.5])
	
	
	def update(frame):
		line_bound.set_data(bound_x[frame], bound_y[frame])
		line_trailing.set_data(trailing_x[frame], trailing_y[frame])
		return line_bound, line_trailing
	
	
	ani = animation.FuncAnimation(fig=fig, func=update, frames=cycles_res*cycles, interval=30)
	ani.save("../results/unit_tests/convection.mp4")

