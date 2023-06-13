from src.Geometry import Geometry
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

test = {
	"convection": False,
	"lhs": False,
	"trailing_induction": False,
	"no_wake_interaction": True
}

if test["convection"]:
	plate_length = 1
	plate_res = 5
	cycles = 4
	cycles_res = 100
	inflow_speed = 2
	max_angle = 80  # degree
	angle_of_attacks = np.deg2rad(max_angle)*np.sin(np.linspace(0, 2*np.pi*cycles, cycles_res*cycles))
	
	geom = Geometry()
	geom.set(dt=0.1, n_time_steps=cycles_res*cycles)
	geom.set_blade(plate_length=plate_length, plate_res=plate_res)
	bound_x = list()
	bound_y = list()
	trailing_x = list()
	trailing_y = list()
	plate_normal_x = list()
	plate_normal_y = list()

	for i, aoa in enumerate(angle_of_attacks):
		geom.update_flow(aoa, inflow_speed)
		geom.advect_trailing_vortices(0.5*(np.random.random((i, 2))-0.5))
		positions = geom.get_positions()
		bound, trailing, _ = positions["bound_vortices"], positions["trailing_vortices"], positions["control_points"]
		bound_x.append(bound[:, 0])
		bound_y.append(bound[:, 1])
		trailing_x.append(trailing[::-1][:, 0])
		trailing_y.append(trailing[::-1][:, 1])

		normals = geom.get_normals()
		plate_normal_x.append(normals["plate"][0])
		plate_normal_y.append(normals["plate"][1])

	fig, ax = plt.subplots()
	line_bound, = ax.plot(bound_x[0], bound_y[0], "xr")
	line_trailing, = ax.plot(trailing_x[0], trailing_y[0], "xg")
	line_normal, = ax.plot([bound_x[0][-1], bound_x[0][-1]+plate_normal_x[0]], [bound_y[0][-1], bound_y[0][-1]+plate_normal_y[0]])
	ax.axis("equal")
	ax.set(xlim=[-1, 2], ylim=[-1.5, 1.5])
	# ax.set(xlim=[-2, 30], ylim=[-1.5, 1.5])

	
	def update(frame):
		line_bound.set_data(bound_x[frame], bound_y[frame])
		line_trailing.set_data(trailing_x[frame], trailing_y[frame])
		line_normal.set_data([bound_x[frame][-1], bound_x[frame][-1]+plate_normal_x[frame]],
							 [bound_y[frame][-1], bound_y[frame][-1]+plate_normal_y[frame]])
		return line_bound, line_trailing
	
	
	ani = animation.FuncAnimation(fig=fig, func=update, frames=cycles_res*cycles, interval=30)
	plt.show()
	# ani.save("../results/unit_tests/convection.mp4")

if test["lhs"]:
	plate_length = 1
	plate_res = 5
	geom = Geometry()
	geom.set(dt=0.1, n_time_steps=1)
	geom.set_blade(plate_length=plate_length, plate_res=plate_res)
	print(geom.get_lhs_matrix())
	geom.update_flow(angle_of_attack=0, inflow_speed=1)
	print(geom.get_lhs_matrix())

if test["trailing_induction"]:
	plate_length = 1
	plate_res = 1
	geom = Geometry()
	n_advection = 5
	geom.set(dt=0.1, n_time_steps=n_advection)
	geom.set_blade(plate_length=plate_length, plate_res=plate_res)
	geom.update_flow(angle_of_attack=0, inflow_speed=1)
	for i in range(n_advection):
		geom.advect_trailing_vortices(np.zeros((i, 1)), new_trailing_fac=1)
	print(geom.get_trailing_induction_matrix())

if test["no_wake_interaction"]:
	inflow_speed = 1
	plate_length = 1
	plate_res = 5
	geom = Geometry()
	n_time_steps = 100
	geom.set(dt=0.2, n_time_steps=n_time_steps)
	geom.set_blade(plate_length=plate_length, plate_res=plate_res)
	angle_of_attacks = np.zeros(n_time_steps)
	angle_of_attacks[10:] = np.deg2rad(10)
	geom.update_flow(angle_of_attack=0, inflow_speed=inflow_speed)
	lhs = geom.get_lhs_matrix()

	inv_lhs = np.linalg.inv(lhs)
	# lhs = lhs[:-1, :-1]
	# inv_lhs = np.linalg.inv(lhs)
	plot_circulation = list()
	plot_lift = list()
	trailing_circulation = np.zeros((n_time_steps, 1))
	plate_normal = geom.get_normals()["plate"]
	normal_inflow = inflow_speed*plate_normal[0]*np.ones((plate_res, 1))
	# exit()

	for n_t in range(n_time_steps):
		aoa = angle_of_attacks[n_t]
		if n_t > 1 and aoa != angle_of_attacks[n_t-1]:
			geom.update_flow(angle_of_attack=aoa, inflow_speed=inflow_speed)
			lhs = geom.get_lhs_matrix()
			inv_lhs = np.linalg.inv(lhs)

			plate_normal = geom.get_normals()["plate"]
			normal_inflow = inflow_speed*plate_normal[0]*np.ones((plate_res, 1))

		geom.advect_trailing_vortices(np.zeros((n_t, 1)))
		trailing_induction_mat = geom.get_trailing_induction_matrix()
		normal_induction = trailing_induction_mat@trailing_circulation[:n_t+1]

		circulation = -inv_lhs@(np.append(normal_inflow+normal_induction, np.sum(trailing_circulation)))

		# print(normal_inflow, normal_induction, lhs@circulation)
		# circulation = -inv_lhs@normal_inflow

		plot_circulation.append(np.sum(circulation[:plate_res]))
		trailing_circulation[n_t, 0] = circulation[plate_res]

		#
		# print(trailing_induction_mat)
		# print(trailing_circulation[:n_t+1])

	plt.plot(plot_circulation)
	# plt.ylim([-0.6, 0.6])
	plt.grid()
	plt.show()


