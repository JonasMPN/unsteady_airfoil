from src.Geometry import Geometry
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from helper_functions import Helper
helper = Helper()

test = {
		"convection"               : False,
		"lhs"                      : False,
		"trailing_induction"       : False,
		"no_wake_interaction"      : False,
		"wake_interaction_matrices": False,
		"with_wake_interaction"    : True
}


def steady_solution(inflow_speed, plate_length, angle_of_attack):
	return -plate_length*np.pi*angle_of_attack*inflow_speed


if test["convection"]:
	plate_length = 1
	plate_res = 5
	cycles = 4
	cycles_res = 100
	inflow_speed = 1
	max_angle = 80  # degree
	angle_of_attacks = np.deg2rad(max_angle)*np.sin(np.linspace(0, 2*np.pi*cycles, cycles_res*cycles))
	
	geom = Geometry()
	geom.set_constants(dt=0.1, n_time_steps=cycles_res*cycles)
	geom.set_blade(plate_length=plate_length, plate_res=plate_res)
	bound_x = list()
	bound_y = list()
	trailing_x = list()
	trailing_y = list()
	plate_normal_x = list()
	plate_normal_y = list()
	
	for i, aoa in enumerate(angle_of_attacks):
		geom.update_flow(aoa, inflow_speed)
		geom.add_trailing_vortex()
		geom.displace_trailing_vortices(0.5*(np.random.random((i+1, 2))-0.5))
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
	line_normal, = ax.plot([bound_x[0][-1], bound_x[0][-1]+plate_normal_x[0]],
						   [bound_y[0][-1], bound_y[0][-1]+plate_normal_y[0]])
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
	geom.set_constants(dt=0.1, n_time_steps=1)
	geom.set_blade(plate_length=plate_length, plate_res=plate_res)
	print(geom.get_lhs_matrix())
	geom.update_flow(angle_of_attack=0, inflow_speed=1)
	print(geom.get_lhs_matrix())

if test["trailing_induction"]:
	plate_length = 1
	plate_res = 1
	geom = Geometry()
	n_advection = 5
	geom.set_constants(dt=1, n_time_steps=n_advection)
	geom.set_blade(plate_length=plate_length, plate_res=plate_res)
	geom.update_flow(angle_of_attack=0, inflow_speed=1)
	for i in range(n_advection):
		geom.add_trailing_vortex()
		geom.displace_trailing_vortices(np.zeros((i+1, 2)))
	print(1/(2*np.pi*geom.get_trailing_induction_matrix()))

if test["no_wake_interaction"]:
	aoa_jump = 10
	inflow_speed = 1
	plate_length = 1
	plate_res = 1
	geom = Geometry()
	n_time_steps = 250
	dt = 0.02
	geom.set_constants(dt=dt, n_time_steps=n_time_steps)
	geom.set_blade(plate_length=plate_length, plate_res=plate_res)
	angle_of_attacks = np.zeros(n_time_steps)
	angle_of_attacks[1:] = np.deg2rad(aoa_jump)
	aoa = angle_of_attacks[0]
	
	plot_circulation = list()
	plot_lift = list()
	trailing_circulation = np.zeros((n_time_steps, 1))
	
	for n_t in range(n_time_steps):
		if n_t == 0 or aoa != angle_of_attacks[n_t]:
			aoa = angle_of_attacks[n_t]
			geom.update_flow(angle_of_attack=aoa, inflow_speed=inflow_speed)
			geom.add_trailing_vortex()
			lhs = geom.get_lhs_matrix()
			inv_lhs = np.linalg.inv(lhs)
			plate_normal = geom.get_normals()["plate"]
			normal_inflow = inflow_speed*plate_normal[0]*np.ones((plate_res, 1))
		else:
			geom.add_trailing_vortex()
		
		trailing_induction_mat = geom.get_trailing_induction_matrix()
		normal_induction = trailing_induction_mat@trailing_circulation[:n_t+1]
		circulation = -inv_lhs@(np.append(normal_inflow+normal_induction, np.sum(trailing_circulation)))
		plot_circulation.append(np.sum(circulation[:plate_res]))
		trailing_circulation[n_t, 0] = circulation[plate_res]
		geom.displace_trailing_vortices(np.zeros((n_t+1, 2)))
	
	steady_sol = steady_solution(inflow_speed, plate_length, angle_of_attacks[-1])
	fig, ax = plt.subplots()
	ax.plot(inflow_speed/plate_length*np.arange(0, dt*n_time_steps, dt), plot_circulation/steady_sol)
	helper.handle_axis(ax, grid=True, x_label=r"$U_\infty t/c$", y_label=r"$\Gamma (t)/\Gamma_\infty$",
					   line_width=3, font_size=25)
	helper.handle_figure(fig, save_to=f"../results/unit_tests/no_wake_interaction_{aoa_jump}.png")

if test["wake_interaction_matrices"]:
	inflow_speed = 1
	plate_length = 1
	plate_res = 1
	geom = Geometry()
	n_time_steps = 5
	geom.set_constants(dt=1, n_time_steps=n_time_steps)
	geom.set_blade(plate_length=plate_length, plate_res=plate_res)
	geom.update_flow(angle_of_attack=0, inflow_speed=inflow_speed)
	for n_t in range(n_time_steps):
		geom.add_trailing_vortex()
		geom.displace_trailing_vortices(np.zeros((n_t+1, 2)))
	bound, trailing, additional = geom.get_trailing_displacement_matrices()
	print("Bound vortices:")
	with np.errstate(divide="ignore"):
		print(1/(2*np.pi*bound["x"]))
		print(1/(2*np.pi*bound["y"]))
		print("\n", "Trailing vortices:")
		print(1/(2*np.pi*trailing["x"]))
		print(1/(2*np.pi*trailing["y"]))
	geom.plot_vortices()

if test["with_wake_interaction"]:
	inflow_speed_jump = [1, 1]
	angle_of_attack_jump = [0, 10]
	plate_length = 1
	plate_res = 20
	n_time_steps = 700
	dt = 0.05
	
	angle_of_attacks = angle_of_attack_jump[0]*np.ones(n_time_steps)
	angle_of_attacks[10:] = np.deg2rad(angle_of_attack_jump[1])
	inflow_speeds = inflow_speed_jump[0]*np.ones(n_time_steps)
	inflow_speeds[10:] = inflow_speed_jump[1]
	
	geom = Geometry()
	geom.set_constants(dt=dt, n_time_steps=n_time_steps)
	geom.set_blade(plate_length=plate_length, plate_res=plate_res)
	
	aoa = angle_of_attacks[0]
	U = inflow_speeds[0]
	
	plot_circulation = list()
	plot_lift = list()
	trailing_circulation = np.zeros((n_time_steps, 1))
	
	bound_x = list()
	bound_y = list()
	trailing_x = list()
	trailing_y = list()
	
	for n_t in range(n_time_steps):
		if n_t == 0 or aoa != angle_of_attacks[n_t] or U != inflow_speeds[n_t]:
			aoa = angle_of_attacks[n_t]
			U = inflow_speeds[n_t]
			geom.update_flow(angle_of_attack=aoa, inflow_speed=U)
			geom.add_trailing_vortex()
			lhs = geom.get_lhs_matrix()
			inv_lhs = np.linalg.inv(lhs)
			plate_normal = geom.get_normals()["plate"]
			normal_inflow = U*plate_normal[0]*np.ones((plate_res, 1))
		else:
			geom.add_trailing_vortex()
		
		trailing_induction_mat = geom.get_trailing_induction_matrix()
		normal_induction = trailing_induction_mat@trailing_circulation[:n_t+1]
		circulation = -inv_lhs@(np.append(normal_inflow+normal_induction, np.sum(trailing_circulation)))
		trailing_circulation[n_t, 0] = circulation[plate_res]
		
		bound, trailing, _ = geom.get_trailing_displacement_matrices()
		bound_induced = np.c_[bound["x"]@circulation[:-1], bound["y"]@circulation[:-1]]
		trailing_induced = np.c_[trailing["x"]@trailing_circulation[:n_t+1], trailing["y"]@trailing_circulation[:n_t+1]]
		trailing_displacement = (bound_induced+trailing_induced)*dt
		geom.displace_trailing_vortices(trailing_displacement)
		
		plot_circulation.append(np.sum(circulation[:plate_res]))
		positions = geom.get_positions()
		bound_coords, trailing_coords = positions["bound_vortices"], positions["trailing_vortices"]
		bound_x.append(bound_coords[:, 0])
		bound_y.append(bound_coords[:, 1])
		trailing_x.append(trailing_coords[::-1][:, 0])
		trailing_y.append(trailing_coords[::-1][:, 1])
		if n_t%20 == 0:
			print(n_t)
	
	fig, ax = plt.subplots()
	line_bound, = ax.plot(bound_x[0], bound_y[0], "xr")
	line_trailing, = ax.plot(trailing_x[0], trailing_y[0], "xg")
	ax.set(xlim=[-1, 37], ylim=[-0.3, 0.2])
	title = ax.text(0.5, 0, "0")
	
	def update(frame):
		line_bound.set_data(bound_x[frame], bound_y[frame])
		line_trailing.set_data(trailing_x[frame], trailing_y[frame])
		title.set_text(frame)
		return line_bound, line_trailing, title

	ani = animation.FuncAnimation(fig=fig, func=update, frames=n_time_steps, interval=30, blit=True)
	# plt.show()
	ani.save(f"../results/unit_tests/full_ll_aoa{np.round(np.rad2deg(angle_of_attacks[-1]), 2)}.mp4")
	
	# geom.plot_vortices()
	
	steady_sol = steady_solution(inflow_speeds[-1], plate_length, angle_of_attacks[-1])
	normalised_distance = inflow_speeds[-1]/plate_length*np.arange(0, dt*n_time_steps, dt)
	fig, ax = plt.subplots()
	ax.plot(normalised_distance, np.asarray(plot_circulation)/steady_sol)
	helper.handle_axis(ax, grid=True, x_label=r"$U_\infty t/c$", y_label=r"$\Gamma (t)/\Gamma_\infty$",
					   line_width=3, font_size=25)
	helper.handle_figure(fig, save_to=f"../results/unit_tests/with_wake_interaction_{angle_of_attack_jump[1]}_{n_time_steps}.png")
	print(-plot_circulation[-1]*inflow_speeds[-1], steady_sol)


