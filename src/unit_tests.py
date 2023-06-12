from src.Geometry import Geometry
import matplotlib.pyplot as plt
import numpy as np

plate_length = 1
plate_res = 5
aoa_init = 15		# in degree

geom = Geometry()
geom.set(dt=0.1, inflow_speed=10)
geom.set_blade(plate_length=plate_length, plate_res=plate_res, angle_of_attack=np.deg2rad(aoa_init))
positions = geom.get_positions()
bound, trailing, cp = positions["bound_vortices"], positions["trailing_vortices"], positions["control_points"]
plt.plot(bound[:, 0], bound[:, 1], "xg")
# plt.plot(trailing[0], trailing[1], "xo")
plt.plot(cp[:, 0], cp[:, 1], "xr")
plt.show()

