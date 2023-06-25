import numpy as np
from numpy.linalg import norm
import numba as nb
import timeit


@nb.njit(fastmath=True, error_model="numpy", parallel=True)
def induction_matrices_x_y(vortices, induction_points, save_to, normals=None):
    biot_savart_fac = 2*np.pi
    for ip_i in range(induction_points.shape[0]):
        for v_i in nb.prange(vortices.shape[0]):
            vortex_to_ip = induction_points[ip_i] - vortices[v_i]
            distance = norm(vortex_to_ip)
            full_factor = 1/(biot_savart_fac*distance**2)
            save_to[0, ip_i, v_i] = -vortex_to_ip[1]*full_factor
            save_to[1, ip_i, v_i] = vortex_to_ip[0]*full_factor
    if normals is not None:
        save_to[0, :, :] *= normals[0]
        save_to[1, :, :] *= normals[1]
    return save_to


vortices = np.random.random((1000, 2)).astype(np.float32)
ip = np.random.random((20, 2)).astype(np.float32)


def _unit_normal_and_length(unit_normal_for: np.ndarray):
    vector_length = norm(unit_normal_for)
    normalised = unit_normal_for/vector_length
    return np.r_[-normalised[1], normalised[0]], vector_length


start_time = timeit.default_timer()
plate_normal, _ = _unit_normal_and_length(ip[0, :])
normals = (plate_normal[0]*np.ones((ip.shape[0], vortices.shape[0])),
           plate_normal[1]*np.ones((ip.shape[0], vortices.shape[0])))
save_to = np.zeros((2, ip.shape[0], vortices.shape[0]), dtype=vortices.dtype)
for _ in range(100):
    induction = induction_matrices_x_y(vortices, ip, save_to, normals)
end_time = timeit.default_timer()
print(f"Numba used {end_time-start_time}s")


def control_point_induction(plate_control_points: np.ndarray,
                            vortices: np.ndarray,):
    plate_normal, _ = _unit_normal_and_length(plate_control_points[0, :])
    cpi_mat_x = np.zeros((plate_control_points.shape[0], vortices.shape[0]))
    cpi_mat_y = np.zeros((plate_control_points.shape[0], vortices.shape[0]))
    control_points = plate_control_points
    normals = plate_control_points.shape[0]*[plate_normal]
    for (i_cp, cp), normal in zip(enumerate(control_points), normals):
        for i_v, vortex in enumerate(vortices):
            ind = normal*_induction(vortex, cp)
            cpi_mat_x[i_cp, i_v] = ind[0]
            cpi_mat_y[i_cp, i_v] = ind[1]
    return cpi_mat_x, cpi_mat_y


def _induction(vortex: np.ndarray, induction_point: np.ndarray):
    direction, distance = _unit_normal_and_length(induction_point-vortex)
    return 1/(2*np.pi*distance)*direction


start_time = timeit.default_timer()
for _ in range(100):
    ind_x, ind_y = control_point_induction(ip, vortices)
end_time = timeit.default_timer()
print(f"Old code used {end_time-start_time}s")

print((induction[0]-ind_x))
print((induction[1]-ind_y))
