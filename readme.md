Parameters:
- trailing_vortices: np.ndarray of size (n_t, 3) with n_t number of time steps. Columns: [x, y, circulation]
- bound_vortices: np.ndarray of size (plate_res, 3) with plate_res number of plate elements. Columns: [x, y, circulation]
- control_points: np.ndarray of size (plate_res, 2) with plate_res number of plate elements. Columns: [x, y]


- Geometry()
- set_blade(plate_length, plate_res, aoa)

- rotate(parameter)
  - Update the "parameter" (could be either "bound_vortices" or "control_points") for its new rotation
  - Uses: Nothing
- advect_trailing_vortices()
