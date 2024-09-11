import numpy as np
import matplotlib.pyplot as plt
from deepc import Controller, DiscreteLTI, data_quality, generate_chirp_with_shift

# Define a system
system = DiscreteLTI(
    A=[[1.9154, -0.915], [1.0, 0.0]],
    B=[[0.00244], [0.0]],
    C=[[1, 0]],
    D=[[0]],
    x_ini=[1, 1],
)

min_input = 0
max_input = 100

levels = [min_input, max_input]
length = 400
num_channels = 1
f0 = 1  # Start frequency in Hz
f1 = 100.0  # End frequency in Hz
shift = 0
samples_n = 200
phi = 0.0  # Initial phase

u_d = generate_chirp_with_shift(length, num_channels, f0, f1, shift, samples_n, phi, levels)


# and applying it to the system
y_d = system.apply_multiple(u_d)

# Define how many steps the controller should look back
# to grasp the current state of the system
T_ini = 10

# Define how many steps the controller should look forward
r_len = 17

data_quality(u_d, y_d, T_ini, r_len, 1.0, 0.001)

# Define the controller
controller = Controller(u_d, y_d, T_ini, r_len, 1.0,0.001, input_constrain_fkt=lambda u: np.clip(u, 0, 100))

# Reset the system
# to sepereate the offline data from the online data
system.set_state([0, 0])  # This is intentionally not the same state as x_ini

# Warm up the controller
while not controller.is_initialized():
    u = [0]
    y = system.apply(u)
    controller.update(u, y)

# Simulate the system
u_online = []
y_online = []
r_online = [[0]] * 20 + [[10]] * 100 + [[7]] * 100 + [[40]] * 100
for i in range(len(r_online) - r_len):
    r = r_online[i: i + r_len]
    u = controller.apply(r)[0]
    #u = controller.apply_trajectory_tracking_version(r)[0]
    y = system.apply(u)
    controller.update(u, y)
    u_online.append(u)
    y_online.append(y)
    r_online.append(r)

plt.figure()
# Plot the results
plt.plot(u_online, label="input")
plt.plot(y_online, label="output")
plt.plot(r_online[:len(y_online)], label="target", color="black", linestyle="--")
plt.legend()

# Plot the results
plt.plot(u_online, label="input")
plt.plot(y_online, label="output")
plt.plot(r_online[:len(y_online)], label="target", color="black", linestyle="--")
plt.legend()

r_online = np.array(r_online[:len(y_online)])  # Trim the targets to match y_online length

# Calculate RMSE for each dimension
rmse = np.sqrt(np.mean((y_online - r_online) ** 2, axis=0))

# Calculate MAE for each dimension
mae = np.mean(np.abs(y_online - r_online), axis=0)

# Print out the results for each dimension
i = 0
print(f"Dimension {i+1}: RMSE = {rmse[i]:.4f}, MAE = {mae[i]:.4f}")

# If you want a single aggregate score across all dimensions
total_rmse = np.sqrt(np.mean((y_online - r_online) ** 2))
total_mae = np.mean(np.abs(y_online - r_online))

print(f"Overall RMSE: {total_rmse:.4f}")
print(f"Overall MAE: {total_mae:.4f}")


plt.show()