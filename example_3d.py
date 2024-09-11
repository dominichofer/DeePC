
import numpy as np
import matplotlib.pyplot as plt
from deepc import Controller, RandomNoiseDiscreteLTI, generate_prbs_with_shift, data_quality, generate_chirp_with_shift

max_input = 5
min_input = -2.0


# Define a system
system = RandomNoiseDiscreteLTI(
    A=[[0.8, 0.05, 0.1], 
       [0.02, 0.8, 0.1], 
       [0.01, 0.02, 0.7]],
    B=[[0.3, 0, 0], 
       [0, 0.2, 0], 
       [0, 0, 0.4]],
    C=[[1.1, 0, 0], 
       [0, 1.3, 0], 
       [0, 0, 0.8]],
    D=[[0, 0, 0], 
       [0, 0, 0], 
       [0, 0, 0]],
    x_ini=[5.0, 5.0, 5.0],
   noise_std=0.05
)

print( "is it stable " ,system.is_stable())


# pribs ..................................................................................
# Usage example: seems to be the minimum "perfect" controller
length = 26
num_channels = 3
levels = [min_input, max_input]
shift = 6 
samples_n = 5

# Usage example: seems to be with noise good controller
length = 90
num_channels = 3
levels = [min_input, max_input]
shift = 7
samples_n = 7

prbs_sequence = generate_prbs_with_shift(length, num_channels, levels, shift, samples_n)



# basic -----------------------------------------
# this approach gives much more intuitive singluar values, however, it takes much more data to get a good controller (which is still worse),
# Gather offline data
N = 10
# by defining a input sequence
u_d = [[0,0,max_input]] * N + [[0,max_input,0]] * N + [[max_input,0,0]] * N
# or this one
iterator = 7
u_d = [[x,y,z] for x in range(iterator) for y in range(iterator) for z in range(iterator) for _ in range(N)]



# chirp -------------------------------------------
length = 100
num_channels = 3
f0 = 1  # Start frequency in Hz
f1 = 100.0  # End frequency in Hz

shift = 25
samples_n = 100
phi = 0.1  # Initial phase

chirp_sequence = generate_chirp_with_shift(length, num_channels, f0, f1, shift, samples_n, phi, levels)





u_d = chirp_sequence #  prbs_sequence#   

# Apply it to the system
y_d = system.apply_multiple(u_d)

u_p = np.array(u_d)
y_p = np.array(y_d)

print("data size ", np.shape(u_d))

# Define how many steps the controller should look back
# to grasp the current state of the system
T_ini = 3

# Define how many steps the controller should look forward
r_len = 3

data_quality(u_d, y_d, T_ini, r_len, 1 , 0.001)


plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(u_p[:, 0], label='u_d[0]')
plt.plot(u_p[:, 1], label='u_d[1]')
plt.plot(u_p[:, 2], label='u_d[2]')
plt.title('Control Input (u_d)')
plt.xlabel('Time Step')
plt.ylabel('u_d')
plt.legend()

# Plotting y_d
plt.subplot(2, 1, 2)
plt.plot(y_p[:, 0], label='y_d[0]')
plt.plot(y_p[:, 1], label='y_d[1]')
plt.plot(y_p[:, 2], label='y_d[2]')
plt.title('System Response (y_d)')
plt.xlabel('Time Step')
plt.ylabel('y_d')
plt.legend()

plt.tight_layout()
#plt.show()


# Define the controller
constraint = lambda u: np.clip(u, min_input, max_input)
controller = Controller(u_d, y_d, T_ini, r_len, 1 , 0.01, input_constrain_fkt=constraint )

# Reset the system
# to sepereate the offline data from the online data
system.set_state([3, 3, 3])  # This is intentionally not the same state as x_ini

# Warm up the controller
while not controller.is_initialized():
    u = [0, 0, 0]
    y = system.apply(u)
    controller.update(u, y)

#exp discarded filter
l = 0.9
u_ss = [0.1, 0.1, 0.1]

# Simulate the system
u_online = []
y_online = []
r_online = [[0, 6, 0]] * 200 + [[0, 0, -1]] * 200 + [[7, 4, 1]] * 200 + [[0, 7, 2]] * 200
for i in range(len(r_online) - r_len):
    r = r_online[i: i + r_len]
    #print("u ss : ",[r, u_ss])
    u = controller.apply_trajectory_tracking_verion(r)[0]#u_ss
    y = system.apply(u)#
    controller.update(u, y)
    u_online.append(u)
    y_online.append(y)
    r_online.append(r)

    u_ss = [(u[i] * (1 - l)) + (u_ss[i] * l) for i in range(len(u))]
    #u_ss = np.zeros_like(u_ss)
    #print("u ss : ", u_ss)



# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot inputs on the first subplot
ax1.plot([u[0] for u in u_online], label="input 1", color="green", linestyle=":")
ax1.plot([u[1] for u in u_online], label="input 2", color="red", linestyle=":")
ax1.plot([u[2] for u in u_online], label="input 3", color="purple", linestyle=":")
ax1.set_title('Inputs')
ax1.legend()

# Plot outputs and targets on the second subplot
ax2.plot([y[0] for y in y_online], label="output 1", color="green")
ax2.plot([y[1] for y in y_online], label="output 2", color="red")
ax2.plot([y[2] for y in y_online], label="output 3", color="purple")
ax2.plot([r[0] for r in r_online[:len(y_online)]], label="target 1", color="green", linestyle="--")
ax2.plot([r[1] for r in r_online[:len(y_online)]], label="target 2", color="red", linestyle="--")
ax2.plot([r[2] for r in r_online[:len(y_online)]], label="target 3", color="purple", linestyle="--")
ax2.set_title('Outputs and Targets')
ax2.legend()

# Adjust layout to prevent overlap
plt.tight_layout()

r_online = np.array(r_online[:len(y_online)])  # Trim the targets to match y_online length

# Calculate RMSE for each dimension
rmse = np.sqrt(np.mean((y_online - r_online) ** 2, axis=0))

# Calculate MAE for each dimension
mae = np.mean(np.abs(y_online - r_online), axis=0)

# Print out the results for each dimension
for i in range(3):
    print(f"Dimension {i+1}: RMSE = {rmse[i]:.4f}, MAE = {mae[i]:.4f}")

# If you want a single aggregate score across all dimensions
total_rmse = np.sqrt(np.mean((y_online - r_online) ** 2))
total_mae = np.mean(np.abs(y_online - r_online))

print(f"Overall RMSE: {total_rmse:.4f}")
print(f"Overall MAE: {total_mae:.4f}")


# Show the plot
plt.show()
