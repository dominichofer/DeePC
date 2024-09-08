
import numpy as np
import matplotlib.pyplot as plt
from deepc import Controller, RandomNoiseDiscreteLTI, data_quality

from scipy.signal import max_len_seq

def generate_prbs_with_shift(length, num_channels=3, levels=[0, 10], shift=10, samples_n=6):
    """
    Generate a PRBS input sequence with a phase shift for each channel.
    
    Args:
    - length (int): Desired length of the PRBS sequence.
    - num_channels (int): Number of input channels.
    - levels (list): Levels that the PRBS can take. Default is [0, 10].
    - shift (int): Number of steps to shift each subsequent channel.
    - samples_n (int): The number of bits in the PRBS sequence.
    
    Returns:
    - prbs_sequence (list): Generated PRBS sequence with shifts.
    """
    # Generate the base PRBS sequence using max_len_seq
    seq = max_len_seq(samples_n)[0]
    N = len(seq)
    
    # Repeat the sequence if necessary to reach the desired length
    base_sequence = np.tile(seq, (length + (num_channels - 1) * shift) // N + 1)[:length + (num_channels - 1) * shift]
    
    # Adjust the levels
    base_sequence = base_sequence * (levels[1] - levels[0]) + levels[0]
    
    # Generate PRBS sequence with phase shifts
    prbs_sequence = []
    for i in range(length):
        step = [base_sequence[i + (j * shift)] for j in range(num_channels)]
        prbs_sequence.append(step)
    
    return prbs_sequence

max_input = 4.5
min_input = -4.5

# Usage example prbs:
length = 1024
num_channels = 10
levels = [min_input,max_input]
shift = 40 #VERY IMPORTANT for data quality  HOW TO QUANTIFY THIS IN DATA QUALITY?
samples_n = 9

prbs_sequence = generate_prbs_with_shift(length, num_channels, levels, shift, samples_n)


# Gather offline data
# Define the input sequence
#prbs_sequence = np.random.choice([min_input, max_input], size=(1000, 10))  # Example PRBS sequence for 10 inputs

# Define a system
system = RandomNoiseDiscreteLTI(
    A=[[0.9, 0.02, 0.05, 0, 0, 0, 0, 0, 0, 0],
       [0.01, 0.85, 0.02, 0.01, 0, 0, 0, 0, 0, 0],
       [0, 0.03, 0.8, 0.05, 0, 0, 0, 0, 0, 0],
       [0, 0, 0.02, 0.75, 0.04, 0, 0, 0, 0, 0],
       [0, 0, 0, 0.03, 0.7, 0.05, 0, 0, 0, 0],
       [0, 0, 0, 0, 0.02, 0.65, 0.06, 0, 0, 0],
       [0, 0, 0, 0, 0, 0.01, 0.6, 0.07, 0, 0],
       [0, 0, 0, 0, 0, 0, 0.02, 0.55, 0.08, 0],
       [0, 0, 0, 0, 0, 0, 0, 0.01, 0.5, 0.09],
       [0, 0, 0, 0, 0, 0, 0, 0, 0.02, 0.45]],
    
    B=[[0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0.3, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0.4, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0.6, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0.7, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0.8, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0.9, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0]],
    
    C=[[1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 1.1, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 1.2, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 1.3, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 1.4, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 1.5, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 1.6, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 1.7, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 1.8, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1.9]],
    
    D=[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    
    x_ini=[5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
    noise_std=0.05
)

# Check system stability
print("Is it stable?", system.is_stable())

# Apply the input sequence to the system
u_d = prbs_sequence
y_d = system.apply_multiple(u_d)

u_p = np.array(u_d)
y_p = np.array(y_d)


print("Data size:", np.shape(u_d))

# Plotting the control input
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
for i in range(10):
    plt.plot(u_p[:, i], label=f'u_d[{i}]')
plt.title('Control Input (u_d)')
plt.xlabel('Time Step')
plt.ylabel('u_d')
plt.legend()

# Plotting the system response
plt.subplot(2, 1, 2)
for i in range(10):
    plt.plot(y_p[:, i], label=f'y_d[{i}]')
plt.title('System Response (y_d)')
plt.xlabel('Time Step')
plt.ylabel('y_d')
plt.legend()

plt.tight_layout()

# Define the controller parameters
T_ini = 15
r_len = 5

data_quality(u_d, y_d, T_ini, r_len)

# Define the controller
constraint = lambda u: np.clip(u, min_input, max_input)
controller = Controller(u_d, y_d, T_ini, r_len, 1, 0.001, input_constrain_fkt=constraint)

# Reset the system state
system.set_state([3] * 10)  # Reset to a different initial state

# Warm up the controller
while not controller.is_initialized():
    u = [0] * 10
    y = system.apply(u)
    controller.update(u, y)

# Exp discarded filter
l = 0.9
u_ss = [0.1] * 10

# Simulate the system
u_online = []
y_online = []
r_online = [[0] * 10] * 200 + \
           [[0, 0, -5, 0, 0, 0, 0, 0, 0, 0]] * 200 + \
           [[5, 4, -3, 0, 0, 0, 0, 0, 0, 0]] * 200 + \
           [[0, -7, 2, 0, 0, 0, 0, 0, 0, 0]] * 200 + \
           [[0, 0, 0, -9, 1, 0, 0, 0, 0, 0]] * 200 + \
           [[0, 0, 0, 0, 6, -8, 0, 0, 0, 0]] * 200 + \
           [[0, 0, 0, 0, 0, 0, -10, 0, 3, 0]] * 200 + \
           [[0, 0, 0, 0, 0, 0, 0, -5, 4, 0]] * 200 + \
           [[0, 0, 0, 0, 0, 0, 0, 0, -8, 7]] * 200 + \
           [[1, -2, 0, 0, 0, 0, 0, 0, 0, 9]] * 200

for i in range(len(r_online) - r_len):
    r = r_online[i: i + r_len]
    u = controller.apply(r)[0]
    y = system.apply(u)
    controller.update(u, y)
    u_online.append(u)
    y_online.append(y)

    u_ss = [(u[i] * (1 - l)) + (u_ss[i] * l) for i in range(len(u))]

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot inputs
for i in range(10):  # Plot only first 3 inputs for readability
    ax1.plot([u[i] for u in u_online], label=f"input {i+1}", linestyle=":")
ax1.set_title('Inputs')
ax1.legend()

# Plot outputs and targets
for i in range(10):  # Plot only first 3 outputs for readability
    ax2.plot([y[i] for y in y_online], label=f"output {i+1}")
    ax2.plot([r[i] for r in r_online[:len(y_online)]], label=f"target {i+1}", linestyle="--")
ax2.set_title('Outputs and Targets')
ax2.legend()

# Adjust layout to prevent overlap
plt.tight_layout()



y_online = np.array(y_online)
r_online = np.array(r_online[:len(y_online)])  # Trim the targets to match y_online length

# Calculate RMSE for each dimension
rmse = np.sqrt(np.mean((y_online - r_online) ** 2, axis=0))

# Calculate MAE for each dimension
mae = np.mean(np.abs(y_online - r_online), axis=0)

# Print out the results for each dimension
for i in range(10):
    print(f"Dimension {i+1}: RMSE = {rmse[i]:.4f}, MAE = {mae[i]:.4f}")

# If you want a single aggregate score across all dimensions
total_rmse = np.sqrt(np.mean((y_online - r_online) ** 2))
total_mae = np.mean(np.abs(y_online - r_online))

print(f"Overall RMSE: {total_rmse:.4f}")
print(f"Overall MAE: {total_mae:.4f}")




# Show the plot
plt.show()
