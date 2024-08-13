
import numpy as np
import matplotlib.pyplot as plt
from deepc import Controller, RandomNoisyLTI


def generate_prbs(length, num_channels=3, levels=[0, 10], shift=5):
    """
    Generate a PRBS input sequence with a phase shift for each channel.
    
    Args:
    - length (int): Desired length of the PRBS sequence.
    - num_channels (int): Number of input channels.
    - levels (list): Levels that the PRBS can take. Default is [0, 10].
    - shift (int): Number of steps to shift each subsequent channel.
    
    Returns:
    - prbs_sequence (list): Generated PRBS sequence with shifts.
    """
    base_sequence = [np.random.choice(levels) for _ in range(length + (num_channels - 1) * shift)]
    prbs_sequence = []

    for i in range(length):
        step = [base_sequence[i + (j * shift)] for j in range(num_channels)]
        prbs_sequence.append(step)
    
    return prbs_sequence




# Define a system
system = RandomNoisyLTI(
    A=[[0.88, 0.1, 0.0], 
       [0.1, 0.8, 0.1], 
       [0.0, 0.1, 0.88]],
    B=[[0.01, 0, 0], 
       [0, 0.01, 0], 
       [0, 0, 0.01]],
    C=[[10, 0, 0], 
       [0, 10, 0], 
       [0, 0, 10]],
    D=[[0, 0, 0], 
       [0, 0, 0], 
       [0, 0, 0]],
    x_ini=[1, 1, 1],
   noise_std=0.0
)

# Gather offline data
N = 21
# by defining a input sequence
u_d = [[0,0,0]] * N + [[0,0,10]] * N + [[0,10,0]] * N + [[10,7,1]] * N
# and applying it to the system

shift = 3  # Number of steps to shift each channel

# Generate PRBS sequence for 3 channels with shift
#u_d = generate_prbs(N*4, num_channels=3, levels=[0, 10], shift=shift)

print(u_d)
# Apply it to the system
y_d = system.apply_multiple(u_d)

# Define how many steps the controller should look back
# to grasp the current state of the system
T_ini = 17

# Define how many steps the controller should look forward
r_len = 13

# Define the controller
constraint = lambda u: np.clip(u, 0, 10)
controller = Controller(u_d, y_d, T_ini, r_len, control_constrain_fkt=constraint)

# Reset the system
# to sepereate the offline data from the online data
system.set_state([0, 0, 0])  # This is intentionally not the same state as x_ini

# Warm up the controller
while not controller.is_initialized():
    u = [0, 0, 0]
    y = system.apply(u)
    controller.update(u, y)

# Simulate the system
u_online = []
y_online = []
r_online = [[0, 0, 0]] * 20 + [[10, 15, 20]] * 200 + [[20, 5, 10]] * 300
for i in range(len(r_online) - r_len):
    r = r_online[i: i + r_len]
    u = controller.apply(r)[0]
    y = system.apply(u)
    controller.update(u, y)
    u_online.append(u)
    y_online.append(y)
    r_online.append(r)

print(f"Type of r_online: {type(r_online)}")
print(f"Type of first element in r_online: {type(r_online[0])}")

# Plot the results
plt.plot([u[0] for u in u_online], label="input 1", color="green",  linestyle=":")
plt.plot([u[1] for u in u_online], label="input 2", color="red",  linestyle=":")
plt.plot([u[2] for u in u_online], label="input 3", color="purple",  linestyle=":")
plt.plot([y[0] for y in y_online], label="output 1", color="green")
plt.plot([y[1] for y in y_online], label="output 2", color="red")
plt.plot([y[2] for y in y_online], label="output 3", color="purple")
plt.plot([r[0] for r in r_online[:len(y_online)]], label="target 1", color="green", linestyle="--")
plt.plot([r[1] for r in r_online[:len(y_online)]], label="target 2", color="red", linestyle="--")
plt.plot([r[2] for r in r_online[:len(y_online)]], label="target 3", color="purple", linestyle="--")

plt.legend()
plt.show()
plt.legend()
plt.show()
