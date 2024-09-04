import numpy as np

from scipy.signal import chirp

import matplotlib.pyplot as plt
from deepc import Controller, RandomNoisyLTI, linear_chirp



# Define a system
system = RandomNoisyLTI(
    A=[[1.9154297228892199, -0.9159698592594919], [1.0, 0.0]],
    B=[[0.0024407501942859677], [0.0]],
    C=[[1, 0]],
    D=[[0]],
    x_ini=[1, 1],
    noise_std=0.0
)
# Gather offline data
N = 500

#frequency sweep input 
u_d = linear_chirp(0, N / 2, N)
# and applying it to the system
y_d = system.apply_multiple(u_d)

# Define how many steps the controller should look back
# to grasp the current state of the system
T_ini = 17 # seems like should be bigger than r_len
# Define how many steps the controller should look forward
r_len = 11

# Define the controller
constraint = lambda u: np.clip(u, 0, 50)
controller = Controller(u_d, y_d, T_ini, r_len, control_constrain_fkt=constraint)#

# Reset the system
# to sepereate the offline data from the online data
system.set_state([0, 0])  # This is intentionally not the same state as x_ini

# Warm up the controller
for i in range(T_ini):
    u = 0
    y = system.apply(u)
    controller.update(u, y)


# Check that the controller is initialized
assert controller.is_initialized()


# Simulate the system
u_online = []
y_online = []
r_online = [0.5] * 40 + [10] * 100 + [10] * 100 + [14]*150
for i in range(len(r_online) - r_len):
    r = r_online[i: i + r_len]
    u = controller.apply(r)[0]
    y = system.apply(u)
    controller.update(u, y)
    u_online.append(u)
    y_online.append(y)
    r_online.append(r)

# Plot the results
plt.plot(u_online, label="input")
plt.plot(y_online, label="output")
plt.plot(r_online[:len(y_online)], label="target", color="black", linestyle="--")
plt.legend()
plt.show()