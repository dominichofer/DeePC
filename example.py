import numpy as np

from scipy.signal import chirp
import math
import matplotlib.pyplot as plt
from deepc import Controller, DescreteLTI, clamp




# Define a system
system = DescreteLTI(
    A=[[1.9154297228892199, -0.9159698592594919], [1.0, 0.0]],
    B=[[0.0024407501942859677], [0.0]],
    C=[[1, 0]],
    D=[[0]],
    x_ini=[1, 1],
)

# Gather offline data
N = 1000

#frequency sweep input 
t = np.linspace(0, 1, N)
u_chirp = chirp(t, f0=0, f1=1000, t1=1, method='linear')
u_d = u_chirp

# and applying it to the system
y_d = system.apply_multiple(u_d)

# Define how many steps the controller should look back
# to grasp the current state of the system
T_ini = 10

# Define how many steps the controller should look forward
r_len = 10

# Define the controller
constraint = lambda u: clamp(u, 0, 5)
controller = Controller(u_d, y_d, T_ini, r_len, control_constrain_fkt=constraint)

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


# Generate noise 
offset = -0.2

# Simulate the system
u_online = []
y_online = []
r_online = [0] * 20 + [10] * 200 + [7] * 100
for i in range(len(r_online) - r_len):
    noise_with_offset = np.random.random() + offset
    noise_with_offset = np.clip(noise_with_offset, -1, 1)

    r = r_online[i: i + r_len]
    u = controller.control(r)[0]
    y = system.apply(u) + noise_with_offset

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