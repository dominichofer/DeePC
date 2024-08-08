import numpy as np
import math

from scipy.signal import chirp
import matplotlib.pyplot as plt
from deepc import Controller, DescreteLTI, clamp
import numpy as np
import math
import matplotlib.pyplot as plt
from deepc import Controller, DescreteLTI, clamp

# Define a 3-room temperature control system
system = DescreteLTI(
    A=[[0.5, 0.1, 0], 
       [0.1, 0.5, 0.1], 
       [0, 0.1, 0.5]],
    B=[[0.1, 0, 0], 
       [0, 0.1, 0], 
       [0, 0, 0.1]],
    C=[[1, 0, 0], 
       [0, 1, 0], 
       [0, 0, 1]],
    D=[[0, 0, 0], 
       [0, 0, 0], 
       [0, 0, 0]],
    x_ini=[0, 0, 0],
)

# Gather offline data
N = 1000

# Frequency sweep input (using chirp as in your original example)
t = np.linspace(0, 1, N)
u_chirp = chirp(t, f0=0, f1=1000, t1=1, method='linear')

# Generate input sequences for the 3-room system
u_d = [[i * math.sin(i * i / 100) for i in range(N)] for _ in range(3)]
u_d = np.array(u_d).T  # Transpose to shape (1000, 3)

# Ensure the shape is correct
print(f"u_d shape: {u_d.shape}")  # Should print (1000, 3)

# Applying the input to the system
y_d = system.apply_multiple(u_d)

# Define how many steps the controller should look back
T_ini = 9

# Define how many steps the controller should look forward
r_len = 7

# Define the controller
constraint = lambda u: clamp(u, 0, 5)
controller = Controller(u_d, y_d, T_ini, r_len, control_constrain_fkt=constraint)

# Reset the system to separate the offline data from the online data
system.set_state([0, 0, 0])

# Warm up the controller
for i in range(T_ini):
    u = [0, 0, 0]
    y = system.apply(u)
    controller.update(u, y)

# Check that the controller is initialized
assert controller.is_initialized()

# Generate noise
offset = -0.2

# Simulate the system
u_online = []
y_online = []
r_online = [[0, 0, 0]] * 20 + [[10, 10, 10]] * 200 + [[7, 7, 7]] * 100

for i in range(len(r_online) - r_len):
    noise_with_offset = np.random.random(3) + offset
    noise_with_offset = np.clip(noise_with_offset, -1, 1)

    r = r_online[i: i + r_len]
    u = controller.control(r)[0]
    y = system.apply(u) + noise_with_offset

    controller.update(u, y)
    u_online.append(u)
    y_online.append(y)
    r_online.append(r)

# Plot the results
y_online_flat = [y[0] for y in y_online]  # Just plotting the first room's temperature for simplicity
plt.plot([u[0] for u in u_online], label="input")
plt.plot(y_online_flat, label="output")
plt.plot([r[0] for r in r_online[:len(y_online)]], label="target", color="black", linestyle="--")
plt.legend()
plt.show()
