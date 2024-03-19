import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
grid_size = 50
time_step = 0.1
total_time = 4
timesteps = np.arange(0, total_time, time_step)

# Define your spatial grid (adjust as necessary)
x, y = np.meshgrid(np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size))

# Sample eigenmodes and eigenvalues (replace with your actual solutions)
# For example, we use simple sine functions as placeholders
eigenmodes = [
    np.sin(np.pi * x) * np.sin(np.pi * y),   # First mode
    np.sin(2 * np.pi * x) * np.sin(np.pi * y),  # Second mode
    np.sin(np.pi * x) * np.sin(2 * np.pi * y)   # Third mode
]
eigenvalues = [np.pi**2, (2*np.pi)**2, (2*np.pi)**2]  # Corresponding eigenvalues

# Initialize figure with subplots
fig, axes = plt.subplots(1, len(eigenmodes), figsize=(15, 5))

# Make sure axes is an array for consistency
if not isinstance(axes, np.ndarray):
    axes = np.array([axes])

# Create a contour plot for each mode
contours = []
for ax, mode in zip(axes, eigenmodes):
    contour = ax.contourf(x, y, mode, 100, cmap='viridis')
    contours.append(contour)

# Update function for the animation
def animate(t):
    for ax, contour, mode, value in zip(axes, contours, eigenmodes, eigenvalues):
        z = mode * np.sin(np.sqrt(value) * t)  # Time-dependent solution
        for c in contour.collections:
            c.remove()  # Remove old contours
        ax.contourf(x, y, z, 100, cmap='viridis')

# Create animation
ani = FuncAnimation(fig, animate, frames=timesteps, repeat=True)



# Display the animation
plt.show()

