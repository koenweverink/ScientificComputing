import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from matplotlib.animation import FuncAnimation, PillowWriter

# Grid size
n, m = 10, 10
N = n * m

# Create the Laplacian matrix for a 10x10 grid
def create_reflective_laplacian(n, m):
    N = n * m
    M = np.zeros((N, N))
    for i in range(N):
        M[i, i] = -4
        row, col = divmod(i, m)
        if col != 0:
            M[i, i - 1] = 1
        if col != (m - 1):
            M[i, i + 1] = 1
        if row != 0:
            M[i, i - m] = 1
        if row != (n - 1):
            M[i, i + m] = 1
    return M

# Create the matrix
L = create_reflective_laplacian(n, m)

# Compute the eigenvalues and eigenvectors
eigvals, eigvecs = eigh(L)

# Constants for the T(t) function, adjust as necessary
A, B, c = 1, 1, 1  # Example values, can be adjusted

# Time-dependent function T(t)
def T(t, lam):
    return A * np.cos(c * lam * t) + B * np.sin(c * lam * t)

# Number of modes and time points to simulate
num_modes = 4
time_points = 100
t = np.linspace(0, 10, time_points)  # 0 to 10 seconds

# Animation setup
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
modes = [eigvecs[:, i].reshape(n, m) for i in range(num_modes)]
ims = []

for i in range(num_modes):
    # Initial plot of the modes
    img = axes[i // 2, i % 2].imshow(modes[i], animated=True, extent=[0, m, 0, n], vmin=-2, vmax=2, cmap='viridis')
    axes[i // 2, i % 2].set_title(f'Mode {i+1}')
    ims.append([img])  # Add the initial image to the list


# Update function for animation
def update_fig(frame):
    for i in range(num_modes):
        updated_mode = modes[i] * T(t[frame], eigvals[i])
        ims[i][0].set_array(updated_mode)
    return [im for sublist in ims for im in sublist]


ani = FuncAnimation(fig, update_fig, frames=len(t), interval=50, blit=True)

# Save the animation as a GIF using Pillow
writer = PillowWriter(fps=20)  # Adjust fps as needed for smoother animation
ani.save('time_dependent_modes.gif', writer=writer)
