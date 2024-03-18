# %%
import numpy as np
from scipy.linalg import eigh, eig
from scipy.sparse import diags
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
import time

# %%
def create_reflective_laplacian(n, m):
    N = n * m  # Total number of points in the grid
    M = np.zeros((N, N))  # Initialize matrix with zeros

    for i in range(N):
        # Default value for cells
        M[i, i] = -4
        
        # Calculate row and column in the grid
        row, col = divmod(i, m)

        # Reflective boundary adjustments
        # if col == 0 or col == (m - 1):  # Left or right edge
        #     M[i, i] += 1
        # if row == 0 or row == (n - 1):  # Top or bottom edge
        #     M[i, i] += 1

        # Set connections for adjacent cells, considering the grid layout
        if col != 0:  # Not on the left edge
            M[i, i - 1] = 1  # Left neighbor
        if col != (m - 1):  # Not on the right edge
            M[i, i + 1] = 1  # Right neighbor
        if row != 0:  # Not on the top edge
            M[i, i - m] = 1  # Top neighbor
        if row != (n - 1):  # Not on the bottom edge
            M[i, i + m] = 1  # Bottom neighbor

    return M

# Example usage
n, m = 4, 4  # Change these to create a different size grid
L = create_reflective_laplacian(n, m)
print(L)

# %%
def create_circular_laplacian(n, m, radius):
    # Total points
    N = n * m
    # Center of the circle, adjusted for the rectangular grid
    center = np.array([n / 2 - 0.5, m / 2 - 0.5])
    # Initialize the matrix
    M = np.zeros((N, N))

    # Identify points inside the circle
    inside_circle = []
    for i in range(n):
        for j in range(m):
            if (np.array([i, j]) - center).dot(np.array([i, j]) - center) <= radius**2:
                inside_circle.append(i * m + j)

    # Now build the Laplacian only for these points
    for idx in inside_circle:
        x, y = divmod(idx, m)
        M[idx, idx] = -4
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < m and nx * m + ny in inside_circle:
                M[idx, nx * m + ny] = 1

    # Filter out the unused rows and columns
    M = M[np.ix_(inside_circle, inside_circle)]

    return M

# Example usage
n, m = 5, 6  # Change these for a different size grid
radius = 2  # Adjust radius as needed
L_circle_corrected = create_circular_laplacian(n, m, radius)

# Check if the matrix is square
if L_circle_corrected.shape[0] == L_circle_corrected.shape[1]:
    print("The matrix is square.")
    try:
        vals_circle, vecs_circle = eigh(L_circle_corrected)
        # Continue with your analysis or plotting
    except ValueError as e:
        print("Error in eigenvalue computation:", e)
else:
    print("The matrix is not square, check the Laplacian construction.")


# %%
# Square grid (already done)
n, m = 20, 20 
L_square = create_reflective_laplacian(n, m)

# Rectangle grid
n, m = 10, 20  # Change these for a different size grid
L_rect = create_reflective_laplacian(n, m)

# Circular grid
n, m = 5, 6  # Change these for a different size grid
radius = 2  # Make sure L and n are defined
L_circle = create_circular_laplacian(n, m, radius)

# %%
def solve_eigenproblem(L_square):
    # Solve the eigenvalue problem
    eigenvalues, eigenvectors = eig(L_square)

    # Convert eigenvalues to real numbers
    eigenvalues = np.real(eigenvalues)

    # Create the dictionary mapping eigenvalues to eigenvectors
    eigen_dict = {}
    for i, val in enumerate(eigenvalues):
        vec = eigenvectors[:, i]
        # Convert complex eigenvectors to real and make them lists for easy representation
        vec_list = np.real(np.abs(vec)).tolist()
        # Add to dictionary
        eigen_dict[val] = vec_list

    return eigen_dict

# %%
def plot_eigenvectors(eigen_dict, nrows=None, ncols=None, num_eigenvectors=5, target_eigenvalue=0):
    # Sort eigenvalues based on closeness to the target_eigenvalue, select the closest num_eigenvectors
    sorted_eigenvals = sorted(eigen_dict.keys(), key=lambda x: abs(x - target_eigenvalue))
    closest_eigenvals = sorted_eigenvals[:num_eigenvectors]

    # Set up the plot
    fig, axes = plt.subplots(1, num_eigenvectors, figsize=(5 * num_eigenvectors, 5))
    if num_eigenvectors == 1:
        axes = [axes]  # Ensure axes is iterable for a single subplot

    # Plot each of the closest eigenvectors as a heatmap
    for ax, eigenvalue in zip(axes, closest_eigenvals):
        eigenvector = eigen_dict[eigenvalue]
        # Reshape eigenvector to a rectangle or square array
        if nrows is not None and ncols is not None:
            matrix = np.reshape(eigenvector, (nrows, ncols))
        else:
            # Fallback to square shape if dimensions are not specified
            side_length = int(np.sqrt(len(eigenvector)))
            matrix = np.reshape(eigenvector, (side_length, side_length))
        
        # Plot the heatmap
        c = ax.imshow(matrix, cmap='viridis', aspect='auto')
        fig.colorbar(c, ax=ax)
        ax.set_title(f'Eigenvalue: {eigenvalue:.2e}')  # Using scientific notation for clarity

    plt.tight_layout()
    plt.show()


# %%
eigen_dict = solve_eigenproblem(L_square)
print(len(eigen_dict))
plot_eigenvectors(eigen_dict)

# %%
n, m = 10, 20  # Change these for a different size grid
eigen_dict = solve_eigenproblem(L_rect)
plot_eigenvectors(eigen_dict, nrows=n, ncols=m)

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

def plot_circular_eigenvectors(vals_circle, vecs_circle, n, m, radius, num_eigenvectors=5):
    # Calculate the positions inside the circle to properly embed eigenvectors later
    center = np.array([n / 2 - 0.5, m / 2 - 0.5])
    inside_circle = [i * m + j for i in range(n) for j in range(m) 
                     if np.linalg.norm(np.array([i, j]) - center) <= radius]

    # Sort eigenvalues and pick the indices of the closest to zero
    indices_closest = np.argsort(np.abs(vals_circle))[:num_eigenvectors]

    # Set up the plot
    fig, axes = plt.subplots(1, num_eigenvectors, figsize=(5 * num_eigenvectors, 5))
    if num_eigenvectors == 1:
        axes = [axes]  # Make sure axes is iterable

    # Plot the specified number of closest eigenvectors
    for i, idx in enumerate(indices_closest):
        # Embed the eigenvector back into the full grid
        full_grid = embed_eigenvector_in_grid(np.abs(vecs_circle[:, idx]), n, m, inside_circle)
        
        # Plot the heatmap
        c = axes[i].imshow(full_grid, cmap='viridis', extent=[0, m, 0, n], origin='lower')
        fig.colorbar(c, ax=axes[i])
        axes[i].set_title(f'Eigenvalue: {vals_circle[idx]:.2f}')
        axes[i].axis('on')

    plt.tight_layout()
    plt.show()

# Usage example
n, m = 20, 26  # Grid size
radius = 10  # Circle radius
L_circle_corrected = create_circular_laplacian(n, m, radius)
vals_circle, vecs_circle = eigh(L_circle_corrected)  # Compute eigenvalues and eigenvectors

plot_circular_eigenvectors(vals_circle, vecs_circle, n, m, radius)  # Adjust num_eigenvectors if needed


# %% [markdown]
# In the provided code, eigh() from scipy.linalg was used to solve the eigenvalue problem for the Laplacian matrix representing the reflective boundary conditions. The choice of eigh() over other options like eig() or eigs() was made because eigh() is specifically designed to efficiently solve Hermitian eigenvalue problems for symmetric matrices.
# 
# Since the Laplacian matrix representing the reflective boundary conditions is symmetric due to the nature of the problem, using eigh() is appropriate and efficient. This function is optimized for symmetric matrices, leading to faster computation times compared to other methods when dealing with such matrices. Therefore, eigh() was selected for its efficiency and suitability for the given problem.

# %%
# Example usage
n, m = 20, 20  # Change these for a different size grid
L = create_reflective_laplacian(n, m)

# Solve the eigenvalue problem using scipy.linalg.eig()
start_time = time.time()
eigenvalues_eig, eigenvectors_eig = eig(L)
end_time = time.time()
eig_time = end_time - start_time

# Solve the eigenvalue problem using scipy.linalg.eigh()
start_time = time.time()
eigenvalues_eigh, eigenvectors_eigh = eigh(L)
end_time = time.time()
eigh_time = end_time - start_time


# Solve the eigenvalue problem using scipy.sparse.linalg.eigs() for sparse matrices
start_time = time.time()
eigenvalues_eigs, eigenvectors_eigs = eigs(L)
end_time = time.time()
eigs_time = end_time - start_time

# %%
# Computation times from your results
method_names = ['eig()', 'eigh()', 'eigs()']
times = [eig_time, eigh_time, eigs_time]

# Create a bar chart
plt.figure(figsize=[10,6])
bars = plt.bar(method_names, times, color=['blue', 'green', 'red'])

# Add the exact computation time above each bar for clarity
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', va='bottom')  # Adjust the format as needed

# Set chart title and labels
plt.title('Comparison of Eigenvalue Computation Times')
plt.ylabel('Time (seconds)')
plt.xlabel('Method')

# Show the plot
plt.show()


# %%



