# V3-3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.linalg import toeplitz
from scipy.sparse import lil_matrix, csr_matrix, eye as speye
from scipy.sparse import kron

# Function to create nucleus
def nucleus(Nx, Ny, seed):
    phi = np.zeros((Nx, Ny))
    tempr = np.zeros((Nx, Ny))
    for i in range(Nx):
        for j in range(Ny):
            if ((i - Nx / 2) ** 2 + (j - Ny / 2) ** 2) < seed:
                phi[i, j] = 1.0
    return phi, tempr

# Function to create flat nucleus
def flatnucleus(Nx, Ny, seed):
    phi = np.zeros((Nx, Ny))
    tempr = np.zeros((Nx, Ny))
    for i in range(Nx):
        for j in range(Ny):
            if j < seed:
                phi[i, j] = 1.0
    return phi, tempr

# Function to create sine nucleus
def sinenucleus(Nx, Ny, seed, a, lamda, theta0, const):
    phi = np.zeros((Nx, Ny))
    tempr = np.zeros((Nx, Ny))
    
    # Generate y values from 0 to Ny
    j = np.linspace(0, Ny, Ny)
    
    # Calculate x values for the sine wave
    x = a * np.sin((2 * np.pi * j / lamda) + theta0) + const
    
    # Fill the grid based on the sine wave pattern
    for i in range(Nx):
        for j in range(Ny):
            if j < x[i % len(x)]:  # Ensuring that we correctly wrap around the x values
                phi[i, j] = 1.0

    return phi, tempr

# Function to compute laplacian
def laplacian(nx, ny, dx, dy):
    r = np.zeros(nx)
    r[0:2] = [2, -1]
    T = toeplitz(r)
    E = speye(nx, format='csr')
    grad = -(kron(T, E, format='csr') + kron(E, T, format='csr')).tolil()

    # For periodic boundaries
    for i in range(nx):
        ii = i * nx
        jj = ii + nx - 1
        grad[ii, jj] = 1.0
        grad[jj, ii] = 1.0
        kk = nx * ny - nx + i
        grad[i, kk] = 1.0
        grad[kk, i] = 1.0

    return grad.tocsr() / (dx * dy)

# Function to compute gradient
def gradient_mat(matx, Nx, Ny, dx, dy):
    matdx, matdy = np.gradient(matx)

    # For periodic boundaries
    matdx[:, 0] = 0.5 * (matx[:, 1] - matx[:, -1])
    matdx[:, -1] = 0.5 * (matx[:, 0] - matx[:, -2])
    matdy[0, :] = 0.5 * (matx[1, :] - matx[-1, :])
    matdy[-1, :] = 0.5 * (matx[0, :] - matx[-2, :])

    matdx *= 2.0 / dx
    matdy *= 2.0 / dy

    return matdx, matdy

# Function to convert vector to matrix
def vec2matx(V, N):
    R = int(np.ceil(len(V) / N))
    return np.reshape(V, (R, N))

# Function to update the animation
def update(frame, phi, tempr, Nx, Ny, dx, dy, dtime, a, lamda, const, tau, epsilonb, delta, aniso, theta0, alpha, gamma, teq, kappa, laplacian_matrix, img, constepsder):
    phi_old = phi.copy()

    # Calculate the laplacians and epsilon
    phi2 = phi.T.flatten()
    lap_phi2 = laplacian_matrix @ phi2
    lap_phi = vec2matx(lap_phi2, Nx).T

    temp2 = tempr.T.flatten()
    lap_tempx = laplacian_matrix @ temp2
    lap_tempr = vec2matx(lap_tempx, Nx).T

    # Gradients of phi
    phidy, phidx = gradient_mat(phi, Nx, Ny, dx, dy)

    # Calculate angle
    theta = np.arctan2(phidy, phidx)

    # Epsilon and its derivative
    epsilon = epsilonb * (1.0 + delta * np.cos(aniso * (theta - theta0)))
    epsilon_deriv = -constepsder * epsilonb * aniso * delta * np.sin(aniso * (theta - theta0))

    # First term
    dummyx = epsilon * epsilon_deriv * phidx
    term1, _ = gradient_mat(dummyx, Nx, Ny, dx, dy)

    # Second term
    dummyy = -epsilon * epsilon_deriv * phidy
    _, term2 = gradient_mat(dummyy, Nx, Ny, dx, dy)

    # Factor m
    m = (alpha / np.pi) * np.arctan(gamma * (teq - tempr))

    # Time integration
    phi += (dtime / tau) * (term1 + term2 + epsilon ** 2 * lap_phi + phi_old * (1.0 - phi_old) * (phi_old - 0.5 + m))

    # Evolve temperature
    tempr += dtime * lap_tempr + kappa * (phi - phi_old)

    # Update the image
    img.set_array(phi)
    return img,

# Main function to simulate dendritic growth and create animation
def dendritic_growth():
    # Simulation parameters
    Nx = 300
    Ny = 300
    dx = 0.03
    dy = 0.03
    nstep = 4000
    dtime = 1.e-4
    a = 5.0
    lamda = 20.0
    const = 6.0

    # Material specific parameters
    tau = 0.0003
    epsilonb = 0.01
    delta = 0.02
    aniso = 4.0
    theta0 = 0.2
    alpha = 0.9
    gamma = 10.0
    teq = 1.0
    kappa = 1.8
    seed = 12.0
    constepsder = 0

    # Initialize and introduce initial nuclei
    #phi, tempr = nucleus(Nx, Ny, seed)

    # Initialize and introduce initial flat nuclei
    #phi, tempr = flatnucleus(Nx, Ny, seed)
    
    # Initialize and introduce initial sine nuclei
    phi, tempr = sinenucleus(Nx, Ny, seed, a, lamda, theta0, const)
    
    # Laplacian template
    laplacian_matrix = laplacian(Nx, Ny, dx, dy)

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(6, 6))  # Ensures the figure is a square
    img = ax.imshow(phi, cmap='bwr', animated=True)
    
    # Force the plot to be square
    ax.set_aspect('equal', adjustable='box')  # Equal aspect ratio
    ax.set_xlim(0, Nx)
    ax.set_ylim(0, Ny)
    
    # Hide the axis labels and ticks for a cleaner look
    ax.set_xticks([])
    ax.set_yticks([])

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=nstep, fargs=(phi, tempr, Nx, Ny, dx, dy, dtime, a, lamda, const, tau, epsilonb, delta, aniso, theta0, alpha, gamma, teq, kappa, laplacian_matrix, img, constepsder), interval=1, blit=True)

    # Save animation as MP4
    ani.save('V3_3.mp4', writer='ffmpeg', fps=30)

    plt.show()

if __name__ == "__main__":
    dendritic_growth()
