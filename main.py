import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from disegno import DisegnaTu



# Define the navier_stokes function
def navier_stokes(U_init, V_init, dx, dy, p, nt, dt, contour):
    # Define fluid properties
    rho = 1.225  # Fluid density
    nu = 15*(10**-6)  # Kinematic viscosity

    # Initialize velocity arrays
    u = U_init.copy()
    v = V_init.copy()

    # Define arrays to store intermediate velocity values
    un = np.zeros_like(u)
    vn = np.zeros_like(v)

    # Iterate over time steps
    for n in range(nt):
        # Copy current velocity values to temporary arrays
        un[:] = u[:]
        vn[:] = v[:]

        # Compute velocity gradients
        dudx = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dx)
        dvdy = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2 * dy)

        # Compute pressure gradient
        dpdx = -(p[:, 1:] - p[:, :-1]) / dx
        dpdy = -(p[1:, :] - p[:-1, :]) / dy

        
        # Update velocity components
        for i in range(1, u.shape[0] - 1):
            for j in range(1, u.shape[1] - 1):
                if (i, j) not in contour:  # Apply zero-slip condition except along the contour
                    u[i, j] = un[i, j] \
                              - un[i, j] * (dt / dx) * (un[i, j] - un[i, j - 1]) \
                              - vn[i, j] * (dt / dy) * (un[i, j] - un[i - 1, j]) \
                              - dt / (2 * rho * dx) * (p[i, j + 1] - p[i, j - 1]) \
                              + nu * (  (dt / dx**2) * (un[i, j + 1] - 2*un[i, j] + un[i, j - 1]) \
                                      + (dt / dy**2) * (un[i + 1, j] - 2*un[i, j] + un[i - 1, j])  )
                else:  # Apply zero velocity (no-slip) condition along the contour
                    u[i, j] = 0

        for i in range(1, v.shape[0] - 1):
            for j in range(1, v.shape[1] - 1):
                if (i, j) not in contour:  # Apply zero-slip condition except along the contour
                    v[i, j] = vn[i, j] \
                              - un[i, j] * (dt / dx) * (vn[i, j] - vn[i, j - 1]) \
                              - vn[i, j] * (dt / dy) * (vn[i, j] - vn[i - 1, j]) \
                              - dt / (2 * rho * dy) * (p[i + 1, j] - p[i - 1, j]) \
                              + nu * (  (dt / dx**2) * (vn[i, j + 1] - 2*vn[i, j] + vn[i, j - 1]) \
                                      + (dt / dy**2) * (vn[i + 1, j] - 2*vn[i, j] + vn[i - 1, j])  )
                else:  # Apply zero velocity (no-slip) condition along the contour
                    v[i, j] = 0

        print(f"{(n/nt*100):.4}%")

        # Apply boundary conditions
        # For simplicity, let's assume no-slip boundary conditions on all boundaries
        u[0, :] = 0
        u[-1, :] = 0
        u[:, 0] = 0
        u[:, -1] = 0

        v[0, :] = 0
        v[-1, :] = 0
        v[:, 0] = 0
        v[:, -1] = 0

    
    return u, v




def integrate_streamlines(u, v, x0, y0_array, dt, num_steps, X, Y):
    streamlines = []

    # Iterate over each seed point along the left boundary
    for y0 in y0_array:
        # Initialize arrays to store streamline coordinates
        x_streamline = [x0]
        y_streamline = [y0]

        print(u[0, :][5])

        # Iterate over time steps
        for i in range(num_steps):
            # Interpolate velocity components at current position
            #u_interp = np.interp(\\ x_streamline, X[0, :])
            #v_interp = np.interp(v, y_streamline, Y[:, 0])

            # Compute new position using Euler's method
            x_new = x_streamline[i] + u * dt
            y_new = y_streamline[i] + v * dt

            # Append new position to streamline arrays
            x_streamline.append(x_new)
            y_streamline.append(y_new)

        # Store streamline coordinates
        streamlines.append((x_streamline, y_streamline))

    streamlines = np.array(streamlines)
    return streamlines






def detect_contours():
    # Draw the image
    DisegnaTu()

    # Load the image
    body_image = mpimg.imread('ala.jpeg')  # Replace with the path to your PNG file

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(body_image, cv2.COLOR_RGB2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray_image, 100, 200)  # Adjust the threshold values as needed

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour = contours[0].reshape(-1, 2)  # Reshape to (n_points, 2)

    # Plot the detected contours
    plt.plot(contour[:, 0], contour[:, 1], color='red')

    return contour




def define_conditions(domain_height, domain_width):
    
    # Define velocity magnitude (order of magnitude of m/s)
    velocity_magnitude = 10  # m/s

    # Define initial condition function
    def initial_condition(x, y):
        # Set initial velocity field
        u = velocity_magnitude * np.ones_like(x)
        v = np.zeros_like(y)
        return u, v

    # Create meshgrid for domain
    num_points_x = 200
    num_points_y = 300
    x = np.linspace(0, domain_width, num_points_x)
    y = np.linspace(0, domain_height, num_points_y)
    X, Y = np.meshgrid(x, y)

    # Apply initial condition
    U_init, V_init = initial_condition(X, Y)

    # Grid Spacing (assuming uniform grid spacing)
    dx = domain_width / (num_points_x - 1)
    dy = domain_height / (num_points_y - 1)

    # Pressure Field (initialize with zeros)
    p = np.zeros((num_points_y, num_points_x))

    # Number of Time Steps
    nt = 30 # Adjust as needed based on convergence and accuracy considerations
    # Define simulation parameters
    dt = 1  # Time step

    return U_init, V_init, X, Y, dx, dy, p, nt, dt




def plot_streamlines(streamlines):
    

    for i in range(streamlines.shape[0]):
        x_streamline , y_streamline = streamlines[i]  # Extract y coordinates
        plt.plot(x_streamline, y_streamline)

    plt.gca().invert_yaxis()
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.title('Final Velocity Field')
    plt.show()




def main():
    contour=detect_contours()
    # Define domain size in meters
    domain_width = 1500  # mm
    domain_height = 1000  # mm

    U_init, V_init, X, Y, dx, dy, p, nt, dt= define_conditions(domain_height, domain_width)
    # Solve Navier-Stokes equations
    u, v = navier_stokes(U_init, V_init, dx, dy, p, nt, dt, contour)

    number_of_streamlines = 10

    x0 = 0.0  # x-coordinate for the left boundary
    y0_array = np.linspace(0, domain_height, number_of_streamlines)  # Array of y-coordinates for seed points
    # Integrate streamlines
    streamlines = integrate_streamlines(u, v, x0, y0_array, dt, nt, X, Y)


    plot_streamlines(streamlines)

main()