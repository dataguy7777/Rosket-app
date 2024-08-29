import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from scipy.integrate import solve_ivp

def atmospheric_density(altitude):
    """
    Returns the atmospheric density at a given altitude based on the U.S. Standard Atmosphere model.

    Args:
        altitude (float): Altitude in meters.

    Returns:
        float: Air density in kg/m^3.
    """
    if altitude < 11000:
        return 1.225 * (1 - 0.0000225577 * altitude) ** 4.2561
    elif altitude < 20000:
        return 0.36391 * np.exp(-0.0001577 * (altitude - 11000))
    elif altitude < 32000:
        return 0.08803 * (1 + 0.0000341632 * (altitude - 20000)) ** -35.1632
    elif altitude < 47000:
        return 0.01322 * (1 - 0.0000511750 * (altitude - 32000)) ** 12.2011
    elif altitude < 51000:
        return 0.00143 * np.exp(-0.000147 * (altitude - 47000))
    elif altitude < 71000:
        return 0.00086 * (1 - 0.0000666270 * (altitude - 51000)) ** 12.2011
    elif altitude < 84852:
        return 0.000064 * np.exp(-0.0002233 * (altitude - 71000))
    else:
        return 0

def simulate_flight(mass, thrust, drag_coefficient, cross_sectional_area, burn_time, initial_height):
    """
    Simulates the rocket flight based on input parameters, accounting for atmospheric density.

    Args:
        mass (float): Mass of the rocket in kg.
        thrust (float): Thrust force in N.
        drag_coefficient (float): Drag coefficient (dimensionless).
        cross_sectional_area (float): Cross-sectional area of the rocket in m^2.
        burn_time (float): Time for which thrust is applied in seconds.
        initial_height (float): Initial height above sea level in meters.

    Returns:
        dict: Contains time, x, y, z positions, velocities, and thrust status.
    """
    g = 9.81  # m/s^2, gravitational acceleration

    def equations(t, y):
        x, vx, y_pos, vy, z, vz = y
        altitude = y_pos + initial_height  # Calculate current altitude
        density = atmospheric_density(altitude)
        v = np.sqrt(vx**2 + vy**2 + vz**2)
        drag = 0.5 * density * drag_coefficient * cross_sectional_area * v**2

        # Thrust is applied only during the burn time
        if t <= burn_time:
            ax = (thrust / mass) - (drag / mass)
        else:
            ax = -drag / mass

        ay = -g
        az = 0  # No lateral forces in this simple model

        return [vx, ax, vy, ay, vz, az]

    # Initial conditions: [x0, vx0, y0, vy0, z0, vz0]
    initial_conditions = [0, 0, initial_height, 0, 0, 0]

    # Time array
    t_span = (0, 500)  # Simulate for 500 seconds
    t_eval = np.linspace(t_span[0], t_span[1], 1000)

    # Solve ODEs
    solution = solve_ivp(equations, t_span, initial_conditions, t_eval=t_eval)

    thrust_active = solution.t <= burn_time

    return {
        "time": solution.t,
        "x": solution.y[0],
        "y": solution.y[2],
        "z": solution.y[4],
        "vx": solution.y[1],
        "vy": solution.y[3],
        "vz": solution.y[5],
        "thrust_active": thrust_active
    }

def plot_2d_path(data):
    """
    Plots the 2D flight path with gradient color indicating thrust status.

    Args:
        data (dict): Dictionary containing time, x, y, z positions, and thrust status.
    """
    plt.figure(figsize=(10, 5))
    
    for i in range(len(data["x"]) - 1):
        color = plt.cm.autumn(data["thrust_active"][i])  # Red to yellow-white gradient
        plt.plot(data["x"][i:i+2], data["y"][i:i+2], color=color, lw=2)

    plt.xlabel('Horizontal Distance (m)')
    plt.ylabel('Altitude (m)')
    plt.title('Rocket Flight Path in 2D')
    plt.grid(True)
    st.pyplot(plt)

def plot_3d_path(data):
    """
    Plots the 3D flight path with gradient color indicating thrust status.

    Args:
        data (dict): Dictionary containing time, x, y, z positions, and thrust status.
    """
    colors = ['rgba(255,0,0,{})'.format(t) if thrust else 'rgba(255,255,0,{})'.format(1-t) 
              for t, thrust in zip(np.linspace(0, 1, len(data["x"])), data["thrust_active"])]
    
    trace = go.Scatter3d(
        x=data["x"],
        y=data["z"],
        z=data["y"],
        mode='lines',
        line=dict(color=colors, width=5)
    )

    layout = go.Layout(
        title='Rocket Flight Path in 3D',
        scene=dict(
            xaxis_title='X Axis (m)',
            yaxis_title='Z Axis (m)',
            zaxis_title='Altitude (m)'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig = go.Figure(data=[trace], layout=layout)
    st.plotly_chart(fig)

def main():
    st.title("Rocket Flight Path Simulation")

    # Sidebar inputs for the rocket parameters
    st.sidebar.header("Rocket Parameters")
    mass = st.sidebar.number_input("Mass (kg)", min_value=1.0, max_value=5000.0, value=1000.0)
    thrust = st.sidebar.number_input("Thrust (N)", min_value=0.0, max_value=500000.0, value=150000.0)
    drag_coefficient = st.sidebar.number_input("Drag Coefficient (dimensionless)", min_value=0.0, max_value=1.0, value=0.5)
    cross_sectional_area = st.sidebar.number_input("Cross-Sectional Area (m^2)", min_value=0.0, max_value=10.0, value=1.0)
    burn_time = st.sidebar.number_input("Burn Time (s)", min_value=0.0, max_value=100.0, value=10.0)
    initial_height = st.sidebar.number_input("Initial Height (m)", min_value=0.0, max_value=100000.0, value=0.0)

    # Simulate the flight
    st.header("Flight Simulation")
    flight_data = simulate_flight(mass, thrust, drag_coefficient, cross_sectional_area, burn_time, initial_height)

    # Plot 2D path
    st.subheader("2D Flight Path")
    plot_2d_path(flight_data)

    # Plot 3D path
    st.subheader("3D Flight Path")
    plot_3d_path(flight_data)

if __name__ == "__main__":
    main()