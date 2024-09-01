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
    Simulates the rocket flight based on input parameters, accounting for atmospheric density and gravity.

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
        altitude = z  # Altitude is directly the z-axis value
        density = atmospheric_density(altitude)
        v = np.sqrt(vx**2 + vy**2 + vz**2)
        drag = 0.5 * density * drag_coefficient * cross_sectional_area * v**2

        # Thrust is applied only during the burn time
        if t <= burn_time:
            ax = (thrust / mass) - (drag / mass)
            az = (thrust / mass) - g - (drag / mass)
        else:
            ax = -drag / mass
            az = -g - (drag / mass)

        ay = 0  # Assuming no lateral forces affecting y direction

        return [vx, ax, vy, ay, vz, az]

    # Initial conditions: [x0, vx0, y0, vy0, z0, vz0]
    initial_conditions = [0, 0, 0, 0, initial_height, 0]

    # Time array
    t_span = (0, 500)  # Simulate for 500 seconds
    t_eval = np.linspace(t_span[0], t_span[1], 1000)

    # Solve ODEs
    solution = solve_ivp(equations, t_span, initial_conditions, t_eval=t_eval)

    # Filter out points where the rocket would be below ground level (z < 0)
    z_filtered = np.maximum(solution.y[4], 0)

    thrust_active = solution.t <= burn_time

    return {
        "time": solution.t,
        "x": solution.y[0],
        "y": solution.y[2],
        "z": z_filtered,
        "vx": solution.y[1],
        "vy": solution.y[3],
        "vz": solution.y[5],
        "thrust_active": thrust_active
    }

def plot_2d_path(data):
    """
    Plots the 2D flight path with gradient color indicating thrust status and marks the launch and impact points.

    Args:
        data (dict): Dictionary containing time, x, y, z positions, and thrust status.
    """
    plt.figure(figsize=(10, 5))
    
    for i in range(len(data["x"]) - 1):
        color = plt.cm.autumn(data["thrust_active"][i])  # Red to yellow-white gradient
        plt.plot(data["x"][i:i+2], data["z"][i:i+2], color=color, lw=2)  # Plot x vs z for altitude

    # Mark the launch point
    plt.scatter(data["x"][0], data["z"][0], color='blue', label='Launch Point', zorder=5)
    
    # Mark the impact point (last point where z is zero or above)
    impact_index = np.where(data["z"] == 0)[0][-1]
    plt.scatter(data["x"][impact_index], data["z"][impact_index], color='red', label='Impact Point', zorder=5)

    plt.xlabel('Horizontal Distance (m)')
    plt.ylabel('Altitude (m)')
    plt.title('Rocket Flight Path in 2D')
    plt.axhline(0, color='black', lw=1, ls='--')  # Ground level
    plt.ylim(0, None)  # Ensure no plotting below ground level
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)

def plot_3d_path(data):
    """
    Plots the 3D flight path with gradient color indicating thrust status and marks the launch and impact points.

    Args:
        data (dict): Dictionary containing time, x, y, z positions, and thrust status.
    """
    colors = ['rgba(255,0,0,{})'.format(t) if thrust else 'rgba(255,255,0,{})'.format(1-t) 
              for t, thrust in zip(np.linspace(0, 1, len(data["x"])), data["thrust_active"])]
    
    trace_path = go.Scatter3d(
        x=data["x"],
        y=data["y"],
        z=data["z"],
        mode='lines',
        line=dict(color=colors, width=5)
    )

    trace_launch = go.Scatter3d(
        x=[data["x"][0]],
        y=[data["y"][0]],
        z=[data["z"][0]],
        mode='markers',
        marker=dict(color='blue', size=5),
        name='Launch Point'
    )

    impact_index = np.where(data["z"] == 0)[0][-1]
    trace_impact = go.Scatter3d(
        x=[data["x"][impact_index]],
        y=[data["y"][impact_index]],
        z=[data["z"][impact_index]],
        mode='markers',
        marker=dict(color='red', size=5),
        name='Impact Point'
    )

    layout = go.Layout(
        title='Rocket Flight Path in 3D',
        scene=dict(
            xaxis_title='X Axis (m)',
            yaxis_title='Y Axis (m)',
            zaxis=dict(title='Altitude (m)', range=[0, max(data["z"]) + 100]),  # Ensure no plotting below ground level
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig = go.Figure(data=[trace_path, trace_launch, trace_impact], layout=layout)
    st.plotly_chart(fig)

def rocket_simulation_page():
    st.title("Rocket Flight Simulation")

    # User inputs for the rocket parameters
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