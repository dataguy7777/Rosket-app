import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from scipy.integrate import solve_ivp

def simulate_flight(mass, thrust, drag_coefficient, cross_sectional_area, burn_time):
    """
    Simulates the rocket flight based on input parameters.

    Args:
        mass (float): Mass of the rocket in kg.
        thrust (float): Thrust force in N.
        drag_coefficient (float): Drag coefficient (dimensionless).
        cross_sectional_area (float): Cross-sectional area of the rocket in m^2.
        burn_time (float): Time for which thrust is applied in seconds.

    Returns:
        dict: Contains time, x, y, z positions, and velocities.
    """
    g = 9.81  # m/s^2, gravitational acceleration

    def equations(t, y):
        x, vx, y, vy, z, vz = y
        v = np.sqrt(vx**2 + vy**2 + vz**2)
        drag = 0.5 * drag_coefficient * cross_sectional_area * v**2

        # Thrust is applied only during the burn time
        ax = (thrust / mass) - (drag / mass) if t <= burn_time else -drag / mass
        ay = -g
        az = 0  # No lateral forces in this simple model

        return [vx, ax, vy, ay, vz, az]

    # Initial conditions: [x0, vx0, y0, vy0, z0, vz0]
    initial_conditions = [0, 0, 0, 0, 0, 0]

    # Time array
    t_span = (0, 50)  # Simulate for 50 seconds
    t_eval = np.linspace(t_span[0], t_span[1], 500)

    # Solve ODEs
    solution = solve_ivp(equations, t_span, initial_conditions, t_eval=t_eval)

    return {
        "time": solution.t,
        "x": solution.y[0],
        "y": solution.y[2],
        "z": solution.y[4],
        "vx": solution.y[1],
        "vy": solution.y[3],
        "vz": solution.y[5],
    }

def plot_2d_path(data):
    """
    Plots the 2D flight path.

    Args:
        data (dict): Dictionary containing time, x, y, and z positions.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(data["x"], data["y"], label="2D Flight Path")
    plt.xlabel('Horizontal Distance (m)')
    plt.ylabel('Vertical Distance (m)')
    plt.title('Rocket Flight Path in 2D')
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)

def plot_3d_path(data):
    """
    Plots the 3D flight path.

    Args:
        data (dict): Dictionary containing time, x, y, and z positions.
    """
    fig = go.Figure(data=[go.Scatter3d(
        x=data["x"],
        y=data["z"],
        z=data["y"],
        mode='lines',
        line=dict(color='blue', width=5)
    )])

    fig.update_layout(
        title='Rocket Flight Path in 3D',
        scene=dict(
            xaxis_title='X Axis (m)',
            yaxis_title='Z Axis (m)',
            zaxis_title='Y Axis (m)'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

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

    # Simulate the flight
    st.header("Flight Simulation")
    flight_data = simulate_flight(mass, thrust, drag_coefficient, cross_sectional_area, burn_time)

    # Plot 2D path
    st.subheader("2D Flight Path")
    plot_2d_path(flight_data)

    # Plot 3D path
    st.subheader("3D Flight Path")
    plot_3d_path(flight_data)

if __name__ == "__main__":
    main()