import streamlit as st
import numpy as np
import plotly.graph_objs as go

def simulate_plane_and_missile(plane_params, missile_params, target_params):
    """
    Simulates the scenario where a plane detects, tracks, and potentially intercepts a target with a missile.
    
    Args:
        plane_params (dict): Parameters of the plane including radar, altitude, and direction.
        missile_params (dict): Parameters of the missile including speed, seeker, and intercept probability.
        target_params (dict): Parameters of the target including speed, cross section, altitude, and direction.
        
    Returns:
        dict: Contains the time series data for plane, missile, and target positions, and whether interception occurs.
    """
    # Unpacking parameters
    plane_speed = plane_params['speed']
    plane_altitude = plane_params['altitude']
    plane_direction = plane_params['direction']
    radar_range = plane_params['radar_range']

    missile_speed = missile_params['speed']
    seeker_prob = missile_params['seeker_prob']

    target_speed = target_params['speed']
    target_altitude = target_params['altitude']
    target_direction = target_params['direction']

    # Time settings
    time_step = 0.1  # seconds
    max_time = 300  # seconds
    time = np.arange(0, max_time, time_step)

    # Initialize positions
    plane_position = np.zeros((len(time), 3))
    missile_position = np.full((len(time), 3), np.nan)
    target_position = np.zeros((len(time), 3))

    # Initial positions
    plane_position[0] = np.array([0, 0, plane_altitude])
    missile_position[0] = np.array([0, 0, plane_altitude])  # Missile starts from the plane
    target_position[0] = np.array([10000, 0, target_altitude])  # Target starts 10 km away

    # Simulation loop
    for i in range(1, len(time)):
        # Update target position
        target_position[i, 0] = target_position[i-1, 0] + target_speed * time_step * np.cos(np.deg2rad(target_direction))
        target_position[i, 1] = target_position[i-1, 1] + target_speed * time_step * np.sin(np.deg2rad(target_direction))
        target_position[i, 2] = target_altitude

        # Update plane position
        plane_position[i, 0] = plane_position[i-1, 0] + plane_speed * time_step * np.cos(np.deg2rad(plane_direction))
        plane_position[i, 1] = plane_position[i-1, 1] + plane_speed * time_step * np.sin(np.deg2rad(plane_direction))
        plane_position[i, 2] = plane_altitude

        # Radar detection
        distance_to_target = np.linalg.norm(plane_position[i] - target_position[i])
        if distance_to_target <= radar_range:
            # Track target and launch missile
            missile_position[i] = plane_position[i]  # Missile launched
            break

    # Missile engagement
    for j in range(i, len(time)):
        if np.isnan(missile_position[j, 0]):
            break
        # Missile guidance (simple proportional navigation)
        missile_to_target_vector = target_position[j] - missile_position[j]
        missile_to_target_distance = np.linalg.norm(missile_to_target_vector)
        missile_direction = missile_to_target_vector / missile_to_target_distance

        # Update missile position
        missile_position[j] += missile_speed * time_step * missile_direction

        # Check for interception
        if missile_to_target_distance < 50:  # Assuming interception if within 50 meters
            interception_probability = seeker_prob * (50 / missile_to_target_distance)
            if np.random.rand() < interception_probability:
                return {
                    "plane_position": plane_position[:j+1],
                    "missile_position": missile_position[:j+1],
                    "target_position": target_position[:j+1],
                    "intercepted": True,
                    "interception_time": time[j]
                }

    # If missile misses
    return {
        "plane_position": plane_position,
        "missile_position": missile_position,
        "target_position": target_position,
        "intercepted": False,
        "interception_time": None
    }

def plot_simulation_results(data):
    """
    Plots the 3D simulation results for the plane, missile, and target.

    Args:
        data (dict): Contains positions of plane, missile, and target, and interception status.
    """
    trace_plane = go.Scatter3d(
        x=data['plane_position'][:, 0],
        y=data['plane_position'][:, 1],
        z=data['plane_position'][:, 2],
        mode='lines',
        line=dict(color='blue', width=2),
        name='Plane'
    )

    trace_missile = go.Scatter3d(
        x=data['missile_position'][:, 0],
        y=data['missile_position'][:, 1],
        z=data['missile_position'][:, 2],
        mode='lines',
        line=dict(color='red', width=2),
        name='Missile'
    )

    trace_target = go.Scatter3d(
        x=data['target_position'][:, 0],
        y=data['target_position'][:, 1],
        z=data['target_position'][:, 2],
        mode='lines',
        line=dict(color='green', width=2),
        name='Target'
    )

    layout = go.Layout(
        title='Plane and Missile Engagement Simulation',
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Altitude (m)'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig = go.Figure(data=[trace_plane, trace_missile, trace_target], layout=layout)
    st.plotly_chart(fig)

def plane_simulation_page():
    st.title("Plane and Missile Engagement Simulation")

    # User input for plane parameters
    st.sidebar.header("Plane Parameters")
    plane_speed = st.sidebar.number_input("Plane Speed (m/s)", min_value=0.0, max_value=500.0, value=250.0)
    plane_altitude = st.sidebar.number_input("Plane Altitude (m)", min_value=0.0, max_value=20000.0, value=10000.0)
    plane_direction = st.sidebar.number_input("Plane Direction (degrees)", min_value=0, max_value=360, value=90)
    radar_range = st.sidebar.number_input("Radar Range (m)", min_value=0.0, max_value=50000.0, value=20000.0)

    # User input for missile parameters
    st.sidebar.header("Missile Parameters")
    missile_speed = st.sidebar.number_input("Missile Speed (m/s)", min_value=0.0, max_value=2000.0, value=1000.0)
    seeker_prob = st.sidebar.number_input("Seeker Probability of Intercept", min_value=0.0, max_value=1.0, value=0.8)

    # User input for target parameters
    st.sidebar.header("Target (Missile) Parameters")
    target_speed = st.sidebar.number_input("Target Speed (m/s)", min_value=0.0, max_value=1000.0, value=300.0)
    target_altitude = st.sidebar.number_input("Target Altitude (m)", min_value=0.0, max_value=20000.0, value=10000.0)
    target_direction = st.sidebar.number_input("Target Direction (degrees)", min_value=0, max_value=360, value=270)

    # Parameters dictionary
    plane_params = {
        'speed': plane_speed,
        'altitude': plane_altitude,
        'direction': plane_direction,
        'radar_range': radar_range
    }

    missile_params = {
        'speed': missile_speed,
        'seeker_prob': seeker_prob
    }

    target_params = {
        'speed': target_speed,
        'altitude': target_altitude,
        'direction': target_direction
    }

    # Run simulation
    st.header("Simulation Results")
    sim_data = simulate_plane_and_missile(plane_params, missile_params, target_params)

    if sim_data['intercepted']:
        st.success(f"Target intercepted at time {sim_data['interception_time']:.2f} seconds!")
    else:
        st.error("Missile missed the target.")

    # Plot the results
    plot_simulation_results(sim_data)