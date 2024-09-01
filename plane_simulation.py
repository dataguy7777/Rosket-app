import streamlit as st
import numpy as np
import plotly.graph_objs as go

# Add your existing plane and missile simulation imports and functions here

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
        st.error("Missile missed the target")

    # Plot the results
    plot_simulation_results(sim_data)