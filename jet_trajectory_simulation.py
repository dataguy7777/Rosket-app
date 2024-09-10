import streamlit as st
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim

def geocode_address(address):
    """
    Convert an address into geographical coordinates (latitude, longitude).

    Args:
        address (str): The address to geocode.

    Returns:
        tuple: (latitude, longitude) or None if not found.
    """
    geolocator = Nominatim(user_agent="jet_trajectory_simulation")
    location = geolocator.geocode(address)
    if location:
        return location.latitude, location.longitude
    else:
        return None

def generate_realistic_trajectory(start_coords, end_coords, total_time, speed):
    """
    Simulate a realistic jet trajectory between two points (start and end).

    Args:
        start_coords (tuple): (latitude, longitude) for the start point.
        end_coords (tuple): (latitude, longitude) for the end point.
        total_time (float): Total flight time in seconds.
        speed (float): Jet speed in km/h.

    Returns:
        pd.DataFrame: A DataFrame containing the simulated latitude, longitude, and altitude.
    """
    lat_start, lon_start = start_coords
    lat_end, lon_end = end_coords

    # Time settings
    time_steps = np.linspace(0, total_time, num=100)

    # Linear interpolation of coordinates between start and end
    latitudes = np.linspace(lat_start, lat_end, num=len(time_steps))
    longitudes = np.linspace(lon_start, lon_end, num=len(time_steps))

    # Simple altitude profile (cruise altitude at halfway)
    altitudes = np.concatenate([np.linspace(0, 10000, num=len(time_steps) // 2),
                                np.linspace(10000, 0, num=len(time_steps) // 2)])

    return pd.DataFrame({
        'latitude': latitudes,
        'longitude': longitudes,
        'altitude': altitudes
    })

def plot_jet_trajectory(df):
    """
    Plot jet trajectory on a 3D world map using Plotly.

    Args:
        df (pd.DataFrame): DataFrame with jet trajectory data (latitude, longitude, altitude)

    Returns:
        plotly.graph_objs.Figure: 3D scatter plot with jet trajectory.
    """
    fig = go.Figure(data=go.Scattergeo(
        lon=df['longitude'],
        lat=df['latitude'],
        mode='markers+lines',
        marker=dict(
            size=5,
            color=df['altitude'],  # Color by altitude
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Altitude (m)")
        ),
        line=dict(width=2, color='blue'),
    ))

    fig.update_layout(
        geo=dict(
            projection_type="orthographic",
            showcoastlines=True,
            coastlinecolor="Black",
            showland=True,
            landcolor="rgb(243, 243, 243)",
            showocean=True,
            oceancolor="rgb(204, 204, 255)",
        ),
        title="Jet Trajectory on 3D World Map",
        height=600
    )
    return fig

def jet_trajectory_page():
    """
    Jet Trajectory Simulation page in the Streamlit app.
    """
    st.title("Jet Trajectory Simulation on 3D World Map")

    # Collect user inputs for start and destination addresses and flight parameters
    col1, col2 = st.columns(2)

    with col1:
        start_address = st.text_input("Enter Start Address", "New York, USA")
        start_coords = geocode_address(start_address)
        if start_coords:
            st.write(f"Start Coordinates: {start_coords}")
        else:
            st.error("Start address could not be geocoded.")

    with col2:
        end_address = st.text_input("Enter Destination Address", "Los Angeles, USA")
        end_coords = geocode_address(end_address)
        if end_coords:
            st.write(f"Destination Coordinates: {end_coords}")
        else:
            st.error("Destination address could not be geocoded.")

    if start_coords and end_coords:
        col1, col2 = st.columns(2)
        with col1:
            speed = st.number_input("Jet Speed (km/h)", min_value=500, max_value=1500, value=900)
        with col2:
            flight_time = st.number_input("Flight Time (seconds)", min_value=60, max_value=10000, value=3600)
        
        # Generate and plot the realistic jet trajectory
        st.write(f"Simulating jet with speed: {speed} km/h and flight time: {flight_time} seconds.")

        trajectory_data = generate_realistic_trajectory(start_coords, end_coords, flight_time, speed)

        # Plot the jet trajectory
        jet_trajectory_fig = plot_jet_trajectory(trajectory_data)
        st.plotly_chart(jet_trajectory_fig)

        # Display a slider to control simulation speed (real-time or faster)
        sim_speed = st.slider("Simulation Speed Multiplier", 1, 100, 10)
        st.write(f"Simulation running at {sim_speed}x normal speed.")
        
        # GIF or video simulation can be implemented here if required.

# Run the jet trajectory page
jet_trajectory_page()