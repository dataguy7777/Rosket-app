# jet_trajectory_simulation.py
import streamlit as st
import plotly.graph_objs as go
import pandas as pd
import numpy as np

def generate_trajectory_data():
    """
    Simulate jet trajectory data points. These could represent a jet's latitude, longitude, and altitude.
    Returns:
        pd.DataFrame: DataFrame containing latitude, longitude, and altitude for jet trajectory simulation.
    """
    np.random.seed(0)
    latitudes = np.cumsum(np.random.randn(50))  # Simulate latitude changes
    longitudes = np.cumsum(np.random.randn(50))  # Simulate longitude changes
    altitudes = np.abs(np.random.randn(50)) * 10000  # Simulate altitude in meters
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

    st.sidebar.title("Jet Trajectory Settings")
    speed = st.sidebar.slider("Jet Speed (km/h)", 500, 1500, 900)
    altitude = st.sidebar.slider("Jet Altitude (m)", 1000, 12000, 8000)

    st.write(f"Simulating jet with speed: {speed} km/h and altitude: {altitude} meters.")

    # Generate simulated trajectory data
    trajectory_data = generate_trajectory_data()

    # Plot the jet trajectory
    jet_trajectory_fig = plot_jet_trajectory(trajectory_data)

    # Display the plot in Streamlit
    st.plotly_chart(jet_trajectory_fig)