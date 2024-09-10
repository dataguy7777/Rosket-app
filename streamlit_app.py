import streamlit as st
from rocket_simulation import rocket_simulation_page
from plane_simulation import plane_simulation_page  # Ensure the correct import here
from jet_trajectory_simulation import jet_trajectory_page  # Import the jet trajectory page

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Rocket Simulation", "Plane and Missile Simulation", "Jet Trajectory Simulation"])

# Navigation
if page == "Rocket Simulation":
    rocket_simulation_page()
elif page == "Plane and Missile Simulation":
    plane_simulation_page()  # Ensure the function name matches the import
elif page == "Jet Trajectory Simulation":  # Add this condition for jet trajectory simulation
    jet_trajectory_page()