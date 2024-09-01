import streamlit as st
from rocket_simulation import rocket_simulation_page
from plane_simulation import plane_simulation_page

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Rocket Simulation", "Plane and Missile Simulation"])

# Navigation
if page == "Rocket Simulation":
    rocket_simulation_page()
elif page == "Plane and Missile Simulation":
    plane_simulation_page()