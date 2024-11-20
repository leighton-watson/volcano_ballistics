import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Define ballistic physics functions
def calculate_reynolds_number(v, D, rho_f, mu_f):
    return (rho_f * v * D) / mu_f

def calculate_drag_coefficient(Re):
    if Re < 0.1:
        return 24 / Re
    elif Re < 1000:
        return 24 / Re**(0.6)
    else:
        return 0.47

def calculate_trajectory(g, rho_f, mu_f, D, m, theta, v0, time_step=0.001, max_time=10):
    theta_rad = np.radians(theta)
    vx, vy = v0 * np.cos(theta_rad), v0 * np.sin(theta_rad)
    x, y = 0, 0
    t = 0
    trajectory = []
    
    while y >= 0 and t <= max_time:
        v = np.sqrt(vx**2 + vy**2)
        Re = calculate_reynolds_number(v, D, rho_f, mu_f)
        C = calculate_drag_coefficient(Re)
        Fd = 0.5 * C * rho_f * np.pi * (D / 2)**2 * v**2
        ax = -Fd * vx / (m * v)
        ay = -g - (Fd * vy / (m * v))
        vx += ax * time_step
        vy += ay * time_step
        x += vx * time_step
        y += vy * time_step
        trajectory.append((t, v, Re, C, vx, vy, x, y))
        t += time_step

    return np.array(trajectory, dtype=[
        ('t', 'f4'), ('v', 'f4'), ('Re', 'f4'), ('C', 'f4'),
        ('vx', 'f4'), ('vy', 'f4'), ('x', 'f4'), ('y', 'f4')
    ])

# Streamlit application
st.title("Ballistic Trajectory Simulator")

# Input fields
st.sidebar.header("Input Parameters")
g = st.sidebar.number_input("Gravity (m/s²)", value=9.81, step=0.01)
rho_f = st.sidebar.number_input("Fluid Density (kg/m³)", value=1.2257, step=0.01)
mu_f = st.sidebar.number_input("Viscosity (Pa·s)", value=0.000018, step=0.000001, format="%.6f")
D = st.sidebar.number_input("Diameter (m)", value=0.01746, step=0.0001)
m = st.sidebar.number_input("Mass (kg)", value=0.0218, step=0.0001)
theta = st.sidebar.number_input("Launch Angle (°)", value=25.0, step=1.0)
v0 = st.sidebar.number_input("Launch Velocity (m/s)", value=5.5, step=0.1)

if st.sidebar.button("Simulate"):
    trajectory = calculate_trajectory(g, rho_f, mu_f, D, m, theta, v0)
    
    # Display results
    st.subheader("Trajectory Data")
    st.write(f"{'Time (s)':<10}{'Velocity (m/s)':<15}{'Re':<10}{'Drag Coeff.':<15}{'Vx (m/s)':<10}{'Vy (m/s)':<10}{'X (m)':<10}{'Y (m)':<10}")
    for point in trajectory[:100]:  # Limit output to 100 rows
        st.text(f"{point['t']:<10.3f}{point['v']:<15.3f}{point['Re']:<10.3f}{point['C']:<15.3f}{point['vx']:<10.3f}{point['vy']:<10.3f}{point['x']:<10.3f}{point['y']:<10.3f}")
    
    # Plot trajectory
    x_values = trajectory['x']
    y_values = trajectory['y']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_values, y_values, label="Trajectory")
    ax.set_title("Ballistic Trajectory")
    ax.set_xlabel("Horizontal Distance (m)")
    ax.set_ylabel("Vertical Distance (m)")
    ax.grid()
    ax.legend()
    st.pyplot(fig)
