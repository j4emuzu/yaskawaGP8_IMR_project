from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import matplotlib.pyplot as plt

client = RemoteAPIClient()
sim = client.require('sim')
sim.setStepping(True)

# --- Time ---
startTime = sim.getSimulationTime()
dt = sim.getSimulationTimeStep()
time_duration = 23.0
elapsedTime = 0
time_log = []

# --- Cup handle ---
cup = sim.getObject('/conveyorSystem/Cup')

# --- Cup handle ---
cup_px, cup_py, cup_pz = [], [], [] # Linear pos
cup_vx, cup_vy, cup_vz = [], [], [] # Linear vel

cup_rx, cup_ry, cup_rz = [], [], [] # Orientation
cup_wx, cup_wy, cup_wz = [], [], [] # Angular vel

sim.startSimulation()

# --- Plotting ---
def plotting_data():
    # Fig 1: Linear Motion (Position & Velocity)
    fig1, axs1 = plt.subplots(2, 3, figsize=(15, 8))
    fig1.suptitle('Fig 1: Linear Motion', fontsize=16)

    labels = ['X', 'Y', 'Z']
    p_data = [cup_px, cup_py, cup_pz]
    v_data = [cup_vx, cup_vy, cup_vz]

    for i in range(3):
        # Row 1: Position (x, y, z) vs t
        axs1[0, i].plot(time_log, p_data[i], 'b', label=f'pos_{labels[i].lower()}')
        axs1[0, i].set_title(f'Position {labels[i]} vs Time')
        axs1[0, i].set_ylabel('Position (m)')
        axs1[0, i].grid(True)
        axs1[0, i].legend()

        # Row 2: Velocity (vx, vy, vz) vs t
        axs1[1, i].plot(time_log, v_data[i], 'r', label=f'vel_{labels[i].lower()}')
        axs1[1, i].set_title(f'Velocity {labels[i]} vs Time')
        axs1[1, i].set_ylabel('Velocity (m/s)')
        axs1[1, i].set_xlabel('Time (s)')
        axs1[1, i].grid(True)
        axs1[1, i].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


    # Fig 2: Orientation (Angular Position & Velocity)
    fig2, axs2 = plt.subplots(2, 3, figsize=(15, 8))
    fig2.suptitle('Fig 2: Orientation', fontsize=16)

    ang_pos_labels = ['Alpha (Roll)', 'Beta (Pitch)', 'Gamma (Yaw)']
    ang_vel_labels = ['wx', 'wy', 'wz']
    r_data = [cup_rx, cup_ry, cup_rz]
    w_data = [cup_wx, cup_wy, cup_wz]

    for i in range(3):
        # Row 1: Orientation (alpha, beta, gamma) vs t
        axs2[0, i].plot(time_log, r_data[i], 'g', label=f'angle_{labels[i].lower()}')
        axs2[0, i].set_title(f'{ang_pos_labels[i]} vs Time')
        axs2[0, i].set_ylabel('Orientation (rad)')
        axs2[0, i].grid(True)
        axs2[0, i].legend()

        # Row 2: Angular Velocity (wx, wy, wz) vs t
        axs2[1, i].plot(time_log, w_data[i], 'm', label=f'ang_vel_{labels[i].lower()}')
        axs2[1, i].set_title(f'Angular Velocity {ang_vel_labels[i]} vs Time')
        axs2[1, i].set_ylabel('Ang. Velocity (rad/s)')
        axs2[1, i].set_xlabel('Time (s)')
        axs2[1, i].grid(True)
        axs2[1, i].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # 3D Trajectory
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(cup_px, cup_py, cup_pz, 'green', linewidth=1)
    ax.set_title('3D Cup Trajectory Path')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    plt.show()

def main():
    # 1. Calculate the total number of steps based on duration and dt
    # Using round() prevents floating-point precision errors during conversion
    current_dt = sim.getSimulationTimeStep()
    total_steps = int(round(time_duration / current_dt))
    
    print(f"Starting Trajectory Recording...")
    print(f"Target Duration: {time_duration}s | Total Steps: {total_steps}")
    
    # Start the simulation
    sim.startSimulation()

    # 2. Main data collection loop
    for i in range(total_steps):
        # Calculate current timestamp based on the index
        t = i * current_dt
        time_log.append(t)

        # Retrieve linear position [x, y, z] relative to World (-1)
        pos = sim.getObjectPosition(cup, -1)
        cup_px.append(pos[0])
        cup_py.append(pos[1])
        cup_pz.append(pos[2])

        # Retrieve velocities: returns two lists [vx, vy, vz] and [wx, wy, wz]
        lin_vel, ang_vel = sim.getObjectVelocity(cup, -1)
        
        # Store Linear Velocity
        cup_vx.append(lin_vel[0])
        cup_vy.append(lin_vel[1])
        cup_vz.append(lin_vel[2])
        
        # Store Angular Velocity (needed for Jacobian orientation control)
        cup_wx.append(ang_vel[0])
        cup_wy.append(ang_vel[1])
        cup_wz.append(ang_vel[2])

        # Retrieve Orientation (Euler Angles: Alpha, Beta, Gamma) in Radians
        ori = sim.getObjectOrientation(cup, -1)
        cup_rx.append(ori[0])
        cup_ry.append(ori[1])
        cup_rz.append(ori[2])

        # Trigger the next simulation step (required when stepping is enabled)
        sim.step()

        # Print progress update every 100 steps
        if i % 100 == 0:
            print(f"📝 Logging Progress: {i}/{total_steps} steps...")

    # 3. Finalize simulation and display results
    print("Recording complete. Generating plots...")
    sim.stopSimulation()
    
    # Call the plotting function provided in your snippet
    plotting_data()

if __name__ == "__main__":
    main()