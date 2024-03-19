import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import turtlebot_controller as c
from pathlib import Path
import simulation as sim
from env_manager import load_env, load_path, load_waypoints
import time
def main():
    env_num = 9
    path_num = 0
    speed=20
    eps = 1e-1

    env_path = env_parent_dir = Path(f"./envData/env{env_num}").absolute().resolve()
    path_file = env_path / f'path{path_num}' / 'path.dat'
    path, angle = load_path(path_file)
    waypoint_path = path
    path = np.flip(np.hstack((path, np.zeros((path.shape[0],1)))), axis=0)  # convert to 3d

    obs_file = env_path / f'env{env_num}_obstacles.dat'
    boundary_file = env_path / f'env{env_num}_boundary.dat'

    initial_qaut = p.getQuaternionFromEuler([0, 0, angle])
    turtle_ops = {
        'pos' : list(path[0]),
        'orn' : initial_qaut
    }

    turtle = sim.create_sim(turtle_opts=turtle_ops)
    load_env(boundary_file, obs_file)
    load_waypoints(waypoint_path)

    node = 1
    # time.sleep(10)
    while True:
        target = path[node]
        
        # Find Wheel Velocities
        leftWheelVelocity, rightWheelVelocity = c.controller_v2(target, sim.turtle, max_vel=speed, a=3.5, eps=eps)

        # Update Turtle Bot
        sim.step_sim(leftWheelVelocity, rightWheelVelocity)

        curr_pos, curr_orn = p.getBasePositionAndOrientation(turtle)
        error = np.linalg.norm(np.array(curr_pos[:2]) - target[:2])

        if error<eps and node<(len(path)-1):
            node+=1

if __name__=='__main__':
    main()