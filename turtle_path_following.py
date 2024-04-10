import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import turtlebot_controller as c
from pathlib import Path
import simulation as sim
import env_generator as eg
import env_manager as em
import argparse as arg
import time

def main(args):
    run = args.run
    env_num = args.env
    path_num = args.path
    speed=20
    eps = 1e-1

    turtle = sim.create_sim()
    env_size, boundary_params, obstacle_params, sg_params, RRTs_params, lidar_params = eg.load_params('./parameters.dat')

    if env_num is not None:
        env_path = Path(f"./envData/env{env_num}").absolute().resolve()

        obs_file = env_path / f'obstacles.dat'
        boundary_file = env_path / f'boundary.dat'

        boundary, obstacles = em.load_env(boundary_file, obs_file)

        if path_num is not None:
            path_file = env_path / f'path{path_num}' / 'path.dat'
            path, angle = em.load_path(path_file)
            waypoint_path = path
            path = np.hstack((path, np.zeros((path.shape[0],1))))  # convert to 3d
            em.load_waypoints(waypoint_path)
        else:  # Generate Path in given env
            path, angle = eg.generate_paths(boundary, obstacles, sg_params, RRTs_params)
            waypoint_path = path
            path = np.hstack((waypoint_path, np.zeros((path.shape[0],1))))  # convert to 3d
            em.load_waypoints(waypoint_path)
    else:
        boundary, obstacles = eg.generate_env(boundary_params, obstacle_params)
        em.load_boundary(boundary)
        em.load_obstacles(obstacles)

        path, angle = eg.generate_paths(boundary, obstacles, sg_params, RRTs_params)
        waypoint_path = path
        path = np.hstack((waypoint_path, np.zeros((path.shape[0],1))))  # convert to 3d
        em.load_waypoints(waypoint_path)

    start = path[0]
    goal = path[-1]
    initial_qaut = p.getQuaternionFromEuler([0, 0, angle])
    turtle_ops = {
        'pos' : list(start),
        'orn' : initial_qaut
    }
    p.resetBasePositionAndOrientation(turtle, turtle_ops['pos'], turtle_ops['orn'])

    node = 1
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
        elif error<eps and node == len(path)-1:
            time.sleep(2)
            node = 0
            p.resetBasePositionAndOrientation(turtle, turtle_ops['pos'], turtle_ops['orn'])

if __name__=='__main__':
    parser = arg.ArgumentParser()
    parser.add_argument('--run', type=int, default=None, help="The run of the model to be loaded, use None for no model")
    parser.add_argument('--env', type=int, default=None, help="The environment number to test the turtlebot in, use None to generate one")
    parser.add_argument('--path', type=int, default=None, help="The path in the environment, use None to generate one")
    args = parser.parse_args()
    main(args)