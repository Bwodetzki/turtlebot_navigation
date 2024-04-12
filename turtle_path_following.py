import numpy as np
import torch as t
import matplotlib.pyplot as plt
import pybullet as p
import turtlebot_controller as c
from pathlib import Path
from network import PlanningNetwork
from train import to_body_frame, to_inertial_frame
import simulation as sim
import env_generator as eg
import env_manager as em
import argparse as arg
import time

DEVICE = (
    "cuda"
    if t.cuda.is_available()
    else "mps"
    if t.backends.mps.is_available()
    else "cpu"
)

def default_path_following(obs, path, node, speed=20, eps=1e-1):
    curr_pos, _, _ = obs

    target = path[node]
    error = np.linalg.norm(np.array(curr_pos[:2]) - target[:2])
    if error<eps and node<(len(path)-1):
        node+=1
        target = path[node]
    
    # Find Wheel Velocities
    leftWheelVelocity, rightWheelVelocity = c.controller_v2(target, sim.turtle, max_vel=speed, a=3.5, eps=eps)
    return leftWheelVelocity, rightWheelVelocity, node

def fast_planning(obs, goal, waypoint, waypoint_id, speed=20, eps=1e-1, draw_waypoint=True, height=0.25, radius=0.25/2):
    # Format Data
    curr_pos, curr_angle, measurements = obs
    goal_vec = to_body_frame(t.tensor(curr_pos[:2]), t.tensor(goal[:2].astype(np.float32)), t.tensor(curr_angle))

    if t.linalg.norm(goal_vec) < eps:
        return 0, 0, waypoint_id

    # Plot Waypoint
    if draw_waypoint:
        p.removeBody(waypoint_id)
        waypoint_id = sim.create_waypoint(t.concatenate((waypoint, t.tensor([height]))), radius, color=[1, 0, 1, 1])

    # Control to Waypoint
    leftWheelVelocity, rightWheelVelocity = c.controller_v2(waypoint.to('cpu').detach(), sim.turtle, max_vel=speed, a=0.5, eps=eps)
    return leftWheelVelocity, rightWheelVelocity, waypoint_id

def slow_planning():
    return None, None, None

def main(args):
    run = args.run
    env_num = args.env
    path_num = args.path
    corn = args.corn
    speed = 30
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
    
    # Load in model and setup torch
    t.no_grad()
    t.set_default_device(DEVICE)
    print(f"Using {DEVICE} device")

    if run is not None:
        net = PlanningNetwork()
        model_path = model_path = f"./good_models/run{run}"
        data = t.load(model_path, map_location=DEVICE)

        loss = data['loss']
        print(f'Validation loss of model is {loss:0.4f}')
        net.load_state_dict(data['network_params'])

    # Initializations For Loop
    start = path[0]
    goal = path[-1]
    node = 1
    waypoint_id = sim.create_waypoint([0, 0, -1], radius=0.25)
    initial_qaut = p.getQuaternionFromEuler([0, 0, angle])
    turtle_ops = {
        'pos' : list(start),
        'orn' : initial_qaut
    }
    p.resetBasePositionAndOrientation(turtle, turtle_ops['pos'], turtle_ops['orn'])
    sim.initLidar(lidar_params['lidarDist'], lidar_params['lidarAngle'], lidar_params['numMeasurements'])

    # Loop
    i=0
    waypointGenerationPeriod = 0.1 # seconds
    prevWaypointTime = -1*waypointGenerationPeriod
    while True:
        ## Recieve Data
        # There may be something funky with the orientation data. 
        # Does getEulerFromQuaternion return z in between -pi and pi? That is what the network is trained for
        curr_pos, curr_orn = p.getBasePositionAndOrientation(turtle)
        curr_angle = p.getEulerFromQuaternion(curr_orn)[-1] # Pretty sure weve done everything in radians, if something is not working though, maybe our training data collected with degrees instead
        measurements = sim.localLidar(curr_pos, curr_angle)
        obs = (curr_pos, curr_angle, measurements)

        ## Controller Logic
        if run is not None:  # Use network
            if corn==0:
                now = time.time()
                if now - prevWaypointTime >= waypointGenerationPeriod:
                    prevWaypointTime = now
                    # Get waypoint
                    goal_vec = to_body_frame(t.tensor(curr_pos[:2]), t.tensor(goal[:2].astype(np.float32)), t.tensor(curr_angle))
                    with t.no_grad():
                        waypoint = net(goal_vec.reshape(1,-1), t.tensor(measurements).reshape(1,-1)).flatten()
                    waypoint = waypoint/t.linalg.norm(waypoint)
                    waypoint = to_inertial_frame(t.tensor(curr_pos[:2]), waypoint, t.tensor(curr_angle))
                    draw_waypoint=True
                else:
                    draw_waypoint=False
                leftWheelVelocity, rightWheelVelocity, waypoint_id = fast_planning(obs, 
                                                                                goal, 
                                                                                waypoint,
                                                                                waypoint_id, 
                                                                                speed=speed,
                                                                                eps=eps, 
                                                                                draw_waypoint=draw_waypoint)
            else: # corn ==1
                leftWheelVelocity, rightWheelVelocity, waypoint_id = slow_planning()  # not implemented YET
        else:  # Use RRT*
            leftWheelVelocity, rightWheelVelocity, node = default_path_following(obs,
                                                                        path,
                                                                        node,
                                                                        speed=speed,
                                                                        eps=eps)
        ## Sim Logic
        # Update Turtle Bot
        sim.keyboard_control(forward=0, turn=0, speed=speed)
        sim.step_sim(leftWheelVelocity, rightWheelVelocity)

        # Restart Sim if at Goal
        goal_error = np.linalg.norm(np.array(curr_pos[:2]) - goal[:2])
        if goal_error<eps:
            time.sleep(2)
            node = 0
            p.resetBasePositionAndOrientation(turtle, turtle_ops['pos'], turtle_ops['orn'])
        i+=1

if __name__=='__main__':
    parser = arg.ArgumentParser()
    parser.add_argument('--run', type=int, default=None, help="The run of the model to be loaded, use None for no model")
    parser.add_argument('--env', type=int, default=None, help="The environment number to test the turtlebot in, use None to generate one")
    parser.add_argument('--path', type=int, default=None, help="The path in the environment, use None to generate one")
    parser.add_argument('--corn', type=int, default=0, help="UNUSED: int(0, 1) The controller to be used, if not used default controller is used")
    args = parser.parse_args()
    main(args)