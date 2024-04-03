'''
generate_polygon(), random_angle_steps(), clip(), and visualize() taken from
https://stackoverflow.com/questions/8997099/algorithm-to-generate-random-2d-polygon
'''

import math
import pickle
from pathlib import Path
from os import makedirs
from typing import List, Tuple
import pybullet as p
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import simulation as sim
import time
from collision_checker import is_inside_boundary, rectangle_col_checker

env_parent_dir = Path("./envData").absolute().resolve() # parent directory all other environment directories are in
env_dir_prefix = 'env' # prefix for environment directories
env_data_file_prefix = 'env' # prefix for environment data files
env_data_file_suffix = '.dat'
path_dir_prefix = 'path' # prefix for path directories
path_data_file_prefix = 'path' # prefix for path data files
path_data_file_suffix = '.dat'

# Functions
def get_curr_env_idx():
    if env_parent_dir.exists():
        env_idxs = [int(env.stem.strip(env_dir_prefix)) for env in env_parent_dir.iterdir() if env.is_dir()]
        return max(env_idxs) + 1
    return 0

def save_curr_env(boundary, obstacles):
    curr_env_idx = get_curr_env_idx()
    curr_env_dir = env_parent_dir / f'{env_dir_prefix}{curr_env_idx}'
    makedirs(str(curr_env_dir))
    curr_env_boundary_file = curr_env_dir / f'{env_data_file_prefix}_boundary{env_data_file_suffix}'
    curr_env_obstacles_file = curr_env_dir / f'{env_data_file_prefix}_obstacles{env_data_file_suffix}'
    with open(str(curr_env_boundary_file), 'wb') as boundary_fp:
        pickle.dump(boundary, boundary_fp)
    with open(str(curr_env_obstacles_file), 'wb') as obstacles_fp:
        pickle.dump(obstacles, obstacles_fp)

def save_env(boundary, obstacles, env_idx, env_parent_dir=env_parent_dir):
    curr_env_dir = env_parent_dir / f'{env_dir_prefix}{env_idx}'
    try:
        makedirs(str(curr_env_dir))
    except:
        pass
    curr_env_boundary_file = curr_env_dir / f'{env_data_file_prefix}_boundary{env_data_file_suffix}'
    curr_env_obstacles_file = curr_env_dir / f'{env_data_file_prefix}_obstacles{env_data_file_suffix}'
    with open(str(curr_env_boundary_file), 'wb') as boundary_fp:
        pickle.dump(boundary, boundary_fp)
    with open(str(curr_env_obstacles_file), 'wb') as obstacles_fp:
        pickle.dump(obstacles, obstacles_fp)

def load_env(boundary_file, obstacles_file):
    with open(str(boundary_file), 'rb') as boundary_fp:
        boundary = pickle.load(boundary_fp)
    with open(str(obstacles_file), 'rb') as obstacles_fp:
        obstacles = pickle.load(obstacles_fp)
    load_boundary(boundary)
    load_obstacles(obstacles)

def save_start_goal(start, goal, angle, curr_env_idx=0, path_idx=0, env_parent_dir=env_parent_dir):
    path = np.vstack((start, goal))
    sg_file = save_path(path, angle, curr_env_idx, path_idx, env_parent_dir)
    return sg_file

def load_start_goal(sg_file):
    path, angle = load_path(sg_file)
    return(path[0], path[1], angle)  # Tuple of start and goal

def save_path(path, angle, curr_env_idx=0, path_idx=0, env_parent_dir=env_parent_dir):
    data = (path, angle)  # tuple, 1st entry:numpy array of path, second entry: start angle
    path_dir = env_parent_dir / f'{env_dir_prefix}{curr_env_idx}' / f'{path_dir_prefix}{path_idx}'
    try:
        makedirs(str(path_dir))
    except:
        pass
    file_name = f'{path_data_file_prefix}{path_data_file_suffix}'  # Start Goal Configuration
    path_file = path_dir / file_name
    with open(str(path_file), 'wb') as path_fp:
        pickle.dump(data, path_fp)
    
    return path_file

def load_path(path_file):
    with open(str(path_file), 'rb') as p_fp:
        data = pickle.load(p_fp)
    return (data[0], data[1])  # Tuple of path and angle

def save_lidar(measurements, curr_env_idx=0, path_idx=0, env_parent_dir=env_parent_dir):
    path_dir = env_parent_dir / f'{env_dir_prefix}{curr_env_idx}' / f'{path_dir_prefix}{path_idx}'
    try:
        makedirs(str(path_dir))
    except:
        pass
    file_name = f'measurements{path_data_file_suffix}'  # Start Goal Configuration
    measurement_file = path_dir / file_name
    with open(str(measurement_file), 'wb') as measurements_fp:
        pickle.dump(measurements, measurements_fp)
    return measurement_file

def load_lidar(measurement_file):
    with open(str(measurement_file), 'rb') as m_fp:
        data = pickle.load(m_fp)
    return data  # Tuple of start and goal

def save_data(data_points, curr_env_idx=0, path_idx=0, env_parent_dir=env_parent_dir):
    path_dir = env_parent_dir / f'{env_dir_prefix}{curr_env_idx}'
    try:
        makedirs(str(path_dir))
    except:
        pass
    file_name = f'data_{len(data_points)}_points{path_data_file_suffix}'  # Start Goal Configuration
    data_file = path_dir / file_name
    with open(str(data_file), 'wb') as data_fp:
        pickle.dump(data_points, data_fp)
    return data_file

def load_data_points(data_file):
    with open(str(data_file), 'rb') as d_fp:
        data_points = pickle.load(d_fp)
    return data_points  # list of all datapoints

def generate_boundary(center: Tuple[float, float], avg_radius: float,
                     irregularity: float, spikiness: float,
                     num_vertices: int, min_radius: float = 0, seed: int = None) -> List[Tuple[float, float]]:
    """
    Start with the center of the polygon at center, then creates the
    polygon by sampling points on a circle around the center.
    Random noise is added by varying the angular spacing between
    sequential points, and by varying the radial distance of each
    point from the centre.

    Args:
        center (Tuple[float, float]):
            a pair representing the center of the circumference used
            to generate the polygon.
        avg_radius (float):
            the average radius (distance of each generated vertex to
            the center of the circumference) used to generate points
            with a normal distribution.
        irregularity (float):
            variance of the spacing of the angles between consecutive
            vertices.
        spikiness (float):
            variance of the distance of each vertex to the center of
            the circumference.
        num_vertices (int):
            the number of vertices of the polygon.
    Returns:
        List[Tuple[float, float]]: list of vertices, in CCW order.
    """
    if seed is not None:
        np.random.seed(seed)
    # Parameter check
    if irregularity < 0 or irregularity > 1:
        raise ValueError("Irregularity must be between 0 and 1.")
    if spikiness < 0 or spikiness > 1:
        raise ValueError("Spikiness must be between 0 and 1.")

    irregularity *= 2 * math.pi / num_vertices
    spikiness *= avg_radius
    angle_steps = _random_angle_steps(num_vertices, irregularity, seed)

    # now generate the points
    points = []
    angle = np.random.uniform(0, 2 * math.pi)
    for i in range(num_vertices):
        radius = _clip(np.random.normal(avg_radius, spikiness), min_radius, 2 * avg_radius)
        point = np.array((center[0] + radius * math.cos(angle),
                 center[1] + radius * math.sin(angle)))
        points.append(point)
        angle += angle_steps[i]

    return np.array(points)

def _random_angle_steps(steps: int, irregularity: float, seed: int = None) -> List[float]:
    """Generates the division of a circumference in random angles.

    Args:
        steps (int):
            the number of angles to generate.
        irregularity (float):
            variance of the spacing of the angles between consecutive vertices.
    Returns:
        List[float]: the list of the random angles.
    """
    if seed is not None:
        np.random.seed(seed)
    # generate n angle steps
    angles = []
    lower = (2 * math.pi / steps) - irregularity
    upper = (2 * math.pi / steps) + irregularity
    cum_sum = 0
    for i in range(steps):
        angle = np.random.uniform(lower, upper)
        angles.append(angle)
        cum_sum += angle

    # normalize the steps so that point 0 and point n+1 are the same
    cum_sum /= (2 * math.pi)
    for i in range(steps):
        angles[i] /= cum_sum
    return angles

def _clip(value, lower, upper):
    """
    Given an interval, values outside the interval are clipped to the interval
    edges.
    """
    return min(upper, max(value, lower))

def _visualize(vertices):
    black = (0, 0, 0)
    white = (255, 255, 255)
    img = Image.new('RGB', (500, 500), white)
    im_px_access = img.load()
    draw = ImageDraw.Draw(img)

    # either use .polygon(), if you want to fill the area with a solid colour
    draw.polygon(vertices, outline=black, fill=white)

    # or .line() if you want to control the line thickness, or use both methods together!
    # draw.line(vertices + [vertices[0]], width=2, fill=black)

    img.show()

    # now you can save the image (img), or do whatever else you want with it.

def generate_obstacles(vertices, center_bounds=[10, 10], edge_len_bounds=[0.1, 2], seed=None, n=2, max_attempts=1000):
    bounding_box = np.array(center_bounds)
    if seed is not None:
        np.random.seed(seed)
    
    centers = np.zeros((n, 2))
    edge_lengths = np.zeros((n, 2))
    angles = np.zeros((n))

    attempts = 0
    num_done=0
    while attempts < max_attempts:
        # Generate Samples
        center = (2*np.random.rand(2)-1)*bounding_box
        edge_length = np.random.rand(2)*(edge_len_bounds[1]-edge_len_bounds[0]) + edge_len_bounds[0]
        angle = (2*np.random.rand() - 1)*np.pi

        # Check Conditions
        success = all(abs(center) >= edge_length[1]/2 + 1) and is_inside_boundary(vertices, center, radius=0.05)
        if success:
            # Store Samples
            centers[num_done, :] = center
            edge_lengths[num_done, :] = edge_length
            angles[num_done] = angle

            num_done+=1

            if num_done==n:
                break
        attempts+=1
    
    if attempts == max_attempts:
        raise Exception('Failed to generate obstacles')

    # represent obstacles with tuples so they are hashable and enable caching
    return [(tuple(center), tuple(edge_length), angle) for center, edge_length, angle in zip(centers, edge_lengths, angles)]

def plot_boundary(boundary, ax=None):
    if ax is None:
        plt.plot(boundary[:, 0], boundary[:, 1], 'b')
        plt.plot([boundary[0, 0], boundary[-1, 0]], [boundary[0, 1], boundary[-1, 1]], 'b')
    else:
        ax.plot(boundary[:, 0], boundary[:, 1], 'b')
        ax.plot([boundary[0, 0], boundary[-1, 0]], [boundary[0, 1], boundary[-1, 1]], 'b')

def plot_obstacle(obstacle, ax=None):
    center, edge_lengths, angle = obstacle
    x_len = edge_lengths[0]
    y_len = edge_lengths[1]

    obs = 0.5 * np.array([[-x_len, -y_len],
                    [-x_len, y_len],
                    [x_len, y_len],
                    [x_len, -y_len]])
    
    rot_mat = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
    obs = (rot_mat@obs.T).T + np.array(center)
        
    if ax is None:
        plt.plot(obs[:, 0], obs[:, 1], 'g')
        plt.plot([obs[0, 0], obs[-1, 0]], [obs[0, 1], obs[-1, 1]], 'g')
    else:
        ax.plot(obs[:, 0], obs[:, 1], 'g')
        ax.plot([obs[0, 0], obs[-1, 0]], [obs[0, 1], obs[-1, 1]], 'g')

def load_boundary(vertices):
    wallHeight = 2
    wallWidth = 0.01
    for idx in range(-1, np.size(vertices, axis=0)-1):
        vertex1 = vertices[idx, :]
        vertex2 = vertices[idx+1, :]
        diff = vertex2 - vertex1
        segmentCenter = 1/2 * (vertex1 + vertex2)
        segmentCenter = np.concatenate((segmentCenter, np.array([wallHeight/2])))
        segmentLength = np.sqrt(np.dot(diff, diff))
        angle = np.arctan2(diff[1], diff[0])
        sim.create_box(segmentCenter, dimensions=[segmentLength, wallWidth, wallHeight], angles=[0, 0, angle], mass=0)

def load_obstacles(obstacles):
    for center, edge_length, angle in obstacles:
        center = np.concatenate((center, [1/2]))
        edge_length = np.concatenate((edge_length, [1]))
        angle = [0, 0, angle]
        sim.create_box(center, edge_length, angle)

def load_waypoints(path, height=0.25, radius=0.25/2):
    sim.create_waypoint(np.concatenate((path[0], [height])), radius, color = [0, 1, 0, 1])
    for path_point in path[1:-1]:
        path_point = np.concatenate((path_point, [height]))
        sim.create_waypoint(path_point, radius, color = [1, 0, 0, 1])
    sim.create_waypoint(np.concatenate((path[-1], [height])), radius, color = [0, 0, 1, 1])

def generate_start_goal(vertices, obstacles, radius=0.75, center_bounds=np.array((10, 10)), dist=1, seed=None, max_attempts=1000):
    if seed is not None:
        np.random.seed(seed)

    attempts = 0
    while attempts < max_attempts:
        # Generate Sample
        start = (2*np.random.rand(2)-1)*center_bounds

        # Check Conditions for Start
        success = True
        if is_inside_boundary(vertices, start, radius):
            for obstacle in obstacles:
                if rectangle_col_checker(obstacle, start, radius):
                    success = False
                    break
        else:
            success = False
        if success:
            break
        attempts+=1

    if attempts == max_attempts:
        raise Exception('Failed to generate start point')

    attempts = 0
    while attempts < max_attempts:
        # Generate Sample
        goal = (2*np.random.rand(2)-1)*center_bounds

        # Check Conditions for Goal
        success = True
        if np.linalg.norm(start-goal)>dist and is_inside_boundary(vertices, goal, radius):
            for obstacle in obstacles:
                if rectangle_col_checker(obstacle, goal, radius):
                    success = False
                    break
        else:
            success = False
        if success:
            break
        attempts+=1

    if attempts == max_attempts:
        raise Exception('Failed to generate end point')

    start_angle = (2*np.random.rand() - 1)*np.pi

    return (start, goal, start_angle)

def _manualEnvGeneration():
    generate_envs = True # after each iteration, prompts user to save or discard current env
    show_plot = True
    run_sim = False
    num_envs = 10
    boundary_seed = None
    obstacle_seed = None
    sg_seed = None

    for _ in range(num_envs):
        center = (0,0)
        avg_radius = 10
        irregularity = 1.0
        spikiness = 0.4
        num_vertices = 10
        min_radius = 1
        boundary_vertices = generate_boundary(center, avg_radius, irregularity, spikiness, num_vertices, min_radius, boundary_seed)

        center_bounds = [10,10]
        edge_len_bounds = [0.1, 2]
        num_obstacles = 10
        max_iters = 1000
        obstacles = generate_obstacles(boundary_vertices, center_bounds, edge_len_bounds, obstacle_seed, num_obstacles, max_iters)

        radius = 0.75
        center_bounds = np.array([10, 10])
        min_dist_from_start_to_goal = 1
        start, goal, angle = generate_start_goal(boundary_vertices, obstacles, radius, center_bounds, min_dist_from_start_to_goal, sg_seed)

        file = save_start_goal(start, goal, angle)
        load_start_goal(file)

        if show_plot:
            ax = plt.subplot()
            # This is our boundary
            ax.plot(boundary_vertices[:, 0], boundary_vertices[:, 1], 'b')
            ax.plot([boundary_vertices[0, 0], boundary_vertices[-1, 0]], [boundary_vertices[0, 1], boundary_vertices[-1, 1]], 'b')

            # These are our obstacles
            for obstacle in obstacles:
                plot_obstacle(obstacle, ax=ax)

            # Start ang Goal
            ax.plot(start[0], start[1], 'ro', label='start')
            ax.plot(goal[0], goal[1], 'mo', label='goal')

            ax.set_aspect('equal')
            plt.legend()
            plt.show()

        if run_sim:
            # Lets test this out:
            sim.create_sim()
            load_obstacles(obstacles)
            load_boundary(boundary_vertices)

            forward=0
            turn=0
            while sim.connected(): # use sim.connected() instead of plt.isConnected so we don't need a server id
                time.sleep(1./240.)
                leftWheelVelocity, rightWheelVelocity, forward, turn = sim.keyboard_control(forward, turn, speed=50)
                sim.step_sim(leftWheelVelocity, rightWheelVelocity)
            sim.disconnect()

        if generate_envs:
            if input("Save this environment y/(n)? ") in ['y', 'Y', 'yes', 'Yes']:
                save_curr_env(boundary_vertices, obstacles)
                save_start_goal(start, goal, angle, get_curr_env_idx()-1)
                print('Environment saved')
            else:
                print('Environment discarded')

def _testLidarPlotting():
    botPos = [0,0,0]
    botAngle = 0
    lidarDist = 10
    lidarAngle = 2*np.pi
    numMeasurements = 360
    boundaryFile = './envData/env0/env_boundary.dat'
    obstaclesFile = './envData/env0/env_obstacles.dat'
    with open(str(boundaryFile), 'rb') as boundary_fp:
        boundary = pickle.load(boundary_fp)
    with open(str(obstaclesFile), 'rb') as obstacles_fp:
        obstacles = pickle.load(obstacles_fp)
    ax = plt.subplot()
    plot_boundary(boundary, ax=ax)
    for obstacle in obstacles:
        plot_obstacle(obstacle, ax=ax)
    sim.create_sim(gui=False)
    sim.initLidar(lidarDist, lidarAngle, numMeasurements)
    load_boundary(boundary)
    load_obstacles(obstacles)
    measurements = sim.localLidar(botPos, botAngle)
    sim.plotLidar(botPos, botAngle, measurements, ax)
    plt.show()
    sim.disconnect()

# Main code
if __name__=='__main__':
    _testLidarPlotting()