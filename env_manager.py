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
    curr_env_boundary_file = curr_env_dir / f'{env_data_file_prefix}{curr_env_idx}_boundary{env_data_file_suffix}'
    curr_env_obstacles_file = curr_env_dir / f'{env_data_file_prefix}{curr_env_idx}_obstacles{env_data_file_suffix}'
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
    curr_env_boundary_file = curr_env_dir / f'{env_data_file_prefix}{env_idx}_boundary{env_data_file_suffix}'
    curr_env_obstacles_file = curr_env_dir / f'{env_data_file_prefix}{env_idx}_obstacles{env_data_file_suffix}'
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
    env_dir = env_parent_dir / f'{env_dir_prefix}{curr_env_idx}' / f'path{path_idx}'
    try:
        makedirs(str(env_dir))
    except:
        pass
    file_name = f'path{env_data_file_suffix}'  # Start Goal Configuration
    path_file = env_dir / file_name
    with open(str(path_file), 'wb') as p_fp:
        pickle.dump(data, p_fp)
    
    return path_file

def load_path(path_file):
    with open(str(path_file), 'rb') as p_fp:
        data = pickle.load(p_fp)
    return (data[0], data[1])  # Tuple of start and goal

def save_lidar(measurements, curr_env_idx=0, path_idx=0, env_parent_dir=env_parent_dir):
    env_dir = env_parent_dir / f'{env_dir_prefix}{curr_env_idx}' / f'path{path_idx}'
    try:
        makedirs(str(env_dir))
    except:
        pass
    file_name = f'measurements{env_data_file_suffix}'  # Start Goal Configuration
    measurement_file = env_dir / file_name
    with open(str(measurement_file), 'wb') as m_fp:
        pickle.dump(measurements, m_fp)
    
    return measurement_file

def load_lidar(measurement_file):
    with open(str(measurement_file), 'rb') as m_fp:
        data = pickle.load(m_fp)
    return data  # Tuple of start and goal


def generate_polygon(center: Tuple[float, float], avg_radius: float,
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
    angle_steps = random_angle_steps(num_vertices, irregularity, seed)

    # now generate the points
    points = []
    angle = np.random.uniform(0, 2 * math.pi)
    for i in range(num_vertices):
        radius = clip(np.random.normal(avg_radius, spikiness), min_radius, 2 * avg_radius)
        point = np.array((center[0] + radius * math.cos(angle),
                 center[1] + radius * math.sin(angle)))
        points.append(point)
        angle += angle_steps[i]

    return np.array(points)

def random_angle_steps(steps: int, irregularity: float, seed: int = None) -> List[float]:
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

def clip(value, lower, upper):
    """
    Given an interval, values outside the interval are clipped to the interval
    edges.
    """
    return min(upper, max(value, lower))

def visualize(vertices):
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

def generate_obstacles(vertices, center_bounds=[10, 10], edge_len_bounds=[0.1, 2], seed=None, n=2, max_iters=1000):
    bounding_box = np.array([[0, 0], center_bounds])
    if seed is not None:
        np.random.seed(seed)
    
    centers = np.zeros((n, 2))
    edge_lengths = np.zeros((n, 2))
    angles = np.zeros((n))

    iters = 0
    idx=0
    while iters < max_iters:
        # Generate Samples
        center = (2*np.random.rand(2)-1)*bounding_box[1, :]
        edge_length = np.random.rand(2)*(edge_len_bounds[1]-edge_len_bounds[0]) + edge_len_bounds[0]
        angle = (2*np.random.rand() - 1)*np.pi

        # Check Conditions
        condition = all(abs(center) >= edge_len_bounds[1]/2 + 1) and is_inside_boundary(vertices, center, radius=0.05)
        if condition:
            # Store Samples
            centers[idx, :] = center
            edge_lengths[idx, :] = edge_length
            angles[idx] = angle
            # Update Index
            idx+=1
            # Check if sol found
            if idx==n:
                break
        iters+=1
    
    assert (iters != max_iters)  # Did not converge in the given # of iterations

    return list(zip(centers, edge_lengths, angles))

def plot_obstacle(obstacle, ax=None):
    center, edge_lengths, angle = obstacle
    x_len = edge_lengths[0]
    y_len = edge_lengths[1]

    x_cen = center[0]
    y_cen = center[1]
    
    obs = 0.5 * np.array([[-x_len, -y_len],
                    [-x_len, y_len],
                    [x_len, y_len],
                    [x_len, -y_len]])
    
    rot_mat = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
    obs = (rot_mat@obs.T).T + np.array([x_cen, y_cen])
        
    if ax is None:
        plt.plot(obs[:, 0], obs[:, 1], 'g')
        plt.plot([obs[0, 0], obs[-1, 0]], [obs[0, 1], obs[-1, 1]], 'g')
    else:
        ax.plot(obs[:, 0], obs[:, 1], 'g')
        ax.plot([obs[0, 0], obs[-1, 0]], [obs[0, 1], obs[-1, 1]], 'g')

def load_boundary(vertices):
    wallHeight = 1
    precision = 0.1
    maxDim = np.max(abs(vertices))
    heightFieldDim = math.ceil(2*maxDim/precision)
    height_field = np.zeros((heightFieldDim, heightFieldDim))
    for idx in range(-1, np.size(vertices, axis=0)-1):
        vertex1 = vertices[idx, :]
        vertex2 = vertices[idx+1, :]
        diff = vertex2 - vertex1
        dist = np.sqrt(np.dot(diff, diff))
        direction = diff/dist
        for step in range(math.ceil(dist/precision)):
            worldCoords = vertex1 + step*precision*direction
            x, y = worldCoords/precision + np.array([heightFieldDim/2, heightFieldDim/2])
            x = clip(round(x), 0, heightFieldDim-1)
            y = clip(round(y), 0, heightFieldDim-1)
            height_field[y,x] = wallHeight
    height_field = height_field.flatten()
    boundary_shape = p.createCollisionShape(shapeType = p.GEOM_HEIGHTFIELD, meshScale=[precision, precision, 1], heightfieldData=height_field, numHeightfieldRows=heightFieldDim, numHeightfieldColumns=heightFieldDim)
    boundary  = p.createMultiBody(0, boundary_shape)
    p.resetBasePositionAndOrientation(boundary,[0,0,0], [0,0,0,1])

def load_obstacles(obstacles):
    for center, edge_length, angle in obstacles:
        center = np.concatenate((center, [1/2]))
        edge_length = np.concatenate((edge_length, [1]))
        angle = [0, 0, angle]
        sim.create_box(center, edge_length, angle)

def generate_start_goal(vertices, obstacles, radius=0.75, center_bounds=np.array((10, 10)), dist=1, seed=None, max_iters=1000):
    if seed is not None:
        np.random.seed(seed)

    iters = 0
    while iters < max_iters:
        # Generate Sample
        start = (2*np.random.rand(2)-1)*center_bounds

        # Check Conditions for Start
        rect_cols = [rectangle_col_checker(state, start, radius) for state in obstacles]
        condition =  (not any(rect_cols)) and is_inside_boundary(vertices, start, radius)
        if condition:
                break
        iters+=1
    assert (iters != max_iters)  # Did not converge to the start position in the given # of iterations

    iters = 0
    while iters < max_iters:
        # Generate Sample
        goal = (2*np.random.rand(2)-1)*center_bounds

        # Check Conditions for Goal
        rect_cols = [rectangle_col_checker(state, goal, radius) for state in obstacles]
        condition =  (not any(rect_cols)) and is_inside_boundary(vertices, goal, radius) and np.linalg.norm(start-goal)>dist
        if condition:
                break
        iters+=1
    assert (iters != max_iters)  # Did not converge to the goal position in the given # of iterations

    angle = (2*np.random.rand() - 1)*np.pi

    return (start, goal, angle)


def main():
    generate_envs = True # after each iteration, prompts user to save or discard current env
    show_plot = True
    run_sim = False
    num_envs = 10
    barrier_seed = None
    obstacle_seed = None
    sg_seed = None

    for _ in range(num_envs):
        center = (0,0)
        avg_radius = 10
        irregularity = 1.0
        spikiness = 0.4
        num_vertices = 10
        min_radius = 1
        barrier_vertices = generate_polygon(center, avg_radius, irregularity, spikiness, num_vertices, min_radius, barrier_seed)

        center_bounds = [10,10]
        edge_len_bounds = [0.1, 2]
        num_obstacles = 10
        max_iters = 1000
        obstacles = generate_obstacles(barrier_vertices, center_bounds, edge_len_bounds, obstacle_seed, num_obstacles, max_iters)

        radius = 0.75
        center_bounds = np.array([10, 10])
        min_dist_from_start_to_goal = 1
        start, goal, angle = generate_start_goal(barrier_vertices, obstacles, radius, center_bounds, min_dist_from_start_to_goal, sg_seed)

        file = save_start_goal(start, goal, angle)
        load_start_goal(file)

        if show_plot:
            ax = plt.subplot()
            # This is our barrier
            ax.plot(barrier_vertices[:, 0], barrier_vertices[:, 1], 'b')
            ax.plot([barrier_vertices[0, 0], barrier_vertices[-1, 0]], [barrier_vertices[0, 1], barrier_vertices[-1, 1]], 'b')

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
            load_boundary(barrier_vertices)

            forward=0
            turn=0
            while sim.connected(): # use sim.connected() instead of plt.isConnected so we don't need a server id
                time.sleep(1./240.)
                leftWheelVelocity, rightWheelVelocity, forward, turn = sim.keyboard_control(forward, turn, speed=50)
                sim.step_sim(leftWheelVelocity, rightWheelVelocity)
            sim.disconnect()

        if generate_envs:
            if input("Save this environment y/(n)? ") in ['y', 'Y', 'yes', 'Yes']:
                save_curr_env(barrier_vertices, obstacles)
                save_start_goal(start, goal, angle, get_curr_env_idx()-1)
                print('Environment saved')
            else:
                print('Environment discarded')

# Main code
if __name__=='__main__':
    main()