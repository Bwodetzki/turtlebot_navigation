import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import simulation as sim
from rrt import rrt_star
from env_manager import generate_obstacles, generate_polygon, generate_start_goal, plot_obstacle, save_env, save_path, save_lidar, load_obstacles, load_boundary
'''
Aim is to generate a set # of environments (obstacles, boundary, start position (and orientation), and goal position)
'''
def get_lidar_measurements(path, angle, lidar_params):
    measurements = []
    path = np.hstack((path, np.zeros((len(path), 1))))

    hit_fraction = sim.localLidar(path[0], 
                              angle,
                              lidar_params['lidarDist'],
                              lidar_params['lidarAngle'],
                              lidar_params['numMeasurements'])
    measurements.append(hit_fraction)

    for i in range(1, len(path)):
        vector = path[i] - path[i-1]
        angle = np.arctan2(vector[1], vector[0])  # Angle is reletive to previous path points
        hit_fraction = sim.localLidar(path[i], 
                              angle,
                              lidar_params['lidarDist'],
                              lidar_params['lidarAngle'],
                              lidar_params['numMeasurements'])
        measurements.append(hit_fraction)
    # Now we can doctor our measurements here. Do we want to use polar coordinates for each measurement?
    measurements = np.array(measurements)*lidar_params['lidarDist']

    return measurements

def generate_env(by_params, obs_params, plot=False):
    barrier_vertices = generate_polygon(by_params['center'], 
                                        by_params['avg_radius'], 
                                        by_params['irregularity'], 
                                        by_params['spikiness'], 
                                        by_params['num_vertices'], 
                                        by_params['min_radius'], 
                                        by_params['barrier_seed'])
    obstacles = generate_obstacles(barrier_vertices, 
                                   obs_params['center_bounds'], 
                                   obs_params['edge_len_bounds'], 
                                   obs_params['obstacle_seed'], 
                                   obs_params['num_obstacles'])
    if plot:
        try:
            ax = plt.gca()
        except:
            ax = plt.subplots()
        # This is our barrier
        ax.plot(barrier_vertices[:, 0], barrier_vertices[:, 1], 'b')
        ax.plot([barrier_vertices[0, 0], barrier_vertices[-1, 0]], [barrier_vertices[0, 1], barrier_vertices[-1, 1]], 'b')

        # These are our obstacles
        for obstacle in obstacles:
            plot_obstacle(obstacle, ax=ax)

    return (barrier_vertices, obstacles)

def generate_paths(barrier_vertices, obstacles, sg_params, RRTs_params, plot=False):
    start, goal, angle = generate_start_goal(barrier_vertices, 
                                      obstacles, 
                                      sg_params['radius'], 
                                      sg_params['center_bounds'], 
                                      sg_params['dist'], 
                                      sg_params['sg_seed'])
    path = rrt_star(barrier_vertices, obstacles, start, goal, RRTs_params)
    path.reverse()
    path = np.array(path)
    if path is None:
        # print('bad')
        path = generate_paths(barrier_vertices, obstacles, sg_params, RRTs_params, plot=False)  # or something like that

    if plot:
        try:
            # print("here")
            ax = plt.gca()
        except:
            ax = plt.subplots()
        
        # Start ang Goal
        ax.plot(start[0], start[1], 'ro', label='start')
        ax.plot(goal[0], goal[1], 'mo', label='goal')
        ax.plot(path[:, 0], path[:, 1])

    return path, angle


if __name__=='__main__':
    env_size=10
    boundary_params = {
        'center' : (0,0),
        'avg_radius' : env_size,
        'irregularity' : 1.0,
        'spikiness' : 0.4,
        'num_vertices' : 10,
        'min_radius' : 1,
        'barrier_seed' : None
    }

    obstacle_params = {
        'center_bounds' : [-env_size, env_size],
        'edge_len_bounds' : [0.1, 2],
        'num_obstacles' : 10,
        'obstacle_seed' : None
    }

    start_goal_params = {
        "radius" : 0.75,
        "center_bounds" : np.array([-env_size, env_size]),
        "dist" : 1,
        'sg_seed' : None
    }

    RRTs_params = {
        'sample_bounds' : np.array((-env_size, env_size)),
        'turtle_radius' : 0.5,
        'max_iters' : 30
    }

    lidar_params = {
        'lidarDist' : 1,
        'lidarAngle' : 2*np.pi,
        'numMeasurements' : 360
    }

    num_envs = 1
    num_paths = 2
    env_parent_dir = Path("./envData").absolute().resolve() 
    plot_env = True
    plot_path = True

    for i in range(num_envs):
        barrier_vertices, obstacles = generate_env(boundary_params, obstacle_params, plot_env)
        save_env(barrier_vertices, obstacles, env_idx=i, env_parent_dir=env_parent_dir)  # NOTE: This overwrites other envs
        # Connect to physics server (for lidar)
        sim.create_sim(load_turtle=False)
        load_boundary(barrier_vertices)
        load_obstacles(obstacles)

        for j in range(num_paths):
            path, angle = generate_paths(barrier_vertices, obstacles, start_goal_params, RRTs_params, plot_path)
            save_path(path, angle, curr_env_idx=i, path_idx=j, env_parent_dir=env_parent_dir)

            # Generate and save lidar            
            measurements = get_lidar_measurements(path, angle, lidar_params)
            save_lidar(measurements, curr_env_idx=i, path_idx=j, env_parent_dir=env_parent_dir)
            plt.show()
        sim.disconnect()