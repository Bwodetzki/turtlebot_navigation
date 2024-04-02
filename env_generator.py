import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import simulation as sim
from rrt import rrt_star
from data_manager import TrainingPoint
import env_manager as em
'''
Aim is to generate a set # of environments (obstacles, boundary, start position (and orientation), and goal position)
'''
def get_lidar_measurements(path, angle, lidar_params):
    measurements = []
    path = np.hstack((path, np.zeros((len(path), 1))))

    hit_fraction = sim.localLidar(path[0], angle)
    measurements.append(hit_fraction)

    for i in range(1, len(path)):
        vector = path[i] - path[i-1]
        angle = np.arctan2(vector[1], vector[0])  # Angle is relative to previous path points
        hit_fraction = sim.localLidar(path[i], angle)
        measurements.append(hit_fraction)
    # Now we can doctor our measurements here. Do we want to use polar coordinates for each measurement?
    # measurements = np.array(measurements)*lidar_params['lidarDist']

    return measurements

def generate_env(by_params, obs_params, plot=False):
    boundary_vertices = em.generate_boundary(by_params['center'], 
                                        by_params['avg_radius'], 
                                        by_params['irregularity'], 
                                        by_params['spikiness'], 
                                        by_params['num_vertices'], 
                                        by_params['min_radius'], 
                                        by_params['boundary_seed'])
    obstacles = em.generate_obstacles(boundary_vertices, 
                                   obs_params['center_bounds'], 
                                   obs_params['edge_len_bounds'], 
                                   obs_params['obstacle_seed'], 
                                   obs_params['num_obstacles'])
    if plot:
        try:
            ax = plt.gca()
        except:
            ax = plt.subplots()
        # This is our boundary
        ax.plot(boundary_vertices[:, 0], boundary_vertices[:, 1], 'b')
        ax.plot([boundary_vertices[0, 0], boundary_vertices[-1, 0]], [boundary_vertices[0, 1], boundary_vertices[-1, 1]], 'b')

        # These are our obstacles
        for obstacle in obstacles:
            em.plot_obstacle(obstacle, ax=ax)

    return (boundary_vertices, obstacles)

def generate_paths(boundary_vertices, obstacles, sg_params, RRTs_params, plot=False, max_iters=20):
    i=0
    path = None
    while path is None and i<max_iters:
        # print('here')
        start, goal, angle = em.generate_start_goal(boundary_vertices, 
                                        obstacles, 
                                        sg_params['radius'], 
                                        sg_params['center_bounds'], 
                                        sg_params['dist'], 
                                        sg_params['sg_seed'])
        path = rrt_star(boundary_vertices, obstacles, start, goal, RRTs_params)
        i+=1
    assert i!=max_iters  # Did not converge

    path.reverse()
    path = np.array(path)

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

def data_from_path(path, angle, lidar_measurements):
    goal = path[-1]

    # Initial Data Point
    data_point = TrainingPoint(
        currPosition=path[0],
        currAngle=angle,
        lidarMeasurements=lidar_measurements[0],
        targetPosition=path[1],
        endPosition=goal
    )
    data_points = [data_point]

    for i in range(1, len(path)-1):  # Begins 1 after start, Stops one before the goal
        vector = path[i] - path[i-1]
        angle = np.arctan2(vector[1], vector[0])

        data_point = TrainingPoint(
            currPosition=path[i],
            currAngle=angle,
            lidarMeasurements=lidar_measurements[i],
            targetPosition=path[i+1],
            endPosition=goal
        )
        data_points.append(data_point)
    return data_points

if __name__=='__main__':
    env_size=10
    boundary_params = {
        'center' : (0,0),
        'avg_radius' : env_size,
        'irregularity' : 1.0,
        'spikiness' : 0.4,
        'num_vertices' : 10,
        'min_radius' : 1,
        'boundary_seed' : None
    }

    obstacle_params = {
        'center_bounds' : [env_size*2, env_size*2],
        'edge_len_bounds' : [0.1, 2],
        'num_obstacles' : 15,
        'obstacle_seed' : None
    }

    start_goal_params = {
        "radius" : 0.75,
        "center_bounds" : np.array([env_size*2, env_size*2]),
        "dist" : 1,
        'sg_seed' : None
    }

    RRTs_params = {
        'sample_bounds' : np.array((env_size*2, env_size*2)),
        'turtle_radius' : 0.5,
        'max_iters' : 100,
        'max_replan' : 0,
        'downsample_size' : 3
    }

    lidar_params = {
        'lidarDist' : 1,
        'lidarAngle' : 2*np.pi,
        'numMeasurements' : 360
    }

    num_envs = 5
    num_paths = 1
    num_data_points = 4
    env_parent_dir = Path("./envData").absolute().resolve() 
    plot_env = False
    plot_path = False

    sim.initLidar(lidar_params['lidarDist'], lidar_params['lidarAngle'], lidar_params['numMeasurements'])
    for env_idx in range(num_envs):
        print(f'Generating environment {env_idx}')
        boundary_vertices, obstacles = generate_env(boundary_params, obstacle_params, plot_env)
        em.save_env(boundary_vertices, obstacles, env_idx=env_idx, env_parent_dir=env_parent_dir)  # NOTE: This overwrites other envs

        sim.create_sim(gui=False, load_turtle=False)
        em.load_boundary(boundary_vertices)
        em.load_obstacles(obstacles)

        # for path_idx in range(num_paths):
        num_points = 0
        path_idx = 0
        data_points = []
        while num_points < num_data_points:
            print(f'Generating path {path_idx}')
            path, angle = generate_paths(boundary_vertices, obstacles, start_goal_params, RRTs_params, plot_path)
            em.save_path(path, angle, curr_env_idx=env_idx, path_idx=path_idx, env_parent_dir=env_parent_dir)
            
            # Generate and save lidar            
            measurements = get_lidar_measurements(path, angle, lidar_params)
            em.save_lidar(measurements, curr_env_idx=env_idx, path_idx=path_idx, env_parent_dir=env_parent_dir)

            # Generate Nice Data
            data_points = data_points + data_from_path(path, angle, measurements)

            # Step 
            path_idx += 1
            num_points += len(data_points)

            # Plotting
            sim.clear_sim()
            if plot_env or plot_path:
                plt.gca().axis('equal')
                plt.show()
        # Save Data
        data_points = data_points[:num_data_points]  # Trim the excess
        em.save_data(data_points, curr_env_idx=env_idx, path_idx=path_idx, env_parent_dir=env_parent_dir)
    sim.disconnect()

    file = env_parent_dir / f'env{0}' / f'data_{num_data_points}_points.dat'
    data_points = em.load_data_points(file)