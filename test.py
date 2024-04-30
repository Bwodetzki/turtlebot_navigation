import pickle
import RNNnetwork
import torch as t
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import argparse as arg
import simulation as sim
import env_generator as eg
import env_manager as em
from timeit import default_timer as timer
from pathlib import Path
from network import PlanningNetwork
from rrt import rrt_star, neural_rrt

DEVICE = (
    "cuda"
    if t.cuda.is_available()
    else "mps"
    if t.backends.mps.is_available()
    else "cpu"
)

def RRT_star(boundary, obstacles, start, goal, RRTs_params):  # Change RRTS_params?
    try:
        path = rrt_star(boundary, obstacles, start, goal, RRTs_params)
    except: 
        path = None
    success = 1 if path is not None else 0 # May not return None
    return success, path

def informed_sampling(boundary, obstacles, start, goal, RRTs_params, net, rnn, use_rrt):
    try:
        path = neural_rrt(boundary, obstacles, start, goal, RRTs_params, net, rnn, use_rrt)
    except:
        path = None
    success = 1 if path is not None else 0
    return success, path

def load_vars(env_num, path_num, params):
    # Load Environment and Other vars
    sim.create_sim(load_turtle=False, gui=False)  # Will load turtle
    boundary_params, obstacle_params, sg_params, lidar_params = params
    sim.initLidar(lidar_params['lidarDist'], lidar_params['lidarAngle'], lidar_params['numMeasurements'])

    # Select environment
    if env_num is not None:
        env_path = Path(f"./envData/env{env_num}").absolute().resolve()

        obs_file = env_path / f'obstacles.dat'
        boundary_file = env_path / f'boundary.dat'

        boundary, obstacles = em.load_env(boundary_file, obs_file)
    else:
        boundary, obstacles = eg.generate_env(boundary_params, obstacle_params)
        em.load_boundary(boundary)
        em.load_obstacles(obstacles)
    
    # Select start and goal
    if path_num is not None:
            path_file = env_path / f'path{path_num}' / 'path.dat'
            path, angle = em.load_path(path_file)
            start = path[0]
            goal = path[-1]
    else:  # Generate Path in given env

        start, goal, angle = em.generate_start_goal(boundary, 
                                                        obstacles, 
                                                        sg_params['radius'], 
                                                        sg_params['center_bounds'], 
                                                        sg_params['dist'], 
                                                        sg_params['sg_seed'])

    return boundary, obstacles, start, goal

def generate_test_data(args):
    run = 2 # net to use
    env_num = args.env
    path_num = args.path - 1 if args.path is not None else None
    # corn = args.corn
    num_its = 1000 # CHANGEME

    # env_num=None
    # path_num=None

    env_size, boundary_params, obstacle_params, sg_params, RRTs_params, lidar_params = eg.load_params('./parameters.dat') # Load generation params
    params = (boundary_params, obstacle_params, sg_params, lidar_params)

    # Load Network
    if run is not None:
        net = PlanningNetwork()
        model_path = model_path = f"./good_models/run{run}"
        data = t.load(model_path, map_location=DEVICE)

        loss = data['loss']
        print(f'Validation loss of model is {loss:0.4f}')
        net.load_state_dict(data['network_params'])

    rnn_net = RNNnetwork.PlanningNetwork()
    model_path = model_path = f"./RNN_good_models/complicatedModels/run6"
    data = t.load(model_path, map_location=DEVICE)

    loss = data['loss']
    print(f'Validation loss of RNN model is {loss:0.4f}')
    rnn_net.load_state_dict(data['network_params'])

    RRT_star_data = []
    MPNET_data = []
    RNN_data = []

    paths_per_env = 3
    change_env = True
    change_path = True
    for i in range(num_its):
        # Step env_num if applicable
        if env_num is not None:
            if change_env and path_num >= paths_per_env:
                env_num += 1
                path_num = -1
        if (env_num is not None) and (path_num is not None):
            path_num += 1

        try:
            boundary, obstacles, start, goal = load_vars(env_num, path_num, params)
        except:
            boundary, obstacles, start, goal = load_vars(env_num, path_num, params)

        if args.rrts:
            print('Starting RRT Planning')
            start_t = timer()
            success, path = RRT_star(boundary, obstacles, start, goal, RRTs_params)
            end_t = timer()
            rrts_data = [success, end_t-start_t]
            RRT_star_data.append(rrts_data)

        if args.net:
            print('Starting Informed Planning')
            start_t = timer()
            success, path = informed_sampling(boundary, obstacles, start, goal, RRTs_params, net, rnn=False, use_rrt=args.use_rrt)
            end_t = timer()
            mpnet_data = [success, end_t-start_t]
            MPNET_data.append(mpnet_data)

        if args.rnn:
            print('Starting RNN Informed Planning')
            start_t = timer()
            success, path = informed_sampling(boundary, obstacles, start, goal, RRTs_params, rnn_net, rnn=True, use_rrt=args.use_rrt)
            end_t = timer()
            rnn_data = [success, end_t-start_t]
            RNN_data.append(rnn_data)

        # # Plot Path Found
        # # This is our boundary
        # fig, ax = plt.subplots()
        # em.plot_boundary(boundary, ax=ax)

        # # These are our obstacles
        # for obstacle in obstacles:
        #     em.plot_obstacle(obstacle, ax=ax)
        
        # path = np.array(path)
        # ax.plot(start[0], start[1], 'ro', label='start')
        # ax.plot(goal[0], goal[1], 'mo', label='goal')
        # ax.plot(path[:, 0], path[:, 1], 'ro-')
        # plt.show()

    data_list = []
    if args.rrts:
        data_list.append(RRT_star_data)
    if args.net:
        data_list.append(MPNET_data)
    if args.rnn:
        data_list.append(RNN_data)
    # data_list = [RRT_star_data, MPNET_data, RNN_data]
    # data_list = [RRT_star_data, MPNET_data]
    # data_list = [RNN_data,]
    success_percs = []
    time_stats = []
    for data in data_list:
        successes = np.array(data)[:, 0]
        success_perc = 100*sum(successes) / len(successes)
        success_percs.append(success_perc)

        time = np.array(data)[:, 1]
        time_mean = np.mean(time)
        time_sigma = np.std(time)
        time_stats.append([time_mean, time_sigma])

    # Save Data
    fname=args.fname # 'test_data/rnn_.pk'
    data_store = (success_percs, time_stats, data_list)
    with open(fname, 'wb') as f: # create file if needed
        pickle.dump(data_store, f)

    return success_percs, time_stats
    
if __name__ == "__main__":
    parser = arg.ArgumentParser()
    parser.add_argument('--env', type=int, default=None, help="The environment number to test the turtlebot in, use None to generate one")
    parser.add_argument('--path', type=int, default=None, help="The path in the environment, use None to generate one")
    parser.add_argument('--fname', type=str, default='test_data/general_test.pk', help="file to save data")
    parser.add_argument('--rrts', type=int, default=0, help="whether or not to use rrtstar, 1 or 0")
    parser.add_argument('--net', type=int, default=0, help="whether or not to use mpnet, 1 or 0")
    parser.add_argument('--rnn', type=int, default=1, help="whether or not to use rnn, 1 or 0")
    parser.add_argument('--use_rrt', type=int, default=1, help="whether or not to use rrt with the networks, 1 or 0")
    args = parser.parse_args()
    generate_test_data(args)