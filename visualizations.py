import env_generator as eg
import env_manager as em
import collision_checker as cc
import numpy as np
import simulation as sim
import matplotlib.pyplot as plt
from pathlib import Path
import time

def steerTo(point1, point2, obstacle, res=100):
    vec = point2 - point1
    vec_mag = np.linalg.norm(vec)
    unit_vec = vec/vec_mag

    test_mags = np.linspace(0, vec_mag, res)

    for test_mag in test_mags:
        point = point1 + unit_vec*test_mag
        col = cc.rectangle_col_checker(obstacle, point , radius=0.01)
        if col:
            return False
    return True

'''
animation steps:
1. upsampled lines plotted
2. shows true connection overlayed and different color
3. connects and redraws path
4. downsamples path
'''

def lvc(path_node_list, obstacle, ax, line):
    if len(path_node_list) <= 2:
        return path_node_list
    point = ax.plot(path_node_list[0][0], path_node_list[0][1], 'bo')
    
    for i in range(len(path_node_list)-1, 0+1, -1):
        # Plot Test Path
        connection = steerTo(path_node_list[0], path_node_list[i], obstacle)
        if connection:
            line2 = plt.plot([path_node_list[0][0], path_node_list[i][0]], [path_node_list[0][1], path_node_list[i][1]], 'b')
            plt.pause(0.5)
            # delete elements from (0+1, i]
            path_node_list = [path_node_list[idx] for idx in range(len(path_node_list)) if (idx == 0) or (idx >= i)]
            l = line.pop(0)
            l.remove()
            l2 = line2.pop(0)
            l2.remove()
            line = plt.plot(np.array(path_node_list)[:, 0], np.array(path_node_list)[:, 1], 'r.-')
            plt.pause(0.5)

            # Plot Path
            path_node_list = upsample(path_node_list)
            l = line.pop(0)
            l.remove()
            plt.plot(np.array(path_node_list)[:2, 0], np.array(path_node_list)[:2, 1], 'r.-')
            line = plt.plot(np.array(path_node_list)[1:, 0], np.array(path_node_list)[1:, 1], 'r.-')
            plt.pause(0.5)
            break
    p = point.pop(0)
    p.remove()
    upper_list = lvc(path_node_list[1:], obstacle, ax, line)
    path_node_list = [path_node_list[0]] + upper_list
    return path_node_list

def upsample(path, upsample_size=0.5):
    # plt.clf()
    n_path = np.array(path)
    # plt.plot(n_path[:, 0], n_path[:, 1])
    new_ind = 0
    for i in range(len(path)-1):
        vector = n_path[i+1] - n_path[i]
        length = np.linalg.norm(vector)
        num_subsamples = int(length//upsample_size)
        subsample_vec_mag = length/(num_subsamples+1)

        j = 0
        for j in range(1, num_subsamples+1):
            subsample = subsample_vec_mag*j*vector/length + n_path[i]
            # plt.plot(subsample[0], subsample[1], 'ro')
            path.insert(new_ind+j, np.array(subsample))
        new_ind = new_ind+j+1
    return path
    
def upsampling_demo():
    path = np.array([[1.5, 0],
                    [0, 4],
                    [-1.5, 0]])
    center = np.array((0, 0))
    edge_lengths = np.array((1, 1))
    angle = 0
    obs_data = (tuple(center), tuple(edge_lengths), angle)

    # Begin Plots
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    em.plot_obstacle(obs_data, ax=ax)
    ax.plot(path[0, 0], path[0, 1], 'm^', label='Goal')
    ax.plot(path[-1, 0], path[-1, 1], 'g^', label='Start')
    line = ax.plot(path[:, 0], path[:, 1], 'r.-')
    plt.legend()
    plt.pause(0.5)
    time.sleep(20)
    
    # Upsample
    path = np.array(upsample(list(path)))
    # Plot Upsampled
    l = line.pop(0)
    l.remove()
    line = ax.plot(path[:, 0], path[:, 1], 'r.-')
    plt.pause(0.5)

    # LVC
    path = np.array(lvc(path, obs_data, ax=ax, line=line))

    # Run LVC
    # ax.plot(path[:, 0], path[:, 1], 'ro-')
    plt.pause(0.5)
    plt.show()

def turtlebot_anim():
    sim.create_sim()

    speed = np.array([1., 3.])
    i = 0
    while 1:
        leftWheelVelocity, rightWheelVelocity = speed
        sim.step_sim(leftWheelVelocity, rightWheelVelocity)

        i+=1e-10
        speed = speed + speed*i

def environment_plotter():
    env_nums = [2]
    path_nums = [3, 4]

    for env_num in env_nums:
        for path_num in path_nums:
            # Load Environment
            env_path = Path(f"./envData/env{env_num}").absolute().resolve()

            obs_file = env_path / f'obstacles.dat'
            boundary_file = env_path / f'boundary.dat'

            boundary, obstacles = em.load_env(boundary_file, obs_file, use_sim=False)

            # Load Path
            path_file = env_path / f'path{path_num}' / 'path.dat'
            path, angle = em.load_path(path_file)

            # Plot
            fig, ax = plt.subplots()
            em.plot_boundary(boundary, ax=ax)
            for obstacle in obstacles:
                    em.plot_obstacle(obstacle, ax=ax)
            ax.plot(path[:, 0], path[:, 1], 'ro-')

            ax.set_title(f'Env {env_num}, Path {path_num}')
            plt.show()

if __name__=="__main__":
    turtlebot_anim()