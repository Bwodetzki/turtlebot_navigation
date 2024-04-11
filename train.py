import pickle
import os
import torch as t
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import env_manager as em
from network import PlanningNetwork
from tqdm import tqdm

DEVICE = (
    "cuda"
    if t.cuda.is_available()
    else "mps"
    if t.backends.mps.is_available()
    else "cpu"
)

def rot_mat(angle):
    return t.tensor([[t.cos(angle), -t.sin(angle)],
                     [t.sin(angle), t.cos(angle)]])

def to_body_frame(pos, target, angle):
    vector = target - pos
    return rot_mat(angle)@vector

def to_inertial_frame(pos, vector, angle):
    return pos + rot_mat(-angle)@vector

def save_loss_data(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    return filename

def load_loss_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def batches(env_num_max, batch_size, test_size, starting_env_num=0):
    newEnvFlag = False
    batchIdx = 0
    for envIdx in range(starting_env_num, env_num_max+1):
        data_file = em.data_file_fpath(envIdx)
        datapoints = em.load_data_points(data_file=data_file)
        datapoints = datapoints[:-test_size]
        for batchStartIdx in range(0, len(datapoints), batch_size):
            batchEndIdx = batchStartIdx + batch_size
            batch = datapoints[batchStartIdx:batchEndIdx]
            x, lidarMeasurements, targets = format_data(batch)  # Goal is to have x be only the goal vector...
            yield (envIdx, batchIdx, newEnvFlag), x, lidarMeasurements, targets
            newEnvFlag = False
            batchIdx += 1
        if batchStartIdx < len(datapoints):
            yield (envIdx, batchIdx, newEnvFlag), x, lidarMeasurements, targets
            newEnvFlag = False
            batchIdx += 1
        newEnvFlag = True

def test_batches(envIdx, batch_size, test_size):
    batchIdx = 0
    data_file = em.data_file_fpath(envIdx)
    datapoints = em.load_data_points(data_file=data_file)
    datapoints = datapoints[-test_size:]
    for batchStartIdx in range(0, len(datapoints), batch_size):
        batchEndIdx = batchStartIdx + batch_size
        batch = datapoints[batchStartIdx:batchEndIdx]
        x, lidarMeasurements, targets = format_data(batch)  # Goal is to have x be only the goal vector...
        yield (batchIdx), x, lidarMeasurements, targets
        batchIdx += 1
    if batchStartIdx < len(datapoints):
        yield (batchIdx), x, lidarMeasurements, targets
        batchIdx += 1

def load_datapoints(env_num, env_num_max, test_size):
    # TODO: Make it load test size with a fraction or something
    # This implementation of the function will load one environment worh of datapoints at a time
    datafile = f"./envData/env{env_num}/data_points.dat"
    data_points = em.load_data_points(data_file=datafile)
    env_num +=1
    all_data_loaded = False if env_num < env_num_max else True  # I think this logic is good
    return data_points[:-test_size], env_num, all_data_loaded

def load_test_datapoints(env_num, env_num_max, test_size):
    # TODO: Make it load test size with a fraction or something
    # This implementation of the function will load one environment worh of datapoints at a time
    datafile = f"./envData/env{env_num}/data_points.dat"
    data_points = em.load_data_points(data_file=datafile)
    all_data_loaded = False if env_num < env_num_max else True  # I think this logic is good
    return data_points[-test_size:], all_data_loaded

def format_data(data_points):
    '''
    inputs:
        list of datapoints
    outputs:
        x: tensor containing (angle, curr_pos, goal)
        z: tensor of lidar measurements
        true: tensor of RRT* sampled point
    '''
    # If network not learning well, try:
    # Possibly Normalize positions
    # Rotate lidar data through pre-processing
    # Shuffle data to get rid of correlation
    x = []
    z = []
    true = []
    for point in data_points:
        angle = t.tensor(float(point.currAngle), device=DEVICE).reshape(1)
        pos = t.tensor(point.currPosition.astype(np.float32), device=DEVICE)
        end = t.tensor(point.endPosition.astype(np.float32), device=DEVICE)
        goal_vec = to_body_frame(pos, end, angle)
        x.append(goal_vec)

        lidar = t.tensor(point.lidarMeasurements, device=DEVICE)
        z.append(lidar)

        target = t.tensor(point.targetPosition.astype(np.float32), device=DEVICE)
        target_vec = to_body_frame(pos, target, angle)
        true.append(target_vec)   
    return t.stack(x), t.stack(z), t.stack(true)

def main():
    # Definitions
    epochs = 2
    batch_size = 20
    learning_rate = 1e-3
    env_num_max = 550
    freq = 1
    test_size = 100
    run_num = 2
    model_path = f"./models/run{run_num}"
    load_policy=True

    # Torch Definitions
    t.set_default_device(DEVICE)
    print(f"Using {DEVICE} device")

    # Define NN
    network = PlanningNetwork(mlp_input_size=2+28, mlp_output_size=2, AE_input_size=360*2, AE_output_size=28)
    loss_fun = nn.MSELoss(reduction='mean')  # Theres an improvement to be made here
    '''
    Loss function improvement:
    Our network should get the direction of the target vector correct, however, 
    the magnitude of the target vector as in some cases it is somewhat arbitrary.
    A custom loss function with a higher weight corresponding to variations in directions and
    a lower weight for variations in magnitude may have better performance.
    '''

    # Define Optimizer
    optim = t.optim.Adam(list(network.encoder_network.parameters())+list(network.fc_network.parameters()), lr=learning_rate)

    # Loading Policy Logic
    if load_policy:
        data = t.load(model_path)

        epoch = data['epoch']
        print(f'Starting from epoch {epoch}')
        loss = data['loss']
        print(f'Validation loss of model is {loss:0.4f}')
        network.load_state_dict(data['network_params'])
        optim.load_state_dict(data['optimizer_state'])
    else:
        epoch=0
    network.to(DEVICE)

    # Train Policy
    losses = []
    test_losses = []

    # Get initial test_loss point
    network.eval()
    with t.no_grad():
        curr_test_loss=0
        for iterInfo, x, obs, true in test_batches(0, batch_size, test_size):
            batchIdx = iterInfo
            # Calculate test_loss
            predictions = network.forward(x, obs)
            test_loss = loss_fun(predictions, true)
            curr_test_loss += test_loss.to('cpu').detach()
        test_losses.append(test_loss)
    network.train()
    optim.zero_grad()

    for e in tqdm(range(epoch, epochs)):
        for iterInfo, x, obs, true in tqdm(batches(env_num_max, batch_size, test_size, starting_env_num=0)):
            envIdx, batchIdx, newEnvFlag = iterInfo
            # Calculate Loss
            predictions = network.forward(x, obs)

            # Converts to unit vector
            # predictions = predictions / t.linalg.norm(predictions, dim=1).reshape(-1,1)  # Unit vectors
            # true = true / t.linalg.norm(true, dim=1).reshape(-1,1)

            loss = loss_fun(predictions, true)
            losses.append(loss)

            # Step Parameters
            loss.backward()
            optim.step()

            # Reset Gradients in Optimizer
            optim.zero_grad()

            if newEnvFlag:
                network.eval()
                with t.no_grad():
                    curr_test_loss=0
                    for iterInfo, x, obs, true in test_batches(envIdx-1, batch_size, test_size):
                        batchIdx = iterInfo
                        # Calculate test_loss
                        predictions = network.forward(x, obs)

                        # Converts to unit vector
                        # predictions = predictions / t.linalg.norm(predictions, dim=1).reshape(-1,1)  # Unit vectors
                        # true = true / t.linalg.norm(true, dim=1).reshape(-1,1)

                        test_loss = loss_fun(predictions, true)
                        curr_test_loss += test_loss.to('cpu').detach()
                    test_losses.append(test_loss)
                network.train()
                optim.zero_grad()
                # Log Data
                t.save({
                    'epoch': e,
                    'batch_index' : batchIdx,
                    'network_params': network.state_dict(),
                    'optimizer_state': optim.state_dict(),
                    'loss': curr_test_loss / (batchIdx+1),
                    }, model_path)
    losses = np.array([l.to('cpu').detach() for l in losses])
    plt.plot(losses)
    plt.show()
    test_losses = np.array([l.to('cpu').detach() for l in test_losses])
    plt.plot(test_losses)
    plt.show()

    test_loss_dir = f'./models/model_perfs'
    os.makedirs(test_loss_dir, exist_ok=True)
    test_loss_file = f'{test_loss_dir}/test_loss{run_num}.dat'
    save_loss_data(test_losses, test_loss_file)

if __name__ == "__main__":
    main()