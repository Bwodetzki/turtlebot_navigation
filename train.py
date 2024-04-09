import torch as t
import torch.nn as nn
import env_manager as em
from network import PlanningNetwork
from tqdm import tqdm

def load_datapoints(env_num, env_num_max, test_size):
    # TODO: Make it load test size with a fraction or something
    # This implementation of the function will load one environment worh of datapoints at a time
    datafile = f"./envData/env{env_num}/data_points.dat"
    data_points = em.load_data_points(data_file=datafile)
    env_num +=1
    all_data_loaded = False if env_num < env_num_max else True  # I think this logic is good
    return data_points[:-test_size], env_num, all_data_loaded

def format_data(data_points):
    '''
    inputs:
        list of datapoints
    outputs:
        x: tensor containing (angle, curr_pos, target)
        z: tensor of lidar measurements
        true: tensor of RRT* sampled point
    '''
    pass

def main():
    # Definitions
    epochs = 10
    batch_size = 20
    learning_rate = 0
    env_num_max = 500
    freq = 1
    test_size = 100
    run_num=1
    model_path = f"./models/run{run_num}"

    # Torch Definitions
    device = (
        "cuda"
        if t.cuda.is_available()
        else "mps"
        if t.backends.mps.is_available()
        else "cpu"
    )
    t.set_default_device(device)
    print(f"Using {device} device")

    # Define NN
    network = PlanningNetwork(mlp_input_size=5+28, mlp_output_size=2, AE_input_size=360*2, AE_output_size=28)

    # Define Optimizer
    optim = t.optim.Adam(network.params(), lr=learning_rate)

    # TODO: Implement Loading Policy Logic

    # Train Policy
    for e in tqdm(range(epochs)):
        all_data_loaded=False
        while not all_data_loaded:
            # Get Batch Data
            loaded_datapoints, counter, all_data_loaded = load_datapoints(counter, env_num_max, test_size)

            for i in range(0, len(loaded_datapoints), batch_size):
                batch = loaded_datapoints[i:i+batch_size]
                x, z, true = format_data(batch)

                # Calculate Loss
                predictions = network.forward(x, z)
                loss_fun = nn.MSELoss(reduction='sum')
                loss = loss_fun(predictions, true)
                
                loss_fun(predictions, true)

                # Step Parameters
                loss.backwards()
                optim.step()

                # Reset Gradients in Optimizer
                optim.zero_grad()
        # TODO: Add Validation Loss
        validation_loss=None

        # Log Data
        if e % freq == 0:
            t.save({
                'epoch': e,
                'network_params_dict': network.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'loss': validation_loss,
                }, model_path)
    


if __name__ == "__main__":
    main()