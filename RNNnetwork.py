import torch as t
import torch.nn as nn

class PlanningNetwork(nn.Module):
    def __init__(self, mlp_input_size=2+28, mlp_output_size=2, AE_input_size=360*2, AE_output_size=28):
        super().__init__()
        params=0.5
        self.mlp_input_size = mlp_input_size

        self.encoder_network = nn.Sequential(
            nn.Linear(AE_input_size, 512),
            nn.PReLU(),
            nn.Linear(512, 256),
            nn.PReLU(),
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Linear(128, AE_output_size)
        )

        self.fc_network = nn.Sequential(
            nn.Linear(mlp_input_size, 1280),nn.PReLU(), nn.Dropout(p=params),
            nn.Linear(1280, 1024),nn.PReLU(), nn.Dropout(p=params),
            nn.Linear(1024, 896),nn.PReLU(), nn.Dropout(p=params),
            nn.Linear(896, 768),nn.PReLU(), nn.Dropout(p=params),
            nn.Linear(768, 512),nn.PReLU(), nn.Dropout(p=params),
            nn.Linear(512, 384),nn.PReLU(), nn.Dropout(p=params),
            nn.Linear(384, 256),nn.PReLU(), nn.Dropout(p=params),
            nn.Linear(256, 256),nn.PReLU(), nn.Dropout(p=params),
            nn.Linear(256, 128),nn.PReLU(), nn.Dropout(p=params),
            nn.Linear(128, 64),nn.PReLU(), nn.Dropout(p=params),
            nn.Linear(64, 32),nn.PReLU(),
            nn.Linear(32, mlp_output_size)
        )

        self.recurrent_network = nn.Sequential(
            nn.Linear(mlp_input_size, 64), nn.PReLU(),
            nn.Linear(64, 128), nn.PReLU(),
            nn.Linear(128, 128), nn.PReLU(),
            nn.Linear(128, 64), nn.PReLU(),
            nn.Linear(64, mlp_input_size), nn.Tanh()
        )

    def forward(self, x, obs, hiddenState):
        z = self.encoder_network(obs)
        hiddenLayer = t.concatenate((x, z), dim=1)
        hiddenLayer += hiddenState
        hiddenState = self.recurrent_network(hiddenLayer)
        pred = self.fc_network(hiddenLayer)
        return pred, hiddenState
