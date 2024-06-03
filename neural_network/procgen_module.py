from collections import OrderedDict
from typing import Tuple

import torch
from torch import nn

from neural_network.impala import ImpalaCNN
from procgen_wrapper.action_space import LeaperAction, MazeAction


class Module(torch.nn.Module):
    def __init__(self, model:torch.nn.Module, name: str):
        super().__init__()
        self.model = model
        self.name = name

    def forward(self, x):
        return self.model(x)


class ProcgenModule(torch.nn.Module):
    """
    A Qsa head module for N-MCTS
    """

    def __init__(self, env_name: str, init_model_path: str):
        super().__init__()

        if env_name == 'leaper':
            ACTION_SPACE_SIZE = len(LeaperAction)
        else:
            ACTION_SPACE_SIZE = len(MazeAction)

        self.qsa_network = Module(ImpalaCNN([16, 32, 32]), 'backbone')
        self.mean_head = Module(nn.Sequential(OrderedDict([
            ('fc_mean1', nn.Linear(in_features=256, out_features=256)),
            ('relu_mean', nn.ReLU()),
            ('fc_mean2', nn.Linear(in_features=256, out_features=ACTION_SPACE_SIZE)),
        ])), 'mean_head')
        self.std_head = Module(nn.Sequential(OrderedDict([
                ('fc_std1', nn.Linear(in_features=256, out_features=256)),
                ('relu_std', nn.ReLU()),
                ('fc_std2', nn.Linear(in_features=256, out_features=ACTION_SPACE_SIZE)),
            ])), 'std_head')

        self.load_model(init_model_path)
        self.to('cpu')
        self.eval()

    def forward(self, nn_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        See documentation for BaseHead
        """

        if nn_input.dim() == 3:
            nn_input = torch.unsqueeze(nn_input, 0)
        assert nn_input.dim() == 4

        qsa_network_out = self.qsa_network(nn_input)

        qsa_mean = self.mean_head(qsa_network_out)
        qsa_std = self.std_head(qsa_network_out.detach())

        return qsa_mean, qsa_std

    def load_model(self, model_path: str, strict: bool = True):
        state_dict = torch.load(model_path, map_location='cpu')
        self.load_state_dict(state_dict=state_dict, strict=strict)
