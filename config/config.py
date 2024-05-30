from dataclasses import dataclass


@dataclass
class Config:
    policy_network_temperature_scaling = 2.  # 0.2 # a temperature scaling for the probabilities entered to the P-UCB formula
    pucb_constant = 1.414  # a constant determining the level of exploration in the P-UCB formula
