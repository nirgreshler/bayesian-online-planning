from dataclasses import dataclass


@dataclass
class Config:
    policy_network_temperature_scaling = 2.  # a temperature scaling for the probabilities entered to the P-UCB formula
    pucb_constant = 1.414  # a constant determining the level of exploration in the P-UCB formula

    select_percentile_init = 0.5  # the percentile by which an action is selected in the first visit of a node.
    # Relevant only when select_action_to_explore_by_percentile is True.
    select_percentile_scale = 3.  # the percentile by which an action is selected depends on the number of visits of
    # the node which is normalized by this number. Relevant only when select_action_to_explore_by_percentile is True.

    softmax_action_commitment = False  # whether to use softmax action commitment in all algorithms
    action_commitment_softmax_temperature = 2.  # the softmax temperature for action commitment
    action_commitment_percentile = 0.5  # the percentile to use for non-softmax action commitment in BTS and TSTS

    use_gt_std = False  # Whether to use ground-truth uncertainty estimations
