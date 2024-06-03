from typing import List

import numpy as np

from planners.bts import BTS
from planners.buct_node import BUCTNode
from procgen_wrapper.action_space import ProcgenAction
from procgen_wrapper.extended_state import ExtendedState
from procgen_wrapper.procgen_simulator import ProcgenSimulator

APPROXIMATE_GAUSSIAN = True


class TSTS(BTS):
    def __init__(self, simulator: ProcgenSimulator, nn_model_path: str, deterministic: bool = False):
        """
        If 'deterministic' is True, use a constant seed (this is TSTS_det).

        """
        super().__init__(simulator, nn_model_path)
        # Create a random generator
        self._prng = np.random.RandomState()
        self._is_deterministic = deterministic

    def _search(self, root_state: ExtendedState, max_iterations: int) -> BUCTNode:
        if self._is_deterministic:
            # Set a constant seed for reproducible Thompson sampling
            self._prng = np.random.RandomState(seed=0)
        return super()._search(root_state, max_iterations)

    def _select_action_to_explore(self, node: BUCTNode, actions: List[ProcgenAction]) -> ProcgenAction:
        """
        Select the action to explore by using thompson sampling on their max-induced distributions.
        :param node: the node to choose an action from
        :param actions: list of actions to choose from
        :return: the action with the highest sample from thompson sampling.
        """
        if APPROXIMATE_GAUSSIAN:
            distribution_samples = np.array([self._prng.normal(node.qsa_posterior_max[action].expectation,
                                                               node.qsa_posterior_max[action].std)
                                             for action in actions])
        else:
            distribution_samples = np.zeros_like(actions)
            for i, a in enumerate(actions):
                bins, pdf = node.qsa_posterior_max[a].pdf
                bin_size = bins[1] - bins[0]
                distribution_samples[i] = self._prng.choice(a=bins, p=pdf * bin_size) + \
                                          self._prng.uniform(low=0, high=bin_size)

        return actions[np.argmax(distribution_samples)]
