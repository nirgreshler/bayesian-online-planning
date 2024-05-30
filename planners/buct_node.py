from typing import Dict, Optional, List

from planners.tree_node import TreeNode
from procgen_wrapper.action_space import ProcgenAction
from procgen_wrapper.extended_state import ExtendedState
from utils.distribution import ScalarDistribution


class BUCTNode(TreeNode):
    """
        A Bayesian UCT node is a node in the Bayesian UCT resulted tree.
        In addition of being a TreeNode, it holds the reward per action, the prior and posteriror distribution of Q(s,a) per
        action, the prior and posterior distribution of the value, and the number of visits of the node.
        """

    def __init__(self, state: ExtendedState, parent: 'BUCTNode' = None, is_terminal_state: bool = False):
        """
        ctr
        :param state: the state of the node
        :param parent: the parent node (None for a root node)
        :param terminal_state: an indicator whether the state is terminal and it's type
        """
        super().__init__(state=state, parent=parent, is_terminal_state=is_terminal_state)

        self._qsa_prior: Dict[ProcgenAction, ScalarDistribution] = dict()
        self._qsa_posterior: Dict[ProcgenAction, ScalarDistribution] = dict()
        self._qsa_posterior_max: Dict[ProcgenAction, ScalarDistribution] = dict()

        self._value_prior: Optional[ScalarDistribution] = None
        self._value_posterior: Optional[ScalarDistribution] = None
        self._value_posterior_max: Optional[ScalarDistribution] = None

        if parent is None:
            # Initialize the root node with a single visit
            self._num_visits = 1

    @property
    def rewards(self) -> Dict[ProcgenAction, float]:
        """
        Return a dictionary of the reward per action
        """
        return self._rewards

    @property
    def qsa_prior(self) -> Dict[ProcgenAction, ScalarDistribution]:
        """
        Return a dictionary of prior distribution of the Q(s,a) per action
        """
        return self._qsa_prior

    @property
    def qsa_posterior(self) -> Dict[ProcgenAction, ScalarDistribution]:
        """
        Return a dictionary of posterior distribution of the Q(s,a) per action
        """
        return self._qsa_posterior

    @property
    def qsa_posterior_max(self) -> Dict[ProcgenAction, ScalarDistribution]:
        """
        Return a dictionary of posterior distribution of the Q(s,a) per action
        """
        return self._qsa_posterior_max

    @property
    def value_prior(self) -> Optional[ScalarDistribution]:
        """
        Return a prior distribution of the state value
        """
        return self._value_prior

    @value_prior.setter
    def value_prior(self, value: ScalarDistribution):
        """
        Set the prior distribution of the state value
        """
        self._value_prior = value

    @property
    def value_posterior(self) -> Optional[ScalarDistribution]:
        """
        Return a posterior distribution of the state value
        """
        return self._value_posterior

    @value_posterior.setter
    def value_posterior(self, value: ScalarDistribution):
        """
        Set the posterior distribution of the state value
        """
        self._value_posterior = value

    @property
    def value_posterior_max(self) -> Optional[ScalarDistribution]:
        """
        Return a posterior distribution of the state value
        """
        return self._value_posterior_max

    @value_posterior_max.setter
    def value_posterior_max(self, value: ScalarDistribution):
        """
        Set the posterior distribution of the state value
        """
        self._value_posterior_max = value

    @property
    def num_visits(self) -> int:
        """
        Return the number of visits of the node
        """
        return self._num_visits

    @num_visits.setter
    def num_visits(self, value: int):
        """
        Set the number of visits of the node
        """
        self._num_visits = value

    def get_sorted_actions(self, also_unexplored: bool = False) -> List[ProcgenAction]:
        # TODO get_sorted_actions shouldn't be a part of the node (in UCT as well)
        """
        Sort the explored actions by a pre-defined percentile of their Q(s,a) distributions, from highest to lowest.
        :return: list of the sorted action
        """
        qsas_percentile = {}
        marginalization_distribution = self.qsa_posterior_max if Config().mcts.bayesian_uct.use_max_distribution_in_marginalization else self.qsa_posterior
        actions = self.available_actions if also_unexplored else self.explored_actions
        # TODO keep marginalize_using_thompson_sampling ?
        # if Config().mcts.bayesian_uct.marginalize_using_thompson_sampling:
        #     qsa_values = np.array([marginalization_distribution[a].expectation for a in actions])
        #     shifted_qsa_values = qsa_values - np.max(qsa_values)
        #     scaled_shifted_qsa_values = shifted_qsa_values / Config().mcts.policy_network_temperature_scaling
        #     # When using fp32 np.exp(-89.) causes underflow errors, this is used to avoid this underflow
        #     # since the exp_qsa_values are divided by their sum, which can be larger than 1, we use a value of -80
        #     # which provides enough margin even if their sum is 1500 (i.e 1500 actions with similar qsa).
        #     scaled_shifted_qsa_values = np.clip(scaled_shifted_qsa_values.astype(np.float32), a_min=-80., a_max=0.)
        #     exp_qsa_values = np.exp(scaled_shifted_qsa_values)
        #     action_probs = exp_qsa_values / np.sum(exp_qsa_values)
        #     chosen_idx = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        #     return [actions[chosen_idx]]
        # else:

        marginalization_percentile = 0.5  # TODO in config?

        for action in actions:
            qsas_percentile[action] = marginalization_distribution[action].interpolate_inverse_cdf(marginalization_percentile)

        return sorted(actions, key=lambda k: qsas_percentile[k], reverse=True)

    def __str__(self):
        return "Expected value: {:.2f}, STD value {:.2f}, Num visits: {:.2f}".format(self.value_posterior.expectation,
                                                                                     self.value_posterior.std,
                                                                                     self.num_visits)
