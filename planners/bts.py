from typing import List

import numpy as np
import torch

from config.config import Config
from planners.buct_node import BUCTNode
from planners.planner_base import PlannerBase
from procgen_wrapper.action_space import ProcgenAction
from procgen_wrapper.procgen_simulator import ProcgenSimulator
from utils.distribution import ScalarDistribution, DistributionTransformationUtils, MIN_STD
from utils.get_gt_qsa import get_gt_qsa

APPROXIMATE_GAUSSIAN_PERCENTILE = True  # True if to approximate the percentile calculation in the select step
MAX_NODE_VISITS = 150


class BTS(PlannerBase):
    def __init__(self, simulator: ProcgenSimulator, nn_model_path: str):
        super().__init__(simulator, nn_model_path, BUCTNode)
        self._transform_utils = DistributionTransformationUtils()

    def _commit_action(self, root_node: BUCTNode) -> ProcgenAction:
        """
        Sort the explored actions by a pre-defined percentile of their Q(s,a) distributions, from highest to lowest.
        :return: list of the sorted action
        """
        qsas_percentile = {}
        actions = root_node.actions
        if Config().softmax_action_commitment:
            qsa_values = np.array([root_node.qsa_posterior[a].expectation for a in actions])
            chosen_idx = self._sample_best_action(qsa_values)
            return actions[chosen_idx]

        for action in actions:
            qsas_percentile[action] = \
                root_node.qsa_posterior[action].interpolate_inverse_cdf(Config().action_commitment_percentile)

        return sorted(actions, key=lambda k: qsas_percentile[k], reverse=True)[0]

    def _select(self, node: BUCTNode) -> ProcgenAction:
        """
        Select an action to explore
        :param node: the node from which an action is selected
        :return: the selected action
        """
        # Generate the available actions and the Qsa priors if needed
        actions = self._get_actions(node=node)
        if len(node.qsa_prior) == 0:
            self._generate_qsa_priors(node=node)

        return self._select_action_to_explore(node=node, actions=actions)

    def _select_action_to_explore(self, node: BUCTNode, actions: List[ProcgenAction]) -> ProcgenAction:
        # Select action to explore according to the Pth percentile of the Qsa
        percentile = self._calculate_node_exploration_percentile(node=node)
        exploration_action = self._select_action_by_percentile(node=node, actions=actions, percentile=percentile,
                                                               use_max_distribution_for_selection=True)
        return exploration_action

    def _backup(self, node: BUCTNode) -> None:
        """
        Perform the backup step in the algorithm - updating the posterior Q(s,a) and V(s)
        :param node: the leaf node from which we start the backup
        """

        # If we expanded a new node we should update posteriors, otherwise only the number of visits should be updated
        should_update_posteriors = node.num_visits == 0

        # Propagate new information in the upwards in the tree
        while node.parent is not None:
            if should_update_posteriors:

                # Calculate posterior of the value of this node
                node.value_posterior = self._calculate_value_posterior(node=node)

                parent_node: BUCTNode = node.parent

                # Update posterior Q(s,a) of the parent node
                action = parent_node.get_action_leading_to_child(node)

                parent_node.qsa_posterior[action] = self._calculate_qsa_posterior(node=parent_node,
                                                                                  action=action,
                                                                                  value_posterior=node.value_posterior)

                # Perform Max-Backup
                node.value_posterior_max = self._calculate_value_posterior_max(node=node)
                parent_node.qsa_posterior_max[action] = self._calculate_qsa_posterior(node=parent_node,
                                                                                      action=action,
                                                                                      value_posterior=node.value_posterior_max)

            node.num_visits += 1
            node: BUCTNode = node.parent

        # Update num_visits also for root node
        node.num_visits += 1

    def _calculate_value_posterior(self, node: BUCTNode) -> ScalarDistribution:
        """
        Calculate the posterior of the value of node. This value is used to propagate new information upwards in the
        tree
        :param node: the node to calculate its new value posterior
        :return: the new value posterior of the node
        """
        # In case this is a new expanded node, calculate its value prior or alternatively its action-value priors
        if node.num_visits == 0:
            self._generate_value_prior(node=node)

        # In case we have only information on the value prior, the value posterior is the value prior
        if len(node.qsa_posterior) == 0:
            return node.value_prior

        # Calculate the value distribution stems from the tree
        # Select action to exploit according to argmax{mean Q(s,a)}
        exploitation_action = self._select_action_by_percentile(node=node,
                                                                actions=node.actions,
                                                                percentile=0.5)

        # the value distribution stems from the tree is the distribution of the Q(s,a) of the exploited action
        value_tree_distribution = node.qsa_posterior[exploitation_action]
        # just return value_tree_distribution
        return value_tree_distribution

    def _calculate_value_posterior_max(self, node: BUCTNode) -> ScalarDistribution:
        """
        Calculate the posterior of the value of node. This value is used to propagate new information upwards in the
        tree
        :param node: the node to calculate its new value posterior
        :return: the new value posterior of the node
        """

        # In case we have only information on the value prior, the value posterior is the value prior
        if len(node.qsa_posterior_max) == 0:
            return node.value_prior

        # Calculate the value distribution stems from the tree
        posterior_distributions = list(node.qsa_posterior_max.values())

        # the value distribution stems from the tree is the distribution of the max of the Q(s,a)
        value_tree_distribution = self._transform_utils.calculate_max_distribution(distributions=posterior_distributions)

        # just return value_tree_distribution
        return value_tree_distribution

    def _calculate_qsa_posterior(self,
                                 node: BUCTNode,
                                 action: ProcgenAction,
                                 value_posterior: ScalarDistribution) -> ScalarDistribution:
        """
        Calculate the posterior of action value given the posterior of the value of the child node
        :param node: the node from which the action is taken
        :param action: the action
        :param value_posterior: the posterior of the value of the child node
        :return: the new action-value posterior
        """
        discount_factor = self._simulator.get_discount_factor()
        return ScalarDistribution.linear_transform(distribution=value_posterior,
                                                   bias=node.rewards[action],
                                                   scale=discount_factor)

    def _generate_qsa_priors(self, node: BUCTNode) -> None:
        """
        Generate the Q(s,a) priors of the node
        :param node: the node from which the the Q(s,a) priors are generated
        """
        if len(node.qsa_prior) == 0:
            # Use neural network to get Q(s,a) prior
            distributions = self._neural_network_predict(node)

            # Assign Q(s,a) priors into the node
            for action, distribution in zip(node.actions, distributions):
                node.qsa_prior[action] = distribution
                node.qsa_posterior[action] = distribution
                node.qsa_posterior_max[action] = distribution  # when the prior is modified, the posterior is initialized with the prior

    def _select_action_by_percentile(self, node: BUCTNode, actions: List[ProcgenAction], percentile: float,
                                     use_max_distribution_for_selection: bool = False) -> ProcgenAction:
        """
        Select the action to explore which has the highest Q(s,a) percentile
        :param node: the node to choose an action from
        :param actions: list of actions to choose from
        :param percentile: the percentile
        :return: the action which has the highest Q(s,a) percentile
        """

        posteriors = node.qsa_posterior_max if use_max_distribution_for_selection else node.qsa_posterior
        if APPROXIMATE_GAUSSIAN_PERCENTILE:
            # Calculate approximately assuming the posterior Q(s,a) are Gaussian
            means = np.array([posteriors[action].expectation for action in actions])
            stds = np.array([posteriors[action].std for action in actions])
            percentile_values = self._transform_utils.calculate_approximate_percentile_for_gaussian(means=means,
                                                                                                    stds=stds,
                                                                                                    percentile=percentile)
        else:
            # Calculate exactly by the inverse CDF
            percentile_values = np.array([posteriors[action].interpolate_inverse_cdf(percentile) for action in actions])

        return actions[np.argmax(percentile_values)]

    def _generate_value_prior(self, node: BUCTNode) -> None:
        """
        Generate the value prior of the node
        :param node: the node
        """
        if node.value_prior is None:

            if node.is_terminal_state:
                # This mode is being used in offline analysis
                node.value_prior = self._transform_utils.create_gaussian_distribution(mean=0., std=MIN_STD)

            else:
                # Use neural network to get value prior
                distributions = self._neural_network_predict(node)
                node.value_prior = distributions[np.argmax([dist.expectation for dist in distributions])]

            node.value_posterior = node.value_prior  # when the prior is modified, the posterior is initialized with the prior

    def _neural_network_predict(self, node: BUCTNode) -> List[ScalarDistribution]:
        # Use neural network to get Q(s,a) prior
        nn_input = torch.from_numpy(np.reshape(node.state.observation, (3, 64, 64))).float() / 255.
        qsa_mean, qsa_std = self._nn.forward(nn_input)

        qsa_mean = qsa_mean.detach().cpu().numpy().ravel()
        if Config().use_gt_std:
            # Estimate the std from the GT qsa
            gt_qsa = [get_gt_qsa(self._simulator, node.state, action) for action in self._simulator.get_actions()]
            qsa_std = np.abs(qsa_mean - gt_qsa)
        else:
            # Extract the std from the NN
            qsa_std = np.exp(qsa_std.detach().cpu().numpy().ravel())
            qsa_std = np.maximum(qsa_std, MIN_STD)  # make sure we don't have zero std

        # Create distributions per action
        return [self._transform_utils.create_gaussian_distribution(qsa_mean[i], qsa_std[i])
                for i in range(len(qsa_mean))]

    @classmethod
    def _calculate_node_exploration_percentile(cls, node: BUCTNode) -> float:
        """
        Calculate exploration percentile which depends on the number of visits of the node
        :param node: the node
        :return: the exploration percentile
        """
        num_visits = min(node.num_visits, MAX_NODE_VISITS)
        return 1. - (1. - Config().select_percentile_init) * np.exp(
            -(num_visits - 1.) / Config().select_percentile_scale)
