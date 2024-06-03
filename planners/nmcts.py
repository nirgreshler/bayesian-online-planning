import math
from typing import Optional, List

import numpy as np
import torch

from config.config import Config
from planners.planner_base import PlannerBase
from planners.uct_node import UCTNode
from procgen_wrapper.action_space import ProcgenAction
from procgen_wrapper.procgen_simulator import ProcgenSimulator


class NMCTS(PlannerBase):
    def __init__(self, simulator: ProcgenSimulator, nn_model_path: str):
        super().__init__(simulator, nn_model_path, UCTNode)

    def _commit_action(self, root_node: UCTNode) -> ProcgenAction:
        actions = root_node.available_actions

        # in case we have un-explored actions at the root, add their predicted Q(s,a)
        qsas = dict(zip(actions, list(root_node.predicted_qsa)))
        qsas.update(root_node.q_sa)

        if Config().softmax_action_commitment:
            qsa_values = np.array([qsas[a] for a in actions])
            chosen_idx = self._sample_best_action(qsa_values)
            return actions[chosen_idx]

        return sorted(actions, key=lambda k: qsas[k], reverse=True)[0]

    def _select(self, node: UCTNode) -> ProcgenAction:
        """
        Returns the best action according to pucb
        Comparison is based on the P-UCB or UCB formula.
        :return: the best action
        """

        # Initially, all possible actions are available for exploration
        available_actions = self._get_actions(node)

        # Get normalized Qsa values
        actions_normalized_values = self._get_actions_qsa(node=node, actions=available_actions)
        # Extract the prior probabilities for actions
        actions_probs = self._calculate_prior_probabilities(node=node, actions=available_actions)
        # Extract the number of visits for each action
        actions_num_visits = np.array([node.get_child(action=action).num_visits
                                       if action in node.q_sa else 0.
                                       for action in available_actions])
        # Calculate exploration term, based on prior probability and number of visits, for each action
        action_exploration_terms = self._calculate_exploration_terms_pucb(actions_priors=actions_probs,
                                                                          num_visits=node.num_visits,
                                                                          actions_num_visits=actions_num_visits)
        # Calculate the P-UCB value for each action
        action_pucb_values = actions_normalized_values + action_exploration_terms

        # Choose best action
        max_action_idx = int(np.argmax(action_pucb_values))
        return available_actions[max_action_idx]

    def _backup(self, node: UCTNode) -> None:
        node.num_visits += 1
        backed_up_value = self._calculate_value(node=node)

        while node.parent is not None:

            child = node
            node: UCTNode = node.parent

            node.num_visits += 1

            action = node.get_action_leading_to_child(child)

            # Use Bellman's equation to calculate the bootstrapped Q value to update parent node
            r = node.rewards[action]
            discount_factor = self._simulator.get_discount_factor()
            target_qsa = r + discount_factor * backed_up_value

            # update q_sa, the value is the average over time
            node.q_sa[action] += (target_qsa - node.q_sa[action]) / min(child.num_visits, 1)

            # update the backup value q
            backed_up_value = self._calculate_value(node)

    def _get_actions_qsa(self, node: UCTNode, actions: List[ProcgenAction]) -> np.ndarray:
        """
        Calculates normalized action Q(s,a) values so it can be used in PUCB. Unexplored actions get zero normalized
        Q(s,a)
        :param node: Node with actions to normalize
        :param actions: List of actions to calculate Qsas for
        :return: numpy array of normalized values
        """
        # For explored actions, take the measured Q(s,a)
        explored_actions_cond = np.array([action in node.q_sa for action in actions])
        explored_actions_qsas = np.array([node.q_sa[actions[action_idx]]
                                          for action_idx in np.where(explored_actions_cond)[0]])

        # For un-explored actions, take the predicted Q(s,a) from the NN
        actions_qsas = self._calculate_predicted_qsa(node=node)
        actions_qsas[explored_actions_cond] = explored_actions_qsas

        return actions_qsas

    def _calculate_prior_probabilities(self, node: UCTNode, actions: List[ProcgenAction]) -> np.ndarray:
        """
        Extract prior probabilities for the given actions. These probabilities can be used in PUCB. In case we don't use
        the policy network, we extract a uniform probability distribution
        :param node:
        :param actions: a list of actions
        :return: prior probabilities for the given actions
        """

        # Use network to extract initial Q(s,a)
        predicted_qsa = self._calculate_predicted_qsa(node=node)

        # Extract Qsa values for given actions
        actions_indices = []
        for action in actions:
            actions_indices.append(node.available_actions.index(action))
        actions_qsa_values = predicted_qsa[actions_indices]

        # Convert to prior probabilities
        shifted_qsa_values = actions_qsa_values - np.max(actions_qsa_values)
        scaled_shifted_qsa_values = shifted_qsa_values / Config().policy_network_temperature_scaling
        # When using fp32 np.exp(-89.) causes underflow errors, this is used to avoid this underflow
        # since the exp_qsa_values are divided by their sum, which can be larger than 1, we use a value of -80
        # which provides enough margin even if their sum is 1500 (i.e 1500 actions with similar qsa).
        scaled_shifted_qsa_values = np.clip(scaled_shifted_qsa_values.astype(np.float32), a_min=-80., a_max=0.)
        exp_qsa_values = np.exp(scaled_shifted_qsa_values)
        action_probs = exp_qsa_values / np.sum(exp_qsa_values)
        return action_probs

    @classmethod
    def _calculate_exploration_terms_pucb(cls,
                                          actions_priors: np.ndarray,
                                          num_visits: int,
                                          actions_num_visits: np.ndarray) -> np.ndarray:
        """
        Calculates the exploration terms of multiple actions
        :param actions_priors: numpy array with prior for every action
        :param num_visits: number of visits for the parent node
        :param actions_num_visits: number of visits per action
        :return: numpy array of exploration terms for all actions, including PUCB constant
        """
        action_exploration_terms = actions_priors * math.sqrt(1e-3 + num_visits) / (1. + actions_num_visits)
        effective_branching_factor = np.exp(-np.sum(actions_priors * np.log(actions_priors + 1e-12)))
        return Config().pucb_constant * effective_branching_factor * action_exploration_terms

    def _calculate_predicted_qsa(self, node: UCTNode) -> np.ndarray:
        if node.predicted_qsa is None:
            nn_input = torch.from_numpy(np.reshape(node.state.observation, (3, 64, 64))).float() / 255.
            qsa_mean, _ = self._nn.forward(nn_input)

            # Squeeze the batch dimension if needed
            node.predicted_qsa = qsa_mean.detach().cpu().numpy().ravel()

        return node.predicted_qsa.copy()

    def _calculate_value(self, node: UCTNode) -> Optional[float]:
        """
        :return: Returns the value of this node in case of an optimal policy, i.e., maxQ(s,a)
        """
        if node.is_terminal_state:
            return 0.
        # we explored this node before, return the max Q(s,a) over its actions
        return np.max(self._get_actions_qsa(node, self._get_actions(node)))


if __name__ == '__main__':
    simulator = ProcgenSimulator(env_name='maze', rand_seed=1)
    init_state = simulator.reset()
    model_path = '../neural_network/models/Maze/model_1/maze_1.bin'
    planner = NMCTS(simulator=simulator, nn_model_path=model_path)
    search_iterations = 100
    action = planner.plan(init_state, search_budget=search_iterations)
