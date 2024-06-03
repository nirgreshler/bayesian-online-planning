from abc import ABC, abstractmethod
from typing import Optional, Type, List

import numpy as np

from config.config import Config
from neural_network.procgen_module import ProcgenModule
from planners.tree_node import TreeNode
from procgen_wrapper.action_space import ProcgenAction
from procgen_wrapper.extended_state import ExtendedState
from procgen_wrapper.procgen_simulator import ProcgenSimulator


class PlannerBase(ABC):
    """
    An abstract class for a search-based planner.

    """

    def __init__(self, simulator: ProcgenSimulator, nn_model_path: str, node_type: Type):
        """
        Initialize a search-based planner using a neural network.

        :param simulator: the simulator used in the search process
        :param nn_model_path: the path to load a model for the NN.
        :param node_type: the type of the node used in the search process (subclass of TreeNode).

        """
        self._simulator = simulator
        self._nn = ProcgenModule(env_name=simulator.env_name, init_model_path=nn_model_path)

        self._root_node: Optional[node_type] = None
        self._node_type = node_type

    def plan(self,
             root_state: ExtendedState,
             search_budget: int) -> ProcgenAction:
        root_node = self._search(root_state, search_budget)
        return self._commit_action(root_node)

    def _search(self,
                root_state: ExtendedState,
                max_iterations: int) -> TreeNode:

        self._root_node = self._node_type(state=root_state)
        self._simulator.reset()
        self._simulator.set_raw_state(root_state.raw_state)

        iter_counter = 0

        while iter_counter < max_iterations:
            # Perform selection and expansion stage of MCTS
            node = self._select_and_expand(self._root_node)

            # Perform backup stage of MCTS
            self._backup(node)

            iter_counter += 1

        return self._root_node

    def _select_and_expand(self, node: TreeNode) -> TreeNode:
        """
        Perform the selection and expansion steps in the algorithm - traversing the tree until we add a new node or
        reach a terminal state.
        :param node: the node from which we perform the selection and expansion step
        :return: the leaf node we reach in the end of the selection and expansion steps.
        """
        while not node.is_terminal_state:
            action = self._select(node=node)

            # Action doesn't exist in tree already, need to perform expansion
            if action not in node.explored_actions:
                node = self._expand(node, action)
                # We perform expansion only once
                break

            node = node.get_child(action=action)

        return node

    @abstractmethod
    def _select(self, node: TreeNode) -> ProcgenAction:
        raise NotImplementedError

    def _expand(self, node: TreeNode, action: ProcgenAction) -> TreeNode:
        """
        This method generates a child node given a node and an action
        :param node: the node
        :param action: the action
        :return: the generated child node
        """
        next_state, reward, is_terminal_state, infos = self._simulator.step(state=node.state, action=action)

        next_node = self._node_type(state=next_state, parent=node, is_terminal_state=is_terminal_state)

        node.add_child_with_reward(action=action, reward=float(reward), node=next_node)

        return next_node

    @abstractmethod
    def _backup(self, node: TreeNode) -> None:
        raise NotImplementedError

    @abstractmethod
    def _commit_action(self, root_node: TreeNode) -> ProcgenAction:
        raise NotImplementedError  # TODO test this for both planners

    def _get_actions(self, node: TreeNode) -> List[ProcgenAction]:
        """
        Generate the available actions from the node
        :param node: the node from which the available actions are generated
        """
        if node.available_actions is None:
            node.available_actions = self._simulator.get_actions()
        return node.available_actions

    def _sample_best_action(self, qsa_values: np.ndarray) -> int:
        shifted_qsa_values = qsa_values - np.max(qsa_values)
        scaled_shifted_qsa_values = shifted_qsa_values / Config().action_commitment_softmax_temperature
        # When using fp32 np.exp(-89.) causes underflow errors, this is used to avoid this underflow
        # since the exp_qsa_values are divided by their sum, which can be larger than 1, we use a value of -80
        # which provides enough margin even if their sum is 1500 (i.e 1500 actions with similar qsa).
        scaled_shifted_qsa_values = np.clip(scaled_shifted_qsa_values.astype(np.float32), a_min=-80., a_max=0.)
        exp_qsa_values = np.exp(scaled_shifted_qsa_values)
        action_probs = exp_qsa_values / np.sum(exp_qsa_values)
        return np.random.choice(np.arange(len(action_probs)), p=action_probs)
