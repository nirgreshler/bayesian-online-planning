from typing import Dict, Optional, List

import numpy as np

from planners.tree_node import TreeNode
from procgen_wrapper.action_space import ProcgenAction
from procgen_wrapper.extended_state import ExtendedState

INITIAL_QSA_VALUE = -100000.


class UCTNode(TreeNode):
    def __init__(self, state: ExtendedState, parent: 'UCTNode' = None, is_terminal_state: bool = False):
        super().__init__(state=state, parent=parent, is_terminal_state=is_terminal_state)

        self._q_sa: Dict[ProcgenAction, float] = dict()
        self._predicted_qsa = None

    @property
    def rewards(self) -> Dict[ProcgenAction, float]:
        return self._rewards

    @property
    def q_sa(self) -> Dict[ProcgenAction, float]:
        return self._q_sa

    @property
    def predicted_qsa(self) -> Optional[np.ndarray]:
        return self._predicted_qsa

    @predicted_qsa.setter
    def predicted_qsa(self, value: np.ndarray):
        self._predicted_qsa = value

    # @property
    # def predicted_value(self) -> Optional[float]:
    #     return self._predicted_value
    #
    # @predicted_value.setter
    # def predicted_value(self, value: float):
    #     self._predicted_value = value

    @property
    def num_visits(self) -> int:
        return self._num_visits

    @num_visits.setter
    def num_visits(self, value):
        self._num_visits = value

    def add_child_with_reward(self: 'UCTNode', action: ProcgenAction, reward: float, node: 'UCTNode') -> None:
        super().add_child(action=action, node=node)
        self._rewards[action] = reward
        self._q_sa[action] = INITIAL_QSA_VALUE

    def get_sorted_actions(self) -> List[ProcgenAction]:
        """
        Retrieves the best action according to some marginalization type. Current supported options are maximum number
        of visits (MaxN) and maximum cumulative value MaxN.
        :param also_unexplored: whether to sort by the NN prediction or by the measured qsa till now.
        :return:
        """
        actions = self.available_actions
        qsas = dict(zip(actions, list(self.predicted_qsa)))
        qsas.update(self.q_sa)
        return sorted(actions, key=lambda k: qsas[k], reverse=True)
