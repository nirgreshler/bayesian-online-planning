from typing import Dict, Optional

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

    def add_child_with_reward(self: 'UCTNode', action: ProcgenAction, reward: float, node: 'UCTNode') -> None:
        super().add_child_with_reward(action=action, reward=reward, node=node)
        self._q_sa[action] = INITIAL_QSA_VALUE
