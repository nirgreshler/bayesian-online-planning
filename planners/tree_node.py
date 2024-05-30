from typing import Dict, List, Optional

from procgen_wrapper.action_space import ProcgenAction
from procgen_wrapper.extended_state import ExtendedState


class TreeNode:
    def __init__(self,
                 state: ExtendedState,
                 parent: Optional['TreeNode'] = None,
                 is_terminal_state: bool = False):
        self._state = state
        self._parent = parent
        self._is_terminal_state = is_terminal_state
        # self._mdp_experience: Optional[MDPExperience] = None
        self._children: Dict[ProcgenAction, 'TreeNode'] = dict()

        self._rewards: Dict[ProcgenAction, float] = dict()
        self._num_visits = 0

        self._available_actions = None

    @property
    def state(self) -> ExtendedState:
        return self._state

    @property
    def parent(self: 'TreeNode') -> 'TreeNode':
        return self._parent

    @property
    def children(self: 'TreeNode') -> List['TreeNode']:
        return list(self._children.values())

    # @property
    # def mdp_experience(self) -> MDPExperience:
    #     """
    #     Returns the MDP experience from root, i.e. a list of states and actions.
    # 
    #     :return:
    #     """
    # 
    #     if self._mdp_experience is not None:
    #         return self._mdp_experience
    # 
    #     if self.parent is None:
    #         self._mdp_experience = MDPExperience([self.state], [])
    #     else:
    #         leading_action = self.parent.get_action_leading_to_child(self)
    #         self._mdp_experience = self.parent.mdp_experience + \
    #                                MDPExperience([self.parent.state, self.state], [leading_action])
    # 
    #     return self._mdp_experience

    @property
    def is_terminal_state(self) -> bool:
        return self._is_terminal_state

    @property
    def explored_actions(self) -> List[ProcgenAction]:
        return list(self._children.keys())

    @property
    def available_actions(self) -> Optional[List[ProcgenAction]]:
        return self._available_actions

    @available_actions.setter
    def available_actions(self, value: List[ProcgenAction]) -> None:
        self._available_actions = value

    def add_child(self: 'TreeNode', action: ProcgenAction, node: 'TreeNode') -> None:
        assert action not in self._children, 'Action was added to node already'
        self._children[action] = node

    def add_child_with_reward(self: 'TreeNode', action: ProcgenAction, reward: float, node: 'TreeNode') -> None:
        """
        Add a new child node and set reward for the leading action to the new child node
        :param action: the leading action to the new child node
        :param reward: the reward for that action
        :param node: the new child node
        """
        self.add_child(action=action, node=node)
        self._rewards[action] = reward

    def get_child(self: 'TreeNode', action: ProcgenAction) -> 'TreeNode':
        assert action in self._children, 'Unknown action to node'
        return self._children[action]

    def get_action_leading_to_child(self: 'TreeNode', child: 'TreeNode') -> ProcgenAction:
        """
        Given a child node, find the action leading to it
        :param child: to which an action should be found
        :return: the found action
        """
        optional_actions = [action for action, c in self._children.items() if c == child]
        assert len(optional_actions) == 1, "{} actions found leading to child {}, while this has to be 1" \
            .format(len(optional_actions), child)
        action = optional_actions[0]
        return action

    # @property
    # def action_leading_to_self(self) -> Optional[ProcgenAction]:
    #     if not self._parent:
    #         return None
    #
    #     optional_actions = [action for action, c in self.parent._children.items() if c == self]
    #     num_actions = len(optional_actions)
    #     assert num_actions == 1, f"{num_actions} actions found leading to self, while this has to be 1"
    #     action = optional_actions[0]
    #     return action

    # def print_children_data(self):
    #     strings = []
    #     for action, child_node in self._children.items():
    #         strings.append("Action {} : {}".format(action, child_node))
    #
    #     strings = sorted(strings)
    #     for s in strings:
    #         print(s)

    def get_sorted_actions(self) -> List[ProcgenAction]:
        """
        Retrieves the best action according to some marginalization type. Current supported options are maximum number
        of visits (MaxN) and maximum cumulative value MaxN.
        :return:
        """
        raise NotImplementedError()