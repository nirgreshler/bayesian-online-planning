from collections import deque
from typing import Optional

from procgen_wrapper.extended_state import ExtendedState
from procgen_wrapper.procgen_simulator import ProcgenSimulator
from utils.hash_2d_array import hash_2d_array


class BFSNode:
    def __init__(self, state: ExtendedState, parent: Optional['BFSNode'] = None):
        self.state = state
        self.hash = hash_2d_array(state.observation)
        self.parent = parent

    def __eq__(self, other):
        return self.hash == other.hash

    def __hash__(self):
        return self.hash

    @property
    def depth(self):
        d = 0
        node = self
        while node.parent is not None:
            d += 1
            node = node.parent
        return d


def bfs(start_state: ExtendedState, simulator: ProcgenSimulator) -> int:
    visited = set()
    queue = deque([BFSNode(start_state)])

    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            # Add unvisited neighbors to the queue
            for a in simulator.get_actions():
                neighbor_state, _, is_terminal = simulator.step(node.state, a)
                if is_terminal:
                    # Return the number of steps to reach the goal
                    return node.depth + 1
                new_node = BFSNode(neighbor_state, parent=node)
                if new_node not in visited:
                    queue.append(new_node)
