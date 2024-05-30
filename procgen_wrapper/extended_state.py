import dataclasses

import numpy as np


@dataclasses.dataclass
class ExtendedState:
    raw_state: object
    observation: np.array
    is_terminal: bool
    is_solved: bool
    depth: int
    total_reward: float
    agent_y: int

    def __setstate__(self, state):
        self.__dict__.update(state)

        if not hasattr(self, 'total_reward'):  # TODO
            self.total_reward = 0.

        if not hasattr(self, 'agent_y'):  # TODO
            self.agent_y = 0
