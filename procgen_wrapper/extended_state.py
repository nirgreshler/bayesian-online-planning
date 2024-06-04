import dataclasses

import numpy as np


@dataclasses.dataclass
class ExtendedState:
    raw_state: object
    observation: np.array
    is_terminal: bool
    is_solved: bool
    depth: int
    agent_y: int
