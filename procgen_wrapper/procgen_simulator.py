from typing import Optional, Tuple

import numpy as np
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecNormalize
)
from matplotlib import pyplot as plt
from procgen import ProcgenEnv

from procgen_wrapper.action_space import LeaperAction, MazeAction, ProcgenAction
from procgen_wrapper.extended_state import ExtendedState

MAX_SIMULATOR_STEPS = {'maze': 500,  # after this number of step, the simulator creates a new sample
                       'leaper': 500}  # TODO this is hard-coded copied from ProcGen's timeout values

PROCGEN_DISTRIBUTION_MODE = 'easy'

MAZE_REWARD = 10.
MAZE_STEP_PENALTY = -1.

LEAPER_UP_PENALTY = -0.1
LEAPER_STEP_PENALTY = -0.2
LEAPER_REWARD = 10.


class ProcgenSimulator:
    def __init__(self, env_name: Optional[str] = None, rand_seed: int = 0):
        # TODO do we need all these?
        self._env_name = env_name
        self._num_env_steps = 6 if self._env_name == 'leaper' else 1  # In leaper, perform 6 "raw" steps for each action
        self._max_simulator_steps = MAX_SIMULATOR_STEPS[env_name]

        # Create the environment
        venv = ProcgenEnv(num_envs=1, env_name=env_name, num_levels=0, start_level=0,
                          distribution_mode=PROCGEN_DISTRIBUTION_MODE, use_backgrounds=False,
                          restrict_themes=True, rand_seed=rand_seed)
        self.venv = VecNormalize(venv=VecMonitor(venv=VecExtractDictObs(venv, "rgb"), filename=None, keep_buf=100),
                                 ob=False, ret=False)

    def step(self, state: ExtendedState, action: ProcgenAction) -> Tuple[
        ExtendedState, float, bool]:
        orig_state = self.get_raw_state()
        self.set_raw_state(state.raw_state)

        for _ in range(self._num_env_steps):
            obs, rews, dones, infos = self.venv.step(np.array([action.value]))
            if dones[0]:
                break

        new_raw_state = self.get_raw_state()
        self.set_raw_state(orig_state)
        depth = state.depth + self._num_env_steps
        is_terminal = dones[0] or (depth == self._max_simulator_steps)
        reward_modification = 0.
        if self._env_name == 'maze':
            reward_modification = MAZE_STEP_PENALTY
        elif self._env_name == 'leaper':
            reward_modification = LEAPER_UP_PENALTY if action == LeaperAction.Up else LEAPER_STEP_PENALTY

        current_reward = rews[0] + reward_modification
        is_solved = is_terminal and current_reward >= MAZE_REWARD + MAZE_STEP_PENALTY
        new_agent_y = self.get_next_agent_y(state.agent_y, action)
        return ExtendedState(observation=obs[0], raw_state=new_raw_state, is_terminal=is_terminal, is_solved=is_solved,
                             depth=depth, agent_y=new_agent_y), \
            current_reward, \
            is_terminal

    def reset(self) -> ExtendedState:
        self.venv.ret = np.zeros(self.venv.num_envs)
        _rew, ob, first = self.venv.venv.venv.venv.env.observe()
        obs = self.venv.venv.venv.process(ob)
        self.venv.venv.eprets = np.zeros(self.venv.venv.num_envs, 'f')
        self.venv.venv.eplens = np.zeros(self.venv.venv.num_envs, 'i')
        return ExtendedState(observation=self.venv._obfilt(obs)[0], raw_state=self.get_raw_state(), is_terminal=False,
                             is_solved=False, depth=0, agent_y=0)

    def render_state(self, state: ExtendedState, path=None):
        # matplotlib.use('Agg')

        plt.imshow(state.observation)

        plt.title('ProcGen - ' + self._env_name)

        # Hide grid lines and axes ticks
        plt.grid(False)
        plt.axis('off')

        if path:
            plt.savefig(path + '.png')
            plt.close()
        else:
            plt.show()

    def get_raw_state(self):
        raw_state = self.venv.venv.venv.venv.env.get_state()[0]
        return raw_state

    def set_raw_state(self, raw_state):
        self.venv.venv.venv.venv.env.set_state([raw_state])

    def get_info(self, state: ExtendedState):
        orig_state = self.get_raw_state()
        self.set_raw_state(state.raw_state)
        info = self.venv.venv.venv.venv.env.get_info()
        self.set_raw_state(orig_state)
        return info

    def get_actions(self):
        if self._env_name == 'leaper':
            return [e for e in LeaperAction]
        else:
            return [e for e in MazeAction]

    def get_discount_factor(self) -> float:
        return 1.

    @property
    def env(self) -> str:
        return self._env_name

    @staticmethod
    def get_next_agent_y(state_agent_y: int, action: ProcgenAction):
        if action == LeaperAction.Up:
            return state_agent_y + 1
        elif action == LeaperAction.Down:
            return max(0, state_agent_y - 1)
        else:
            return state_agent_y


if __name__ == '__main__':
    # Create a simulator and generate a random environment
    simulator = ProcgenSimulator(env_name='maze', rand_seed=np.random.randint(0, 1000))

    # Reset the simulator
    initial_state = simulator.reset()

    # Perform some actions
    state = initial_state
    for a in [MazeAction.Up, MazeAction.Right, MazeAction.Right, MazeAction.Up]:
        state, _, _ = simulator.step(state, a)

    # Render the initial and final state
    simulator.render_state(initial_state)
    simulator.render_state(state)

