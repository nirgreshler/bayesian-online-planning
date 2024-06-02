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


class ProcGenSimulator:
    def __init__(self, num_envs=1, num_levels=0, start_level=0, distribution_mode='easy',
                 use_backgrounds=False, restrict_themes=True, env_name: Optional[str] = None,
                 rand_seed: Optional[int] = None):
        # TODO do we need all these?
        self.env_name = env_name
        self._num_env_steps = 6 if self.env_name == 'leaper' else 1
        self.max_simulator_steps = MAX_SIMULATOR_STEPS[env_name]
        if rand_seed is not None:
            rand_seed = int(rand_seed)
        venv = ProcgenEnv(num_envs=num_envs, env_name=env_name, num_levels=num_levels, start_level=start_level,
                          distribution_mode=distribution_mode, use_backgrounds=use_backgrounds,
                          restrict_themes=restrict_themes, rand_seed=rand_seed)
        self.venv = VecNormalize(venv=VecMonitor(venv=VecExtractDictObs(venv, "rgb"), filename=None, keep_buf=100), ob=False, ret=False)

    def step(self, state: ExtendedState, action: ProcgenAction) -> Tuple[
        ExtendedState, float, bool, object]:
        orig_state = self.get_raw_state()
        self.set_raw_state(state.raw_state)

        for _ in range(self._num_env_steps):
            obs, rews, dones, infos = self.venv.step(np.array([action.value]))
            if dones[0]:
                break

        leaper_up_penalty = -0.1  # TODO
        leaper_step_penalty = -0.2


        new_raw_state = self.get_raw_state()
        self.set_raw_state(orig_state)
        depth = state.depth + self._num_env_steps
        is_terminal = dones[0] or (depth == self.max_simulator_steps)
        reward_modification = 0.
        if self.env_name == 'maze':
            reward_modification = -.1
        elif self.env_name == 'leaper':
            reward_modification = leaper_up_penalty if action == LeaperAction.Up.value else leaper_step_penalty
            # if is_terminal and rews[0] < 9.:
            #     reward_modification += Config().procgen.leaper_crash_penalty
        current_reward = rews[0] + reward_modification
        total_reward = state.total_reward + current_reward
        is_solved = is_terminal and current_reward >= 9.
        # new_agent_y = self.get_next_agent_y(state.agent_y, action)
        return ExtendedState(observation=obs[0], raw_state=new_raw_state, is_terminal=is_terminal, is_solved=is_solved,
                             depth=depth, total_reward=total_reward, agent_y=0), \
            current_reward, \
            is_terminal, \
            infos[0]

    def reset(self, context=None) -> ExtendedState:
        self.venv.ret = np.zeros(self.venv.num_envs)
        _rew, ob, first = self.venv.venv.venv.venv.env.observe()
        obs = self.venv.venv.venv.process(ob)
        self.venv.venv.eprets = np.zeros(self.venv.venv.num_envs, 'f')
        self.venv.venv.eplens = np.zeros(self.venv.venv.num_envs, 'i')
        return ExtendedState(observation=self.venv._obfilt(obs)[0], raw_state=self.get_raw_state(), is_terminal=False,
                             is_solved=False, depth=0, total_reward=context['total_cl_reward'] if context else 0., agent_y=0)

    def render_state(self, state: ExtendedState, path=None, overlay_text: str = ''):
        # matplotlib.use('Agg')

        plt.imshow(state.observation)

        # Add overlay text using the text function
        plt.text(5, 5, overlay_text, color='yellow', fontsize=10, fontweight='bold')

        plt.title('ProcGen - ' + self.env_name)

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

    def is_terminal_state(self, state: ExtendedState) -> bool:
        return state.is_terminal

    def get_info(self, state: ExtendedState):
        orig_state = self.get_raw_state()
        self.set_raw_state(state.raw_state)
        info = self.venv.venv.venv.venv.env.get_info()
        self.set_raw_state(orig_state)
        return info

    def get_available_actions(self):
        """
        Extracts list of available actions per given state
        """
        if self.env_name == 'leaper':
            return [e for e in LeaperAction]
        else:
            return [e for e in MazeAction]

    def get_discount_factor(self) -> float:
        return 1.


if __name__ == '__main__':
    # Create a simulator and generate a random environment
    simulator = ProcGenSimulator(env_name='maze', rand_seed=np.random.randint(0, 1000))

    # Reset the simulator
    initial_state = simulator.reset()

    # Perform some actions
    state = initial_state
    for a in [MazeAction.Up, MazeAction.Right, MazeAction.Right, MazeAction.Up]:
        state, _, _, _ = simulator.step(state, a.value)

    # Render the initial and final state
    simulator.render_state(initial_state)
    simulator.render_state(state)

