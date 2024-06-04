from procgen_wrapper.action_space import ProcgenAction, LeaperAction
from procgen_wrapper.extended_state import ExtendedState
from procgen_wrapper.procgen_simulator import ProcgenSimulator, LEAPER_UP_PENALTY, LEAPER_STEP_PENALTY, LEAPER_REWARD, \
    MAZE_REWARD, MAZE_STEP_PENALTY
from utils.astar import astar
from utils.bfs import bfs


def get_maze_gt_qsa(simulator: ProcgenSimulator, state: ExtendedState, action: ProcgenAction) -> float:
    """
    Return the true Q(s,a) value of a state-action pair in the Maze environment,
    by stepping to the following state and computing its value using BFS search from the following state.

    """
    next_state, reward, is_terminal = simulator.step(state, action)
    if is_terminal:
        return reward
    next_state_value = MAZE_REWARD + bfs(next_state, simulator) * MAZE_STEP_PENALTY
    return float(next_state_value + MAZE_STEP_PENALTY)


def get_leaper_gt_qsa(simulator: ProcgenSimulator, state: ExtendedState, action: ProcgenAction) -> float:
    """
    Return the (estimated) true Q(s,a) value of a state-action pair in the Leaper environment,
    by stepping to the following state and computing its value using A*.

    """
    next_state, reward, is_terminal = simulator.step(state, action)
    if is_terminal:
        return reward
    actions = astar(next_state, simulator)
    rewards = [(LEAPER_UP_PENALTY if a == LeaperAction.Up else LEAPER_STEP_PENALTY)
               for a in actions]

    return LEAPER_REWARD + sum(rewards)


def get_gt_qsa(simulator: ProcgenSimulator, state: ExtendedState, action: ProcgenAction) -> float:
    """
    Calculate GT Q(s,a) values for the Maze and Leaper environments.

    """
    if simulator.env == 'maze':
        return get_maze_gt_qsa(simulator, state, action)
    return get_leaper_gt_qsa(simulator, state, action)
